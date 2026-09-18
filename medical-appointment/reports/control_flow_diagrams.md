# Control flow diagrams

Mermaid diagrams of the system at three levels: the request path as the
evaluator sees it, the pipeline's stage sequencing with its fallback ladder,
and the internals of the QA and evidence stages. The last two diagrams show
the development loop (offline evaluation, calibration, replay) and the
multi-agent build workflow used in the session. Render with any Mermaid
viewer (GitHub, VS Code, mermaid.live).

## 1. High level: one request

```mermaid
flowchart LR
    EV[Evaluation service] -->|"POST /predict: audio_base64, audio_filename, questions"| API["api.predict_endpoint"]
    API --> EX["example.predict<br/>never raises"]
    EX --> PIPE["pipeline.Pipeline.predict_with_trace"]
    PIPE --> DTO["ASRQuestionResponseDto<br/>answers, evidence_start, evidence_end"]
    DTO --> VAL["utils.validate_response"]
    VAL -->|"JSON: booleans, numbers, nulls"| EV
    EX -. "any exception" .-> EMG["response_checks.emergency_response<br/>constant answers, null spans"]
    EMG --> DTO
```

## 2. Pipeline stages and fallback ladder

```mermaid
flowchart TD
    S(["request"]) --> DL["Deadline: 52 s budget, 1.5 s reserve"]
    DL --> DEC{"audio_io.decode_request_audio"}
    DEC -->|"AudioDecodeError, oversize, too long"| EMG["emergency_response<br/>fallback answer, null spans"]
    DEC -->|"waveform 16 kHz, duration, sha256"| ASR["asr.transcribe<br/>lazy whisper segments<br/>stop when remaining < 12 s"]
    ASR -->|"exception"| CTX
    ASR -->|"raw units"| CTX["transcript.build_context<br/>resegment, global word ids, anomalies"]
    CTX -->|"no units"| NT["constant answers, null spans"]
    CTX -->|"AudioContext"| QA["qa.answer(context, questions, deadline)"]
    QA -->|"exception or wrong count"| LEX["LexicalBackend fallback"]
    LEX -->|"exception"| CONST["constant QAResults"]
    QA -->|"n QAResults"| EVD
    LEX -->|"n QAResults"| EVD
    CONST --> EVD["evidence.select_span<br/>for every True answer"]
    EVD --> FIN["response_checks.finalize<br/>False -> nulls, clamp, round"]
    NT --> FIN
    FIN --> GRD{"strict_validate<br/>+ wire JSON parse"}
    GRD -->|"ok"| OUT(["response + trace"])
    GRD -->|"violation"| EMG
    EMG --> OUT
```

## 3. QA stage: RetrievalNLIBackend

```mermaid
flowchart TD
    Q["questions"] --> PREP["prepare each question<br/>question_rewrite.to_statement<br/>focus_terms, is_existence_question"]
    C["AudioContext units"] --> WIN["transcript.windows<br/>every run of 1..3 units"]
    WIN --> EMB["bge-small embeddings<br/>windows and questions"]
    PREP --> EMB
    EMB --> SIM["combined = 0.6 cosine + 0.4 lexical overlap"]
    SIM --> TOPK["top-6 windows per question"]
    TOPK --> FITS{"deadline.fits(3 s)?"}
    FITS -->|"no"| DEG["degrade: LexicalBackend decisions<br/>diagnostics.degraded = true"]
    FITS -->|"yes"| NLI["NLI cross-encoder<br/>(window text, statement) pairs, one batch"]
    NLI --> BEST["best window by entailment"]
    BEST --> FC["factcheck.compare<br/>question vs window + neighbours"]
    FC --> CAL{"calibration loaded?"}
    CAL -->|"yes"| FEAT["question features:<br/>p_ent max/second, p_con, cosine,<br/>lexical, existence, verdict one-hot"]
    FEAT --> LR["logistic decision: p_yes"]
    LR --> HG{"verdict = contradiction<br/>and hard gate on?"}
    HG -->|"yes"| NO["answer = False"]
    HG -->|"no"| ANS["answer = p_yes >= 0.5"]
    ANS --> RANK["linear ranker over candidate windows<br/>p_ent, cosine, lexical, n_units, duration, rank"]
    RANK --> PICK["evidence_unit_ids = top ranked window"]
    CAL -->|"no"| RULE["rule: p_ent >= 0.5 and p_ent > p_con<br/>and factcheck gate, existence uses similarity threshold"]
    RULE --> RES["QAResult per question<br/>answer, evidence_unit_ids, confidence, diagnostics"]
    PICK --> RES
    NO --> RES
    DEG --> RES
    RES --> INV["invariant: exactly len(questions) results, in order"]
```

## 4. Evidence stage: unit ids to seconds

```mermaid
flowchart TD
    R["QAResult: answer True, evidence_unit_ids"] --> VAL{"ids present in context?"}
    VAL -->|"none valid or answer False"| NONE(["None: no span invented"])
    VAL -->|"valid positions"| RUNS["group adjacent positions into runs"]
    RUNS --> MODE{"policy.mode"}
    MODE -->|"union"| BOUNDS
    MODE -->|"first_run / longest_run"| BOUNDS
    MODE -->|"best_run"| DENS["run with highest focus-term density"] --> BOUNDS["bounds from units<br/>or from word ids if all inside"]
    BOUNDS --> FILL["trim filler words at the edges<br/>um, okay, so, yeah ..."]
    FILL --> LONG{"longer than max_span_s?"}
    LONG -->|"yes"| TF["trim_to_focus: sub-run<br/>with most focus terms that fits"] --> PAD
    LONG -->|"no"| PAD["pad, clamp to [0, duration],<br/>expand to min_span_s"]
    PAD --> CHK{"0 <= start < end <= duration?"}
    CHK -->|"yes"| SPAN(["(start, end) rounded to 3 dp"])
    CHK -->|"no"| NONE
```

## 5. One request in time: deadline handling

```mermaid
sequenceDiagram
    participant Ev as Evaluator
    participant Api as api.py
    participant Pipe as Pipeline
    participant ASR as FasterWhisperBackend
    participant QA as RetrievalNLIBackend
    participant Evd as evidence
    participant RC as response_checks
    Ev->>Api: POST /predict (one conversation, ten questions)
    Api->>Pipe: predict_with_trace(audio_base64, questions, filename)
    Pipe->>Pipe: Deadline(52 s) starts, then decode audio
    Pipe->>ASR: transcribe(waveform, 16000, deadline, cache_key=sha256)
    loop next 30 s whisper window
        ASR->>ASR: pull next segment (lazy generator)
        ASR-->>ASR: stop if deadline.remaining() < 12 s (truncated=true)
    end
    ASR-->>Pipe: units with word timings
    Pipe->>Pipe: build_context (resegment)
    Pipe->>QA: answer(context, questions, deadline)
    QA->>QA: retrieval, then NLI only if deadline.fits(3 s), else lexical
    QA-->>Pipe: ten QAResults
    Pipe->>Evd: select_span for each True answer
    Evd-->>Pipe: spans or None
    Pipe->>RC: finalize, strict_validate, wire JSON check
    RC-->>Pipe: DTO (or emergency response on violation)
    Pipe-->>Api: DTO + stage times logged
    Api->>Api: utils.validate_response
    Api-->>Ev: JSON body
```

## 6. Development loop: offline evaluation, calibration, replay

```mermaid
flowchart LR
    DATA["data/audio + question_train.csv"] --> SPLIT["eval_tools.split<br/>manifests/split_v1.json<br/>31 dev / 8 holdout"]
    DATA --> TR["eval_tools.transcribe_all<br/>or import_raw_whisper"]
    TR --> CACHE["work/transcripts/config_hash/contexts/*.json"]
    CACHE --> OFF["eval_tools.offline_eval<br/>QA + evidence, minutes not hours"]
    SPLIT --> OFF
    OFF --> REC["results/runs/run_id/records.jsonl<br/>summary.json, report.txt"]
    REC --> FIT["eval_tools.fit_calibration<br/>grouped 5-fold CV, dev records only"]
    FIT --> CAL["configs/calibration_v1.json"]
    CAL --> OFF
    REC --> ERR["eval_tools.error_analysis"]
    ERR --> QUEUE["reports/error_analysis.md<br/>experiment queue"]
    SRV["api.py server<br/>configs/default.json or prod_cpu.json"] --> REP["eval_tools.replay<br/>unchanged local_evaluator.replay<br/>split aware, full denominators"]
    SPLIT --> REP
    REP --> IDX["results/index.csv"]
    OFF --> IDX
    SRV --> FULL["python local_evaluator.py<br/>all 39 conversations"]
    FREEZE["freeze: manifests/C1_source_hashes.txt<br/>eval_tools.freeze_env"] --> HOLD["replay --split holdout, once"]
```

## 7. Build workflow used in the session

```mermaid
flowchart TD
    USER["user: start building, every part testable"] --> LEAD["integrator (this session)<br/>contract: core_types.py, reports/implementation_plan.md"]
    LEAD --> BUILD["10 builder agents in parallel<br/>one module + its tests each, disjoint files"]
    BUILD --> REV["adversarial reviewer per module<br/>probe, fix confirmed bugs, add regression tests"]
    REV --> INT["integrator: pipeline.py, example.py, configs,<br/>tests/test_pipeline.py, tests/test_security.py"]
    INT --> MEAS["measure: offline runs r001..r005,<br/>NLI comparison, LLM probe, calibration"]
    MEAS --> E2E["HTTP replay: dev 0.675, holdout 0.699, full 0.680"]
    E2E --> FREEZE["freeze C1, readiness checklist, session record"]
    REV -. "interrupted by usage limits" .-> LATE["3 reviewers re-run later<br/>evidence, response_checks, factcheck"]
    LATE --> INT
```
