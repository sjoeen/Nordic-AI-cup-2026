# Session record: medical-appointment build, 2026-09-17 to 2026-09-18

Everything done in this Claude Code session, in order, with the numbers as
measured. Companion documents: `reports/project_state.md` (current state and
decisions), `reports/implementation_plan.md` (interface contract),
`reports/contract_and_security.md`, `reports/error_analysis.md`,
`reports/final_readiness.md`, `reports/how_to_run.md`.

## 1. Starting point and authorization

- Input: the challenge folder (README, `api.py`, `dtos.py`, `example.py`,
  `utils.py`, `local_evaluator.py`, 39 MP3s, `question_train.csv`) and the
  user's brief `claude_medical_appointment_plan.md`.
- User instruction: "Start building based on the md files ... I want every
  part of the system to be able to be tested individually." Treated as BUILD
  authorization. Later: ultracode on, then switched to xhigh effort with
  workflows off; hardware for the serving host given at the end: 6 CPU cores,
  16 GB RAM, 320 GB disk, no GPU.
- Snapshot audit: all nine SHA-256 fingerprints in the plan's Appendix A match
  the working tree. Dataset: 39 recordings, 390 questions (195 positive, 142
  hard negative, 53 off topic), gold spans 0.16 to 14.2 s (median 2.88 s),
  longest clip `sample_20` at 231.9 s.
- Reference files kept byte-identical: `api.py`, `dtos.py`, `utils.py`,
  `local_evaluator.py`. Nothing committed or pushed.

## 2. Environment

| Item | Value |
| --- | --- |
| Dev machine | WSL2, Intel i7-1165G7 (4 cores / 8 threads), 7.6 GB RAM, no NVIDIA GPU visible |
| Python | 3.12.3 in `./.venv` (`bash setup_env.sh`), 1.7 GB |
| Stack | torch 2.14 CPU, transformers 4.57.6, sentence-transformers 5.7.0, faster-whisper 1.2.1, ctranslate2 4.8.2, PyAV 18, pydantic 2.13.5, fastapi 0.141.1, pytest 9, httpx, llama-cpp-python 0.3.35 (built from source, optional) |
| Files added for this | `requirements-ml.txt`, `setup_env.sh`, `prefetch_models.py`, `pytest.ini`, `.gitignore` entries (`work/`, `.venv/`, `results/runs/`) |
| Weights on disk | whisper base and small (HF cache), bge-small-en-v1.5, nli-deberta-v3-small/base, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli, deberta-v3-large-zeroshot-v2.0; Qwen2.5 1.5B/3B GGUF under `work/models/` (3.1 GB, probe only) |

## 3. Benchmarks that shaped the design (dev CPU)

| Measurement | Result |
| --- | --- |
| faster-whisper base int8, 4 threads | RTF 0.035; longest clip 8 s; RSS 0.7 GB |
| faster-whisper small int8, 4 threads | RTF 0.10 to 0.16; longest clip 36 s; RSS 1.2 GB |
| Evidence ceiling on cached transcripts (all 195 positives) | best single unit 0.71 (small) / 0.74 (base); best run of <=3 units 0.82 / 0.85; best word window 0.95 / 0.98; no gold span unreachable |
| Local LLM via llama.cpp, 4 clips | Qwen2.5-1.5B 75 to 84 s per clip, acc 0.425, broken JSON; Qwen2.5-3B 112 to 160 s per clip, acc 0.675. Both over the 60 s budget: LLM kept as a GPU-host option only |
| NLI model comparison (1860 cached pairs, raw questions) | nli-deberta-v3-small 0.813 (20 pairs/s); nli-deberta-v3-base 0.784 (8/s); MoritzLaurer base mnli-fever-anli 0.890 (10/s); deberta-v3-large-zeroshot 0.894 (3/s). Chosen: MoritzLaurer base |
| Learned decision layer, grouped 5-fold CV over dev transcripts | accuracy 0.726 (rule) to 0.868 (r001 features) to 0.923 (r003 features); window ranking tIoU 0.367 (max entailment) to 0.507; oracle among candidates 0.679 |

## 4. Architecture built

Request path: `audio_io.decode_request_audio` (strict base64, 32 MiB / 30 min
bounds, PyAV decode to 16 kHz) -> `asr_backend.FasterWhisperBackend`
(lazy segment generator, stops consuming 12 s before the deadline) ->
`transcript.build_context` (clause-sized units cut at punctuation and pauses,
global word ids, anomaly diagnostics) -> `qa_backend.RetrievalNLIBackend`
(bge-small cosine + lexical retrieval of top-6 windows of <=3 units,
`question_rewrite.to_statement` as NLI hypothesis, MoritzLaurer NLI,
`factcheck.compare` gate, `calibration` learned decision and window pick) ->
`evidence.select_span` (contiguous run, filler trim, clamp) ->
`response_checks.finalize` + always-on `strict_validate` and wire-JSON parse
with emergency fallback -> `ASRQuestionResponseDto`. `pipeline.Pipeline`
owns lifecycle, warm-up at import, the monotonic `Deadline`, stage timing and
the fallback ladder (QA failure -> `LexicalBackend`; ASR/decode failure ->
constant answers with null spans). `example.predict` never raises.

Modules and sizes (lines): core_types 315, audio_io 452, transcript 675,
asr_backend 878, question_rewrite 867, factcheck 1241, qa_backend 1036,
qa_llm_backend 904 (LLM engines: transformers / llama_cpp / fake),
evidence 687, response_checks 583, calibration 311, pipeline 426,
example 48, prefetch_models 25. Configs: `default.json` (dev),
`prod_cpu.json` (the stated serving host, 6 ASR threads),
`prod_gpu.json` (CUDA starting point, unmeasured), `test_fake.json`,
`calibration_v1.json`. Prompt: `prompts/qa_v1.txt`.

Evaluation tooling (`eval_tools/`): `scoring` (parity with
`local_evaluator.Statistics`, full-denominator accounting), `split` (grouped
31/8 split, seed 2026, `sample_20` forced into dev), `transcribe_all`,
`import_raw_whisper`, `offline_eval`, `replay` (split-aware HTTP replay through
the unchanged `local_evaluator.replay`), `evidence_oracle`, `error_analysis`,
`fit_calibration`, `freeze_env`, `bench_asr`.

## 5. Tests

- 15 test files, about 1,060 fast tests (no real models) plus 9 real-model
  tests (`RUN_SLOW=1`), all passing at the end of the session.
- Each stage runs alone: `.venv/bin/python -m pytest tests/test_<module>.py`.
- Integration: `tests/test_pipeline.py` (fake ASR/QA end to end, fallback
  ladder, HTTP round trip through the real FastAPI app, config env overrides,
  boot-time config validation, final-body guard); `tests/test_security.py`
  (7 adversarial fixtures in `security_fixtures/transcript_attacks_v1.json`,
  passing with the lexical backend and with the real NLI backend);
  `tests/test_eval_tools.py` (scoring parity on synthetic records and on the
  gold oracle, denominators, split determinism and group integrity).

## 6. Multi-agent work

- Workflow `build-medical-appointment-modules`: 10 builders (one per module,
  disjoint file ownership) each followed by an adversarial reviewer. It was
  interrupted by usage-limit resets three times; 11 of 18 agents finished
  (builders for all 10 modules effectively landed their files; reviews of
  transcript, asr_backend, audio_io and qa_llm_backend completed). Missing
  test files for `question_rewrite` and `eval_tools` were written by the
  integrator. Subagent tokens for the workflow: about 3.0 M.
- After switching to xhigh effort, three single reviewer agents were run for
  evidence, response_checks and factcheck. Bugs fixed with regression tests:
  transcript 7, asr_backend several (env parsing, corrupt-cache validation),
  audio_io (FLAC-silence decode bomb), evidence 6, response_checks 5,
  factcheck 16 (ten classes of false contradictions, quadratic path made
  linear: 3000-word window 63 ms to 32 ms). Reviews of qa_backend,
  question_rewrite and eval_tools were done by the integrator only.

## 7. Measured runs (dev split = 31 conversations / 310 questions, holdout = 8 / 80)

| Run | Setting | Accuracy | Mean tIoU | Score |
| --- | --- | --- | --- | --- |
| dummy baseline (analytical) | constant true, no spans | 0.500 | 0.000 | 0.200 |
| r001 | retrieval + nli-deberta-v3-small, raw questions, rules | 0.726 | 0.228 | 0.427 |
| r002 | lexical fallback only | 0.684 | 0.249 | 0.423 |
| r003 small | MoritzLaurer NLI + rewriter, rules, whisper small | 0.810 | 0.259 | 0.479 |
| r003 base | same, whisper base | 0.813 | 0.242 | 0.470 |
| r004 | + calibration_v1 (fitted on r003 small; in-sample for dev) | 0.919 | 0.464 | 0.646 |
| r005 | r004 after the three reviews | 0.919 | 0.464 | 0.646 |
| e2e_dev_001 | HTTP, unchanged evaluator, dev, C1 | 0.935 | 0.501 | 0.675 |
| e2e_holdout_001 | HTTP, holdout opened once, C1 | 0.925 | 0.549 | 0.699 |
| full_local_eval_001 | `python local_evaluator.py`, all 39 | 0.933 | 0.511 | 0.680 |

Per type in the full run: positive 0.949 (185/195), hard negative 0.915
(130/142), off topic 0.925 (49/53); no span returned 10/195. Round trip mean
28.8 s, worst 52.0 s (sample_20: ASR 39 s + QA 9 s, with reviewer agents
loading the CPU); 0 timeouts, 0 failed conversations in every HTTP run. Peak
serving-process RSS 2.2 GB. Oracle: 1.000.

## 8. Decisions

1. D1 ASR default whisper `small` int8 (fits the budget on the longest clip
   with ~8 s to spare on the 4-core box); `base` is the documented fallback
   at -0.009 offline score.
2. D2 QA on CPU = retrieval + NLI + fact check; D4 LLM answering is a GPU-host
   option only; D5 NLI = MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli.
3. D3 Evidence = the QA-selected contiguous run, `best_run` mode, no padding.
4. D6 Candidate C1 frozen 2026-09-18 03:30 (hashes in
   `manifests/C1_source_hashes.txt`, environment in
   `manifests/env_C1_candidate.json`); the holdout was opened once for it.
5. Always-on final response guard; boot-time validation of the resegment
   config; filename is metadata only; labels never reach inference code.

## 9. Error analysis and experiment queue (from `reports/error_analysis.md`)

Dev losses: 17 missed positives, 7 hard negatives answered yes, 1 off topic
answered yes, 51 positives localized in the wrong place (typically medicines
mentioned several times where the gold is the confirming turn), 51 with
partial overlap. Queue: E-A richer evidence candidate pool and last-mention
features (largest lever), E-B extra decision features (drug-name mismatch),
E-C dual hypotheses, E-D larger whisper on a stronger host, E-E timing margin
per host.

## 10. Git state at the end

- Modified tracked files: `example.py` (thin adapter), `Dockerfile` (copies
  the modules and prefetches weights), `.gitignore`.
- New, untracked: the modules listed in section 4, `tests/`, `eval_tools/`,
  `configs/`, `prompts/`, `manifests/`, `reports/`, `results/index.csv` and
  `results/full_local_eval_001.txt`, `security_fixtures/`,
  `requirements-ml.txt`, `setup_env.sh`, `prefetch_models.py`, `pytest.ini`.
  `results/runs/` and `work/` are gitignored (per-run records, caches,
  weights).
- Not committed, not pushed, no official attempt made, deployment untouched.

## 11. Commands used (reproducible)

```bash
bash setup_env.sh
.venv/bin/python -m pytest                                   # fast suite
RUN_SLOW=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -m slow -o addopts=""
.venv/bin/python -m eval_tools.split --write
.venv/bin/python -m eval_tools.import_raw_whisper --model small   # or transcribe_all
.venv/bin/python -m eval_tools.offline_eval --transcripts work/transcripts/<hash> --split dev \
    --qa-backend retrieval_nli --qa-option calibration_path=configs/calibration_v1.json --run-id <id>
.venv/bin/python -m eval_tools.fit_calibration --records results/runs/<dev run>/records.jsonl \
    --transcripts work/transcripts/<hash> --out configs/calibration_v1.json
.venv/bin/python -m eval_tools.error_analysis results/runs/<id>/records.jsonl
HF_HUB_OFFLINE=1 .venv/bin/python api.py                     # serve
.venv/bin/python -m eval_tools.replay --split dev --run-id <id>
.venv/bin/python local_evaluator.py ; .venv/bin/python local_evaluator.py --oracle
.venv/bin/python -m eval_tools.freeze_env --run-id <id>
```

## 12. Open items for the user

- Run `eval_tools.replay --split dev` on the 6-core serving host with
  `MA_CONFIG=configs/prod_cpu.json`; if the worst round trip is above ~50 s,
  set `MA_ASR_MODEL_SIZE=base`.
- Verify external reachability of `http://<host>:9054/predict` and the
  request body limit (bodies up to ~5 MB).
- Official Verify, validation and the single evaluation attempt need explicit
  authorization; none has been made.
- Decide what to commit (suggested: everything except `work/`, `.venv/`,
  `results/runs/`).
