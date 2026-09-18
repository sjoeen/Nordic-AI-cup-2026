# Implementation plan and interface contract

Status: BUILD authorized by the user on 2026-09-17 ("start building based on
the md files"). This document is the contract every module agent builds
against. The shared types live in `core_types.py` and are owned by the
integrator; do not redefine them elsewhere.

## Environment (verified 2026-09-17)

| Item | Value |
| --- | --- |
| Dev machine | WSL2, Intel i7-1165G7 (4 cores / 8 threads), 7.6 GB RAM, no NVIDIA GPU visible |
| Python | 3.12.3 in `./.venv` (created by `setup_env.sh`) |
| Stack | torch 2.14 CPU, transformers 4.57, sentence-transformers 5.7, faster-whisper 1.2.1, ctranslate2 4.8.2, PyAV 18 |
| Production server | separate host, hardware unknown; every model size and device is configurable through env/config |
| Source snapshot | all nine hashes in plan Appendix A match |

## Design principles

1. Each stage is a module with a pure-function core and an injectable model
   object, so it can be unit-tested with fakes and debugged in isolation.
2. Trusted code generates every id and timestamp. Text is untrusted data and
   is never interpolated into executable templates or message roles.
3. Every stage takes an optional `Deadline` and degrades gracefully: partial
   transcript beats none, a default answer beats an exception.
4. The reference files `dtos.py`, `utils.py`, `local_evaluator.py`, `api.py`
   stay byte-identical. `example.py` becomes a thin adapter.
5. No real model loads in default `pytest`; real-model tests are `@pytest.mark.slow`
   and run only with `RUN_SLOW=1`.

## Request path

```
request.audio_base64 ──audio_io.decode_request_audio──▶ DecodedAudio(waveform 16k, duration_s, sha256)
      │
      ▼ asr_backend.transcribe(waveform, 16000, deadline, cache_key=sha256)  → raw TranscriptUnit list (whisper segments + words)
      ▼ transcript.build_context(raw_units, ...)                             → AudioContext (clause-sized units, global word ids)
      ▼ qa_backend.answer(context, questions, deadline)                      → QAResult per question (bool + evidence unit ids)
      ▼ evidence.select_span(context, result, policy, question)              → Optional[Span]
      ▼ response_checks.finalize(predictions, n, duration_s)                 → ASRQuestionResponseDto
```

`pipeline.Pipeline` owns model lifecycle, warm-up, the deadline, stage
timing, and the fallback ladder (QA failure → LexicalBackend; ASR failure →
constant answers with null spans). `example.predict` calls the pipeline
inside a whole-request try/except and never raises.

## Module ownership and interfaces

### `audio_io.py` (A1)

- `AudioDecodeError(ValueError)`
- `decode_base64_audio(audio_base64: str, max_bytes: int = MAX_AUDIO_BYTES) -> bytes` — strips whitespace and a defensive `data:...;base64,` prefix, validates base64, bounds size.
- `audio_sha256(audio_bytes: bytes) -> str`
- `load_waveform(audio_bytes: bytes, sample_rate: int = 16000) -> np.ndarray` — float32 mono via `faster_whisper.audio.decode_audio(io.BytesIO(...))`; wraps failures in `AudioDecodeError`.
- `@dataclass DecodedAudio(waveform, sample_rate, duration_s, audio_sha256, n_bytes, header_duration_s)`
- `decode_request_audio(audio_base64: str, sample_rate=16000, max_bytes=..., max_duration_s=...) -> DecodedAudio`
- Tests: synthetic WAV built with `wave` + numpy (PyAV decodes WAV), bad base64, oversize, prefix, empty, optional real MP3 from `data/audio` (skip if missing).

### `transcript.py` (A2)

- `normalize_text(text) -> str`, `tokenize(text) -> list[str]` (keeps decimals like `0.3` and `130/85` intact)
- `resegment(units, max_unit_s=8.0, pause_gap_s=0.7, split_on_punct=True) -> list[TranscriptUnit]` — re-cut whisper segments at sentence punctuation and long pauses using word timings; renumber unit ids 0..n-1; preserve global word ids; enforce max length by splitting at the largest internal gap.
- `assign_word_ids(units) -> list[TranscriptUnit]` — global consecutive ids in time order.
- `build_context(raw_units, audio_sha256, duration_s, asr_model_id, asr_config_hash, resegment_kwargs=None) -> AudioContext` — runs the above, `check_units` anomalies into `diagnostics`.
- `windows(context, max_units=3) -> list[Window]` where `Window(unit_ids: tuple[int,...], text, start_s, end_s)` — all contiguous runs of 1..max_units units, in order.
- `save_context(context, path)`, `load_context(path) -> AudioContext`

### `asr_backend.py` (A3)

- `@dataclass ASRConfig` (model_size='small', device='auto', compute_type='auto', beam_size=1, language='en', word_timestamps=True, vad_filter=False, condition_on_previous_text=False, cpu_threads=4, num_workers=1, temperature=0.0, initial_prompt=None, download_root=None, local_files_only=False, stop_reserve_s=8.0) with `config_hash` (stable_hash of fields) and `model_id`, plus `ASRConfig.from_env(prefix='MA_ASR_')`.
- `segments_to_units(segments) -> list[TranscriptUnit]` — pure conversion from faster-whisper `Segment`/`Word` objects (duck-typed: `.start .end .text .words .avg_logprob .no_speech_prob`, word `.start .end .word .probability`).
- `class FasterWhisperBackend(config)` — implements `ASRBackend`; `warm_up()` runs one short transcription; `transcribe(...)` iterates the lazy segment generator and stops consuming when `deadline.remaining() < config.stop_reserve_s` (records `last_run_info` with `truncated`, `segments`, `seconds`).
- `class FakeASRBackend(units, delay_s=0.0, clock=None)` — returns scripted units.
- `class CachingASRBackend(inner, cache_dir)` — JSON cache keyed by `cache_key` + `inner.config_hash`; pass-through when no key.
- `build_asr_backend(config, cache_dir=None) -> ASRBackend`

### `question_rewrite.py` (A4)

- `strip_tag(question) -> str` — removes tag endings (", didn't it?", ", right?", ", correct?", ", isn't that so?").
- `to_statement(question) -> str` — rule-based yes/no question → declarative (keep the auxiliary after the subject: "Did the patient attend X?" → "The patient did attend X."; "Is there any mention of X?" → "X is mentioned."). Falls back to the question minus `?`.
- `is_existence_question(question) -> bool` ("any mention of", "discussed", "talk about", "come up").
- `focus_terms(question) -> list[str]` — content tokens minus stopwords and question scaffolding, normalized with `transcript.normalize_text`.

### `factcheck.py` (A5)

- `@dataclass Quantity(value: float, unit: str | None, family: str, raw: str)` — families: dose, volume, duration, frequency, pressure, percentage, weight, length, temperature, count, plain.
- `extract_quantities(text) -> list[Quantity]` — digits and number words ("two hundred milligrams", "twice a day", "a week", "half"), unit normalization, blood pressure "130 over 85" / "130/85", duration conversion to days for comparison.
- `compare(question_text, evidence_text) -> FactCheckVerdict(status in {'consistent','contradiction','unverifiable','no_quantities'}, details: list[str])`
- `fuzzy_contains(term, text, threshold=0.8) -> bool` and `entity_terms(question) -> list[str]` (capitalised/drug-like tokens) for name near-miss checks.

### `qa_backend.py` (A6)

- `@dataclass QAConfig` (backend='retrieval_nli', embed_model='BAAI/bge-small-en-v1.5', nli_model='cross-encoder/nli-deberta-v3-small', top_k=6, max_window_units=3, yes_threshold=0.5, off_topic_sim_threshold=0.25, lexical_weight=0.4, use_factcheck=True, device='cpu', batch_size=32, max_seq_len=256) with `from_env('MA_QA_')`.
- `class LexicalBackend` — no ML deps; focus-term overlap over windows; factcheck gate; used as emergency fallback and baseline.
- `class RetrievalNLIBackend(config, embedder=None, nli=None)` — injectable model objects (`embedder.encode(list[str]) -> np.ndarray`, `nli.predict(list[tuple[str,str]]) -> np.ndarray[n,3]` with label order `[contradiction, entailment, neutral]`); retrieval = cosine + lexical; NLI on top-k windows with `to_statement(question)`; decision combines entailment, contradiction, factcheck verdict and existence/off-topic threshold; evidence = best window's unit ids.
- `class FakeQABackend(results)`.
- `build_qa_backend(config) -> QABackend`.
- Invariant: `answer()` returns exactly `len(questions)` results in order, always; per-question exceptions become a default result with `diagnostics['error']`.

### `qa_llm_backend.py` (A7)

- `@dataclass LLMConfig` (engine='transformers'|'llama_cpp'|'fake', model_id='Qwen/Qwen2.5-1.5B-Instruct', dtype='bfloat16', max_new_tokens=400, batch_all_questions=True, prompt_path='prompts/qa_v1.txt', device='auto', n_ctx=4096, max_prompt_units=400).
- `prompts/qa_v1.txt` — trusted task text (see plan §8.4); data goes in a separate JSON block: `{"transcript":[{"id":0,"start":..,"end":..,"text":..}], "questions":[{"q":0,"text":..}]}`.
- `build_messages(context, questions, prompt_text) -> list[dict]`
- `parse_llm_output(text, n_questions, valid_unit_ids) -> tuple[list[QAResult], list[int] missing]` — strict; ignores extra keys/tool calls/commentary; non-bool answers and invalid ids rejected.
- `class LLMBackend(config, generate=None)` — `generate(messages, max_new_tokens) -> str` injectable; one bounded per-question repair for missing items if the deadline allows.
- Engines are lazy imports; `fake` engine for tests.

### `evidence.py` (A8)

- `@dataclass EvidencePolicy(mode='union', max_span_s=20.0, pad_pre_s=0.0, pad_post_s=0.0, trim_filler=True, min_span_s=0.2)`; modes: union, first_run, longest_run, best_run (highest focus-term density).
- `select_span(context, result, policy, question=None) -> Optional[Span]` — validates ids, groups consecutive units into runs, applies mode, optional word-level trim of filler words at the edges, padding, clamp to `[0, duration_s]`, enforce `end > start`; never invents a span when no valid ids.
- `trim_to_focus(context, unit_ids, question, max_span_s) -> list[int]`
- `calibrate_padding(pairs: list[tuple[Span, Span]], grid) -> tuple[float, float]` (offline helper).

### `response_checks.py` (A9)

- `ResponseCheckError(ValueError)`
- `finalize(predictions, n_questions, duration_s, fallback_answer=True) -> ASRQuestionResponseDto` — length, bool, false→nulls, invalid span→nulls (clamp first), round 3 dp.
- `strict_validate(response, n_questions, duration_s=None) -> None` — raises with a precise message.
- `emergency_response(n_questions, answer=True) -> ASRQuestionResponseDto`
- `to_wire_json(response) -> str` and `check_wire_json(text, n_questions) -> None` (real booleans/nulls/finite numbers, no NaN).

### `eval_tools/` (A10)

- `scoring.py` — `score_records(records)` reproducing the reference formula; must match `local_evaluator.Statistics` on identical inputs (tested). Full-denominator variant counts unsent questions wrong.
- `split.py` — grouped 31/8 split, seed 2026, `sample_20` forced into dev; writes `manifests/split_v1.json`.
- `transcribe_all.py` — decode + ASR for every conversation (optionally split-filtered) into `work/transcripts/<config_hash>/<sha>.json` and a manifest with timings.
- `offline_eval.py` — cached contexts → QA → evidence → score for a split; writes `results/runs/<run_id>/`.
- `replay.py` — split-aware HTTP replay through the unchanged `local_evaluator.replay` (monkeypatched conversation list) with full-denominator accounting.
- `evidence_oracle.py` — localization ceiling per policy on cached transcripts.
- `error_analysis.py` — categorise misses from a run's records.
- `freeze_env.py` — environment/source manifest.

### Integration (integrator, after the modules)

- `pipeline.py`, `configs/*.json`, `example.py`, `tests/test_pipeline.py`, `security_fixtures/`, `tests/test_security.py`.

## Test conventions

- `tests/conftest.py` provides `make_context(sentences)` and `clinic_context`.
- Fast tests only by default; `RUN_SLOW=1 .venv/bin/pytest -m slow` for real models.
- Each module test file is runnable alone: `.venv/bin/python -m pytest tests/test_<module>.py`.
