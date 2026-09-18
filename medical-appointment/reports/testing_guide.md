# Step-by-step testing guide, stage by stage

Run everything from `medical-appointment/` with the venv (`bash setup_env.sh`
once). Expected counts are from 2026-09-18; a lower count means a test file
changed, a failure means the stage broke. Debug bottom-up in the order below:
each stage only depends on the ones above it, so the first failing stage is
the cause.

Quick commands:

```bash
.venv/bin/python -m pytest                       # whole fast suite: 1053 passed, 9 skipped, ~4 s
.venv/bin/python -m pytest tests/test_<stage>.py -o addopts="" -q   # one stage with a summary line
RUN_SLOW=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -m slow -o addopts=""   # 9 real-model tests, ~25 s
```

## 0. Harness self-check (before trusting any number)

1. `.venv/bin/python local_evaluator.py --oracle` -> `Accuracy 1.000, Mean tIoU 1.000, Score 1.000`. Anything else: the data or the reference files changed; check `sha256sum` against `manifests/C1_source_hashes.txt` and the plan's Appendix A.
2. `.venv/bin/python -m pytest tests/test_eval_tools.py` -> 7 passed. Covers: our scorer equals `local_evaluator.Statistics` on synthetic records and on the gold oracle, unsent questions count wrong, the split is deterministic and keeps duplicate recordings together.

## 1. core_types (shared types, deadline)

- No own test file; exercised by every other test. Smoke:
  `.venv/bin/python -c "from core_types import Deadline; d=Deadline(5); print(d.remaining()>4, d.fits(1))"` -> `True True`.
- Failure here breaks every import: check `python -c "import core_types"`.

## 2. audio_io (base64 -> waveform)

1. `pytest tests/test_audio_io.py` -> 99 passed.
2. Manual smoke with a real file:
   `.venv/bin/python -c "import base64; from audio_io import decode_request_audio; from utils import load_sample_audio; d=decode_request_audio(base64.b64encode(load_sample_audio('conversation_sample_4.mp3')).decode()); print(round(d.duration_s,2), d.header_duration_s, d.waveform.dtype, d.audio_sha256[:12])"`
   -> `106.48 106.5x float32 <hash>`; decoded and header durations within 1 s.
3. Negative checks (must raise `AudioDecodeError`, never crash): empty string, `'%%%'`, a data-URI prefix only, 50 MB of base64.
4. If it fails: PyAV missing (`import av`), or the file is not audio. Errors are logged on logger `audio_io`.

## 3. transcript (units, resegmentation, windows)

1. `pytest tests/test_transcript.py` -> 113 passed.
2. Manual smoke on a cached context:
   `.venv/bin/python -c "from transcript import load_context, windows; c=load_context('work/transcripts/a0fafd700b24a45f411eb1940777f0a86a158b417480fe79655ab8042466ea6e/contexts/sample_4.json'); print(len(c.units), len(windows(c,3)), c.diagnostics.get('unit_anomalies'))"`
   -> `43 <n> []` (no anomalies). Units should be 1 to 8 s long; print `c.units[8]` and expect text about the annual follow-up starting near 21.6 s.
3. If units look wrong (one giant unit, or one word each): check `configs/*.json` `resegment` values; `pipeline` validates them at boot and raises `ValueError` on bad ones.

## 4. asr_backend (faster-whisper wrapper, cache, deadline)

1. `pytest tests/test_asr_backend.py` -> 76 passed, 1 skipped (the skipped one is the real-model test).
2. Real model: `RUN_SLOW=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_asr_backend.py -m slow -o addopts=""` -> 1 passed (~20 s): transcribes `sample_4` with whisper `base`, and checks that a short deadline returns a strict prefix with `truncated=True`.
3. Speed on this machine: `.venv/bin/python -m eval_tools.bench_asr --models base small --clips sample_4 sample_20` -> RTF about 0.035 (base) and 0.10 to 0.16 (small); longest clip must stay well under 40 s for `small`.
4. If it fails: model not in the HF cache (run `prefetch_models.py` with network), `HF_HUB_OFFLINE=1` set while weights are missing, or `compute_type`/`device` not supported on the host.

## 5. question_rewrite (question -> statement)

1. `pytest tests/test_question_rewrite.py` -> 35 passed (30-question table plus tags, existence detection, focus terms).
2. Manual: `.venv/bin/python -c "from question_rewrite import to_statement as t; print(t('Should the daily dose be 100 mg?'), '|', t('The lipid profile came back normal, didn\'t it?'), '|', t('Is there any mention of attending a concert?'))"`
   -> `The daily dose should be 100 mg. | The lipid profile came back normal. | Attending a concert is mentioned.`
3. A wrong rewrite degrades NLI quietly; if dev accuracy drops after a change here, run `offline_eval` (section 12) and compare `statement` in `records.jsonl`.

## 6. factcheck (quantities, contradictions)

1. `pytest tests/test_factcheck.py` -> 300 passed (extraction table, comparison table, ranges, ages, decimal commas, performance bounds).
2. Manual: `.venv/bin/python -c "from factcheck import compare; print(compare('Should the daily dose be 100 mg?', 'penicillin, 100 milligrams daily').status, compare('Was the dose 200 mg daily?', 'penicillin, 100 milligrams daily').status, compare('Will it last two weeks?', 'for fourteen days').status)"`
   -> `consistent contradiction consistent`.
3. Performance: a 3000-word window must compare in under 100 ms (`test_compare_scales_linearly_and_is_fast_on_a_3000_word_window`).
4. A false `contradiction` forces a NO on a true statement: if a positive is missed with `verdict = contradiction` in the diagnostics, this is the stage to fix.

## 7. qa_backend (retrieval, NLI, calibration wiring, lexical fallback)

1. `pytest tests/test_qa_backend.py` -> 40 passed, 1 skipped. All with fake embedder/NLI: decisions, order, per-question error isolation, deadline degradation, calibration wiring, hard contradiction gate, missing/invalid calibration file.
2. Real models: `RUN_SLOW=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest tests/test_qa_backend.py -m slow -o addopts=""` -> 1 passed: loads bge-small and the NLI model on the clinic context and checks label-order handling.
3. Manual end-to-end on a cached transcript (loads real models, ~30 s):
   ```bash
   .venv/bin/python - <<'PY'
   from transcript import load_context
   from qa_backend import QAConfig, build_qa_backend
   c = load_context('work/transcripts/a0fafd700b24a45f411eb1940777f0a86a158b417480fe79655ab8042466ea6e/contexts/sample_4.json')
   qa = build_qa_backend(QAConfig(calibration_path='configs/calibration_v1.json')); qa.warm_up()
   for r in qa.answer(c, ['Did the patient attend for an annual asthma follow-up?', 'Should the asthma medication dose be increased?', 'Is there any discussion about cooking?']):
       print(r.answer, r.evidence_unit_ids, round(r.confidence, 2), r.diagnostics['verdict'])
   PY
   ```
   -> `True [8, 9] ...`, `False ...`, `False ...`; `diagnostics['calibrated']` must be `True` (otherwise the calibration file did not load: see the `qa_backend` log line).
4. Timing: `answer()` for ten questions on a 230 s clip should take under 10 s on 4 cores (see `seconds` in the offline records).

## 8. qa_llm_backend (optional LLM answerer)

1. `pytest tests/test_qa_llm_backend.py` -> 73 passed (fake engine only: prompt construction keeps data inside the JSON block, strict parser, repair budget, unit merging).
2. Real LLM is not part of the CPU candidate; to try one: `QAConfig(backend='llm')` with `LLMConfig(engine='llama_cpp', model_path='work/models/qwen2.5-3b-instruct-q4_k_m.gguf')`. On this CPU expect over 100 s per clip (documented as infeasible).

## 9. calibration (learned decision and window ranking)

1. `pytest tests/test_calibration.py` -> 8 passed (features, linear models, round trip, fit recovers a separable rule).
2. Refit and read the cross-validated numbers (dev records only):
   `.venv/bin/python -m eval_tools.fit_calibration --records results/runs/r003_nli_mfa_rules_small_dev/records.jsonl --transcripts work/transcripts/a0fafd700b24a45f411eb1940777f0a86a158b417480fe79655ab8042466ea6e --out /tmp/cal_check.json`
   -> `cv_accuracy` about 0.92, `cv_window_tiou` about 0.50. Compare with `configs/calibration_v1.json` before replacing it.

## 10. evidence (unit ids -> seconds)

1. `pytest tests/test_evidence.py` -> 69 passed (modes, invalid ids, word override, filler trim, max-span narrowing, padding/clamping, calibrate_padding).
2. Manual: `.venv/bin/python -c "from transcript import load_context; from evidence import EvidencePolicy, select_span; from core_types import QAResult; c=load_context('work/transcripts/a0fafd700b24a45f411eb1940777f0a86a158b417480fe79655ab8042466ea6e/contexts/sample_4.json'); print(select_span(c, QAResult(0, True, [8, 9]), EvidencePolicy(mode='best_run'), question='Did the patient attend for an annual asthma follow-up?'))"`
   -> `(22.28, 26.2)`: gold is 21.62 to 26.24; the leading "So," is dropped by `trim_filler`, which is why the filler trim is under measurement (r006).
3. Localization ceiling for a transcript cache: `.venv/bin/python -m eval_tools.evidence_oracle --transcripts work/transcripts/<hash> --split dev` -> best run of <=3 units about 0.82; if this drops, the ASR timing or resegmentation changed.

## 11. response_checks (wire safety)

1. `pytest tests/test_response_checks.py` -> 212 passed (finalize never raises, NaN/inf/numpy handling, wire JSON checks, fuzz).
2. Manual: `.venv/bin/python -c "from response_checks import finalize, check_wire_json, to_wire_json; from core_types import Prediction; r=finalize([Prediction(True,(1.0,float('nan'))), Prediction(False,(1.0,2.0))], 2, 100.0); print(r.answers, r.evidence_start, r.evidence_end); check_wire_json(to_wire_json(r), 2)"`
   -> `[True, False] [None, None] [None, None]` and no exception.

## 12. pipeline + example (integration with fakes)

1. `pytest tests/test_pipeline.py` -> 13 passed: scripted QA end to end, order preserved, QA/ASR/decode failures, wrong counts, filename is metadata, zero questions, `example.predict` never raises, HTTP round trip through the real FastAPI app, env overrides, bad resegment config fails at boot, invalid final body replaced by the emergency response.
2. Config check for a host: `MA_CONFIG=configs/prod_cpu.json .venv/bin/python -c "from pipeline import PipelineConfig; c=PipelineConfig.load(); print(c.asr, c.qa['nli_model'], c.qa['calibration_path'])"`.

## 13. security fixtures

1. `pytest tests/test_security.py` -> 8 passed, 7 skipped (the 7 real-NLI variants run with `RUN_SLOW=1`: expect 15 passed).
2. Fixture file: `security_fixtures/transcript_attacks_v1.json` (7 cases: benign control, spoken override, authority claim in a question, delimiter/role spoofing, evidence fabrication, exfiltration request, unicode/bidi). Add cases there; the test picks them up automatically.

## 14. Offline evaluation (QA + evidence on cached transcripts, minutes)

1. Cache transcripts once: `.venv/bin/python -m eval_tools.transcribe_all --split all --asr-model-size small` (or `import_raw_whisper` from an existing dump).
2. Evaluate: `.venv/bin/python -m eval_tools.offline_eval --transcripts work/transcripts/<hash> --split dev --qa-backend retrieval_nli --qa-option calibration_path=configs/calibration_v1.json --run-id r0XX_name --notes "..."`
   -> reference: r005 = accuracy 0.919, mean tIoU 0.464, score 0.646 on dev.
3. Inspect: `.venv/bin/python -m eval_tools.error_analysis results/runs/r0XX_name/records.jsonl` (per type, confusion, tIoU histogram, worst positives, false positives with diagnostics).
4. Every run appends to `results/index.csv`; never edit old rows.

## 15. End to end over HTTP (the real path)

1. Terminal 1: `HF_HUB_OFFLINE=1 .venv/bin/python api.py` and wait for `Application startup complete` (models load and a synthetic warm-up request runs first).
2. Health: `curl http://localhost:9054/` -> `"Your endpoint is running!"`.
3. One conversation by hand:
   ```bash
   .venv/bin/python - <<'PY'
   import json, requests
   from utils import encode_audio, load_sample_audio, group_questions_by_conversation
   fn, rows = group_questions_by_conversation()[0]
   r = requests.post('http://localhost:9054/predict', json={'audio_base64': encode_audio(load_sample_audio(fn)), 'audio_filename': fn, 'questions': [x['question'] for x in rows]}, timeout=60)
   print(r.status_code, json.dumps(r.json())[:300])
   PY
   ```
   -> `200` and three lists of length 10 with real booleans and nulls.
4. Dev split: `.venv/bin/python -m eval_tools.replay --split dev --run-id e2e_dev_00X` -> reference e2e_dev_001: score 0.675, 0 timeouts, worst round trip 50.7 s on a loaded 4-core box.
5. Whole set (the CLAUDE.md check): `.venv/bin/python local_evaluator.py` -> reference 0.680; read the `Round trip` block: worst must stay clearly under 60000 ms.
6. Server-side stage times are logged per request on logger `pipeline` (`stage_times`, `fallbacks`, `asr_info.truncated`); grep the server log when a request is slow.
7. Holdout: `--split holdout` was already run once for C1 (0.699). Do not tune on it.

## 16. Reproducibility and freeze

1. `.venv/bin/python -m eval_tools.freeze_env --run-id <id>` -> `manifests/env_<id>.json` with pip freeze and source hashes.
2. `sha256sum -c` style check against `manifests/C1_source_hashes.txt` to confirm the frozen sources are what is being served.
