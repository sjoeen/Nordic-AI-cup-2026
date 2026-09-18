# How to run, test and evaluate

All commands from `medical-appointment/` with the venv created by `bash setup_env.sh`.

## Serve

```bash
.venv/bin/python api.py                       # loads configs/default.json, warms up, serves :9054
MA_CONFIG=configs/prod_gpu.json .venv/bin/python api.py
MA_ASR_MODEL_SIZE=base MA_QA_BACKEND=lexical .venv/bin/python api.py   # env overrides any field
```

Config precedence: `configs/<file>.json` (or `MA_CONFIG`) → env vars `MA_PIPELINE_*`,
`MA_ASR_*`, `MA_QA_*`, `MA_EVIDENCE_*`, `MA_RESEGMENT_*`.

## Tests (each stage in isolation)

```bash
.venv/bin/python -m pytest                                  # all fast tests, no real models
.venv/bin/python -m pytest tests/test_factcheck.py -q       # one stage
RUN_SLOW=1 OMP_NUM_THREADS=2 .venv/bin/python -m pytest -m slow   # real whisper/NLI smoke tests
.venv/bin/python -m pytest tests/test_pipeline.py tests/test_security.py   # integration + adversarial fixtures
```

Stage → test file: `audio_io`, `transcript`, `asr_backend`, `question_rewrite`,
`factcheck`, `qa_backend`, `qa_llm_backend`, `evidence`, `response_checks`,
`calibration`, `pipeline` (+ `example.py`, HTTP round trip), `security`, `eval_tools`.

## Offline evaluation (fast iteration on QA / evidence without re-running ASR)

```bash
.venv/bin/python -m eval_tools.transcribe_all --split all --asr-model-size small   # cache contexts
.venv/bin/python -m eval_tools.offline_eval --transcripts work/transcripts/<hash> --split dev \
    --qa-backend retrieval_nli --run-id r00X_name --notes "what changed"
.venv/bin/python -m eval_tools.error_analysis results/runs/r00X_name/records.jsonl
.venv/bin/python -m eval_tools.evidence_oracle --transcripts work/transcripts/<hash> --split dev
.venv/bin/python -m eval_tools.fit_calibration --records results/runs/r00X_name/records.jsonl \
    --transcripts work/transcripts/<hash> --out configs/calibration_v1.json     # dev records only
```

Runs land in `results/runs/<run_id>/` and one line in `results/index.csv`.

## End-to-end (the real path, HTTP, unchanged reference evaluator)

```bash
.venv/bin/python api.py &                                    # terminal 1
.venv/bin/python -m eval_tools.replay --split dev --run-id e2e_dev_001   # terminal 2
.venv/bin/python local_evaluator.py                          # full 39-conversation reference run
.venv/bin/python local_evaluator.py --oracle                 # harness self-check, prints 1.000
```

`eval_tools.replay` reports both the as-replayed score and the full-denominator
score (unsent conversations counted wrong), which is what the service would show.

## Splits and holdout

`manifests/split_v1.json`: 31 dev / 8 holdout, seed 2026. Tune on `dev` only;
run `--split holdout` once for the frozen candidate and record it.
