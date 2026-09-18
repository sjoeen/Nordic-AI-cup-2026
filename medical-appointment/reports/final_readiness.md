# Final readiness checklist (candidate C1)

Status legend: PASS / FAIL / UNKNOWN, with the evidence reference. Updated
2026-09-18 (dev machine). Items marked UNKNOWN need the serving host.

| Gate | Status | Evidence |
| --- | --- | --- |
| Scope and rules | PASS | `/predict` route, DTOs, `utils.py`, `local_evaluator.py`, `api.py` byte-identical to the snapshot; local-only inference (`HF_HUB_OFFLINE=1` server serves) |
| Reference scoring | PASS | `local_evaluator.py --oracle` = 1.000; `tests/test_eval_tools.py` parity with `Statistics`; full-denominator accounting in `eval_tools.replay` |
| Data integrity | PASS | Appendix A hashes match; `manifests/split_v1.json` (31/8, seed 2026); labels only in `eval_tools` evaluation views |
| Candidate validity | PASS | `configs/default.json` + `configs/calibration_v1.json`; `manifests/env_C1_candidate.json`; e2e_dev_001 |
| Generalization | PASS (n=8 recordings, wide uncertainty) | e2e_holdout_001: score 0.699 (acc 0.925: pos 0.974 / hard_neg 0.844 / off_topic 1.000; tIoU 0.549; no span 1/39); 0 timeouts; worst 47.3 s. Holdout opened once; any later tuning must be disclosed |
| Timing | PASS on dev CPU (margin thin) | e2e_dev_001: worst 50.7 s (sample_20, 232 s audio) with reviewer agents loading the CPU; 0 timeouts; ASR stops consuming segments 12 s before the budget |
| Memory | PASS on dev CPU | peak serving-process RSS 2.2 GB during e2e_holdout_001 (136 samples, 2 s apart) on a 7.6 GB host; GPU host VRAM not measured |
| Timeline | PASS | evidence ceiling study (no unreachable gold span); `tests/test_transcript.py`, `tests/test_evidence.py` |
| Failure recovery | PASS | `tests/test_pipeline.py` (ASR/QA/decode failures, wrong counts, final-body guard), `tests/test_security.py` oversize/malformed inputs |
| Injection resilience | PASS (suite only) | `security_fixtures/transcript_attacks_v1.json` 7/7 lexical and 7/7 real NLI; no tools on the inference path |
| Offline inference | PASS | server started with `HF_HUB_OFFLINE=1`; `prefetch_models.py` for a fresh host |
| Reproducibility | PASS | `setup_env.sh`, `requirements*.txt`, `manifests/env_C1_candidate.json`, `results/index.csv`, `reports/how_to_run.md` |
| Operational readiness | PARTIAL | serving host stated by the user 2026-09-18: 6 CPU cores, 16 GB RAM, 320 GB disk, no GPU (use `MA_CONFIG=configs/prod_cpu.json`); peak RSS 2.2 GB fits; external reachability and body-size limits still to verify on the host |
| Official checks | UNKNOWN | no Verify/validation attempt made; needs the user's authorization |
| Final evaluation | UNKNOWN | needs explicit user authorization |

## Launch (serving host)

```bash
bash setup_env.sh                       # or TORCH_INDEX=<cuda index> bash setup_env.sh
.venv/bin/python prefetch_models.py configs/default.json    # once, with network
HF_HUB_OFFLINE=1 .venv/bin/python api.py                    # serves http://<host>:9054/predict
```

On the stated CPU host use `MA_CONFIG=configs/prod_cpu.json`; on a CUDA host `configs/prod_gpu.json`. Measure with
`eval_tools.replay --split dev` before any official attempt. If the worst
round trip exceeds ~50 s, set `MA_ASR_MODEL_SIZE=base`.

## Rollback

The constant-answer emergency path is built in; to fall back to the
dependency-free answerer set `MA_QA_BACKEND=lexical`, and to skip ASR-heavy
models set `MA_ASR_MODEL_SIZE=base`.
