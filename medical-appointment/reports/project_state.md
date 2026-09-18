# Project state

Mode: BUILD (authorized 2026-09-17). Lead: Claude (Fable 5.1) with workflow agents.

## Incumbent

C1 (frozen 2026-09-18): offline dev 0.646, HTTP dev 0.675, HTTP holdout 0.699.

## Completed

- Snapshot audit and hash match (Appendix A).
- Environment: `.venv` with torch CPU, faster-whisper, sentence-transformers (`setup_env.sh`).
- ASR benchmark on the dev CPU: base RTF 0.035, small RTF 0.10–0.16 (`eval_tools/bench_asr.py`).
- Interface contract: `reports/implementation_plan.md`, `core_types.py`.
- Raw whisper transcripts for all 39 clips (base, small) under `work/raw_whisper/` (dev cache, not committed).
- Integration layer drafted: `pipeline.py`, `example.py`, `configs/*.json`, `tests/test_pipeline.py`, `tests/test_security.py`.

## Measurements

- Evidence ceiling (oracle-assisted, all 195 positives, cached transcripts, resegment max_unit_s=8 pause_gap_s=0.7): best single unit mean tIoU 0.71 (small) / 0.74 (base); best run of <=3 units 0.82 / 0.85; best word window 0.95 / 0.98; no gold span unreachable. Units per clip mean 53, median unit length 1.4 s. Conclusion: localization is bounded by unit selection (QA), not timing; padding is likely ~0.

- LLM on this CPU (llama.cpp Q4_K_M, 4 threads, 4 clips, cached small transcripts): Qwen2.5-1.5B 75-84 s per clip, acc 0.425 with JSON failures; Qwen2.5-3B 112-160 s per clip, acc 0.675. Both far over the 60 s budget (prompt ~2000-3000 tokens at ~25 tok/s). Measured under concurrent agent load; even 4x faster would not fit with ASR. DECISION D4: LLM backend is a GPU-host option only; CPU serving candidate = retrieval + NLI + factcheck.
- Quick offline QA on cached small transcripts (before question rewriting was wired): lexical 0.436 (acc 0.697, tIoU 0.262); retrieval_nli 0.437 (acc 0.736, positives 0.53, hard_neg 0.95, off_topic 0.91, tIoU 0.238, tIoU-when-yes 0.446; QA time mean 8 s, max 28 s = too slow next to 36 s ASR on the longest clip).

- Split: `manifests/split_v1.json`, seed 2026, 31 dev / 8 holdout, no multi-recording groups found; holdout = sample_5, 17, 33, 42, 47, 52, 64, 84 (label counts dev 156/110/44, holdout 39/32/9).
- r001 (dev, small transcripts, retrieval_nli, raw questions as hypotheses because question_rewrite was mid-build): score 0.427 (acc 0.726; positives 0.52, hard_neg 0.955, off_topic 0.886; tIoU 0.228; tIoU-when-yes 0.44). r002 lexical: 0.423.
- Window-selection study on r001 candidates (156 dev positives): picking by p_ent 0.368, by cosine/lexical/combined ~0.387, oracle best-of-top-k 0.677, oracle single unit inside the p_ent window 0.526, gold region missing from top-k for 19/156. Conclusion: a learned/shorter-window selector is worth up to +0.29 tIoU-when-yes; retrieval recall loses 12%.
- Entailment alone (nli-deberta-v3-small) separates dev questions at best 0.784 accuracy (threshold 0.05); the yes_threshold of 0.5 costs ~6 points.

- Calibration prototype (`calibration.py`, `eval_tools/fit_calibration.py`, fitted on r001 dev records, grouped 5-fold CV over transcripts): decision accuracy 0.868 (positive recall 0.85, negative 0.88) vs 0.726 rule; learned window ranking tIoU 0.507 vs 0.368 by max entailment (oracle 0.677). Saved `configs/calibration_v1.json`. Not yet wired into the backend.

- NLI model comparison on r001 candidate pairs (1860 pairs, raw questions as hypotheses, CPU under agent load): nli-deberta-v3-small 0.813 best acc (20 pairs/s); nli-deberta-v3-base 0.784 (8/s); MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli 0.890 (10/s, pos recall 0.86 / neg 0.92 at th 0.1); deberta-v3-large-zeroshot-v2.0 0.894 (3/s). DECISION D5: default NLI = MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (large is +0.004 for 3x the time).
- Calibration wired into `qa_backend.RetrievalNLIBackend` (`QAConfig.calibration_path`, `hard_contradiction_gate`), configs point at `configs/calibration_v1.json`. Module workflow ended with 11/18 agents done (7 hit the usage limit: reviews of response_checks, evidence, factcheck, audio_io, qa_backend and re-runs of eval_tools/question_rewrite builders); missing test files for question_rewrite and eval_tools written by the integrator; suite green.

- r003 (dev, MoritzLaurer NLI + rewriter, rule decision): small 0.479 (acc 0.810; pos 0.654 / hard_neg 0.973 / off_topic 0.955; tIoU 0.259), base 0.470 (acc 0.813; tIoU 0.242). ASR size is within noise for QA; small kept as default.
- Calibration refit on r003 small records: grouped-CV accuracy 0.923 (pos recall 0.904, neg 0.942), window tIoU 0.500 (oracle 0.679, p_ent pick 0.367).
- r004 (dev, calibrated, in-sample for the decision layer): score 0.646 (acc 0.919; pos 0.891 / hard_neg 0.936 / off_topic 0.977; tIoU 0.464; tIoU-when-yes 0.521; no span 17/156). Incumbent candidate C1 = small whisper int8 + bge-small retrieval + MoritzLaurer NLI + factcheck + calibration_v1 + best_run evidence.

- e2e_dev_001 (HTTP, unchanged local_evaluator via eval_tools.replay, dev split, C1 config, dev CPU with 3 reviewer agents running): score 0.675 (acc 0.935; pos 0.942 / hard_neg 0.936 / off_topic 0.909; tIoU 0.501; no span 9/156); 31/31 conversations, 0 timeouts, 0 failures; round trip mean 29.3 s, worst 50.7 s (sample_20: ASR 39.3 s + QA 9.0 s), no ASR truncation, no fallbacks. Full-denominator score identical (nothing unsent).

- r005 (dev, after the evidence/factcheck/response_checks reviews): identical to r004 (0.6464). DECISION D6 (2026-09-18 03:30): C1 frozen; source hashes in `manifests/C1_source_hashes.txt`, environment in `manifests/env_C1_candidate.json`. Holdout opened once for C1 (e2e_holdout_001).

- e2e_holdout_001 (HTTP, 8 holdout conversations never used for tuning): score 0.699 (acc 0.925; pos 0.974 / hard_neg 0.844 / off_topic 1.000; tIoU 0.549; no span 1/39); 0 timeouts; round trip mean 32.4 s, worst 47.3 s. Dev 0.675 vs holdout 0.699: no sign of overfitting; n=8 recordings so the interval is wide.

- full_local_eval_001 (`python local_evaluator.py`, all 39 conversations incl. the holdout, unchanged harness, C1): accuracy 0.933, mean tIoU 0.511, score 0.680; round trip mean 28844 ms, worst 51958 ms; log in `results/full_local_eval_001.txt`.

- r006 (dev, r005 with `trim_filler` off): score 0.649 vs 0.646 (tIoU 0.469 vs 0.464, accuracy unchanged). Decision: inconclusive on 31 recordings (+0.003); C1 retained unchanged, logged as a candidate simplification for the next promotion round.

## In progress

- Nothing running. Next: experiment queue in `reports/error_analysis.md`; serving-host measurements.

## Pending

- Full test suite green; split manifest; offline eval on cached transcripts (base vs small; lexical vs retrieval_nli vs llm); evidence oracle ceiling; HTTP replay via the unchanged local evaluator; oracle run; error analysis; hardening; freeze.

## Decisions

- D1: ASR default `small` int8 on CPU (fits the 60 s budget on the longest clip with ~24 s to spare); `base` as the fast fallback. Reversal: if QA needs more than ~15 s on the longest clip, drop to `base`.
- D2: QA default = retrieval + NLI cross-encoder + deterministic fact check (CPU-feasible). LLM backend implemented as an optional candidate to be measured.
- D3: Evidence = contiguous run of clause-level units chosen by QA, `best_run` mode, padding calibrated offline on the dev split only.

## Resource ledger

- Official attempts used: 0. Deployment: untouched.
