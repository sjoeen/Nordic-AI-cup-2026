# Error analysis and experiment queue

Based on r004 (offline, dev, calibrated) and e2e_dev_001 (HTTP, dev). Dev split
only; the holdout has not been opened. Updated 2026-09-18.

## Where the score is lost (dev, 310 questions, 156 positives)

| Loss | Count | Score cost (approx) | Cause |
| --- | --- | --- | --- |
| Positives answered no | 17 | 17/310 x 0.4 + 17/156 x 0.6 = 0.022 + 0.065 | decision layer; mostly inference-heavy paraphrases with low entailment |
| Hard negatives answered yes | 7 | 0.009 | NLI entailed a near-miss (drug name swap, body part, "want to avoid"); factcheck has no quantity to catch |
| Off-topic answered yes | 1 | 0.001 | existence question matched a loosely related unit |
| Positives answered yes, wrong place (tIoU < 0.1) | 51 | 51/156 x 0.6 x ~0.55 = 0.108 | window picked in another part of the conversation; typical for medicines mentioned several times ("Ibumetin renewed as well", "both prescriptions issued") where the gold is the confirming turn, not the first mention |
| Positives answered yes, partial overlap (0.1 to 0.7) | 51 | ~0.05 | window boundaries wider or narrower than the annotation |

The localisation ceiling study (best unit run of <=3 units 0.82; oracle among
the six retrieved windows 0.68) says that the retrieval pool misses the gold
region for ~12% of positives and the ranker picks the wrong candidate in about
a quarter of the rest.

## Ranked experiment queue

1. **E-A Candidate pool for evidence** (expected +0.03 to +0.06 tIoU): add the
   single units and pairs inside each top window as ranked candidates and add
   "position of the last mention" features (gold is often the confirming turn
   late in the conversation). Cost: +10 to 15 NLI pairs per question on CPU.
   Control: r005; metric: mean tIoU with fixed decision layer.
2. **E-B Decision features** (expected +0.01 to +0.02 accuracy): add the
   second-best window's contradiction, a drug-name mismatch flag from
   `factcheck.entity_terms`/`fuzzy_contains`, and question length interactions.
   Refit with grouped CV; promote only if CV accuracy rises.
3. **E-C NLI hypothesis variants**: score both the statement and the raw
   question and feed both entailments to the calibration (cheap, +60 pairs).
4. **E-D Whisper size on the serving host**: large-v3 on a GPU host is expected
   to fix name/dose errors (e.g. "Airomir", "Ibumetin") that no text model can
   recover from; measure with `transcribe_all` + `offline_eval` there.
5. **E-E Timing margin**: sample_20 at 48.8 s server-side on this CPU with
   `small`; `base` is 8 s ASR at -0.009 offline score. Decide per host.

6. **E-F Filler trim off** (measured r006: +0.003 score on dev, inconclusive): gold
   spans keep leading "So," / "Okay," in at least some annotations; bundle
   with E-A rather than promoting alone.

## Not worth pursuing now

- Local LLM answering on CPU (75 to 160 s per clip, see project_state).
- Larger NLI (deberta-v3-large): +0.004 accuracy for 3x the time.
