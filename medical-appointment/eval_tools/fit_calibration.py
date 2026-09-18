"""Fit the learned decision/window models from an offline run's records.

    .venv/bin/python -m eval_tools.fit_calibration \
        --records results/runs/r001_nli_small_dev/records.jsonl \
        --transcripts work/transcripts/<hash> --out configs/calibration_v1.json

Uses ONLY the records given (run it on a dev-split run; never on holdout).
The records must carry the retrieval_nli diagnostics (``top_windows``), which
``offline_eval`` writes. Reports grouped cross-validation numbers; the saved
models are refit on all given records.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def candidates_from_record(record, context):
    from calibration import Candidate

    cands = []
    for rank, w in enumerate(record['diagnostics']['qa'].get('top_windows') or []):
        units = [context.unit_by_id(int(i)) for i in w['unit_ids']]
        units = [u for u in units if u is not None]
        if not units:
            continue
        cands.append(Candidate(
            unit_ids=tuple(int(i) for i in w['unit_ids']),
            start_s=min(u.start_s for u in units), end_s=max(u.end_s for u in units),
            p_ent=float(w.get('p_ent', 0.0)), p_con=float(w.get('p_con', 0.0)),
            cosine=float(w.get('cosine', 0.0)), lexical=float(w.get('lexical', 0.0)), rank=rank,
        ))
    return cands


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--transcripts', required=True)
    parser.add_argument('--out', default='configs/calibration_v1.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--C', type=float, default=1.0)
    parser.add_argument('--ridge-alpha', type=float, default=1.0)
    args = parser.parse_args()

    from calibration import DecisionExample, WindowExample, fit_calibration, question_features, window_features
    from transcript import load_context
    from utils import temporal_iou

    records = [json.loads(line) for line in open(args.records, encoding='utf-8') if line.strip()]
    contexts = {}
    decision, windows = [], []
    for r in records:
        tid = r['transcript_id']
        if tid not in contexts:
            contexts[tid] = load_context(Path(args.transcripts) / 'contexts' / f'{tid}.json')
        qa = r['diagnostics'].get('qa') or {}
        cands = candidates_from_record(r, contexts[tid])
        if not cands:
            continue
        x = question_features(cands, bool(qa.get('existence', False)), str(qa.get('verdict', 'no_quantities')), r['question'])
        decision.append(DecisionExample(x, int(r['label']), tid))
        if int(r['label']) == 1 and r.get('gold_span'):
            gold = tuple(r['gold_span'])
            tious = np.array([temporal_iou(gold, (c.start_s, c.end_s)) for c in cands])
            windows.append(WindowExample(window_features(cands), tious, tid))

    cal, report = fit_calibration(
        decision, windows, decision_threshold=args.threshold, C=args.C, ridge_alpha=args.ridge_alpha,
        provenance={'records': args.records, 'transcripts': args.transcripts, 'n_records': len(records)},
    )
    path = cal.save(Path(args.out))
    print(json.dumps(report, indent=1))
    print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
