"""Split-aware HTTP replay through the unchanged reference evaluator.

    .venv/bin/python -m eval_tools.replay --split dev --url http://localhost:9054/predict
    .venv/bin/python -m eval_tools.replay --split holdout --run-id final_holdout --verbose

``local_evaluator.replay`` is the closest thing to the evaluation service we
have, so its bytes stay identical and this tool changes only what it sends:
``local_evaluator.group_questions_by_conversation`` is temporarily replaced
by a function returning the selected split's conversations, in the original
CSV order. The reference loop, timeouts, abort rules and report are all
untouched.

The one thing the reference deliberately leaves out is the score of the
conversations it never sent (after an abort). The service counts those
questions wrong, so this tool completes the number analytically from the
sufficient statistics: ``correct`` and ``sum(tious)`` are what the endpoint
earned, and the expected ``N``/``P`` come from the split's rows. Both the
as-replayed and the full-denominator figures are written to
``results/runs/<run_id>/replay_summary.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import local_evaluator
from eval_tools import scoring
from eval_tools.offline_eval import (
    DEFAULT_INDEX_PATH,
    DEFAULT_OUT_DIR,
    append_index_row,
    default_run_id,
    json_default,
    utc_now,
)
from eval_tools.split import DEFAULT_SPLIT_PATH, SPLIT_NAMES, conversations_for, safe_id, select_transcripts
from utils import audio_filename_for_transcript, gold_evidence

Conversation = Tuple[str, List[Dict[str, Any]]]


@contextlib.contextmanager
def patched_conversations(conversations: Sequence[Conversation]) -> Iterator[None]:
    """Make ``local_evaluator.replay`` iterate exactly ``conversations``.

    ``replay`` looks the name up in its module globals on every call, so
    rebinding the attribute is enough; the original is restored on exit even
    if the replay raises, so a later full run in the same process is not
    silently filtered.
    """
    original = local_evaluator.group_questions_by_conversation
    snapshot = [(audio_filename, list(rows)) for audio_filename, rows in conversations]
    local_evaluator.group_questions_by_conversation = lambda: list(snapshot)
    try:
        yield
    finally:
        local_evaluator.group_questions_by_conversation = original


def evaluator_conversations(transcript_ids: Sequence[str]) -> List[Conversation]:
    """``(audio_filename, rows)`` pairs in the shape ``local_evaluator.replay`` expects."""
    return [
        (audio_filename_for_transcript(transcript_id), rows)
        for transcript_id, rows in conversations_for(transcript_ids)
    ]


def expected_totals(conversations: Sequence[Conversation]) -> Tuple[int, int]:
    """``(N, P)``: questions expected, and annotated yes questions expected."""
    n = p = 0
    for _, rows in conversations:
        for row in rows:
            n += 1
            if scoring.as_int(row['label'], 'label') == local_evaluator.YES and gold_evidence(row) is not None:
                p += 1
    return n, p


def full_denominator_summary(
    statistics: local_evaluator.Statistics,
    conversations: Sequence[Conversation],
) -> Dict[str, Any]:
    """Both readings of one replay: as the reference printed it, and completed.

    When every conversation was sent the two agree exactly. After an abort the
    completed figure is what the service would report, because the unsent
    questions add to both denominators and contribute nothing to the
    numerators.
    """
    n_expected, p_expected = expected_totals(conversations)
    tiou_sum = float(sum(statistics.tious))
    as_replayed = scoring.score_from_totals(
        statistics.correct, tiou_sum, statistics.total, len(statistics.tious),
    )
    completed = scoring.score_from_totals(
        statistics.correct, tiou_sum, max(n_expected, statistics.total), max(p_expected, len(statistics.tious)),
    )
    latencies = list(statistics.latencies_ms)
    return {
        'conversations_expected': len(conversations),
        'conversations_sent': statistics.conversations,
        'failed_conversations': statistics.failed_conversations,
        'timeouts': statistics.timeouts,
        'aborted': bool(statistics.aborted),
        'unanswered': statistics.errors,
        'unsent_questions': max(0, n_expected - statistics.total),
        'by_type': {key: list(value) for key, value in statistics.by_type.items()},
        'missing_spans': statistics.missing_spans,
        'mean_tiou_answered_yes': statistics.mean_tiou_answered_yes,
        'latency_ms': {
            'mean': sum(latencies) / len(latencies) if latencies else None,
            'worst': max(latencies) if latencies else None,
            'n': len(latencies),
        },
        'as_replayed': as_replayed,
        'full_denominator': completed,
        'comparable_to_competition': not statistics.aborted and statistics.total == n_expected,
    }


def format_completion(summary: Mapping[str, Any]) -> str:
    """The lines that extend the reference report with the completed score."""
    full = summary['full_denominator']
    lines = ['', 'Full-denominator accounting (unsent questions scored wrong)']
    lines.append(f'  conversations sent     {summary["conversations_sent"]}/{summary["conversations_expected"]}')
    lines.append(f'  unsent questions       {summary["unsent_questions"]}')
    lines.append(f'  Accuracy:  {full["accuracy"]:.3f}  ({full["correct"]}/{full["n"]})')
    lines.append(f'  Mean tIoU: {full["mean_tiou"]:.3f}  (over {full["p"]} annotated yes questions)')
    lines.append(f'  Score:     {full["score"]:.3f}')
    if not summary['comparable_to_competition']:
        lines.append('  NOTE: the run stopped early; the completed figures above are the comparable ones.')
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--url', default=local_evaluator.DEFAULT_URL)
    parser.add_argument('--split', choices=SPLIT_NAMES, default='dev')
    parser.add_argument('--split-file', type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--notes', default='')
    parser.add_argument('--no-wait', action='store_true', help='Do not poll the endpoint before replaying.')
    args = parser.parse_args(argv)

    run_id = safe_id(args.run_id, 'run_id') if args.run_id else default_run_id('replay', args.split)
    conversations = evaluator_conversations(select_transcripts(args.split, args.split_file))
    if not conversations:
        print('nothing selected', file=sys.stderr)
        return 1
    if not args.no_wait and not local_evaluator.wait_for_endpoint(args.url):
        print(f'Nothing answering at {args.url}. Start it with "python api.py".', file=sys.stderr)
        return 1

    with patched_conversations(conversations):
        statistics = local_evaluator.replay(args.url, args.verbose)

    completion = full_denominator_summary(statistics, conversations)
    report = statistics.report() + '\n' + format_completion(completion)
    print(report)

    summary = {
        'run_id': run_id,
        'timestamp': utc_now(),
        'url': args.url,
        'split': args.split,
        'split_file': str(args.split_file),
        'transcripts': [audio_filename for audio_filename, _ in conversations],
        'notes': args.notes,
        **completion,
    }
    run_dir = Path(args.out) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'replay_summary.json').write_text(
        json.dumps(summary, indent=1, default=json_default) + '\n', encoding='utf-8',
    )
    (run_dir / 'report.txt').write_text(report + '\n', encoding='utf-8')
    full = completion['full_denominator']
    append_index_row(args.index, {
        'run_id': run_id,
        'timestamp': summary['timestamp'],
        'split': args.split,
        'qa_backend': 'replay',
        'asr_config_hash': '',
        'accuracy': f'{full["accuracy"]:.4f}',
        'mean_tiou': f'{full["mean_tiou"]:.4f}',
        'score': f'{full["score"]:.4f}',
        'notes': ' '.join(part for part in (f'replay {args.url}', args.notes) if part),
    })
    print(f'\nwrote {run_dir / "replay_summary.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
