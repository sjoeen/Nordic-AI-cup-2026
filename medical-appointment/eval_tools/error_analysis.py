"""Categorise the misses in one offline run.

    .venv/bin/python -m eval_tools.error_analysis results/runs/<run_id>/records.jsonl
    .venv/bin/python -m eval_tools.error_analysis results/runs/<run_id>/records.jsonl \\
        --transcripts work/transcripts/<hash> --top 20

Prints accuracy and a label/prediction confusion per question type, a tIoU
histogram over the annotated yes questions, the worst positives that were
answered yes (lowest tIoU: the localisation failures, as opposed to the
answer failures) with gold vs predicted spans and the transcript text under
each, and the false positives on hard negatives and off-topic questions
with the backend's diagnostics. Transcript text is untrusted ASR output, so
control characters are stripped before anything reaches the terminal.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core_types import AudioContext, Span
from eval_tools import scoring
from eval_tools.split import safe_id
from local_evaluator import UNANSWERED, YES

DEFAULT_EDGES: Tuple[float, ...] = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
DEFAULT_TOP = 15
_CONTROL = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f]')


# --------------------------------------------------------------------------- #
# Loading and small helpers
# --------------------------------------------------------------------------- #

def load_records(path: Path) -> List[Dict[str, Any]]:
    """Records from a ``records.jsonl``; blank lines are skipped."""
    records: List[Dict[str, Any]] = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clean_text(text: Any, limit: int = 200) -> str:
    """Printable, single-line, bounded: transcript text goes to a terminal."""
    flat = ' '.join(str(text or '').split())
    flat = _CONTROL.sub('', flat)
    return flat if len(flat) <= limit else flat[:limit - 3] + '...'


def record_tiou(record: Mapping[str, Any]) -> Optional[float]:
    """The record's stored tIoU, recomputed from its spans when absent."""
    stored = record.get('tiou')
    if isinstance(stored, (int, float)) and not isinstance(stored, bool) and math.isfinite(stored):
        return float(stored)
    return scoring.record_tiou(record)


def is_positive(record: Mapping[str, Any]) -> bool:
    return scoring.as_int(record['label'], 'label') == YES and scoring.as_span(record.get('gold_span')) is not None


# --------------------------------------------------------------------------- #
# Aggregations (pure)
# --------------------------------------------------------------------------- #

def accuracy_by_type(records: Iterable[Mapping[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """``{question_type: (correct, total)}`` in first-seen order."""
    return {key: (value[0], value[1]) for key, value in scoring.score_records(records)['by_type'].items()}


def confusion_by_type(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[int, Dict[int, int]]]:
    """``{question_type: {label: {prediction: count}}}``; UNANSWERED is its own column."""
    table: Dict[str, Dict[int, Dict[int, int]]] = collections.OrderedDict()
    for record in records:
        question_type = str(record.get('question_type', 'unknown'))
        label = scoring.as_int(record['label'], 'label')
        prediction = scoring.as_int(record['prediction'], 'prediction')
        by_label = table.setdefault(question_type, {})
        by_prediction = by_label.setdefault(label, {})
        by_prediction[prediction] = by_prediction.get(prediction, 0) + 1
    return table


def bucket_labels(edges: Sequence[float] = DEFAULT_EDGES) -> List[str]:
    """Half-open buckets ``[a, b)`` with the last one closed, so 1.0 has a home."""
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f'edges must be strictly increasing with at least two values, got {edges!r}')
    labels = [f'[{a:.1f}, {b:.1f})' for a, b in zip(edges[:-2], edges[1:-1])]
    labels.append(f'[{edges[-2]:.1f}, {edges[-1]:.1f}]')
    return labels


def bucket_index(value: Any, edges: Sequence[float] = DEFAULT_EDGES) -> Optional[int]:
    """Index of the bucket holding ``value``; ``None`` for NaN, None or out of range."""
    bucket_labels(edges)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        return None
    if value < edges[0] or value > edges[-1]:
        return None
    for index, (low, high) in enumerate(zip(edges, edges[1:])):
        if low <= value < high:
            return index
    return len(edges) - 2  # value == edges[-1]


def tiou_histogram(
    records: Iterable[Mapping[str, Any]],
    edges: Sequence[float] = DEFAULT_EDGES,
) -> List[Tuple[str, int]]:
    """Bucket counts of the scored tIoU over annotated yes questions."""
    counts = [0] * (len(edges) - 1)
    invalid = 0
    for record in records:
        if not is_positive(record):
            continue
        index = bucket_index(record_tiou(record), edges)
        if index is None:
            invalid += 1
        else:
            counts[index] += 1
    histogram = list(zip(bucket_labels(edges), counts))
    if invalid:
        histogram.append(('invalid', invalid))
    return histogram


def worst_positives(records: Iterable[Mapping[str, Any]], k: int = DEFAULT_TOP) -> List[Dict[str, Any]]:
    """The ``k`` annotated yes questions answered yes with the lowest tIoU.

    Restricting to yes answers separates localisation errors from answer
    errors: a positive answered no already shows up in the confusion table.
    Ties are broken by question id so the list is stable across runs.
    """
    answered = [
        dict(record) for record in records
        if is_positive(record) and scoring.as_int(record['prediction'], 'prediction') == YES
    ]
    answered.sort(key=lambda record: (record_tiou(record) or 0.0, str(record['question_id'])))
    return answered[:max(0, k)]


def false_positives(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Negatives answered yes, in record order (hard negatives and off-topic)."""
    return [
        dict(record) for record in records
        if scoring.as_int(record['label'], 'label') != YES
        and scoring.as_int(record['prediction'], 'prediction') == YES
    ]


def span_text(context: AudioContext, span: Optional[Any]) -> str:
    """The words whose timing overlaps ``span``, in order; unit text when no words."""
    valid = scoring.as_span(span)
    if valid is None:
        return ''
    start, end = valid
    words = [
        word.text.strip() for word in context.words()
        if word.start_s < end and word.end_s > start
    ]
    if words:
        return ' '.join(words)
    units = [
        unit.text.strip() for unit in context.units
        if unit.end_s > start and unit.start_s < end
    ]
    return ' '.join(units)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def _fmt_span(span: Optional[Any]) -> str:
    valid = scoring.as_span(span)
    return f'{valid[0]:.2f}-{valid[1]:.2f}' if valid else 'none'


def _prediction_name(prediction: int) -> str:
    return {YES: 'yes', 0: 'no', UNANSWERED: 'unanswered'}.get(prediction, str(prediction))


def render(
    records: Sequence[Mapping[str, Any]],
    contexts: Optional[Mapping[str, AudioContext]] = None,
    top: int = DEFAULT_TOP,
) -> str:
    lines: List[str] = []
    scores = scoring.score_records(records)
    lines.append(f'{scores["n"]} questions, accuracy {scores["accuracy"]:.3f}, mean tIoU {scores["mean_tiou"]:.3f}, '
                 f'score {scores["score"]:.3f}, unanswered {scores["unanswered"]}')

    lines += ['', 'Accuracy by type']
    for question_type, (correct, total) in accuracy_by_type(records).items():
        lines.append(f'  {question_type:<16} {correct / total if total else 0.0:.3f}  ({correct}/{total})')

    lines += ['', 'Confusion (label -> prediction)']
    for question_type, by_label in confusion_by_type(records).items():
        for label, by_prediction in sorted(by_label.items()):
            cells = ', '.join(f'{_prediction_name(pred)}={count}' for pred, count in sorted(by_prediction.items()))
            lines.append(f'  {question_type:<16} label {_prediction_name(label):<3} -> {cells}')

    lines += ['', 'tIoU histogram (annotated yes questions)']
    for label, count in tiou_histogram(records):
        lines.append(f'  {label:<12} {count:4d} {"#" * count}')

    lines += ['', f'Worst {top} positives answered yes (lowest tIoU)']
    for record in worst_positives(records, top):
        tiou = record_tiou(record) or 0.0
        lines.append(f'  {record["question_id"]:<24} tIoU {tiou:.3f}  gold {_fmt_span(record.get("gold_span"))}  '
                     f'pred {_fmt_span(record.get("pred_span"))}')
        lines.append(f'      Q: {clean_text(record.get("question"))}')
        context = (contexts or {}).get(str(record['transcript_id']))
        if context is not None:
            lines.append(f'      gold text: {clean_text(span_text(context, record.get("gold_span")))}')
            lines.append(f'      pred text: {clean_text(span_text(context, record.get("pred_span")))}')

    lines += ['', 'False positives (label no, answered yes)']
    for record in false_positives(records):
        diagnostics = record.get('diagnostics') or {}
        lines.append(f'  {record["question_id"]:<24} {record["question_type"]:<14} '
                     f'pred {_fmt_span(record.get("pred_span"))}')
        lines.append(f'      Q: {clean_text(record.get("question"))}')
        lines.append(f'      diagnostics: {clean_text(json.dumps(diagnostics, default=str), limit=400)}')
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('records', type=Path,
                        help="A run's records.jsonl, or the run directory that holds it.")
    parser.add_argument('--transcripts', type=Path, default=None,
                        help='transcribe_all output dir; shows the transcript text under each span.')
    parser.add_argument('--top', type=int, default=DEFAULT_TOP)
    args = parser.parse_args(argv)

    records_path = args.records / 'records.jsonl' if args.records.is_dir() else args.records
    records = load_records(records_path)
    if not records:
        print('no records', file=sys.stderr)
        return 1
    contexts: Optional[Dict[str, AudioContext]] = None
    if args.transcripts is not None:
        from eval_tools.offline_eval import load_contexts

        transcript_ids = sorted({safe_id(record['transcript_id']) for record in records})
        contexts, missing = load_contexts(Path(args.transcripts) / 'contexts', transcript_ids)
        if missing:
            print(f'no cached context for {missing}', file=sys.stderr)
    print(render(records, contexts, args.top))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
