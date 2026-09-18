"""Pure re-implementation of the reference score over plain record dicts.

``local_evaluator.Statistics`` is the ground truth for what a number means,
but it is built around an HTTP replay loop: it scores as it goes and keeps its
state in a mutable object. Offline tools want the same arithmetic over a list
they can filter, group, re-score and serialise, so this module scores a list
of *records* and is tested for parity against ``Statistics.record`` on
identical inputs. The weights and sentinel values are imported from the
reference module rather than copied, so a change there cannot silently drift
from what is measured here.

A record is a plain dict::

    {
        'question_id': 'sample_4_yes_q03',
        'transcript_id': 'sample_4',
        'question_type': 'positive',
        'label': 1,                    # int, 1 = yes
        'prediction': 1,               # int, -1 (UNANSWERED) when nothing arrived
        'gold_span': (21.62, 26.24),   # tuple | list | None
        'pred_span': (21.0, 26.0),     # tuple | list | None
    }

Extra keys (diagnostics, question text, timings) are carried along untouched.
Spans are read leniently through ``utils.evidence_interval`` exactly as the
reference does, so a NaN, a reversed interval or a half-filled pair scores 0
instead of raising.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from local_evaluator import ACCURACY_WEIGHT, TIOU_WEIGHT, UNANSWERED, YES, Statistics
from utils import Span, evidence_interval, gold_evidence, temporal_iou

RECORD_KEYS: Tuple[str, ...] = (
    'question_id', 'transcript_id', 'question_type', 'label', 'prediction',
    'gold_span', 'pred_span',
)

# Order the reference report uses; unknown types are appended after these.
QUESTION_TYPES: Tuple[str, ...] = ('positive', 'hard_negative', 'off_topic')


# --------------------------------------------------------------------------- #
# Record construction and normalisation
# --------------------------------------------------------------------------- #

def as_span(value: Any) -> Optional[Span]:
    """A valid ``(start, end)`` from a pair-like value, else ``None``.

    Lenient on purpose: records come back from JSON as lists, from the CSV
    as strings and from predictions as anything at all, and the reference
    scores every unusable span as 0 rather than failing the run.
    """
    if value is None:
        return None
    try:
        start, end = value
    except (TypeError, ValueError):
        return None
    return evidence_interval(start, end)


def as_int(value: Any, name: str) -> int:
    """``label``/``prediction`` as an int; bools are accepted (JSON answers)."""
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f'{name} must be an integer, got {value!r}') from None


def make_record(
    row: Mapping[str, Any],
    prediction: int,
    pred_span: Optional[Any] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """A record for one CSV row (the evaluation view) and one prediction."""
    record: Dict[str, Any] = {
        'question_id': str(row['question_id']),
        'transcript_id': str(row['transcript_id']),
        'question_type': str(row['question_type']),
        'label': as_int(row['label'], 'label'),
        'prediction': as_int(prediction, 'prediction'),
        'gold_span': gold_evidence(row),
        'pred_span': as_span(pred_span),
    }
    record.update(extra)
    return record


def unanswered_record(row: Mapping[str, Any], **extra: Any) -> Dict[str, Any]:
    """The record for a question the endpoint never answered."""
    return make_record(row, UNANSWERED, None, **extra)


def record_tiou(record: Mapping[str, Any]) -> Optional[float]:
    """The scored temporal IoU of one record, or ``None`` when it has no gold.

    Only annotated yes questions contribute to the tIoU mean; for anything
    else there is nothing to compare against and the result is ``None`` so
    callers cannot mistake "not applicable" for "scored zero".
    """
    label = as_int(record['label'], 'label')
    gold = as_span(record.get('gold_span'))
    if label != YES or gold is None:
        return None
    return temporal_iou(gold, as_span(record.get('pred_span')))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_from_totals(correct: int, tiou_sum: float, n: int, p: int) -> Dict[str, Any]:
    """The reference formula from its four sufficient statistics.

    ``n`` is the accuracy denominator (every question, sent or not) and ``p``
    the tIoU denominator (every annotated yes question). Both are fixed by
    the annotations, which is what lets a replay that stopped early be
    completed analytically: unsent questions add to ``n``/``p`` and nothing to
    ``correct``/``tiou_sum``. An empty denominator scores 0.0, as in the
    reference.
    """
    if n < 0 or p < 0 or correct < 0 or correct > n:
        raise ValueError(f'inconsistent totals: correct={correct} n={n} p={p}')
    accuracy = correct / n if n else 0.0
    mean_tiou = tiou_sum / p if p else 0.0
    return {
        'accuracy': accuracy,
        'mean_tiou': mean_tiou,
        'score': ACCURACY_WEIGHT * accuracy + TIOU_WEIGHT * mean_tiou,
        'n': n,
        'p': p,
        'correct': correct,
        'tiou_sum': tiou_sum,
    }


def score_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Score a list of records exactly as ``Statistics.record`` would.

    Returns ``accuracy``, ``mean_tiou``, ``score``, ``by_type`` (``{type:
    [correct, total]}`` in first-seen order), ``n`` (questions), ``p``
    (annotated yes questions), ``correct``, ``unanswered``, ``missing_spans``,
    ``mean_tiou_answered_yes`` (diagnostic only, not scored),
    ``n_answered_yes`` and ``tiou_sum``.
    """
    n = correct = unanswered = missing_spans = 0
    tiou_sum = 0.0
    p = 0
    answered_yes: List[float] = []
    by_type: Dict[str, List[int]] = collections.OrderedDict()

    for record in records:
        label = as_int(record['label'], 'label')
        prediction = as_int(record['prediction'], 'prediction')
        question_type = str(record.get('question_type', 'unknown'))

        n += 1
        if prediction == UNANSWERED:
            unanswered += 1
        is_correct = int(prediction == label)
        correct += is_correct
        bucket = by_type.setdefault(question_type, [0, 0])
        bucket[0] += is_correct
        bucket[1] += 1

        gold = as_span(record.get('gold_span'))
        if label != YES or gold is None:
            continue
        predicted = as_span(record.get('pred_span'))
        iou = temporal_iou(gold, predicted)
        p += 1
        tiou_sum += iou
        if predicted is None:
            missing_spans += 1
        if prediction == YES:
            answered_yes.append(iou)

    summary = score_from_totals(correct, tiou_sum, n, p)
    summary.update({
        'by_type': {key: list(value) for key, value in by_type.items()},
        'unanswered': unanswered,
        'missing_spans': missing_spans,
        'mean_tiou_answered_yes': (
            sum(answered_yes) / len(answered_yes) if answered_yes else 0.0
        ),
        'n_answered_yes': len(answered_yes),
    })
    return summary


def full_denominator(
    records: Sequence[Mapping[str, Any]],
    expected_question_rows: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """``records`` plus an UNANSWERED record for every expected row not in it.

    The evaluation service scores a conversation it never sent as ten wrong
    answers with no evidence; a local run that stopped early (or skipped a
    transcript with no cached context) must be completed the same way before
    its number can be compared with a competition score. Records are kept in
    their original order and the appended rows follow in CSV order, so the
    result is deterministic. Records for questions that are not expected are
    kept as they are: dropping data silently is worse than an odd total.
    """
    seen = {str(record['question_id']) for record in records}
    completed: List[Dict[str, Any]] = [dict(record) for record in records]
    for row in expected_question_rows:
        if str(row['question_id']) in seen:
            continue
        seen.add(str(row['question_id']))
        completed.append(unanswered_record(row, unsent=True))
    return completed


def per_transcript(records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``score_records`` for each transcript, in first-seen order."""
    groups: Dict[str, List[Mapping[str, Any]]] = collections.OrderedDict()
    for record in records:
        groups.setdefault(str(record['transcript_id']), []).append(record)
    return {transcript_id: score_records(rows) for transcript_id, rows in groups.items()}


# --------------------------------------------------------------------------- #
# Bridges to the reference object (parity tests and the reference report)
# --------------------------------------------------------------------------- #

def to_statistics(records: Iterable[Mapping[str, Any]]) -> Statistics:
    """Feed records through the reference ``Statistics`` object.

    Used for the parity test and for the report, whose layout is then the
    reference's own rather than a copy of it. Records are grouped by
    transcript to fill the conversation counters; a transcript whose every
    prediction is UNANSWERED counts as a failed conversation, which is what
    it would have been on the wire.
    """
    statistics = Statistics()
    groups: Dict[str, List[Mapping[str, Any]]] = collections.OrderedDict()
    for record in records:
        groups.setdefault(str(record['transcript_id']), []).append(record)
    for rows in groups.values():
        failed = all(as_int(r['prediction'], 'prediction') == UNANSWERED for r in rows)
        statistics.record_request(len(rows), None, failed=failed)
        for record in rows:
            statistics.record(
                str(record.get('question_type', 'unknown')),
                as_int(record['label'], 'label'),
                as_int(record['prediction'], 'prediction'),
                as_span(record.get('gold_span')),
                as_span(record.get('pred_span')),
            )
    return statistics


def format_per_transcript_table(records: Sequence[Mapping[str, Any]]) -> str:
    """One line per transcript: accuracy, mean tIoU, score, n, p, unanswered."""
    header = f'  {"transcript":<12} {"acc":>6} {"tIoU":>6} {"score":>6} {"n":>3} {"p":>3} {"unans":>5}'
    lines = ['', 'Per transcript', header]
    for transcript_id, summary in per_transcript(records).items():
        lines.append(
            f'  {transcript_id:<12} {summary["accuracy"]:6.3f} {summary["mean_tiou"]:6.3f} '
            f'{summary["score"]:6.3f} {summary["n"]:3d} {summary["p"]:3d} {summary["unanswered"]:5d}'
        )
    return '\n'.join(lines)


def render_report(records: Sequence[Mapping[str, Any]]) -> str:
    """The reference report layout followed by the per-transcript table."""
    return to_statistics(records).report() + '\n' + format_per_transcript_table(records)
