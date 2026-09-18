"""Localisation ceiling: the best temporal IoU any policy could reach.

    .venv/bin/python -m eval_tools.evidence_oracle --transcripts work/transcripts/<hash> --split dev

QA picks units; the evidence stage turns them into a span. Before tuning
either it helps to know how much of the tIoU is even reachable from the
cached transcript's unit and word boundaries, because ASR timing drift and
unit cuts put a ceiling on every downstream choice. For each annotated yes
question this computes the best achievable tIoU when the span is:

* one unit;
* a contiguous run of at most k units (k = 2, 3, 5 by default);
* a contiguous window of at most ``--max-words`` words inside the best run
  (the run found for the largest k), which bounds the search at
  ``words * max_words`` IoU evaluations instead of ``words ** 2``;

and, for each of those candidates, the best fixed ``(pad_pre, pad_post)``
from a grid via ``evidence.calibrate_padding``. Means are printed as a table
and everything is written to ``results/evidence_oracle_<asr_hash>_<split>.json``.
It is an oracle-assisted diagnostic: gold spans are read here and nowhere
near inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core_types import AudioContext, Span, TranscriptUnit, Word, is_finite_number
from eval_tools import scoring
from eval_tools.offline_eval import json_default, load_contexts, utc_now
from eval_tools.split import DEFAULT_SPLIT_PATH, SPLIT_NAMES, conversations_for, select_transcripts
from local_evaluator import YES
from utils import gold_evidence, temporal_iou

DEFAULT_KS: Tuple[int, ...] = (2, 3, 5)
DEFAULT_MAX_WORDS = 40
# -0.5 s .. +1.0 s in 0.1 s steps: the grid ``evidence.calibrate_padding`` uses by default.
DEFAULT_PAD_GRID: Tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(-5, 11))
DEFAULT_RESULTS_DIR = Path('results')

Calibrator = Callable[[Sequence[Tuple[Span, Span]], Sequence[float]], Tuple[float, float]]


@dataclass(frozen=True)
class Candidate:
    """The best span of one kind for one gold span.

    ``ids`` names what the span covers: one unit id, the unit ids of a run,
    or ``(first_word_id, last_word_id)`` of a word window. ``span`` is
    ``None`` (and ``tiou`` 0.0) when the context offered nothing usable, so
    "nothing to point at" stays distinguishable from "pointed at the wrong
    place".
    """

    tiou: float
    ids: Tuple[int, ...]
    span: Optional[Span]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'tiou': float(self.tiou),
            'ids': list(self.ids),
            'span': None if self.span is None else [float(self.span[0]), float(self.span[1])],
        }


NO_CANDIDATE = Candidate(0.0, (), None)


# --------------------------------------------------------------------------- #
# Candidate spans
# --------------------------------------------------------------------------- #

def _unit_span(unit: TranscriptUnit) -> Optional[Span]:
    if is_finite_number(unit.start_s) and is_finite_number(unit.end_s) and unit.end_s >= unit.start_s:
        return (float(unit.start_s), float(unit.end_s))
    return None


def _timed_words(units: Iterable[TranscriptUnit]) -> List[Word]:
    return [
        word for unit in units for word in unit.words
        if is_finite_number(word.start_s) and is_finite_number(word.end_s) and word.end_s >= word.start_s
    ]


def best_unit_tiou(context: AudioContext, gold: Span) -> Candidate:
    """The single unit whose bounds best match ``gold``; ties go to the earlier unit."""
    best = NO_CANDIDATE
    for unit in context.units:
        span = _unit_span(unit)
        if span is None:
            continue
        tiou = temporal_iou(gold, span)
        if best.span is None or tiou > best.tiou:
            best = Candidate(tiou, (unit.unit_id,), span)
    return best


def best_run_tiou(context: AudioContext, gold: Span, max_units: int) -> Candidate:
    """The best contiguous run of 1..``max_units`` units in ``context.units`` order.

    A run's span covers all of its units. Candidates are visited by start
    position, then by length, and only a strictly better tIoU replaces the
    current best, so ties go to the earlier start and then the shorter run.
    A unit without usable bounds ends every run through it.
    """
    if max_units < 1:
        raise ValueError(f'max_units must be >= 1, got {max_units}')
    units = context.units
    best = NO_CANDIDATE
    for first in range(len(units)):
        start = end = None
        for length in range(1, min(max_units, len(units) - first) + 1):
            span = _unit_span(units[first + length - 1])
            if span is None:
                break
            start = span[0] if start is None else min(start, span[0])
            end = span[1] if end is None else max(end, span[1])
            tiou = temporal_iou(gold, (start, end))
            if best.span is None or tiou > best.tiou:
                ids = tuple(unit.unit_id for unit in units[first:first + length])
                best = Candidate(tiou, ids, (start, end))
    return best


def best_word_window_tiou(
    context: AudioContext,
    gold: Span,
    unit_ids: Sequence[int],
    max_words: int = DEFAULT_MAX_WORDS,
) -> Candidate:
    """The best contiguous window of at most ``max_words`` words in the named units.

    Words are taken in order from the units that exist; unknown ids and
    words without usable timing are skipped. The cap keeps the search at
    ``len(words) * max_words`` IoU evaluations; gold spans are a few seconds
    long, so a window of forty words (fifteen seconds or so) loses nothing.
    """
    if max_words < 1:
        raise ValueError(f'max_words must be >= 1, got {max_words}')
    units = [unit for unit in (context.unit_by_id(unit_id) for unit_id in unit_ids) if unit is not None]
    words = _timed_words(units)
    best = NO_CANDIDATE
    for first in range(len(words)):
        start = end = None
        for last in range(first, min(first + max_words, len(words))):
            word = words[last]
            start = word.start_s if start is None else min(start, word.start_s)
            end = word.end_s if end is None else max(end, word.end_s)
            span = (float(start), float(end))
            tiou = temporal_iou(gold, span)
            if best.span is None or tiou > best.tiou:
                best = Candidate(tiou, (words[first].word_id, word.word_id), span)
    return best


def candidate_names(ks: Sequence[int]) -> List[str]:
    """The candidate keys of an oracle entry, in table order."""
    return ['unit'] + [f'run{k}' for k in ks] + ['words']


def oracle_for_positive(
    context: AudioContext,
    gold: Span,
    ks: Sequence[int] = DEFAULT_KS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> Dict[str, Any]:
    """Every ceiling for one annotated yes question, as a JSON-ready dict.

    The word window is searched inside the run found for the largest k: that
    search covers every shorter run, so its result is the best run overall
    and the one with the most words to choose from.
    """
    if not ks:
        raise ValueError('ks must name at least one run length')
    candidates: Dict[str, Candidate] = {'unit': best_unit_tiou(context, gold)}
    for k in ks:
        candidates[f'run{k}'] = best_run_tiou(context, gold, k)
    widest = candidates[f'run{max(ks)}']
    candidates['words'] = best_word_window_tiou(context, gold, widest.ids, max_words)
    entry: Dict[str, Any] = {'gold': [float(gold[0]), float(gold[1])]}
    entry.update({name: candidate.to_dict() for name, candidate in candidates.items()})
    return entry


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def padded_mean_tiou(
    pairs: Sequence[Tuple[Span, Span]],
    pre: float,
    post: float,
    denominator: Optional[int] = None,
) -> float:
    """Mean tIoU after widening every predicted span, start clamped at zero.

    ``denominator`` lets the caller average over every positive rather than
    only the pairs that have a span, so the number is comparable with the
    unpadded means (a positive with no candidate span scores 0 in both).
    """
    total = sum(
        temporal_iou(gold, (max(0.0, start - pre), max(0.0, end + post)))
        for gold, (start, end) in pairs
    )
    count = len(pairs) if denominator is None else denominator
    return total / count if count else 0.0


def _default_calibrator(pairs: Sequence[Tuple[Span, Span]], grid: Sequence[float]) -> Tuple[float, float]:
    from evidence import calibrate_padding

    return calibrate_padding(pairs, grid)


def summarise(
    entries: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = DEFAULT_KS,
    pad_grid: Sequence[float] = DEFAULT_PAD_GRID,
    calibrate: Optional[Calibrator] = None,
) -> Dict[str, Any]:
    """Means over the per-question entries plus the best fixed padding per candidate.

    ``calibrate`` defaults to ``evidence.calibrate_padding`` and is injectable
    so the aggregation can be tested without the evidence module. Padded
    means use every positive as the denominator (see ``padded_mean_tiou``).
    """
    calibrate = calibrate or _default_calibrator
    names = candidate_names(ks)
    means = {name: _mean([float(entry[name]['tiou']) for entry in entries]) for name in names}
    padding: Dict[str, Any] = {}
    for name in names:
        pairs = [
            (tuple(entry['gold']), tuple(entry[name]['span']))
            for entry in entries if entry[name].get('span') is not None
        ]
        if not pairs:
            padding[name] = {'pre': 0.0, 'post': 0.0, 'mean_tiou': 0.0, 'n': 0}
            continue
        pre, post = calibrate(pairs, list(pad_grid))
        padding[name] = {
            'pre': float(pre),
            'post': float(post),
            'mean_tiou': padded_mean_tiou(pairs, pre, post, denominator=len(entries)),
            'n': len(pairs),
        }
    return {
        'n_positives': len(entries),
        'zero_overlap': sum(1 for entry in entries if float(entry['unit']['tiou']) <= 0.0),
        'means': means,
        'padding': padding,
    }


def format_summary_table(summary: Mapping[str, Any]) -> str:
    lines = [
        f'{"candidate":<12} {"mean tIoU":>9} {"+ best pad":>10}   pad (pre, post)',
        f'n={summary["n_positives"]} annotated yes questions, '
        f'{summary["zero_overlap"]} with no overlapping unit',
    ]
    for name, value in summary['means'].items():
        pad = summary['padding'].get(name)
        if pad is None:
            lines.append(f'{name:<12} {value:9.3f}')
            continue
        lines.append(
            f'{name:<12} {value:9.3f} {pad["mean_tiou"]:10.3f}   ({pad["pre"]:+.1f}s, {pad["post"]:+.1f}s)'
        )
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--transcripts', type=Path, required=True,
                        help='Directory holding contexts/<transcript_id>.json (a transcribe_all output).')
    parser.add_argument('--split', choices=SPLIT_NAMES, default='dev')
    parser.add_argument('--split-file', type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument('--ks', type=int, nargs='+', default=list(DEFAULT_KS))
    parser.add_argument('--max-words', type=int, default=DEFAULT_MAX_WORDS)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    transcript_ids = select_transcripts(args.split, args.split_file)
    contexts, missing = load_contexts(Path(args.transcripts) / 'contexts', transcript_ids)
    if missing:
        print(f'no cached context for {missing}; skipped', file=sys.stderr)

    entries: List[Dict[str, Any]] = []
    for transcript_id, rows in conversations_for(transcript_ids):
        context = contexts.get(transcript_id)
        if context is None:
            continue
        for row in rows:
            gold = gold_evidence(row)
            if scoring.as_int(row['label'], 'label') != YES or gold is None:
                continue
            entry = oracle_for_positive(context, gold, args.ks, args.max_words)
            entry.update(question_id=str(row['question_id']), transcript_id=transcript_id)
            entries.append(entry)

    summary = summarise(entries, args.ks, DEFAULT_PAD_GRID)
    print(format_summary_table(summary))

    hashes = {context.asr_config_hash for context in contexts.values()}
    asr_hash = next(iter(hashes)) if len(hashes) == 1 else Path(args.transcripts).name
    out = Path(args.out_dir) / f'evidence_oracle_{asr_hash[:12]}_{args.split}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'timestamp': utc_now(),
        'transcripts_dir': str(args.transcripts),
        'asr_config_hash': asr_hash,
        'split': args.split,
        'missing_contexts': missing,
        'ks': list(args.ks),
        'max_words': args.max_words,
        'pad_grid': list(DEFAULT_PAD_GRID),
        **summary,
        'per_question': entries,
    }, indent=1, default=json_default) + '\n', encoding='utf-8')
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
