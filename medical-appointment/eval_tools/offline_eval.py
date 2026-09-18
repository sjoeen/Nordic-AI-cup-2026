"""Score a QA backend and evidence policy on cached transcripts, offline.

    .venv/bin/python -m eval_tools.offline_eval --transcripts work/transcripts/<hash> --split dev
    .venv/bin/python -m eval_tools.offline_eval --transcripts work/transcripts/<hash> \\
        --qa-backend retrieval_nli --evidence-mode best_run --pad-pre 0.2 --qa-option top_k=4

The ASR stage is the expensive one and does not change between QA
experiments, so it is read from ``<transcripts>/contexts/<transcript_id>.json``
(written by ``transcribe_all``) and only QA + evidence selection run here.
Each run lands in ``results/runs/<run_id>/`` as ``records.jsonl`` (one line
per question with the backend's diagnostics, for ``error_analysis``),
``summary.json`` and ``report.txt`` (the reference report layout plus a
per-transcript table), and one line is appended to ``results/index.csv`` so
runs can be compared without opening them.

A transcript with no cached context is scored as an unsent conversation:
every one of its questions counts wrong, as the service would count it. The
labels come from the CSV rows (the evaluation view) and are used only to
score; nothing here feeds them to the backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from core_types import AudioContext, QABackend, QAResult, Span
from eval_tools import scoring
from eval_tools.split import (
    DEFAULT_SPLIT_PATH,
    SPLIT_NAMES,
    conversations_for,
    safe_id,
    select_transcripts,
)

DEFAULT_OUT_DIR = Path('results') / 'runs'
DEFAULT_INDEX_PATH = Path('results') / 'index.csv'
INDEX_COLUMNS: Tuple[str, ...] = (
    'run_id', 'timestamp', 'split', 'qa_backend', 'asr_config_hash',
    'accuracy', 'mean_tiou', 'score', 'notes',
)
QA_BACKENDS: Tuple[str, ...] = ('lexical', 'retrieval_nli', 'llm')

SpanSelector = Callable[[AudioContext, QAResult, Any, Optional[str]], Optional[Span]]


# --------------------------------------------------------------------------- #
# Small pure helpers
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def default_run_id(qa_backend: str, split: str, now: Optional[datetime] = None) -> str:
    """``YYYYmmddTHHMMSS_<backend>_<split>``: sortable, unique enough, path-safe."""
    stamp = (now or datetime.now(timezone.utc)).strftime('%Y%m%dT%H%M%S')
    return safe_id(f'{stamp}_{qa_backend}_{split}', 'run_id')


def parse_qa_options(options: Optional[Sequence[str]], prefix: str = 'MA_QA_') -> Dict[str, str]:
    """``['top_k=4', 'yes_threshold=0.6']`` -> the env mapping ``QAConfig.from_env`` reads.

    Reusing the env parser means every ``QAConfig`` field is settable from
    the command line with the exact validation the server applies, and this
    module never has to know the field list.
    """
    env: Dict[str, str] = {}
    for option in options or ():
        key, sep, value = str(option).partition('=')
        key = key.strip()
        if not sep or not key or not key.replace('_', '').isalnum():
            raise ValueError(f'--qa-option expects FIELD=VALUE, got {option!r}')
        env[f'{prefix}{key.upper()}'] = value.strip()
    return env


def json_default(value: Any) -> Any:
    """Make numpy scalars and other odd diagnostics serialisable."""
    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)


def append_index_row(path: Path, row: Mapping[str, Any]) -> Path:
    """Append one run to the CSV index, writing the header on first use."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new_file = not target.exists() or target.stat().st_size == 0
    with open(target, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS), extrasaction='ignore')
        if new_file:
            writer.writeheader()
        writer.writerow({column: row.get(column, '') for column in INDEX_COLUMNS})
    return target


def index_row(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``index.csv`` line for a run summary (numbers rounded for reading)."""
    scores = summary['scores']
    return {
        'run_id': summary['run_id'],
        'timestamp': summary['timestamp'],
        'split': summary['split'],
        'qa_backend': summary['qa_backend'],
        'asr_config_hash': summary.get('asr_config_hash', ''),
        'accuracy': f'{scores["accuracy"]:.4f}',
        'mean_tiou': f'{scores["mean_tiou"]:.4f}',
        'score': f'{scores["score"]:.4f}',
        'notes': summary.get('notes', ''),
    }


# --------------------------------------------------------------------------- #
# Evaluation core (models are injected so it is testable with fakes)
# --------------------------------------------------------------------------- #

def evaluate_conversation(
    context: AudioContext,
    rows: Sequence[Mapping[str, Any]],
    qa: QABackend,
    policy: Any,
    select_span: SpanSelector,
) -> List[Dict[str, Any]]:
    """Records for one conversation: QA, then a span for every yes answer.

    A failure anywhere in the conversation scores all of its questions as
    unanswered, mirroring what an exception inside ``predict`` would cost on
    the wire, and the error is kept in the diagnostics so it can be found.
    """
    questions = [str(row['question']) for row in rows]
    started = time.monotonic()
    try:
        results = qa.answer(context, questions, None)
        if len(results) != len(questions):
            raise ValueError(f'QA returned {len(results)} results for {len(questions)} questions')
        spans = [
            select_span(context, result, policy, question) if result.answer else None
            for result, question in zip(results, questions)
        ]
    except Exception as exc:  # noqa: BLE001 - one bad conversation must not stop the run
        seconds = time.monotonic() - started
        failed: List[Dict[str, Any]] = []
        for row, question in zip(rows, questions):
            record = scoring.unanswered_record(
                row, question=question, correct=False,
                diagnostics={'error': f'{type(exc).__name__}: {exc}', 'seconds': seconds},
            )
            record['tiou'] = scoring.record_tiou(record)
            failed.append(record)
        return failed
    seconds = time.monotonic() - started

    records: List[Dict[str, Any]] = []
    for row, question, result, span in zip(rows, questions, results, spans):
        record = scoring.make_record(
            row, int(bool(result.answer)), span, question=question,
            diagnostics={
                'backend': result.backend,
                'confidence': result.confidence,
                'evidence_unit_ids': list(result.evidence_unit_ids),
                'evidence_word_ids': (
                    None if result.evidence_word_ids is None else list(result.evidence_word_ids)
                ),
                'qa': dict(result.diagnostics),
                'seconds': seconds / max(1, len(questions)),
            },
        )
        record['tiou'] = scoring.record_tiou(record)
        record['correct'] = record['prediction'] == record['label']
        records.append(record)
    return records


def load_contexts(
    contexts_dir: Path,
    transcript_ids: Sequence[str],
) -> Tuple[Dict[str, AudioContext], List[str]]:
    """Cached contexts by transcript id, and the ids that have none."""
    from transcript import load_context

    contexts: Dict[str, AudioContext] = {}
    missing: List[str] = []
    for transcript_id in transcript_ids:
        path = Path(contexts_dir) / f'{safe_id(transcript_id)}.json'
        if not path.exists():
            missing.append(transcript_id)
            continue
        contexts[transcript_id] = load_context(path)
    return contexts, missing


def evaluate_split(
    conversations: Sequence[Tuple[str, Sequence[Mapping[str, Any]]]],
    contexts: Mapping[str, AudioContext],
    qa: QABackend,
    policy: Any,
    select_span: SpanSelector,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Records for every conversation, in CSV order; missing contexts score wrong."""
    records: List[Dict[str, Any]] = []
    for transcript_id, rows in conversations:
        context = contexts.get(transcript_id)
        if context is None:
            for row in rows:
                record = scoring.unanswered_record(
                    row, question=str(row['question']), correct=False,
                    diagnostics={'error': 'no cached context'},
                )
                record['tiou'] = scoring.record_tiou(record)
                records.append(record)
            print(f'  {transcript_id}: no cached context, {len(rows)} questions scored wrong', file=sys.stderr)
            continue
        conversation_records = evaluate_conversation(context, rows, qa, policy, select_span)
        records.extend(conversation_records)
        if verbose:
            for record in conversation_records:
                mark = 'ok  ' if record['prediction'] == record['label'] else 'WRONG'
                said = {1: 'yes', 0: 'no'}.get(record['prediction'], '-')
                tiou = record.get('tiou')
                evidence = f' tIoU {tiou:.3f}' if tiou is not None else ''
                print(f'  {mark} {record["question_id"]:<24} {record["question_type"]:<14} said {said:<3}{evidence}')
    return records


# --------------------------------------------------------------------------- #
# Persisting a run
# --------------------------------------------------------------------------- #

def write_run(run_dir: Path, records: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> Dict[str, Path]:
    """Write ``records.jsonl``, ``summary.json`` and ``report.txt`` into ``run_dir``."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        'records': run_dir / 'records.jsonl',
        'summary': run_dir / 'summary.json',
        'report': run_dir / 'report.txt',
    }
    with open(paths['records'], 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, default=json_default, ensure_ascii=False) + '\n')
    paths['summary'].write_text(
        json.dumps(summary, indent=1, default=json_default, ensure_ascii=False) + '\n', encoding='utf-8',
    )
    header = [
        f'run {summary["run_id"]}  ({summary["timestamp"]})',
        f'split {summary["split"]}  qa {summary["qa_backend"]}  asr {summary.get("asr_config_hash", "")[:12]}',
        f'evidence {summary.get("evidence_policy")}',
    ]
    if summary.get('missing_contexts'):
        header.append(f'missing contexts (scored wrong): {summary["missing_contexts"]}')
    paths['report'].write_text('\n'.join(header) + '\n' + scoring.render_report(records) + '\n', encoding='utf-8')
    return paths


def build_summary(
    run_id: str,
    split: str,
    qa_backend_name: str,
    qa_config: Mapping[str, Any],
    policy: Any,
    transcripts_dir: Path,
    asr_config_hash: str,
    records: Sequence[Mapping[str, Any]],
    missing_contexts: Sequence[str],
    timing: Mapping[str, float],
    notes: str,
    argv: Sequence[str],
) -> Dict[str, Any]:
    return {
        'run_id': run_id,
        'timestamp': utc_now(),
        'split': split,
        'qa_backend': qa_backend_name,
        'qa_config': dict(qa_config),
        'evidence_policy': asdict(policy) if hasattr(policy, '__dataclass_fields__') else str(policy),
        'transcripts_dir': str(transcripts_dir),
        'asr_config_hash': asr_config_hash,
        'n_transcripts': len({record['transcript_id'] for record in records}),
        'missing_contexts': list(missing_contexts),
        'scores': scoring.score_records(records),
        'per_transcript': scoring.per_transcript(records),
        'timing': dict(timing),
        'notes': notes,
        'argv': list(argv),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--transcripts', type=Path, required=True,
                        help='Directory holding contexts/<transcript_id>.json (a transcribe_all output).')
    parser.add_argument('--split', choices=SPLIT_NAMES, default='dev')
    parser.add_argument('--split-file', type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument('--qa-backend', choices=QA_BACKENDS, default='lexical')
    parser.add_argument('--qa-option', action='append', default=[], metavar='FIELD=VALUE',
                        help='Override a QAConfig field, e.g. top_k=4 (repeatable).')
    parser.add_argument('--evidence-mode', default='best_run')
    parser.add_argument('--max-span-s', type=float, default=15.0)
    parser.add_argument('--pad-pre', type=float, default=0.0)
    parser.add_argument('--pad-post', type=float, default=0.0)
    parser.add_argument('--no-trim-filler', action='store_true')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--index', type=Path, default=DEFAULT_INDEX_PATH)
    parser.add_argument('--notes', default='')
    parser.add_argument('--limit', type=int, default=None, help='Only the first N transcripts of the split.')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)

    from evidence import EvidencePolicy, select_span
    from qa_backend import QAConfig, build_qa_backend

    env = parse_qa_options(args.qa_option)
    env['MA_QA_BACKEND'] = args.qa_backend
    qa_config = QAConfig.from_env(prefix='MA_QA_', env=env)
    policy = EvidencePolicy(
        mode=args.evidence_mode, max_span_s=args.max_span_s, pad_pre_s=args.pad_pre,
        pad_post_s=args.pad_post, trim_filler=not args.no_trim_filler,
    )
    run_id = safe_id(args.run_id, 'run_id') if args.run_id else default_run_id(args.qa_backend, args.split)

    transcript_ids = select_transcripts(args.split, args.split_file)
    if args.limit is not None:
        transcript_ids = transcript_ids[:max(0, args.limit)]
    conversations = conversations_for(transcript_ids)
    contexts_dir = Path(args.transcripts) / 'contexts'
    contexts, missing = load_contexts(contexts_dir, transcript_ids)
    asr_hashes = {context.asr_config_hash for context in contexts.values()}
    asr_config_hash = next(iter(asr_hashes)) if len(asr_hashes) == 1 else Path(args.transcripts).name

    timing: Dict[str, float] = {}
    started = time.monotonic()
    qa = build_qa_backend(qa_config)
    qa.warm_up()
    timing['warm_up_s'] = time.monotonic() - started
    started = time.monotonic()
    records = evaluate_split(conversations, contexts, qa, policy, select_span, verbose=args.verbose)
    timing['evaluate_s'] = time.monotonic() - started

    summary = build_summary(
        run_id, args.split, args.qa_backend, asdict(qa_config), policy, args.transcripts,
        asr_config_hash, records, missing, timing, args.notes, list(argv if argv is not None else sys.argv[1:]),
    )
    paths = write_run(Path(args.out) / run_id, records, summary)
    append_index_row(args.index, index_row(summary))
    print(paths['report'].read_text(encoding='utf-8'))
    print(f'run {run_id}: {paths["summary"]}  (evaluate {timing["evaluate_s"]:.1f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
