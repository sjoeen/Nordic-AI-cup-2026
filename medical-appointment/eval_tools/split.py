"""Grouped development/holdout split of the supplied conversations.

    .venv/bin/python -m eval_tools.split            # show the split
    .venv/bin/python -m eval_tools.split --write    # write manifests/split_v1.json

Why grouped: two recordings that are the same audio, or that carry (nearly)
the same ten questions, are one script recorded twice; letting one tune the
system while the other measures it would inflate the holdout number. Whole
groups therefore move together. Why frozen in a manifest: the holdout is only
worth something if every tool reads the same membership, so the split is
written once, with the seed and the hashes it was derived from, and refuses
to overwrite itself unless forced.

Transcript ids come from the CSV and end up as file names in other tools, so
``safe_id`` is the one gate every id passes before touching a path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from utils import (
    QUESTIONS_CSV,
    audio_filename_for_transcript,
    load_sample_audio,
    load_sample_questions,
)

DEFAULT_SPLIT_PATH = Path('manifests') / 'split_v1.json'
DEFAULT_SEED = 2026
DEFAULT_N_HOLDOUT = 8
DEFAULT_FORCE_DEV: Tuple[str, ...] = ('sample_20',)
DEFAULT_JACCARD = 0.5
SPLIT_NAMES: Tuple[str, ...] = ('dev', 'holdout', 'all')

_SAFE_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$')
_NON_ALNUM = re.compile(r'[^a-z0-9 ]+')
_SPACES = re.compile(r'\s+')

Rows = Sequence[Mapping[str, Any]]
RowsByTranscript = Mapping[str, Rows]


# --------------------------------------------------------------------------- #
# Ids and normalisation
# --------------------------------------------------------------------------- #

def safe_id(value: Any, what: str = 'transcript_id') -> str:
    """``value`` as a string safe to use as one path component, or ``ValueError``.

    Ids are data: the CSV, a manifest or a command line could hand over
    ``../x``. Nothing in the tools builds a path from an id without this.
    """
    text = str(value)
    if not _SAFE_ID.fullmatch(text) or '..' in text:
        raise ValueError(f'{what} is not a safe path component: {text!r}')
    return text


def normalise_question(text: Any) -> str:
    """Lower-cased alphanumerics with single spaces, for near-duplicate matching."""
    lowered = str(text or '').lower()
    return _SPACES.sub(' ', _NON_ALNUM.sub('', lowered)).strip()


def question_jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity of two question sets; two empty sets are unrelated (0)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# --------------------------------------------------------------------------- #
# Loading the CSV in transcript order
# --------------------------------------------------------------------------- #

def load_rows_by_transcript(rows: Optional[Iterable[Mapping[str, Any]]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """CSV rows grouped by transcript, both in first-appearance order.

    This is the order the evaluator sends conversations and questions in, so
    every tool that iterates the split iterates in this order.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = collections.OrderedDict()
    for row in (load_sample_questions() if rows is None else rows):
        grouped.setdefault(safe_id(row['transcript_id']), []).append(dict(row))
    return grouped


def compute_audio_hashes(transcript_ids: Iterable[str]) -> Dict[str, str]:
    """SHA-256 of each supplied recording, keyed by transcript id."""
    return {
        transcript_id: hashlib.sha256(
            load_sample_audio(audio_filename_for_transcript(safe_id(transcript_id)))
        ).hexdigest()
        for transcript_id in transcript_ids
    }


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #

def build_groups(
    rows_by_transcript: RowsByTranscript,
    audio_hashes: Mapping[str, str],
    jaccard_threshold: float = DEFAULT_JACCARD,
) -> List[List[str]]:
    """Connected components of transcripts that must stay on the same side.

    Two transcripts are linked when their audio bytes hash identically or the
    Jaccard similarity of their normalised question sets is at least
    ``jaccard_threshold``. Components are returned in first-appearance order
    with members in first-appearance order, so the result is deterministic
    for a given input order. A transcript missing from ``audio_hashes`` is
    treated as having unique audio.
    """
    ids = list(rows_by_transcript)
    parent = {transcript_id: transcript_id for transcript_id in ids}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            # Keep the earlier id as the root so representative order is stable.
            if ids.index(root_a) < ids.index(root_b):
                parent[root_b] = root_a
            else:
                parent[root_a] = root_b

    by_hash: Dict[str, str] = {}
    for transcript_id in ids:
        digest = audio_hashes.get(transcript_id)
        if digest is None:
            continue
        if digest in by_hash:
            union(by_hash[digest], transcript_id)
        else:
            by_hash[digest] = transcript_id

    questions = {
        transcript_id: {normalise_question(row.get('question')) for row in rows} - {''}
        for transcript_id, rows in rows_by_transcript.items()
    }
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if question_jaccard(questions[a], questions[b]) >= jaccard_threshold:
                union(a, b)

    groups: Dict[str, List[str]] = collections.OrderedDict()
    for transcript_id in ids:
        groups.setdefault(find(transcript_id), []).append(transcript_id)
    return list(groups.values())


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def make_split(
    rows_by_transcript: Optional[RowsByTranscript] = None,
    audio_hashes: Optional[Mapping[str, str]] = None,
    seed: int = DEFAULT_SEED,
    n_holdout: int = DEFAULT_N_HOLDOUT,
    force_dev: Sequence[str] = DEFAULT_FORCE_DEV,
    jaccard_threshold: float = DEFAULT_JACCARD,
    source_csv_sha256: Optional[str] = None,
    created: Optional[str] = None,
) -> Dict[str, Any]:
    """Sample whole groups into the holdout until ``n_holdout`` transcripts are in it.

    Groups containing a ``force_dev`` transcript are never candidates (the
    longest clip is a stress fixture the development loop needs repeatedly,
    and a holdout that is re-inspected is not a holdout). Candidate groups
    are shuffled with ``random.Random(seed)`` and taken in that order while
    the holdout is short of ``n_holdout``, so it can exceed the target by at
    most one group's excess members and never falls short while candidates
    remain. Defaults load the supplied CSV and hash the supplied audio.
    """
    if rows_by_transcript is None:
        rows_by_transcript = load_rows_by_transcript()
    ids = [safe_id(transcript_id) for transcript_id in rows_by_transcript]
    if audio_hashes is None:
        audio_hashes = compute_audio_hashes(ids)
    if source_csv_sha256 is None:
        source_csv_sha256 = sha256_of_file(QUESTIONS_CSV) if QUESTIONS_CSV.exists() else ''
    if n_holdout < 0 or n_holdout > len(ids):
        raise ValueError(f'n_holdout must be in [0, {len(ids)}], got {n_holdout}')
    forced = [safe_id(item) for item in force_dev]
    unknown = sorted(set(forced) - set(ids))
    if unknown:
        # A typo here would silently drop the protection, so it is an error.
        raise ValueError(f'force_dev ids not in the data: {unknown}')

    groups = build_groups(rows_by_transcript, audio_hashes, jaccard_threshold)
    candidates = [group for group in groups if not set(group) & set(forced)]
    order = list(range(len(candidates)))
    random.Random(seed).shuffle(order)

    holdout: Set[str] = set()
    for index in order:
        if len(holdout) >= n_holdout:
            break
        holdout.update(candidates[index])

    return {
        'version': 1,
        'seed': int(seed),
        'created': created or datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'n_holdout_requested': int(n_holdout),
        'force_dev': forced,
        'jaccard_threshold': float(jaccard_threshold),
        'groups': groups,
        'transcripts': ids,
        'dev': [transcript_id for transcript_id in ids if transcript_id not in holdout],
        'holdout': [transcript_id for transcript_id in ids if transcript_id in holdout],
        'source_csv_sha256': source_csv_sha256,
        'audio_sha256': {transcript_id: audio_hashes.get(transcript_id, '') for transcript_id in ids},
    }


# --------------------------------------------------------------------------- #
# Manifest I/O and selection
# --------------------------------------------------------------------------- #

def write_split(split: Mapping[str, Any], path: Path = DEFAULT_SPLIT_PATH, overwrite: bool = False) -> Path:
    """Write the manifest; refuses to replace an existing one unless ``overwrite``."""
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f'{target} exists; a frozen split is not rewritten without --force')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(split, indent=1, sort_keys=False) + '\n', encoding='utf-8')
    return target


def validate_split(split: Mapping[str, Any]) -> Dict[str, Any]:
    """Check the shape a manifest must have before anything trusts it."""
    for key in ('dev', 'holdout'):
        if not isinstance(split.get(key), list):
            raise ValueError(f'split manifest is missing the {key!r} list')
    dev = [safe_id(item) for item in split['dev']]
    holdout = [safe_id(item) for item in split['holdout']]
    overlap = sorted(set(dev) & set(holdout))
    if overlap:
        raise ValueError(f'transcripts in both dev and holdout: {overlap}')
    if len(set(dev)) != len(dev) or len(set(holdout)) != len(holdout):
        raise ValueError('split manifest lists a transcript twice')
    return dict(split)


def load_split(path: Path = DEFAULT_SPLIT_PATH) -> Dict[str, Any]:
    """Read and validate a manifest written by ``write_split``."""
    return validate_split(json.loads(Path(path).read_text(encoding='utf-8')))


def transcripts_for(split: Mapping[str, Any], name: str) -> List[str]:
    """The transcript ids of ``'dev'``, ``'holdout'`` or ``'all'`` in CSV order."""
    if name not in SPLIT_NAMES:
        raise ValueError(f'split name must be one of {SPLIT_NAMES}, got {name!r}')
    split = validate_split(split)
    if name != 'all':
        return list(split[name])
    everything = split.get('transcripts')
    if isinstance(everything, list) and set(everything) == set(split['dev']) | set(split['holdout']):
        return [safe_id(item) for item in everything]
    return list(split['dev']) + list(split['holdout'])


def select_transcripts(name: str, split_file: Optional[Path]) -> List[str]:
    """Transcript ids for a CLI ``--split`` choice.

    ``'all'`` without a manifest on disk means every transcript in the CSV, so
    the tools work before the split is frozen; ``'dev'``/``'holdout'``
    require the manifest, because guessing a holdout defeats its purpose.
    """
    if name not in SPLIT_NAMES:
        raise ValueError(f'split name must be one of {SPLIT_NAMES}, got {name!r}')
    path = Path(split_file) if split_file is not None else DEFAULT_SPLIT_PATH
    if name == 'all' and not path.exists():
        return list(load_rows_by_transcript())
    if not path.exists():
        raise FileNotFoundError(f'{path} not found; run "python -m eval_tools.split --write" first')
    return transcripts_for(load_split(path), name)


def conversations_for(
    transcript_ids: Iterable[str],
    rows_by_transcript: Optional[RowsByTranscript] = None,
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """``(transcript_id, rows)`` for the selected transcripts, in CSV order."""
    if rows_by_transcript is None:
        rows_by_transcript = load_rows_by_transcript()
    wanted = {safe_id(transcript_id) for transcript_id in transcript_ids}
    missing = sorted(wanted - set(rows_by_transcript))
    if missing:
        raise ValueError(f'transcripts not in the CSV: {missing}')
    return [
        (transcript_id, [dict(row) for row in rows])
        for transcript_id, rows in rows_by_transcript.items()
        if transcript_id in wanted
    ]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def describe(split: Mapping[str, Any], rows_by_transcript: Optional[RowsByTranscript] = None) -> str:
    """Human summary: group sizes, membership and label balance per side."""
    lines = [f'seed {split["seed"]}, {len(split["dev"])} dev / {len(split["holdout"])} holdout']
    multi = [group for group in split.get('groups', []) if len(group) > 1]
    lines.append(f'groups with more than one transcript: {multi if multi else "none"}')
    for name in ('dev', 'holdout'):
        members = split[name]
        lines.append(f'{name} ({len(members)}): {", ".join(members)}')
        if rows_by_transcript is not None:
            counts: Dict[str, int] = collections.Counter(
                str(row.get('question_type', '?'))
                for transcript_id in members
                for row in rows_by_transcript.get(transcript_id, [])
            )
            lines.append(f'  question types: {dict(sorted(counts.items()))}')
    return '\n'.join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--write', action='store_true', help='Write the manifest.')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing manifest.')
    parser.add_argument('--out', type=Path, default=DEFAULT_SPLIT_PATH)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--n-holdout', type=int, default=DEFAULT_N_HOLDOUT)
    parser.add_argument('--force-dev', nargs='*', default=list(DEFAULT_FORCE_DEV))
    parser.add_argument('--jaccard', type=float, default=DEFAULT_JACCARD)
    args = parser.parse_args(argv)

    rows_by_transcript = load_rows_by_transcript()
    split = make_split(
        rows_by_transcript, seed=args.seed, n_holdout=args.n_holdout,
        force_dev=args.force_dev, jaccard_threshold=args.jaccard,
    )
    print(describe(split, rows_by_transcript))
    if args.write:
        path = write_split(split, args.out, overwrite=args.force)
        print(f'wrote {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
