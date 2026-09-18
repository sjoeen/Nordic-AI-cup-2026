"""Turn raw ASR segments into clause-sized, well-formed transcript units.

Why this module exists
----------------------
Gold evidence spans in the training data are short (median about three
seconds) while whisper segments routinely run ten to thirty seconds. If the
evidence stage can only point at whole whisper segments, temporal IoU stays
low even when the QA stage picks the right passage. So the pipeline re-cuts
segments at sentence punctuation and long pauses, using whisper's word
timings, and caps unit length by splitting at the widest internal pause.

Whisper word texts carry a leading space and attached punctuation
(``" hello"``, ``"hello."``); unit text is rebuilt from the stripped words so
that a unit reads exactly like the transcript it was cut from.

Everything here is pure: no models, and no I/O except ``save_context`` /
``load_context``. Text is untrusted data; it is only compared, split and
joined, never interpreted.
"""

from __future__ import annotations

import math
import numbers
import os
import tempfile
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from core_types import AudioContext, TranscriptUnit, Word, check_units

__all__ = [
    'ABBREVIATIONS',
    'DEFAULT_MAX_UNIT_S',
    'DEFAULT_PAUSE_GAP_S',
    'SENTENCE_END_PUNCT',
    'Window',
    'assign_word_ids',
    'build_context',
    'ends_sentence',
    'load_context',
    'normalize_text',
    'resegment',
    'sanitize_units',
    'save_context',
    'tokenize',
    'unit_index',
    'windows',
    'word_ids_valid',
]

#: Longest unit the evidence stage should have to point at. Gold spans are a
#: few seconds long; eight seconds keeps IoU reasonable when a unit is chosen
#: whole, without shredding sentences into fragments.
DEFAULT_MAX_UNIT_S = 8.0
#: A silence longer than this between two words is treated as a clause break.
DEFAULT_PAUSE_GAP_S = 0.7
#: Word endings that close a clause. ':' is opt-in via ``split_on_colon``.
SENTENCE_END_PUNCT = '.?!;'
#: Abbreviations whose trailing period does not end a sentence ("Dr. Smith").
ABBREVIATIONS = frozenset({'dr', 'mr', 'mrs', 'ms', 'prof', 'st', 'vs', 'e.g', 'i.e', 'a.m', 'p.m'})

#: Closing quotes and brackets whisper may append after the sentence mark.
_TRAILING_WRAPPERS = '"\'”’)]}'
#: Tolerance for "did the clamp actually move a timestamp"; matches check_units.
_EPS = 1e-6

# --------------------------------------------------------------------------- #
# Text normalisation
# --------------------------------------------------------------------------- #

# Curly quotes and primes become straight quotes so that "don’t" in a question
# and "don't" from whisper normalise identically. The fraction slash keeps
# NFKC-expanded fractions ("½" -> "1/2") in the numeric form factcheck reads.
_CHAR_TRANSLATION = str.maketrans({
    '‘': "'", '’': "'", '‚': "'", '‛': "'", '′': "'",
    '“': '"', '”': '"', '„': '"', '‟': '"', '″': '"',
    '⁄': '/',
})
# Every hyphen and dash becomes a space: "follow-up" and "follow up" must match.
_HYPHEN_TRANSLATION = str.maketrans({
    ch: ' ' for ch in '-‐‑‒–—―−'
})
# Punctuation that is part of a number when it sits between two digits:
# 0.3, 1,000, 130/85, 10:30. Anywhere else it is stripped.
_NUMERIC_JOINERS = '.,/:'


def normalize_text(text: Optional[str]) -> str:
    """Canonical lowercase form of ``text`` for lexical comparison.

    Lowercases, applies Unicode NFKC, straightens quotes, turns hyphens into
    spaces, drops apostrophes ("don't" -> "dont"), keeps ``. , / :`` only when
    they join two digits, keeps combining marks with the letter they modify,
    replaces all other punctuation by a space and collapses whitespace.
    Deterministic and idempotent; never interprets text.
    """
    if not text:
        return ''
    text = unicodedata.normalize('NFKC', str(text))
    text = text.translate(_CHAR_TRANSLATION).lower().translate(_HYPHEN_TRANSLATION)
    chars: List[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        if ch.isalnum() or ch.isspace() or unicodedata.category(ch).startswith('M'):
            # Combining marks (category M*) are not alphanumeric but belong to
            # the letter before them: without this "İstanbul" (whose lowercase
            # form is "i" + a combining dot) and every Devanagari or Thai word
            # would be cut into pieces.
            chars.append(ch)
        elif (
            ch in _NUMERIC_JOINERS
            and 0 < i < last
            and text[i - 1].isdigit()
            and text[i + 1].isdigit()
        ):
            chars.append(ch)
        elif ch == "'":
            # Apostrophes inside a word vanish rather than split, so "don't"
            # stays one token and "patient's" matches "patients"; between two
            # digits (5'11) they separate like other punctuation, otherwise
            # the numbers would fuse into a value nobody said.
            if 0 < i < last and text[i - 1].isdigit() and text[i + 1].isdigit():
                chars.append(' ')
            continue
        else:
            chars.append(' ')
    return ' '.join(''.join(chars).split())


def tokenize(text: Optional[str]) -> List[str]:
    """Whitespace tokens of ``normalize_text(text)``; decimals stay intact."""
    return normalize_text(text).split()


# --------------------------------------------------------------------------- #
# Timing hygiene
# --------------------------------------------------------------------------- #

def _as_float(value: Any) -> float:
    """``value`` as a finite float, or NaN when it is not a usable number.

    Accepts numpy scalars (they are ``numbers.Real``) but rejects bools and
    strings, because a boolean or textual timestamp is a caller bug, not data.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return math.nan
    value = float(value)
    return value if math.isfinite(value) else math.nan


def _is_integral(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _optional_score(value: Any) -> Optional[float]:
    """A diagnostic score (``avg_logprob``, ``no_speech_prob``) as a plain
    finite float, or ``None`` when absent or unusable.

    These fields are never used for decisions, but they travel into the JSON
    cache, where a numpy scalar or a NaN would make ``save_context`` fail.
    """
    if value is None:
        return None
    score = _as_float(value)
    return None if math.isnan(score) else score


def _optional_limit(value: Any, name: str, zero_is_off: bool) -> Optional[float]:
    """A resegmentation limit as a float, or ``None`` when the rule is off.

    ``None``, non-finite and negative values switch the rule off (``inf`` is
    a natural "no cap"; a negative pause would otherwise cut after every
    word). Zero also means off when ``zero_is_off`` (a zero-second cap is
    meaningless) and is a legitimate "any positive gap" otherwise. Anything
    that is not a real number, including bools and strings such as ``'8'``
    from an untyped config, raises ``TypeError``: a limit that silently
    stops applying would only show up much later as worse evidence spans.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a number or None, got {type(value).__name__}')
    limit = float(value)
    if not math.isfinite(limit) or limit < 0.0 or (zero_is_off and limit == 0.0):
        return None
    return limit


def _effective_limits(max_unit_s: Any, pause_gap_s: Any) -> Tuple[Optional[float], Optional[float]]:
    """The (max length, pause gap) that ``resegment`` will actually apply."""
    return (
        _optional_limit(max_unit_s, 'max_unit_s', zero_is_off=True),
        _optional_limit(pause_gap_s, 'pause_gap_s', zero_is_off=False),
    )


def _flag(value: Any, name: str) -> bool:
    """A resegmentation switch, which must be a real bool.

    A string such as ``'false'`` is truthy, so accepting it would silently
    keep the rule on; failing loudly is the only way a config typo is seen.
    """
    if not isinstance(value, bool):
        raise TypeError(f'{name} must be a bool, got {type(value).__name__}')
    return value


def _sanitize_unit(unit: TranscriptUnit) -> Tuple[TranscriptUnit, List[str]]:
    """Copy of ``unit`` with finite, ordered bounds and words clamped inside.

    Whisper word timings occasionally run past the segment end or come back
    non-monotonic; downstream stages assume ``start <= end`` and words inside
    their unit, so this is where that invariant is established. Every repair
    is reported so the pipeline knows the ASR output was not clean.
    """
    problems: List[str] = []
    label = f'unit {unit.unit_id}'
    words_in = list(unit.words or [])
    start, end = _as_float(unit.start_s), _as_float(unit.end_s)
    if math.isnan(start) or math.isnan(end):
        finite_word_times = [
            t for w in words_in for t in (_as_float(w.start_s), _as_float(w.end_s))
            if not math.isnan(t)
        ]
        if math.isnan(start) and math.isnan(end):
            start = min(finite_word_times, default=0.0)
            end = max(finite_word_times, default=0.0)
        elif math.isnan(start):
            start = min(finite_word_times + [end])
        else:
            end = max(finite_word_times + [start])
        problems.append(f'{label}: non-finite bounds replaced by [{start:.3f}, {end:.3f}]')
    if end < start:
        start, end = end, start
        problems.append(f'{label}: end before start (swapped)')

    words_out: List[Word] = []
    cursor = start  # end of the previous word: the best guess for a lost start
    for word in words_in:
        w_label = f'word {word.word_id}'
        w_start, w_end = _as_float(word.start_s), _as_float(word.end_s)
        if math.isnan(w_start) or math.isnan(w_end):
            problems.append(f'{w_label}: non-finite timing')
            if math.isnan(w_start):
                w_start = cursor if math.isnan(w_end) else min(w_end, cursor)
            if math.isnan(w_end):
                w_end = w_start
        c_start, c_end = _clamp(w_start, start, end), _clamp(w_end, start, end)
        if abs(c_start - w_start) > _EPS or abs(c_end - w_end) > _EPS:
            problems.append(f'{w_label}: clamped into {label}')
        if c_end < c_start:
            problems.append(f'{w_label}: end before start')
            c_end = c_start
        cursor = c_end
        probability = _as_float(word.probability) if word.probability is not None else None
        words_out.append(Word(
            word_id=int(word.word_id) if _is_integral(word.word_id) else word.word_id,
            start_s=c_start,
            end_s=c_end,
            text='' if word.text is None else str(word.text),
            probability=None if probability is None or math.isnan(probability) else probability,
        ))
    avg_logprob = _optional_score(unit.avg_logprob)
    no_speech_prob = _optional_score(unit.no_speech_prob)
    if unit.avg_logprob is not None and avg_logprob is None:
        problems.append(f'{label}: unusable avg_logprob dropped')
    if unit.no_speech_prob is not None and no_speech_prob is None:
        problems.append(f'{label}: unusable no_speech_prob dropped')
    clean = TranscriptUnit(
        unit_id=unit.unit_id,
        start_s=start,
        end_s=end,
        text='' if unit.text is None else str(unit.text),
        words=words_out,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
        speaker=None if unit.speaker is None else str(unit.speaker),
    )
    return clean, problems


def sanitize_units(units: Sequence[TranscriptUnit]) -> Tuple[List[TranscriptUnit], List[str]]:
    """Copies of ``units`` with sane timings, plus a description of each repair.

    Never mutates its input. Unit order and ids are kept; ``resegment`` is
    what renumbers. An empty repair list means the ASR output was clean.
    """
    clean: List[TranscriptUnit] = []
    problems: List[str] = []
    for unit in units:
        fixed, unit_problems = _sanitize_unit(unit)
        clean.append(fixed)
        problems.extend(unit_problems)
    return clean, problems


# --------------------------------------------------------------------------- #
# Word ids
# --------------------------------------------------------------------------- #

def word_ids_valid(units: Sequence[TranscriptUnit]) -> bool:
    """True when every word carries an integer id that is unique across units."""
    seen: set = set()
    for unit in units:
        for word in unit.words or []:
            if not _is_integral(word.word_id) or word.word_id in seen:
                return False
            seen.add(word.word_id)
    return True


def _time_key(unit: TranscriptUnit) -> Tuple[int, float]:
    """Stable sort key: finite starts in order, non-finite ones last."""
    start = _as_float(unit.start_s)
    return (1, 0.0) if math.isnan(start) else (0, start)


def assign_word_ids(units: Sequence[TranscriptUnit]) -> List[TranscriptUnit]:
    """Copies of ``units`` in time order with global word ids 0..n-1.

    Word ids are how QA backends refer to words without naming a unit, so they
    must be unique per request. Ids follow time order: units are stably
    sorted by start, words keep their spoken order inside each unit.
    """
    ordered: List[TranscriptUnit] = []
    next_id = 0
    for unit in sorted(units, key=_time_key):
        words: List[Word] = []
        for word in unit.words or []:
            words.append(replace(word, word_id=next_id))
            next_id += 1
        ordered.append(replace(unit, words=words))
    return ordered


# --------------------------------------------------------------------------- #
# Resegmentation
# --------------------------------------------------------------------------- #

def ends_sentence(word_text: Optional[str], punct: str = SENTENCE_END_PUNCT) -> bool:
    """True when a whisper word closes a clause: it ends in one of ``punct``.

    Trailing quotes/brackets are ignored (``today?"``) and a period after a
    known abbreviation (``Dr.``) does not count, because "Dr. Smith" split in
    two would leave a useless half-second unit.
    """
    # NFKC turns "…" into "..." and full-width "！" into "!", which whisper
    # emits now and then; without it those sentence ends would be missed.
    stripped = unicodedata.normalize('NFKC', word_text or '').strip().rstrip(_TRAILING_WRAPPERS)
    if not stripped or stripped[-1] not in punct:
        return False
    if stripped[-1] == '.' and stripped.rstrip('.').lower() in ABBREVIATIONS:
        return False
    return True


def _join_word_texts(words: Sequence[Word]) -> str:
    """Unit text rebuilt from whisper words (leading spaces stripped)."""
    return ' '.join(t for t in ((w.text or '').strip() for w in words) if t)


def _split_at_boundaries(
    words: Sequence[Word],
    gap_limit: Optional[float],
    punct: str,
) -> List[List[Word]]:
    """Cut ``words`` after sentence punctuation and after pauses longer than
    ``gap_limit`` (``None`` disables the pause rule)."""
    groups: List[List[Word]] = []
    current: List[Word] = []
    for i, word in enumerate(words):
        current.append(word)
        if i == len(words) - 1:
            break
        cut = bool(punct) and ends_sentence(word.text, punct)
        if not cut and gap_limit is not None:
            cut = words[i + 1].start_s - word.end_s > gap_limit
        if cut:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _best_cut(words: Sequence[Word]) -> int:
    """Index after which to cut: the widest internal gap, ties to the middle.

    Continuous speech has (near) zero gaps everywhere, so the tie rule is what
    keeps the halves balanced instead of peeling one word off an end.
    """
    gaps = [words[i + 1].start_s - words[i].end_s for i in range(len(words) - 1)]
    widest = max(gaps)
    middle = (len(gaps) - 1) / 2.0
    return min(
        (i for i, gap in enumerate(gaps) if gap == widest),
        key=lambda i: (abs(i - middle), i),
    )


def _group_span(words: Sequence[Word]) -> Tuple[float, float]:
    """Bounds covering every word (min start, max end).

    For monotonic whisper words this is first-word start / last-word end; the
    min/max form also holds when timings wobble, so words never fall outside
    their unit.
    """
    return min(w.start_s for w in words), max(w.end_s for w in words)


def _enforce_max_length(words: List[Word], limit: Optional[float]) -> List[List[Word]]:
    """Split ``words`` at the widest gaps until every piece fits ``limit``.

    Iterative rather than recursive so a pathological segment with thousands
    of words cannot hit the recursion limit. A single word is never split,
    whatever its length. ``None`` disables the cap.
    """
    if limit is None:
        return [words]
    result: List[List[Word]] = []
    stack: List[List[Word]] = [words]
    while stack:
        group = stack.pop()
        start, end = _group_span(group)
        if len(group) < 2 or end - start <= limit:
            result.append(group)
            continue
        cut = _best_cut(group)
        stack.append(group[cut + 1:])  # pushed first so the left half is emitted first
        stack.append(group[:cut + 1])
    return result


def resegment(
    units: Sequence[TranscriptUnit],
    max_unit_s: Optional[float] = DEFAULT_MAX_UNIT_S,
    pause_gap_s: Optional[float] = DEFAULT_PAUSE_GAP_S,
    split_on_punct: bool = True,
    split_on_colon: bool = False,
) -> List[TranscriptUnit]:
    """Re-cut whisper segments into clause-sized units with global word ids.

    A segment is cut after a word that ends a sentence (``. ? ! ;`` and,
    with ``split_on_colon``, ``:``) and wherever the silence before the next
    word exceeds ``pause_gap_s``; pieces longer than ``max_unit_s`` are then
    split at their widest internal gap (ties go to the middle) until they fit.
    Units without words cannot be cut and are kept as they are, text included.

    Unit ids are renumbered 0..n-1 in time order. Word ids are preserved when
    they are already valid and assigned afresh otherwise. Unit bounds come
    from the words and never extend beyond the original segment because the
    words are clamped into it first. The input is never mutated.

    ``max_unit_s`` and ``pause_gap_s`` accept a number or ``None``; ``None``,
    non-finite and negative values (and a zero ``max_unit_s``) switch that
    rule off. The flags must be bools. Any other type raises ``TypeError``
    rather than silently switching a rule off.
    """
    limit, gap_limit = _effective_limits(max_unit_s, pause_gap_s)
    punct = SENTENCE_END_PUNCT + (':' if _flag(split_on_colon, 'split_on_colon') else '')
    if not _flag(split_on_punct, 'split_on_punct'):
        punct = ''
    clean, _ = sanitize_units(units)
    if not word_ids_valid(clean):
        clean = assign_word_ids(clean)
    pieces: List[TranscriptUnit] = []
    for unit in sorted(clean, key=_time_key):
        if not unit.words:
            pieces.append(unit)
            continue
        groups = [
            piece
            for group in _split_at_boundaries(unit.words, gap_limit, punct)
            for piece in _enforce_max_length(group, limit)
        ]
        for group in groups:
            start, end = _group_span(group)
            pieces.append(TranscriptUnit(
                unit_id=-1,
                start_s=start,
                end_s=end,
                text=_join_word_texts(group),
                words=list(group),
                avg_logprob=unit.avg_logprob,
                no_speech_prob=unit.no_speech_prob,
                speaker=unit.speaker,
            ))
    pieces.sort(key=_time_key)
    for unit_id, piece in enumerate(pieces):
        piece.unit_id = unit_id
    return pieces


# --------------------------------------------------------------------------- #
# Context assembly and windows
# --------------------------------------------------------------------------- #

def build_context(
    raw_units: Sequence[TranscriptUnit],
    audio_sha256: str,
    duration_s: float,
    asr_model_id: str,
    asr_config_hash: str,
    resegment_kwargs: Optional[Dict[str, Any]] = None,
    sample_rate: int = 16000,
) -> AudioContext:
    """Sanitise, resegment and package raw ASR units as an ``AudioContext``.

    ``diagnostics`` records what was repaired in the ASR output
    (``input_anomalies``), what ``check_units`` still flags on the final units
    (``unit_anomalies``), the effective resegmentation parameters and the
    unit/word counts, so a bad transcript can be diagnosed from the cached
    context alone. A non-finite or negative ``duration_s`` is replaced by the
    last unit end (never below zero) rather than poisoning every later clamp.
    """
    raw_list = list(raw_units)
    params: Dict[str, Any] = {
        'max_unit_s': DEFAULT_MAX_UNIT_S,
        'pause_gap_s': DEFAULT_PAUSE_GAP_S,
        'split_on_punct': True,
        'split_on_colon': False,
    }
    params.update(resegment_kwargs or {})

    clean, input_anomalies = sanitize_units(raw_list)
    if not word_ids_valid(clean):
        input_anomalies.append('word ids missing or duplicated; reassigned in time order')
        clean = assign_word_ids(clean)
    units = resegment(clean, **params)

    duration = _as_float(duration_s)
    if math.isnan(duration) or duration < 0.0:
        input_anomalies.append(f'duration_s {duration_s!r} unusable; using last unit end')
        # Floor at zero: a transcript whose every timestamp is negative must
        # not turn into a negative recording length that the evidence stage
        # would clamp every span into.
        duration = max((unit.end_s for unit in units), default=0.0)
        duration = max(duration, 0.0)

    # Effective values (a numpy float, ``inf`` or ``0`` from a caller becomes
    # a plain float, or ``None`` when the rule did not apply) so the cached
    # context stays strict JSON and says what was really done.
    max_unit_s, pause_gap_s = _effective_limits(params['max_unit_s'], params['pause_gap_s'])
    effective_params = {
        'max_unit_s': max_unit_s,
        'pause_gap_s': pause_gap_s,
        'split_on_punct': params['split_on_punct'],
        'split_on_colon': params['split_on_colon'],
    }
    diagnostics: Dict[str, Any] = {
        'n_raw_units': len(raw_list),
        'n_units': len(units),
        'n_words': sum(len(unit.words) for unit in units),
        'resegment': effective_params,
        'input_anomalies': input_anomalies,
        'unit_anomalies': check_units(units, duration),
    }
    return AudioContext(
        audio_sha256=str(audio_sha256),
        duration_s=duration,
        units=units,
        asr_model_id=str(asr_model_id),
        asr_config_hash=str(asr_config_hash),
        sample_rate=int(sample_rate),
        timing_method='whisper_word',
        diagnostics=diagnostics,
    )


def unit_index(context: AudioContext) -> Dict[int, int]:
    """Map unit_id -> position in ``context.units`` (first occurrence wins).

    Evidence grouping needs to know whether two unit ids are neighbours in
    time, which is a question about positions, not ids.
    """
    index: Dict[int, int] = {}
    for position, unit in enumerate(context.units):
        index.setdefault(unit.unit_id, position)
    return index


@dataclass(frozen=True)
class Window:
    """A contiguous run of units offered to QA as one candidate passage."""

    unit_ids: Tuple[int, ...]
    text: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def windows(context: AudioContext, max_units: int = 3) -> List[Window]:
    """Every contiguous run of 1..``max_units`` units, by start position then length.

    Runs are contiguous in ``context.units`` order (time order for a context
    from ``build_context``). For four units and ``max_units=3`` that is
    4 + 3 + 2 = 9 windows. Bounds cover every unit in the run; text joins the
    non-empty unit texts with single spaces.
    """
    # A bool or float here is a config bug: ``True`` would silently mean 1 and
    # ``3.0`` would fail deep inside ``range``; both are reported by name.
    if isinstance(max_units, bool) or not isinstance(max_units, numbers.Integral):
        raise TypeError(f'max_units must be an int, got {type(max_units).__name__}')
    if max_units < 1:
        raise ValueError(f'max_units must be >= 1, got {max_units!r}')
    units = context.units
    result: List[Window] = []
    for first in range(len(units)):
        for length in range(1, min(max_units, len(units) - first) + 1):
            run = units[first:first + length]
            result.append(Window(
                unit_ids=tuple(unit.unit_id for unit in run),
                text=' '.join(t for t in ((unit.text or '').strip() for unit in run) if t),
                start_s=min(unit.start_s for unit in run),
                end_s=max(unit.end_s for unit in run),
            ))
    return result


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #

def save_context(context: AudioContext, path: Union[str, Path]) -> Path:
    """Write ``context`` as strict JSON (no NaN) to ``path`` atomically.

    Cached transcripts are shared by parallel offline tools, so the file is
    written next to its destination and renamed into place: a reader can never
    see a half-written context. Parent directories are created. The file gets
    the permissions a plain ``open(..., 'w')`` would give (the process umask)
    rather than the owner-only mode of a temporary file, so a cache written
    by an offline tool stays readable by the service account.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = context.to_json(ensure_ascii=False, allow_nan=False)
    handle = tempfile.NamedTemporaryFile(
        'w', encoding='utf-8', dir=target.parent, prefix=f'.{target.name}.',
        suffix='.tmp', delete=False,
    )
    try:
        with handle:
            handle.write(payload)
            os.fchmod(handle.fileno(), 0o666 & ~_current_umask())
        os.replace(handle.name, target)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return target


def _current_umask() -> int:
    """The process umask; ``os.umask`` can only be read by setting it."""
    mask = os.umask(0)
    os.umask(mask)
    return mask


def load_context(path: Union[str, Path]) -> AudioContext:
    """Read an ``AudioContext`` written by ``save_context``."""
    return AudioContext.from_json(Path(path).read_text(encoding='utf-8'))
