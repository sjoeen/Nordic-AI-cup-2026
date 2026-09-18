"""Evidence localisation: map QA-selected unit and word ids to an audio span.

Why this module exists
----------------------
The QA stage says *which* transcript units support a yes answer; the score
rewards *where* in the original audio that evidence sits, measured as
temporal IoU against a human-annotated span that is usually only a few
seconds long. This stage turns unit ids into one ``(start_s, end_s)`` span
and makes every choice that trades recall for precision explicit and
configurable through :class:`EvidencePolicy`:

* ids are validated against the context, because a backend (or an LLM
  behind it) can name units that do not exist;
* neighbouring units are grouped into runs so a policy can keep the union,
  the first, the longest or the most question-relevant run;
* filler words at the edges ("um", "okay", "you know") are dropped because
  annotators do not include them in the gold span;
* an over-long span is narrowed to the sub-run with the highest focus-term
  density, then padded, clamped to the audio and given a minimum length.

Everything here is pure and deterministic. Transcript text and question text
are untrusted data: they are normalised and compared, never interpreted.
Positions are indexes into ``context.units``; two units are "consecutive"
when they are neighbours in that list, which after ``transcript.build_context``
is the same as being neighbours in time.
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from core_types import AudioContext, QAResult, Span, TranscriptUnit, Word
from transcript import normalize_text, tokenize, unit_index
from utils import evidence_interval, temporal_iou

__all__ = [
    'DEFAULT_PAD_GRID',
    'DENSITY_FLOOR_S',
    'FILLER',
    'MODES',
    'EvidencePolicy',
    'calibrate_padding',
    'focus_tokens',
    'group_runs',
    'select_span',
    'trim_filler_words',
    'trim_to_focus',
    'valid_positions',
]

logger = logging.getLogger(__name__)

#: Run-selection modes accepted by :class:`EvidencePolicy`.
MODES: Tuple[str, ...] = ('union', 'first_run', 'longest_run', 'best_run')

#: Words annotators leave out of a gold span when they open or close it.
FILLER: FrozenSet[str] = frozenset({
    'um', 'uh', 'hmm', 'mm', 'mhm', 'mm-hmm', 'uh-huh', 'so', 'okay', 'ok',
    'yeah', 'yes', 'no', 'well', 'right', 'alright', 'oh', 'ah', 'like',
    'you know', 'i mean', 'and', 'but',
})
# Matching happens on normalised word text ("Mm-hmm," -> "mm hmm"), so the set
# is normalised once. A word that normalises to nothing (pure punctuation)
# carries no evidence either and is droppable at an edge.
_FILLER_NORMALISED: FrozenSet[str] = (
    frozenset(normalize_text(filler) for filler in FILLER) | frozenset({''})
)
_MAX_FILLER_WORDS: int = max(len(filler.split()) for filler in _FILLER_NORMALISED)

#: Floor on the duration used for focus-term density. A degenerate (near
#: zero-length) unit that happens to hold one focus term must not beat a
#: proper clause holding several; anything shorter than this is timing noise.
DENSITY_FLOOR_S = 0.5

#: Offsets, in seconds, that :func:`calibrate_padding` tries by default.
DEFAULT_PAD_GRID: Tuple[float, ...] = tuple(round(0.1 * i, 1) for i in range(-5, 11))

# Function words and question scaffolding that never locate evidence. Only the
# fallback focus-term extractor uses this; ``question_rewrite`` owns the real
# list. Apostrophes vanish in ``normalize_text`` ("didn't" -> "didnt").
_FALLBACK_STOPWORDS: FrozenSet[str] = frozenset("""
a an the and or but if whether that this these those there here it its
is are was were be been being am do does did done has have had having will
would shall should can could may might must
i you he she they we me him her them us my your his their our
of to in on at for with by from as into about over under after before during
than then so such also ever still yet already again
any some something anything nothing all both each every either neither
not no nor
what which who whom whose when where why how
patient patients doctor doctors mention mentioned mentions mentioning
discuss discussed discusses discussing talk talked talks talking conversation
say said says tell told ask asked
ive im id ill weve well were wed youve youre youll theyve theyre
hes shes thats itll theres heres isnt arent wasnt werent dont
doesnt didnt hasnt havent hadnt wont wouldnt cant couldnt shouldnt
okay ok yes yeah um uh right correct true
""".split())


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #

@dataclass
class EvidencePolicy:
    """Every knob of the unit-ids-to-span mapping, validated on construction.

    ``mode`` picks which run of consecutive units the span covers when the
    QA evidence is not one contiguous block. ``max_span_s`` (``None`` for no
    limit) triggers focus-term narrowing; ``pad_pre_s``/``pad_post_s`` widen
    the span (negative values shrink it); ``min_span_s`` is the shortest span
    ever returned. Validation happens here rather than per call so that a
    bad config fails at pipeline start-up, not on the first request.
    """

    mode: str = 'union'
    max_span_s: Optional[float] = 20.0
    pad_pre_s: float = 0.0
    pad_post_s: float = 0.0
    trim_filler: bool = True
    min_span_s: float = 0.2

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}, got {self.mode!r}')
        self.mode = mode
        for name in ('pad_pre_s', 'pad_post_s', 'min_span_s'):
            value = getattr(self, name)
            if not _finite(value):
                raise ValueError(f'{name} must be a finite number, got {value!r}')
            setattr(self, name, float(value))
        if self.min_span_s < 0.0:
            raise ValueError(f'min_span_s must be >= 0, got {self.min_span_s!r}')
        if self.max_span_s is not None:
            if not _finite(self.max_span_s) or float(self.max_span_s) <= 0.0:
                raise ValueError(
                    f'max_span_s must be a positive number or None, got {self.max_span_s!r}'
                )
            self.max_span_s = float(self.max_span_s)
        self.trim_filler = bool(self.trim_filler)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #

def _finite(value: Any) -> bool:
    """True for a finite real number that is not a bool.

    Wider than ``core_types.is_finite_number`` on purpose: unit timings that
    went through numpy arrive as ``numpy.float32``, which is a ``Real`` but
    not a ``float``, and they are perfectly good timestamps.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return False
    return math.isfinite(float(value))


def _is_index(value: Any) -> bool:
    """True for an integer id (numpy ints included, bools excluded)."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _as_items(value: Any) -> List[Any]:
    """``value`` as a list of candidate ids; empty for ``None`` or a non-iterable.

    Never tests the truth value of the container: a numpy array raises on
    ``bool()``, and a backend that hands one over must not lose its span.
    """
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return []


def _has_finite_bounds(unit: TranscriptUnit) -> bool:
    return _finite(unit.start_s) and _finite(unit.end_s) and unit.end_s >= unit.start_s


def _timed_words(words: Iterable[Word]) -> List[Word]:
    """The words whose timings can be used for a bound."""
    return [w for w in words if _finite(w.start_s) and _finite(w.end_s) and w.end_s >= w.start_s]


def _unit_bounds(units: Sequence[TranscriptUnit]) -> Span:
    return (
        float(min(unit.start_s for unit in units)),
        float(max(unit.end_s for unit in units)),
    )


def _word_bounds(words: Sequence[Word]) -> Optional[Span]:
    timed = _timed_words(words)
    if not timed:
        return None
    return float(min(w.start_s for w in timed)), float(max(w.end_s for w in timed))


# --------------------------------------------------------------------------- #
# Id validation and run grouping
# --------------------------------------------------------------------------- #

def valid_positions(context: AudioContext, unit_ids: Optional[Iterable[Any]]) -> List[int]:
    """Sorted, de-duplicated positions in ``context.units`` for usable ids.

    An id is usable when it is an integer naming a unit of the context whose
    bounds are finite. Anything else (bools, strings, floats, unknown ids,
    units with NaN timing) is dropped silently: evidence ids are produced by
    models and never trusted blindly.
    """
    index = unit_index(context)
    positions: Set[int] = set()
    for unit_id in _as_items(unit_ids):
        if not _is_index(unit_id):
            continue
        position = index.get(int(unit_id))
        if position is not None and _has_finite_bounds(context.units[position]):
            positions.add(position)
    return sorted(positions)


def group_runs(positions: Iterable[int]) -> List[List[int]]:
    """Maximal groups of adjacent positions, in order.

    ``[0, 1, 2, 5, 7, 8]`` becomes ``[[0, 1, 2], [5], [7, 8]]``. Input is
    sorted and de-duplicated first so callers need not care.
    """
    runs: List[List[int]] = []
    for position in sorted(set(positions)):
        if runs and position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])
    return runs


# --------------------------------------------------------------------------- #
# Focus terms
# --------------------------------------------------------------------------- #

def _canonical(token: str) -> str:
    """Plural-insensitive token: "weeks" -> "week", but "dose"/"this" untouched.

    A trailing ``s`` on a word of four letters or more is a plural marker
    unless the word ends in ``ss``/``us``/``is``; questions say "two weeks"
    where the transcript may say "a week". Cheap and good enough for ranking.
    """
    if len(token) > 3 and token.endswith('s') and not token.endswith(('ss', 'us', 'is')):
        return token[:-1]
    return token


def _fallback_focus_terms(question: str) -> List[str]:
    return [token for token in tokenize(question) if token not in _FALLBACK_STOPWORDS]


def _sibling_focus_terms() -> Optional[Callable[[str], Iterable[str]]]:
    """``question_rewrite.focus_terms`` when the sibling provides it, else ``None``.

    Resolved at call time rather than at import time so that this module keeps
    working while the sibling is missing, half-written (any import error, not
    only ``ImportError``) or lacks the function, and picks the real extractor
    up as soon as it exists. The module name is a fixed literal; nothing from
    the question or transcript ever chooses what gets imported.
    """
    try:
        import question_rewrite
    except Exception:  # a broken sibling must not cost the span
        return None
    extractor = getattr(question_rewrite, 'focus_terms', None)
    return extractor if callable(extractor) else None


def focus_tokens(question: Optional[str]) -> FrozenSet[str]:
    """Canonical content tokens of ``question`` used to rank runs.

    Prefers ``question_rewrite.focus_terms`` and falls back to a stopword
    filter when the sibling is missing or raises, because evidence selection
    must never fail on account of a helper. Empty for no question.
    """
    if not isinstance(question, str) or not question.strip():
        return frozenset()
    terms: Optional[List[str]] = None
    sibling = _sibling_focus_terms()
    if sibling is not None:
        try:
            terms = [str(term) for term in sibling(question)]
        except Exception as exc:  # a sibling bug must not cost the span
            logger.debug('question_rewrite.focus_terms failed, using fallback: %r', exc)
    if terms is None:
        terms = _fallback_focus_terms(question)
    return frozenset(_canonical(token) for term in terms for token in tokenize(term))


def _unit_tokens(unit: TranscriptUnit) -> FrozenSet[str]:
    """Canonical tokens of a unit; rebuilt from its words when the text is empty."""
    text = unit.text if isinstance(unit.text, str) and unit.text.strip() else ' '.join(
        word.text for word in unit.words if isinstance(word.text, str)
    )
    return frozenset(_canonical(token) for token in tokenize(text))


def _density(matched: int, duration_s: float) -> float:
    return matched / max(duration_s, DENSITY_FLOOR_S)


def _run_key(context: AudioContext, run: Sequence[int], terms: FrozenSet[str]) -> Tuple[float, int, int]:
    """Ranking key for ``best_run``: density, then matched count, then earliest."""
    units = [context.units[position] for position in run]
    start, end = _unit_bounds(units)
    matched: Set[str] = set()
    for unit in units:
        matched |= _unit_tokens(unit) & terms
    return _density(len(matched), end - start), len(matched), -run[0]


def _select_positions(
    context: AudioContext, runs: List[List[int]], mode: str, terms: FrozenSet[str],
) -> List[int]:
    """The positions the span will cover under ``mode``.

    ``union`` keeps everything (gaps included, the span bridges them);
    ``first_run`` the earliest run; ``longest_run`` the run with the longest
    duration (earliest on a tie); ``best_run`` the run with the highest
    focus-term density (earliest on a tie, so no question degrades to
    ``first_run``). ``max`` returns the first maximal element, which makes
    every tie-break deterministic.
    """
    if mode == 'union':
        return [position for run in runs for position in run]
    if mode == 'first_run':
        return list(runs[0])
    if mode == 'longest_run':
        def duration(run: List[int]) -> float:
            start, end = _unit_bounds([context.units[position] for position in run])
            return end - start
        return list(max(runs, key=duration))
    return list(max(runs, key=lambda run: _run_key(context, run, terms)))


# --------------------------------------------------------------------------- #
# Word-level refinement
# --------------------------------------------------------------------------- #

def _words_of(context: AudioContext, positions: Sequence[int]) -> List[Word]:
    return [word for position in positions for word in context.units[position].words]


def _resolve_word_override(
    context: AudioContext, positions: Sequence[int], word_ids: Optional[Iterable[Any]],
) -> Optional[List[Word]]:
    """Words named by ``word_ids`` when every id resolves inside the selected units.

    A partial resolution is rejected as a whole: a backend that names words
    outside its own units is confused, and the unit bounds are the safer bet.
    """
    items = _as_items(word_ids)
    if not items:
        return None
    wanted: Set[int] = set()
    for word_id in items:
        if not _is_index(word_id):
            return None
        wanted.add(int(word_id))
    chosen = [word for word in _words_of(context, positions) if word.word_id in wanted]
    if {word.word_id for word in chosen} != wanted:
        logger.debug('evidence_word_ids %s do not all resolve inside units; using unit bounds', sorted(wanted))
        return None
    return chosen


def _filler_prefix_length(texts: Sequence[str]) -> int:
    """How many leading tokens form filler (longest phrase first)."""
    i, n = 0, len(texts)
    while i < n:
        for length in range(min(_MAX_FILLER_WORDS, n - i), 0, -1):
            if ' '.join(texts[i:i + length]) in _FILLER_NORMALISED:
                i += length
                break
        else:
            return i
    return i


def _filler_suffix_length(texts: Sequence[str]) -> int:
    """How many trailing tokens form filler (longest phrase first)."""
    j = n = len(texts)
    while j > 0:
        for length in range(min(_MAX_FILLER_WORDS, j), 0, -1):
            if ' '.join(texts[j - length:j]) in _FILLER_NORMALISED:
                j -= length
                break
        else:
            return n - j
    return n - j


def _filler_edge_counts(words: Sequence[Word]) -> Optional[Tuple[int, int]]:
    """``(leading, trailing)`` filler words to drop.

    ``None`` when trimming does not apply: no words at all, or nothing but
    filler, because a span must never be emptied by its own clean-up.
    """
    if not words:
        return None
    texts = [normalize_text(word.text) for word in words]
    lead = _filler_prefix_length(texts)
    if lead >= len(texts):
        return None
    trail = _filler_suffix_length(texts)
    if lead + trail >= len(texts):
        return None
    return lead, trail


def trim_filler_words(words: Sequence[Word]) -> List[Word]:
    """``words`` without filler at either edge; unchanged when only filler remains.

    Multi-word fillers ("you know", "i mean") are matched as phrases, and a
    hyphenated whisper token such as ``"Mm-hmm,"`` matches through
    normalisation. Interior fillers are kept: they sit inside the evidence.
    """
    edges = _filler_edge_counts(words)
    if edges is None:
        return list(words)
    lead, trail = edges
    return list(words[lead:len(words) - trail])


def _refine_bounds(
    units: Sequence[TranscriptUnit], words: Sequence[Word], word_level: bool, trim: bool,
) -> Span:
    """Bounds of ``units``, tightened by the word-level rules that apply.

    Unit bounds are the base. With ``word_level`` (a resolved word override)
    the bounds are those of ``words`` instead. When filler trimming applies
    (there are words and a non-filler word remains) the bounds are recomputed
    from the remaining words, so the span starts and ends on speech that
    carries evidence rather than on a segment boundary.
    """
    start, end = _unit_bounds(units)
    if word_level:
        start, end = _word_bounds(words) or (start, end)
    if trim:
        edges = _filler_edge_counts(words)
        if edges is not None:
            lead, trail = edges
            start, end = _word_bounds(words[lead:len(words) - trail]) or (start, end)
    return start, end


# --------------------------------------------------------------------------- #
# Focus trimming
# --------------------------------------------------------------------------- #

def trim_to_focus(
    context: AudioContext,
    unit_ids: Iterable[Any],
    question: Optional[str],
    max_span_s: float,
) -> List[int]:
    """Unit ids of the contiguous sub-run under ``max_span_s`` that best fits ``question``.

    Candidates are every contiguous sub-run of every run of the valid ids
    whose duration is at most ``max_span_s``, plus every single unit that is
    longer than that on its own (a unit cannot be cut, and a long clause that
    holds the answer must not be discarded for a short one that does not).
    Ranking: a candidate holding at least one focus term beats one holding
    none, a candidate that fits the limit beats one that does not, then the
    highest focus-term density (distinct question terms present / duration)
    wins, ties going to more matched terms, the earliest start and finally
    the longer sub-run. With no question, or none of its terms in the
    transcript, this degrades to the earliest, longest sub-run that fits.
    Returns ``[]`` when no id is valid.
    """
    if not _finite(max_span_s) or float(max_span_s) <= 0.0:
        raise ValueError(f'max_span_s must be a positive number, got {max_span_s!r}')
    limit = float(max_span_s)
    positions = valid_positions(context, unit_ids)
    if not positions:
        return []
    terms = focus_tokens(question)
    tokens_at: Dict[int, FrozenSet[str]] = {p: _unit_tokens(context.units[p]) & terms for p in positions}

    best: List[int] = []
    best_key: Optional[Tuple[bool, bool, float, int, int, int]] = None
    for run in group_runs(positions):
        for i in range(len(run)):
            start = math.inf
            end = -math.inf
            matched: Set[str] = set()
            for j in range(i, len(run)):
                unit = context.units[run[j]]
                start = min(start, float(unit.start_s))
                end = max(end, float(unit.end_s))
                matched |= tokens_at[run[j]]
                duration = end - start
                fits = duration <= limit
                if not fits and j > i:
                    break  # extending only lengthens the sub-run
                key = (
                    bool(matched), fits, _density(len(matched), duration),
                    len(matched), -run[i], j - i + 1,
                )
                if best_key is None or key > best_key:
                    best_key, best = key, run[i:j + 1]
    return [context.units[position].unit_id for position in best]


# --------------------------------------------------------------------------- #
# Padding, clamping, minimum length
# --------------------------------------------------------------------------- #

def _audio_end(context: AudioContext) -> Optional[float]:
    """Upper clamp: the audio duration, or the last unit end when it is unusable."""
    if _finite(context.duration_s) and float(context.duration_s) > 0.0:
        return float(context.duration_s)
    ends = [float(unit.end_s) for unit in context.units if _finite(unit.end_s)]
    return max(ends) if ends else None


def _expand_to_min(start: float, end: float, min_span_s: float, audio_end: float) -> Span:
    """Widen ``[start, end]`` symmetrically to ``min_span_s``, sliding off the audio edges."""
    centre = (start + end) / 2.0
    half = min_span_s / 2.0
    start, end = centre - half, centre + half
    if start < 0.0:
        end -= start
        start = 0.0
    if end > audio_end:
        start -= end - audio_end
        end = audio_end
    return max(start, 0.0), end


def _pad_and_clamp(start: float, end: float, policy: EvidencePolicy, audio_end: Optional[float]) -> Optional[Span]:
    if audio_end is None or not (_finite(start) and _finite(end)):
        return None
    start -= policy.pad_pre_s
    end += policy.pad_post_s
    if start > audio_end or end < 0.0:
        # Nothing of the (padded) evidence lies inside the audio: clamping
        # would collapse it onto an edge and ``min_span_s`` would then grow a
        # span there out of nothing.
        return None
    start = min(max(start, 0.0), audio_end)
    end = min(max(end, 0.0), audio_end)
    if end - start < policy.min_span_s:
        start, end = _expand_to_min(start, end, policy.min_span_s, audio_end)
    if end <= start:
        return None
    start, end = round(start, 3), round(end, 3)
    if end > audio_end:
        # Rounding up crossed the audio end: fall back to the last whole ms.
        end = math.floor(audio_end * 1000.0 + 1e-6) / 1000.0
    return (start, end) if end > start else None


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def select_span(
    context: AudioContext,
    result: QAResult,
    policy: EvidencePolicy,
    question: Optional[str] = None,
) -> Optional[Span]:
    """The audio span for one yes answer, or ``None`` when nothing can be pointed at.

    Order of operations, each one optional and driven by ``policy``:
    validate ids -> group runs -> pick run(s) by mode -> word-id override ->
    filler trim -> focus trim when longer than ``max_span_s`` (needs a
    question) -> pad -> clamp to the audio -> enforce ``min_span_s`` ->
    require ``end > start``. A no answer never gets a span, and neither does
    a result whose ids all fail validation or whose evidence lies entirely
    outside the audio: a span is never invented.
    """
    if not result.answer:
        return None
    positions = valid_positions(context, result.evidence_unit_ids)
    if not positions:
        return None
    question_text = question.strip() if isinstance(question, str) else ''
    selected = _select_positions(context, group_runs(positions), policy.mode, focus_tokens(question_text))

    units = [context.units[position] for position in selected]
    override = _resolve_word_override(context, selected, result.evidence_word_ids)
    words = override if override is not None else _words_of(context, selected)
    start, end = _refine_bounds(units, words, override is not None, policy.trim_filler)

    if policy.max_span_s is not None and end - start > policy.max_span_s and question_text:
        candidates = units
        if override is not None:
            # The backend named words: narrowing may only choose among the
            # units that hold them, or the span would jump to speech the
            # backend never pointed at.
            named = {word.word_id for word in override}
            candidates = [unit for unit in units if any(word.word_id in named for word in unit.words)]
        kept = valid_positions(
            context,
            trim_to_focus(context, [unit.unit_id for unit in candidates], question_text, policy.max_span_s),
        )
        if kept and len(kept) < len(selected):
            units = [context.units[position] for position in kept]
            inside = {word.word_id for unit in units for word in unit.words}
            narrowed = [word for word in words if word.word_id in inside]
            word_level = override is not None and bool(narrowed)
            words = narrowed if word_level else _words_of(context, kept)
            start, end = _refine_bounds(units, words, word_level, policy.trim_filler)

    return _pad_and_clamp(start, end, policy, _audio_end(context))


# --------------------------------------------------------------------------- #
# Offline calibration
# --------------------------------------------------------------------------- #

def _as_span(value: Any) -> Optional[Span]:
    """A valid ``(start, end)`` from a pair-like value, else ``None``."""
    try:
        start, end = value
    except (TypeError, ValueError):
        return None
    return evidence_interval(start, end)


def _grid_values(grid: Iterable[Any], name: str) -> List[float]:
    values: List[float] = []
    for value in grid:
        if not _finite(value):
            raise ValueError(f'{name} must contain finite numbers, got {value!r}')
        values.append(float(value))
    if not values:
        raise ValueError(f'{name} must not be empty')
    return values


def calibrate_padding(
    pairs: Iterable[Tuple[Any, Any]],
    pre_grid: Iterable[Any] = DEFAULT_PAD_GRID,
    post_grid: Optional[Iterable[Any]] = None,
) -> Tuple[float, float]:
    """``(pad_pre_s, pad_post_s)`` from the grids that maximise mean temporal IoU.

    ``pairs`` holds ``(gold_span, predicted_span)`` in the argument order of
    ``utils.temporal_iou``; pairs with a missing or invalid span on either
    side are skipped. Padded starts are clamped at zero as they are at
    inference time. ``post_grid`` defaults to ``pre_grid``. Ties prefer the
    smallest total padding, then grid order, so the answer is deterministic.
    With no usable pair the calibration has nothing to say and returns
    ``(0.0, 0.0)``.
    """
    pre_values = _grid_values(pre_grid, 'pre_grid')
    post_values = pre_values if post_grid is None else _grid_values(post_grid, 'post_grid')
    usable: List[Tuple[Span, Span]] = []
    for pair in pairs:
        try:
            gold, predicted = pair
        except (TypeError, ValueError):
            continue
        gold_span, predicted_span = _as_span(gold), _as_span(predicted)
        if gold_span is not None and predicted_span is not None:
            usable.append((gold_span, predicted_span))
    if not usable:
        return 0.0, 0.0

    best = (0.0, 0.0)
    best_key: Optional[Tuple[float, float]] = None
    for pre in pre_values:
        for post in post_values:
            total = sum(
                temporal_iou(gold_span, (max(0.0, p_start - pre), max(0.0, p_end + post)))
                for gold_span, (p_start, p_end) in usable
            )
            key = (round(total / len(usable), 12), -(abs(pre) + abs(post)))
            if best_key is None or key > best_key:
                best_key, best = key, (pre, post)
    return best
