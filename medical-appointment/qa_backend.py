"""Grounded yes/no question answering over a timed transcript (stage A6).

Two backends share one contract (``core_types.QABackend``):

* ``LexicalBackend`` needs no ML dependencies. It scores every transcript
  window by focus-term overlap, gates the decision with the fact checker and
  is the emergency fallback for everything else.
* ``RetrievalNLIBackend`` retrieves the top-k windows per question with a
  cosine + lexical score, runs a natural-language-inference cross-encoder on
  ``(window, statement)`` pairs, and decides from entailment, contradiction,
  the fact-check verdict and an off-topic threshold.

Design rules that shape this file:

* The sibling modules (``transcript``, ``question_rewrite``, ``factcheck``)
  are optional at import time. Each has a small local fallback so that this
  module, its tests and the pipeline keep working when a sibling is absent or
  broken. The fallbacks are deliberately naive; the siblings own the quality.
* Every path returns exactly ``len(questions)`` results in order. A failure
  in one question's decision never affects another question and never raises.
* Question and transcript text is untrusted data. It is tokenised and scored,
  never interpreted, formatted into code, or used to pick message roles.
* Model objects are injectable so the default tests run with fakes; the real
  ``sentence_transformers`` models load lazily on first use or ``warm_up()``.
"""

from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from core_types import AudioContext, Deadline, QABackend, QAResult, stable_hash

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Optional siblings with local fallbacks
# --------------------------------------------------------------------------- #
# Each import is guarded separately: one broken sibling must not take the
# others down with it. The module-level names are looked up at call time so a
# test (or a hot fix) can swap them with ``monkeypatch.setattr(qa_backend, ...)``.

try:
    from transcript import windows as transcript_windows
except ImportError:  # pragma: no cover - depends on the sibling being present
    transcript_windows = None

try:
    from question_rewrite import (
        focus_terms as rewrite_focus_terms,
        is_existence_question as rewrite_is_existence_question,
        to_statement as rewrite_to_statement,
    )
except ImportError:  # pragma: no cover - depends on the sibling being present
    rewrite_focus_terms = None
    rewrite_is_existence_question = None
    rewrite_to_statement = None

try:
    from factcheck import compare as factcheck_compare
except ImportError:  # pragma: no cover - depends on the sibling being present
    factcheck_compare = None
try:
    from calibration import Calibration, Candidate
except ImportError:  # pragma: no cover - depends on the sibling being present
    Calibration = None
    Candidate = None


@dataclass(frozen=True)
class LocalWindow:
    """Fallback for ``transcript.Window``: a contiguous run of units.

    Only the four attributes below are ever read from a window, so the real
    ``transcript.Window`` and this fallback are interchangeable.
    """

    unit_ids: Tuple[int, ...]
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class LocalVerdict:
    """Fallback for ``factcheck.FactCheckVerdict`` with the same two fields."""

    status: str
    details: List[str] = field(default_factory=list)


NO_QUANTITIES = 'no_quantities'
CONTRADICTION = 'contradiction'
UNVERIFIABLE = 'unverifiable'
FACTCHECK_ERROR = 'error'

# Question scaffolding and function words that never help locate evidence.
# Used only by the fallback focus-term extractor; ``question_rewrite`` owns
# the real list.
_FALLBACK_STOPWORDS = frozenset("""
a an the and or but if whether that this these those there here it its it's
is are was were be been being am do does did done has have had having will
would shall should can could may might must
i you he she they we me him her them us my your his their our
of to in on at for with by from as into about over under after before during
than then so such also ever still yet already again
any some something anything nothing all both each every either neither
not no nor
what which who whom whose when where why how
patient patient's doctor doctor's mention mentioned mentions mentioning
discuss discussed discusses discussing talk talked talks talking conversation
say said says tell told ask asked
i've i'm i'd i'll we've we'll we're we'd you've you're you'll they've they're
he's she's that's it'll there's here's isn't aren't wasn't weren't don't
doesn't didn't hasn't haven't hadn't won't wouldn't can't couldn't shouldn't
okay ok yes yeah um uh
""".split())

# Number words spoken in questions and by the ASR ("two weeks" vs "2 weeks").
# Mapping both sides to digits makes exact-number matching possible without
# a full quantity parser (that is ``factcheck``'s job).
_NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
    'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    'eleven': '11', 'twelve': '12', 'thirteen': '13', 'fourteen': '14',
    'fifteen': '15', 'sixteen': '16', 'seventeen': '17', 'eighteen': '18',
    'nineteen': '19', 'twenty': '20', 'thirty': '30', 'forty': '40',
    'fifty': '50', 'sixty': '60', 'seventy': '70', 'eighty': '80',
    'ninety': '90', 'hundred': '100', 'thousand': '1000',
}

# Spoken unit names versus the abbreviations questions tend to use.
_UNIT_SYNONYMS = {
    'milligram': 'mg', 'milligrams': 'mg', 'mgs': 'mg',
    'microgram': 'mcg', 'micrograms': 'mcg', 'ug': 'mcg', 'µg': 'mcg',
    'millilitre': 'ml', 'millilitres': 'ml', 'milliliter': 'ml', 'milliliters': 'ml',
    'gram': 'g', 'grams': 'g', 'kilogram': 'kg', 'kilograms': 'kg',
    'hr': 'hour', 'hrs': 'hour', 'min': 'minute', 'mins': 'minute',
    'percent': '%', 'pct': '%',
}

_TOKEN_RE = re.compile(r"[a-z0-9%]+(?:[.,/'][a-z0-9]+)*")
_NUMBER_RE = re.compile(r'\d+(?:[.,/]\d+)*')
_CANONICAL_LABELS: Tuple[str, str, str] = ('contradiction', 'entailment', 'neutral')


def is_number_token(token: str) -> bool:
    """True for digit tokens including decimals, thousands and ratios (130/85)."""
    return bool(_NUMBER_RE.fullmatch(token))


def _canonical_token(token: str) -> str:
    """Map a token to the form both question and transcript are compared in.

    Number words become digits and unit names their abbreviation. A trailing
    plural ``s`` is dropped from longer words so "weeks"/"week" and
    "tablets"/"tablet" match; words ending in ``ss``/``us``/``is`` are left
    alone because that ``s`` is not a plural marker.
    """
    token = _NUMBER_WORDS.get(token, token)
    token = _UNIT_SYNONYMS.get(token, token)
    if len(token) >= 4 and token.endswith('s') and not token.endswith(('ss', 'us', 'is')) \
            and not is_number_token(token):
        token = token[:-1]
    return token


def tokenize_for_match(text: str) -> List[str]:
    """Lowercase tokens for lexical matching, canonicalised on both sides.

    Owned here rather than delegated to ``transcript.tokenize`` so that the
    lexical scorer's behaviour does not depend on another module's
    normalisation choices; both sides of every comparison go through this one
    function.
    """
    if not isinstance(text, str) or not text:
        return []
    lowered = text.lower().replace('’', "'")
    return [_canonical_token(t) for t in _TOKEN_RE.findall(lowered)]


def _fallback_focus_terms(question: str) -> List[str]:
    return [t for t in tokenize_for_match(question) if t not in _FALLBACK_STOPWORDS]


_EXISTENCE_CUES = ('any mention', 'mention', 'discussed', 'discuss', 'talk about',
                   'talked about', 'come up', 'came up', 'brought up')
# "...mention of X" / "...discuss X" / "...talk about X"  and  "Did X come up"
_TOPIC_AFTER_CUE_RE = re.compile(
    r'(?:mention(?:ed|s)?\s+of|discuss(?:ed|es)?|talk(?:ed|s)?\s+about|br(?:ing|ought)\s+up)\s+(.+)$',
    re.IGNORECASE)
_TOPIC_BEFORE_CUE_RE = re.compile(r'^(?:did|does|was|were|has|had)\s+(.+?)\s+(?:come|came)\s+up$', re.IGNORECASE)


def _fallback_is_existence(question: str) -> bool:
    lowered = question.lower()
    return any(cue in lowered for cue in _EXISTENCE_CUES)


def _fallback_statement(question: str) -> str:
    """Question minus its trailing question marks, except for existence
    questions, which become "<topic> is mentioned.".

    A question-shaped hypothesis is read by NLI cross-encoders as a
    contradiction of almost anything ("Was there any mention of blood
    pressure" scored 0.98 contradiction against a sentence about blood
    pressure), whereas "blood pressure is mentioned." is entailed cleanly.
    """
    stripped = question.strip().rstrip('?').strip()
    if not stripped:
        return question.strip()
    if _fallback_is_existence(stripped):
        match = _TOPIC_AFTER_CUE_RE.search(stripped) or _TOPIC_BEFORE_CUE_RE.match(stripped)
        topic = match.group(1).strip() if match else ' '.join(_fallback_focus_terms(stripped))
        if topic:
            return f'{topic} is mentioned.'
    return stripped


def focus_tokens(question: str) -> List[str]:
    """Canonical, de-duplicated focus terms of a question (order preserved).

    Uses ``question_rewrite.focus_terms`` when available and re-tokenises its
    output so the terms are comparable with ``tokenize_for_match`` output.
    Falls back to the local stopword filter when the sibling is absent or
    raises.
    """
    terms: Optional[List[str]] = None
    if rewrite_focus_terms is not None:
        try:
            terms = [str(t) for t in rewrite_focus_terms(question)]
        except Exception as exc:  # the sibling is untrusted-quality at this point
            logger.warning('question_rewrite.focus_terms failed, using fallback: %r', exc)
    if terms is None:
        terms = _fallback_focus_terms(question)
    seen: Dict[str, None] = {}
    for term in terms:
        for token in tokenize_for_match(term):
            seen.setdefault(token, None)
    return list(seen)


def statement_for(question: str) -> str:
    """Declarative form of a yes/no question for the NLI hypothesis."""
    if rewrite_to_statement is not None:
        try:
            statement = rewrite_to_statement(question)
            if isinstance(statement, str) and statement.strip():
                return statement
        except Exception as exc:
            logger.warning('question_rewrite.to_statement failed, using fallback: %r', exc)
    return _fallback_statement(question)


def is_existence_question(question: str) -> bool:
    """Whether the question only asks if a topic came up at all."""
    if rewrite_is_existence_question is not None:
        try:
            return bool(rewrite_is_existence_question(question))
        except Exception as exc:
            logger.warning('question_rewrite.is_existence_question failed, using fallback: %r', exc)
    return _fallback_is_existence(question)


def factcheck_verdict(question: str, evidence_text: str) -> LocalVerdict:
    """Quantity consistency between question and evidence, never raising.

    A failing fact checker is reported as status ``'error'`` rather than
    turning the answer into a hard no: the NLI decision still stands and the
    error is visible in the diagnostics.
    """
    if factcheck_compare is None:
        return LocalVerdict(NO_QUANTITIES, ['factcheck unavailable'])
    try:
        verdict = factcheck_compare(question, evidence_text)
    except Exception as exc:
        logger.warning('factcheck.compare failed: %r', exc)
        return LocalVerdict(FACTCHECK_ERROR, [repr(exc)])
    status = str(getattr(verdict, 'status', NO_QUANTITIES))
    details = [str(d) for d in (getattr(verdict, 'details', None) or [])]
    return LocalVerdict(status, details)


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #

def _fallback_windows(context: AudioContext, max_units: int) -> List[LocalWindow]:
    units = context.units
    result: List[LocalWindow] = []
    for start in range(len(units)):
        for length in range(1, max_units + 1):
            run = units[start:start + length]
            if len(run) < length:
                break
            result.append(LocalWindow(
                unit_ids=tuple(u.unit_id for u in run),
                text=' '.join(u.text.strip() for u in run),
                start_s=run[0].start_s,
                end_s=run[-1].end_s,
            ))
    return result


def sorted_windows(context: AudioContext, max_units: int) -> List[Any]:
    """All contiguous 1..max_units windows, ordered by first unit then length.

    The order matters for determinism: score ties are broken by position in
    this list, so two implementations of ``transcript.windows`` that differ
    only in enumeration order still give identical answers.
    """
    max_units = max(1, int(max_units))
    if not context.units:
        return []
    result: Optional[List[Any]] = None
    if transcript_windows is not None:
        try:
            result = list(transcript_windows(context, max_units))
        except Exception as exc:
            logger.warning('transcript.windows failed, using fallback: %r', exc)
    if result is None:
        result = _fallback_windows(context, max_units)
    return sorted(result, key=lambda w: (w.unit_ids[0], len(w.unit_ids)))


def _neighbour_text(context: AudioContext, window: Any) -> str:
    """Window text plus the unit before and after it.

    The fact checker sees a little more than the NLI so that a dose stated in
    the next clause ("...penicillin. 100 milligrams daily.") is still compared
    against the question. Positions are looked up by id, never computed from
    ids, because ids need not be contiguous.
    """
    position = {u.unit_id: i for i, u in enumerate(context.units)}
    first = position.get(window.unit_ids[0])
    last = position.get(window.unit_ids[-1])
    if first is None or last is None:
        return str(window.text)
    lo, hi = max(0, first - 1), min(len(context.units), last + 2)
    return ' '.join(u.text.strip() for u in context.units[lo:hi])


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #

NUMBER_MATCH_BONUS = 0.1
NUMBER_MISMATCH_PENALTY = 0.3


def lexical_score(focus: Sequence[str], window_tokens: Iterable[str]) -> float:
    """Fraction of focus terms found in the window, adjusted for numbers.

    Numbers are the thing near-miss questions get wrong ("50 mg" vs "100 mg"),
    so an exact numeric match earns a bonus, and a window that states other
    numbers while missing the question's number is penalised: it is most
    likely the near-miss passage itself. Without the penalty the lexical
    fallback would say yes to the wrong dose whenever the drug name matches.
    The result is clipped to [0, 1] so it stays comparable across windows.
    """
    if not focus:
        return 0.0
    present = set(window_tokens)
    matched = [t for t in focus if t in present]
    score = len(matched) / len(focus)
    score += NUMBER_MATCH_BONUS * sum(1 for t in matched if is_number_token(t))
    missing_numbers = any(is_number_token(t) and t not in present for t in focus)
    if missing_numbers and any(is_number_token(t) for t in present):
        score -= NUMBER_MISMATCH_PENALTY
    return min(1.0, max(0.0, score))


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax; NaN/inf logits are treated as 0 first."""
    x = np.nan_to_num(np.asarray(logits, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-normalise; zero rows stay zero (cosine 0) instead of becoming NaN."""
    m = np.nan_to_num(np.asarray(matrix, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return m / norms


def _best_index(candidates: Sequence[int], scores: Any, windows: Sequence[Any]) -> int:
    """Argmax over candidate window indices with a deterministic tie-break.

    ``scores`` is indexed by window index (a list or a dict). Ties (after
    rounding away float noise) go to the shortest window, then
    the earliest: gold evidence spans are short, so a window that adds units
    without adding score only dilutes the localisation.
    """
    return max(
        candidates,
        key=lambda j: (round(float(scores[j]), 6), -len(windows[j].unit_ids), -windows[j].unit_ids[0]),
    )


def resolve_nli_label_order(nli: Any) -> Tuple[List[int], str]:
    """Permutation mapping raw NLI columns to [contradiction, entailment, neutral].

    Looks for ``nli.label_names``, then ``nli.config.id2label``, then
    ``nli.model.config.id2label`` (the layout of a ``CrossEncoder``). Returns
    ``(perm, source)`` where ``canonical[:, c] == raw[:, perm[c]]``. When no
    usable labels are found the order is assumed canonical and ``source`` says
    so, because every ``cross-encoder/nli-*`` model uses that order.
    """
    labels: Optional[List[str]] = None
    source = 'assumed'
    names = getattr(nli, 'label_names', None)
    if names:
        labels, source = [str(n) for n in names], 'label_names'
    else:
        for holder in (getattr(nli, 'config', None), getattr(getattr(nli, 'model', None), 'config', None)):
            id2label = getattr(holder, 'id2label', None)
            if isinstance(id2label, Mapping) and id2label:
                try:
                    labels = [str(id2label[k]) for k in sorted(id2label, key=lambda k: int(k))]
                    source = 'id2label'
                except (TypeError, ValueError):
                    labels = None
                break
    if not labels or len(labels) != 3:
        return [0, 1, 2], 'assumed'
    perm: List[int] = []
    for canonical in _CANONICAL_LABELS:
        matches = [i for i, name in enumerate(labels) if name.lower().startswith(canonical)]
        if len(matches) != 1:
            return [0, 1, 2], 'assumed'
        perm.append(matches[0])
    return perm, source


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

_TRUE_STRINGS = frozenset({'1', 'true', 'yes', 'on'})
_FALSE_STRINGS = frozenset({'0', 'false', 'no', 'off'})


@dataclass
class QAConfig:
    """Knobs for both backends; every field is overridable through the env."""

    backend: str = 'retrieval_nli'
    embed_model: str = 'BAAI/bge-small-en-v1.5'
    nli_model: str = 'cross-encoder/nli-deberta-v3-small'
    top_k: int = 6
    max_window_units: int = 3
    yes_threshold: float = 0.5
    off_topic_sim_threshold: float = 0.25
    lexical_weight: float = 0.4
    use_factcheck: bool = True
    device: str = 'cpu'
    batch_size: int = 32
    max_seq_len: int = 256
    unverifiable_means_no: bool = False
    lexical_yes_threshold: float = 0.6
    existence_yes_threshold: float = 0.3
    min_time_for_nli_s: float = 3.0
    local_files_only: bool = False
    # Learned decision/window layer (``calibration.py``). Empty = rule-based.
    calibration_path: str = ''
    hard_contradiction_gate: bool = True

    def __post_init__(self) -> None:
        if self.top_k < 1 or self.max_window_units < 1 or self.batch_size < 1 or self.max_seq_len < 8:
            raise ValueError('top_k, max_window_units, batch_size must be >= 1 and max_seq_len >= 8')
        for name in ('yes_threshold', 'off_topic_sim_threshold', 'lexical_weight',
                     'lexical_yes_threshold', 'existence_yes_threshold'):
            value = getattr(self, name)
            if not (isinstance(value, (int, float)) and math.isfinite(value) and 0.0 <= value <= 1.0):
                raise ValueError(f'{name} must be a finite number in [0, 1], got {value!r}')
        if not (math.isfinite(self.min_time_for_nli_s) and self.min_time_for_nli_s >= 0.0):
            raise ValueError('min_time_for_nli_s must be >= 0')

    @property
    def config_hash(self) -> str:
        return stable_hash(asdict(self))

    @classmethod
    def from_env(cls, prefix: str = 'MA_QA_', env: Optional[Mapping[str, str]] = None) -> 'QAConfig':
        """Build a config from ``<prefix><FIELD>`` variables, e.g. ``MA_QA_TOP_K``.

        Unset variables keep the dataclass default; a set variable that cannot
        be parsed raises ``ValueError`` naming the variable, because a silently
        ignored production setting is worse than a loud startup failure.
        """
        source = os.environ if env is None else env
        values: Dict[str, Any] = {}
        for f in fields(cls):
            key = f'{prefix}{f.name.upper()}'
            if key not in source:
                continue
            raw = str(source[key]).strip()
            try:
                if f.type in ('bool', bool):
                    lowered = raw.lower()
                    if lowered in _TRUE_STRINGS:
                        values[f.name] = True
                    elif lowered in _FALSE_STRINGS:
                        values[f.name] = False
                    else:
                        raise ValueError('expected a boolean')
                elif f.type in ('int', int):
                    values[f.name] = int(raw)
                elif f.type in ('float', float):
                    values[f.name] = float(raw)
                else:
                    values[f.name] = raw
            except ValueError as exc:
                raise ValueError(f'{key}={raw!r}: {exc}') from exc
        return cls(**values)


# --------------------------------------------------------------------------- #
# Result helpers
# --------------------------------------------------------------------------- #

def _default_result(index: int, backend: str, reason: str, **diagnostics: Any) -> QAResult:
    """The answer given when nothing can be decided: no, with no evidence."""
    return QAResult(
        question_index=index, answer=False, evidence_unit_ids=[], confidence=0.0,
        backend=backend, diagnostics={'reason': reason, **diagnostics},
    )


def _error_result(index: int, backend: str, exc: BaseException) -> QAResult:
    return _default_result(index, backend, 'exception', error=repr(exc))


def _round(value: Any) -> float:
    return round(float(value), 4)


def _empty_results(n: int, backend: str, reason: str) -> List[QAResult]:
    return [_default_result(i, backend, reason) for i in range(n)]


@dataclass
class _Prepared:
    """Everything derived from one question before any model runs."""

    question: str
    statement: str
    focus: List[str]
    existence: bool


def _prepare(question: str) -> _Prepared:
    return _Prepared(
        question=question,
        statement=statement_for(question),
        focus=focus_tokens(question),
        existence=is_existence_question(question),
    )


def _gate_passes(verdict: LocalVerdict, config: QAConfig) -> bool:
    """The fact-check gate: a contradiction is always a no; unverifiable is
    a no only when configured so."""
    if verdict.status == CONTRADICTION:
        return False
    if verdict.status == UNVERIFIABLE and config.unverifiable_means_no:
        return False
    return True


# --------------------------------------------------------------------------- #
# Lexical backend
# --------------------------------------------------------------------------- #

class LexicalBackend:
    """Focus-term overlap over windows with a fact-check gate.

    Fast, dependency-free and deterministic. It is the baseline the ML backend
    must beat and the answer of last resort when models or time run out.
    """

    name = 'lexical'

    def __init__(self, config: Optional[QAConfig] = None) -> None:
        self.config = config if config is not None else QAConfig()

    def warm_up(self) -> None:
        """Nothing to load."""

    def answer(
        self,
        context: AudioContext,
        questions: List[str],
        deadline: Optional[Deadline] = None,
    ) -> List[QAResult]:
        del deadline  # the lexical path is far cheaper than any budget
        n = len(questions)
        if n == 0:
            return []
        windows = sorted_windows(context, self.config.max_window_units)
        if not windows:
            return _empty_results(n, self.name, 'no_transcript')
        window_tokens = [set(tokenize_for_match(w.text)) for w in windows]
        results: List[QAResult] = []
        for i, question in enumerate(questions):
            try:
                results.append(self._decide(i, question, context, windows, window_tokens))
            except Exception as exc:
                logger.exception('lexical decision failed for question %d', i)
                results.append(_error_result(i, self.name, exc))
        return results

    def _decide(
        self,
        index: int,
        question: str,
        context: AudioContext,
        windows: Sequence[Any],
        window_tokens: Sequence[set],
    ) -> QAResult:
        if not isinstance(question, str) or not question.strip():
            return _default_result(index, self.name, 'empty_question')
        prepared = _prepare(question)
        if not prepared.focus:
            return _default_result(index, self.name, 'no_focus_terms',
                                   statement=prepared.statement)
        scores = [lexical_score(prepared.focus, toks) for toks in window_tokens]
        best = _best_index(range(len(windows)), scores, windows)
        score = float(scores[best])
        verdict = (factcheck_verdict(question, _neighbour_text(context, windows[best]))
                   if self.config.use_factcheck else LocalVerdict(NO_QUANTITIES, ['factcheck disabled']))
        answer = score >= self.config.lexical_yes_threshold and _gate_passes(verdict, self.config)
        return QAResult(
            question_index=index,
            answer=bool(answer),
            evidence_unit_ids=list(windows[best].unit_ids),
            confidence=_round(min(1.0, max(0.0, score))),
            backend=self.name,
            diagnostics={
                'score': _round(score),
                'best_window': list(windows[best].unit_ids),
                'focus_terms': list(prepared.focus),
                'statement': prepared.statement,
                'existence': prepared.existence,
                'verdict': verdict.status,
                'verdict_details': list(verdict.details),
            },
        )


# --------------------------------------------------------------------------- #
# Model adapters (real sentence-transformers objects behind a tiny protocol)
# --------------------------------------------------------------------------- #

def resolve_device(device: Optional[str]) -> Optional[str]:
    """Translate the config's device string into what sentence-transformers accepts.

    The ASR config uses ``'auto'`` for "let the library pick"; sentence-
    transformers spells that ``None``. Accepting both here means one value
    can be shared across the ``asr`` and ``qa`` sections of a config file
    without a crash at model-load time.
    """
    if device is None:
        return None
    lowered = str(device).strip().lower()
    return None if lowered in ('', 'auto') else lowered


class SentenceTransformerEmbedder:
    """``encode(texts, batch_size) -> (n, d)`` unit vectors from a bi-encoder."""

    def __init__(self, model_name: str, device: str, max_seq_len: int, local_files_only: bool) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model_name, device=resolve_device(device), local_files_only=local_files_only)
        self.model.max_seq_length = int(max_seq_len)

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(self.model.encode(
            list(texts), batch_size=int(batch_size), normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ))


class CrossEncoderNLI:
    """``predict(pairs, batch_size) -> (n, 3)`` raw logits from an NLI cross-encoder.

    The activation is forced to identity so the caller always receives logits
    regardless of what the model card configured. ``resolve_nli_label_order``
    reads the model's own label order through ``self.model.config.id2label``.
    """

    def __init__(self, model_name: str, device: str, max_seq_len: int, local_files_only: bool) -> None:
        import torch
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(
            model_name, device=resolve_device(device), max_length=int(max_seq_len),
            local_files_only=local_files_only, activation_fn=torch.nn.Identity(),
        )

    def predict(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 32) -> np.ndarray:
        return np.asarray(self.model.predict(
            [tuple(p) for p in pairs], batch_size=int(batch_size), apply_softmax=False,
            convert_to_numpy=True, show_progress_bar=False,
        ))


# --------------------------------------------------------------------------- #
# Retrieval + NLI backend
# --------------------------------------------------------------------------- #

class RetrievalNLIBackend:
    """Retrieve top-k windows per question, verify with NLI, gate with facts.

    ``embedder`` and ``nli`` follow the protocols of the adapters above and
    are injectable for tests; when ``None`` the configured models load lazily.
    """

    name = 'retrieval_nli'

    def __init__(
        self,
        config: Optional[QAConfig] = None,
        embedder: Any = None,
        nli: Any = None,
    ) -> None:
        self.config = config if config is not None else QAConfig()
        self._embedder = embedder
        self._nli = nli
        self.nli_label_permutation: Optional[List[int]] = None
        self.nli_label_source: str = 'unresolved'
        self._lexical = LexicalBackend(self.config)
        self._calibration: Any = None
        self._calibration_state: str = 'unloaded'
        if nli is not None:
            self._resolve_labels()

    # -- models ----------------------------------------------------------- #

    def _resolve_labels(self) -> None:
        self.nli_label_permutation, self.nli_label_source = resolve_nli_label_order(self._nli)
        if self.nli_label_source == 'assumed':
            logger.warning('NLI label order not found on model; assuming %s', _CANONICAL_LABELS)

    def _ensure_models(self) -> None:
        """Load whatever was not injected. Called lazily so importing this
        module and constructing the backend stay free of model I/O."""
        cfg = self.config
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder(
                cfg.embed_model, cfg.device, cfg.max_seq_len, cfg.local_files_only)
        if self._nli is None:
            self._nli = CrossEncoderNLI(cfg.nli_model, cfg.device, cfg.max_seq_len, cfg.local_files_only)
        if self.nli_label_permutation is None:
            self._resolve_labels()
        self._ensure_calibration()

    def _ensure_calibration(self) -> None:
        """Load the learned layer once; a missing or broken file means the
        rule-based decision is used and the reason is recorded."""
        if self._calibration_state != 'unloaded':
            return
        path = (self.config.calibration_path or '').strip()
        if not path:
            self._calibration_state = 'disabled'
            return
        if Calibration is None:
            self._calibration_state = 'module_missing'
            logger.warning('calibration_path set but calibration module unavailable; using rules')
            return
        candidates = [path, os.path.join(os.path.dirname(os.path.abspath(__file__)), path)]
        for candidate in candidates:
            if os.path.isfile(candidate):
                try:
                    self._calibration = Calibration.load(candidate)
                    self._calibration_state = 'loaded'
                    logger.info('loaded calibration %s', candidate)
                    return
                except Exception:
                    logger.exception('calibration file %s is unusable; using rules', candidate)
                    self._calibration_state = 'invalid'
                    return
        self._calibration_state = 'missing'
        logger.warning('calibration file %s not found; using rules', path)

    def set_calibration(self, calibration: Any) -> None:
        """Inject a calibration object (tests, offline experiments)."""
        self._calibration = calibration
        self._calibration_state = 'loaded' if calibration is not None else 'disabled'

    def warm_up(self) -> None:
        """Load the models and push one tiny batch through each so the first
        real request does not pay for lazy initialisation."""
        self._ensure_models()
        self._embed(['warm up'])
        self._nli_probabilities([('warm up', 'warm up')])

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._embedder.encode(list(texts), batch_size=self.config.batch_size)
        matrix = _l2_normalize(np.asarray(vectors))
        if matrix.shape[0] != len(texts):
            raise ValueError(f'embedder returned {matrix.shape[0]} rows for {len(texts)} texts')
        return matrix

    def _nli_probabilities(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """Softmaxed ``[p_contradiction, p_entailment, p_neutral]`` per pair."""
        if not pairs:
            return np.zeros((0, 3), dtype=np.float64)
        raw = np.asarray(self._nli.predict(list(pairs), batch_size=self.config.batch_size), dtype=np.float64)
        if raw.ndim != 2 or raw.shape != (len(pairs), 3):
            raise ValueError(f'NLI returned shape {raw.shape}, expected {(len(pairs), 3)}')
        perm = self.nli_label_permutation or [0, 1, 2]
        return softmax(raw[:, perm])

    # -- answering -------------------------------------------------------- #

    def answer(
        self,
        context: AudioContext,
        questions: List[str],
        deadline: Optional[Deadline] = None,
    ) -> List[QAResult]:
        n = len(questions)
        if n == 0:
            return []
        windows = sorted_windows(context, self.config.max_window_units)
        if not windows:
            return _empty_results(n, self.name, 'no_transcript')
        if deadline is not None and deadline.expired():
            return self._degrade(context, questions, 'deadline_expired')
        try:
            self._ensure_models()
            window_texts = [str(w.text) for w in windows]
            window_tokens = [set(tokenize_for_match(t)) for t in window_texts]
            prepared = [_prepare(q if isinstance(q, str) else '') for q in questions]
            cosine = np.clip(self._embed(list(p.question for p in prepared)) @ self._embed(window_texts).T, 0.0, 1.0)
            lexical = np.array([[lexical_score(p.focus, toks) for toks in window_tokens] for p in prepared])
            combined = (1.0 - self.config.lexical_weight) * cosine + self.config.lexical_weight * lexical
        except Exception as exc:
            logger.exception('retrieval stage failed; degrading to lexical decisions')
            return self._degrade(context, questions, 'retrieval_error', error=repr(exc))

        k = min(self.config.top_k, len(windows))
        top_k = [list(np.argsort(-combined[i], kind='stable')[:k]) for i in range(n)]

        if deadline is not None and not deadline.fits(self.config.min_time_for_nli_s):
            return self._degrade(context, questions, 'deadline')
        try:
            pairs = [(window_texts[j], prepared[i].statement) for i in range(n) for j in top_k[i]]
            probs = self._nli_probabilities(pairs)
        except Exception as exc:
            logger.exception('NLI stage failed; degrading to lexical decisions')
            return self._degrade(context, questions, 'nli_error', error=repr(exc))

        results: List[QAResult] = []
        offset = 0
        for i in range(n):
            rows = probs[offset:offset + len(top_k[i])]
            offset += len(top_k[i])
            try:
                results.append(self._decide_question(
                    i, prepared[i], context, windows, top_k[i], rows, cosine[i], lexical[i], combined[i]))
            except Exception as exc:
                logger.exception('decision failed for question %d', i)
                results.append(_error_result(i, self.name, exc))
        return results

    def _degrade(self, context: AudioContext, questions: List[str], reason: str, **extra: Any) -> List[QAResult]:
        """Lexical decisions for every question, flagged as degraded."""
        results = self._lexical.answer(context, questions)
        for result in results:
            result.diagnostics.update({'degraded': True, 'degrade_reason': reason, **extra})
        return results

    def _decide_question(
        self,
        index: int,
        prepared: _Prepared,
        context: AudioContext,
        windows: Sequence[Any],
        candidates: Sequence[int],
        probs: np.ndarray,
        cosine_row: np.ndarray,
        lexical_row: np.ndarray,
        combined_row: np.ndarray,
    ) -> QAResult:
        cfg = self.config
        if not prepared.question.strip():
            return _default_result(index, self.name, 'empty_question')
        best = self._best_by_entailment(candidates, probs, windows)
        row = list(candidates).index(best)
        p_con, p_ent, p_neu = (float(x) for x in probs[row])
        max_similarity = float(combined_row.max()) if combined_row.size else 0.0

        verdict = (factcheck_verdict(prepared.question, _neighbour_text(context, windows[best]))
                   if cfg.use_factcheck else LocalVerdict(NO_QUANTITIES, ['factcheck disabled']))
        gate = _gate_passes(verdict, cfg)
        calibrated = self._calibration is not None and Candidate is not None
        p_yes: Optional[float] = None
        if calibrated:
            # Learned layer: logistic decision over question-level features and
            # a linear ranker choosing the evidence window (see calibration.py).
            cands = [Candidate(
                unit_ids=tuple(int(u) for u in windows[j].unit_ids),
                start_s=float(windows[j].start_s), end_s=float(windows[j].end_s),
                p_ent=float(probs[r, 1]), p_con=float(probs[r, 0]),
                cosine=float(cosine_row[j]), lexical=float(lexical_row[j]), rank=r,
            ) for r, j in enumerate(candidates)]
            answer, p_yes = self._calibration.decide(cands, prepared.existence, verdict.status, prepared.question)
            if cfg.hard_contradiction_gate and verdict.status == CONTRADICTION:
                answer = False
            picked = self._calibration.pick_window(cands)
            if picked is not None:
                best = list(candidates)[cands.index(picked)]
        elif prepared.existence:
            answer = (max_similarity >= cfg.off_topic_sim_threshold
                      and p_ent >= cfg.existence_yes_threshold
                      and verdict.status != CONTRADICTION)
        else:
            answer = p_ent >= cfg.yes_threshold and p_ent > p_con and gate

        top_windows = [{
            'unit_ids': list(windows[j].unit_ids),
            'combined': _round(combined_row[j]),
            'cosine': _round(cosine_row[j]),
            'lexical': _round(lexical_row[j]),
            'p_con': _round(probs[r, 0]),
            'p_ent': _round(probs[r, 1]),
            'p_neu': _round(probs[r, 2]),
        } for r, j in enumerate(candidates)]
        return QAResult(
            question_index=index,
            answer=bool(answer),
            evidence_unit_ids=list(windows[best].unit_ids),
            confidence=_round(p_yes if p_yes is not None else p_ent),
            backend=self.name,
            diagnostics={
                'calibrated': calibrated,
                'p_yes': _round(p_yes) if p_yes is not None else None,
                'statement': prepared.statement,
                'focus_terms': list(prepared.focus),
                'existence': prepared.existence,
                'best_window': list(windows[best].unit_ids),
                'p_ent': _round(p_ent),
                'p_con': _round(p_con),
                'p_neu': _round(p_neu),
                'max_similarity': _round(max_similarity),
                'verdict': verdict.status,
                'verdict_details': list(verdict.details),
                'top_windows': top_windows,
                'nli_label_source': self.nli_label_source,
            },
        )

    @staticmethod
    def _best_by_entailment(candidates: Sequence[int], probs: np.ndarray, windows: Sequence[Any]) -> int:
        scores = {j: float(probs[r, 1]) for r, j in enumerate(candidates)}
        return _best_index(list(candidates), scores, windows)


# --------------------------------------------------------------------------- #
# Test double and factory
# --------------------------------------------------------------------------- #

class FakeQABackend:
    """Returns scripted results; pads with default no-answers to keep the
    one-result-per-question invariant whatever the script length."""

    name = 'fake'

    def __init__(self, results: Sequence[QAResult]) -> None:
        self.results = list(results)
        self.calls: List[List[str]] = []

    def warm_up(self) -> None:
        """Nothing to load."""

    def answer(
        self,
        context: AudioContext,
        questions: List[str],
        deadline: Optional[Deadline] = None,
    ) -> List[QAResult]:
        del context, deadline
        self.calls.append(list(questions))
        out: List[QAResult] = []
        for i in range(len(questions)):
            if i < len(self.results):
                scripted = self.results[i]
                out.append(QAResult(
                    question_index=i, answer=bool(scripted.answer),
                    evidence_unit_ids=list(scripted.evidence_unit_ids),
                    evidence_word_ids=scripted.evidence_word_ids,
                    confidence=scripted.confidence, backend=self.name,
                    diagnostics=dict(scripted.diagnostics),
                ))
            else:
                out.append(_default_result(i, self.name, 'unscripted'))
        return out


def build_qa_backend(config: QAConfig) -> QABackend:
    """Instantiate the backend named by ``config.backend``.

    The LLM backend lives in its own module with heavy optional dependencies,
    so it is imported only when asked for, and a missing module is reported
    as a configuration error rather than a bare ImportError. Its settings
    come from the ``MA_LLM_*`` environment (its own config namespace), since
    ``QAConfig`` only knows about the retrieval/NLI knobs.
    """
    name = str(config.backend).strip().lower()
    if name == 'lexical':
        return LexicalBackend(config)
    if name == 'retrieval_nli':
        return RetrievalNLIBackend(config)
    if name == 'llm':
        try:
            import qa_llm_backend
        except ImportError as exc:
            raise RuntimeError(
                "config.backend='llm' requires qa_llm_backend.py, which is not importable") from exc
        return qa_llm_backend.LLMBackend(qa_llm_backend.LLMConfig.from_env())
    raise ValueError(f"unknown QA backend {config.backend!r}; expected 'lexical', 'retrieval_nli' or 'llm'")
