"""Last gate before a response leaves the process (contract section A9).

WHY this module exists: the evaluation service reads the three response lists
positionally and throws the whole conversation away when their lengths differ,
while a malformed interval (NaN, reversed, half-filled, a bool where a number
belongs) silently scores a temporal IoU of 0. Every upstream stage is allowed
to fail or hand over garbage; this module turns whatever arrives into a
response that is always scoreable (``finalize``), and proves it before it is
sent (``strict_validate``, ``check_wire_json``).

Trust model: predictions are produced by trusted code, but their *values* may
be wrong in any way (an LLM parser letting a string through, a numpy scalar,
a span past the end of the audio). Nothing here interprets text, builds paths
or chooses roles; the only untrusted-looking input is our own JSON output,
which ``check_wire_json`` parses with the standard library and never executes.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from core_types import Prediction, Span, is_finite_number
from dtos import ASRQuestionResponseDto
from utils import validate_response as _reference_validate

logger = logging.getLogger(__name__)

#: Seconds a span may extend past the decoded audio length. Whisper word
#: timings are quantised to 20 ms and the decoder's length can differ from the
#: MP3 header by a frame, so a hard ``end <= duration_s`` would throw away
#: good spans that end on the last word.
END_TOLERANCE_S = 0.05

#: Decimal places kept on the wire. Millisecond precision is finer than any ASR
#: timing and keeps the JSON short and byte-for-byte reproducible.
TIMESTAMP_DECIMALS = 3

#: The exact top-level keys of ``ASRQuestionResponseDto``. The service ignores
#: extra keys today; we still refuse them so a refactor cannot leak internals
#: (diagnostics, transcript text) onto the wire.
WIRE_KEYS = frozenset({'answers', 'evidence_start', 'evidence_end'})

#: Used when a caller passes something that is not a bool as the fallback
#: answer. ``True`` because both label sets are balanced (the choice is
#: arithmetically neutral) and only a True answer can carry evidence.
DEFAULT_FALLBACK_ANSWER = True

# Slack when comparing a 3-dp timestamp against ``duration_s + tolerance``:
# absorbs float representation noise, never a real overrun.
_EPSILON = 1e-9
_REPR_LIMIT = 80


class ResponseCheckError(ValueError):
    """A response would not survive the evaluator, or would score nothing.

    ``violations`` lists every problem found, so one log line shows the whole
    picture instead of the first failure only. Subclasses ``ValueError`` to
    match ``utils.validate_response``.
    """

    def __init__(self, violations: Union[str, Sequence[str]]) -> None:
        if isinstance(violations, str):
            violations = [violations]
        self.violations: List[str] = [str(v) for v in violations] or ['unspecified violation']
        count = len(self.violations)
        noun = 'violation' if count == 1 else 'violations'
        super().__init__(f'{count} response {noun}: ' + '; '.join(self.violations))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _short_repr(value: Any) -> str:
    """Bounded ``repr`` for log and error messages; never raises."""
    try:
        text = repr(value)
    except Exception:  # a hostile __repr__ must not turn a warning into a crash
        text = f'<{type(value).__name__} with unprintable repr>'
    return text if len(text) <= _REPR_LIMIT else text[:_REPR_LIMIT - 3] + '...'


def _is_finite_number(value: Any) -> bool:
    """``core_types.is_finite_number`` that also refuses an ``int`` too large
    to become a float (``math.isfinite`` raises ``OverflowError`` on those,
    and ``float()`` would too)."""
    try:
        return is_finite_number(value)
    except OverflowError:
        return False


def _unwrap_scalar(value: Any) -> Any:
    """Convert a 0-d numpy/torch scalar to the matching Python scalar.

    WHY: the evidence and QA stages compute with numpy, and ``np.float32`` is
    not a ``float`` nor ``np.bool_`` a ``bool``. Without this, every span they
    produce would be dropped as "not a number" and every answer replaced by
    the fallback, silently. Anything that is not a 0-d array-like is returned
    unchanged, so no leniency is added for strings, ints or lists.
    """
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    item = getattr(value, 'item', None)
    shape = getattr(value, 'shape', None)
    if callable(item) and isinstance(shape, tuple) and len(shape) == 0:
        try:
            return item()
        except (TypeError, ValueError):
            return value
    return value


def _check_n_questions(n_questions: Any) -> int:
    """``n_questions`` comes from ``len(request.questions)``; anything else is
    a programming error worth surfacing rather than a value to guess around."""
    if isinstance(n_questions, bool) or not isinstance(n_questions, int) or n_questions < 0:
        raise ResponseCheckError(
            f'n_questions must be a non-negative int, got {_short_repr(n_questions)}'
        )
    return n_questions


def _coerce_fallback(answer: Any) -> bool:
    """The fallback must itself be a real bool; otherwise use the module default."""
    answer = _unwrap_scalar(answer)
    if isinstance(answer, bool):
        return answer
    logger.warning(
        'fallback answer %s is not a bool; using %r',
        _short_repr(answer), DEFAULT_FALLBACK_ANSWER,
    )
    return DEFAULT_FALLBACK_ANSWER


def _usable_duration(duration_s: Any) -> Optional[float]:
    """Audio length for the upper clamp, or ``None`` when it cannot be trusted.

    A NaN, negative or non-numeric duration must not collapse every span to
    null, so it is treated as unknown (no upper bound) and logged.
    """
    duration_s = _unwrap_scalar(duration_s)
    if duration_s is None:
        return None
    if not _is_finite_number(duration_s) or duration_s < 0:
        logger.warning(
            'duration_s=%s is not a usable audio length; spans are not bounded above',
            _short_repr(duration_s),
        )
        return None
    return float(duration_s)


def _clean_span(span: Any, duration_s: Optional[float]) -> Tuple[Optional[Span], Optional[str]]:
    """Clamp, round and validate one candidate span.

    Returns ``(span, None)`` when usable and ``(None, reason)`` when it must be
    dropped. Dropping is right because a True answer with null timestamps is
    legal and only forfeits the evidence part, whereas a span the service
    cannot read forfeits the same points *and* hides the bug.
    """
    if span is None:
        return None, None
    if isinstance(span, (str, bytes, dict)):
        return None, f'span must be a (start, end) pair, got {type(span).__name__}'
    try:
        start, end = span
    except Exception:  # not a pair, or an iterable that raises while unpacking
        return None, f'span must be a (start, end) pair, got {_short_repr(span)}'

    start, end = _unwrap_scalar(start), _unwrap_scalar(end)
    if not (_is_finite_number(start) and _is_finite_number(end)):
        return None, (
            f'span bounds must be finite numbers, got '
            f'({_short_repr(start)}, {_short_repr(end)})'
        )

    start, end = float(start), float(end)
    if start <= 0.0:
        # Explicit assignment rather than max(): max(-0.0, 0.0) keeps the
        # negative zero and the wire would carry "-0.0".
        start = 0.0
    cap = None if duration_s is None else duration_s + END_TOLERANCE_S
    if cap is not None:
        end = min(end, cap)

    start = round(start, TIMESTAMP_DECIMALS)
    end = round(end, TIMESTAMP_DECIMALS)
    if cap is not None and end > cap:
        # round() may step over the cap by < 0.5 ms; flooring keeps the wire
        # value both at 3 dp and inside the tolerance strict_validate checks.
        scale = 10 ** TIMESTAMP_DECIMALS
        end = math.floor(cap * scale) / scale

    if end <= start:
        return None, f'span is empty after clamping and rounding: ({start}, {end})'
    return (start, end), None


def _clean_item(
    item: Any, index: int, fallback: bool, duration: Optional[float],
) -> Tuple[bool, Optional[Span]]:
    """One prediction -> ``(bool answer, clean span or None)``.

    Anything that is not a real bool (after numpy unwrapping) becomes
    ``fallback``; a span is only kept for a True answer.
    """
    try:
        raw_answer = _unwrap_scalar(getattr(item, 'answer', None))
    except Exception:  # a property that raises: treat as "no answer"
        logger.exception('prediction %d: reading the answer raised', index)
        raw_answer = None
    if isinstance(raw_answer, bool):
        answer = bool(raw_answer)
    else:
        logger.warning(
            'prediction %d has non-bool answer %s; using fallback %r',
            index, _short_repr(raw_answer), fallback,
        )
        answer = fallback

    span: Optional[Span] = None
    if answer:
        try:
            raw_span = getattr(item, 'span', None)
        except Exception:
            logger.exception('prediction %d: reading the span raised', index)
            raw_span = None
        span, reason = _clean_span(raw_span, duration)
        if reason is not None:
            logger.warning('prediction %d: %s; sending null evidence', index, reason)
    return answer, span


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def finalize(
    predictions: Optional[Iterable[Any]],
    n_questions: int,
    duration_s: Optional[float],
    fallback_answer: bool = True,
) -> ASRQuestionResponseDto:
    """Turn per-question predictions into a response that is always scoreable.

    Guarantees, whatever comes in:

    * exactly ``n_questions`` entries in each list (short input is padded with
      ``Prediction(fallback_answer, None, source='padded')``, long input is
      truncated; both are logged as errors because they mean a stage broke
      its "one result per question" invariant);
    * every answer is a real ``bool``; anything else becomes ``fallback_answer``;
    * a False answer carries ``(None, None)``;
    * a True answer carries either a clean span (finite, start clamped to
      ``>= 0``, end clamped to ``<= duration_s + END_TOLERANCE_S`` when the
      duration is known, ``end > start``, rounded to 3 dp) or ``(None, None)``.

    Never reads text; never raises for bad prediction *values*. It does raise
    ``ResponseCheckError`` for an impossible ``n_questions`` because that is a
    bug in trusted code, not a data problem.
    """
    n = _check_n_questions(n_questions)
    fallback = _coerce_fallback(fallback_answer)
    duration = _usable_duration(duration_s)

    try:
        items = list(predictions) if predictions is not None else []
    except Exception:
        # Not iterable, or an iterator that blew up: a stage bug, but one that
        # must cost at most the evidence marks, never the conversation.
        logger.exception(
            'predictions (%s) could not be iterated; using the fallback answer for every question',
            type(predictions).__name__,
        )
        items = []
    if len(items) != n:
        logger.error(
            'expected %d predictions, got %d; %s',
            n, len(items), 'padding with the fallback answer' if len(items) < n else 'truncating',
        )
        items = items[:n] + [
            Prediction(fallback, None, source='padded') for _ in range(n - len(items))
        ]

    answers: List[bool] = []
    starts: List[Optional[float]] = []
    ends: List[Optional[float]] = []
    for index, item in enumerate(items):
        try:
            answer, span = _clean_item(item, index, fallback, duration)
        except Exception:
            # Last resort: ``_clean_item`` guards the answer and span reads,
            # so only something exotic (a numpy ``item()`` that throws, say)
            # lands here. Trusted code is misbehaving, but the wire contract
            # comes first.
            logger.exception(
                'prediction %d could not be read; using fallback %r with null evidence',
                index, fallback,
            )
            answer, span = fallback, None

        answers.append(answer)
        starts.append(None if span is None else span[0])
        ends.append(None if span is None else span[1])

    response = ASRQuestionResponseDto(answers=answers, evidence_start=starts, evidence_end=ends)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('finalized response: %s', summarize(response))
    return response


def strict_validate(
    response: ASRQuestionResponseDto,
    n_questions: int,
    duration_s: Optional[float] = None,
) -> None:
    """Raise ``ResponseCheckError`` listing every way ``response`` is wrong.

    Stricter than ``utils.validate_response`` (which is also run, for parity):
    a False answer must carry nulls, a start cannot be negative, an end must be
    after its start and, when ``duration_s`` is given, at most
    ``duration_s + END_TOLERANCE_S``. Timestamps must be real numbers, so a
    bool (a subclass of ``int``) is rejected explicitly.
    """
    n = _check_n_questions(n_questions)
    duration = _unwrap_scalar(duration_s)
    if duration is not None and (not _is_finite_number(duration) or duration < 0):
        # ``finalize`` tolerates a bad duration (it just stops clamping), but a
        # validator fed one would either pass or fail every span for the wrong
        # reason, so here it is a caller bug.
        raise ResponseCheckError(
            f'duration_s must be a non-negative finite number or None, got '
            f'{_short_repr(duration_s)}'
        )
    if not isinstance(response, ASRQuestionResponseDto):
        raise ResponseCheckError(
            f'response must be an ASRQuestionResponseDto, got {type(response).__name__}'
        )

    violations: List[str] = []
    columns: Dict[str, list] = {}
    for name in ('answers', 'evidence_start', 'evidence_end'):
        values = getattr(response, name, None)
        if not isinstance(values, list):
            violations.append(f'{name} must be a list, got {type(values).__name__}')
            values = []
        elif len(values) != n:
            violations.append(f'{name} must have {n} entries, got {len(values)}')
        columns[name] = values

    for index, answer in enumerate(columns['answers']):
        if not isinstance(answer, bool):
            violations.append(
                f'answers[{index}] must be a bool, got {type(answer).__name__} '
                f'{_short_repr(answer)}'
            )

    for name in ('evidence_start', 'evidence_end'):
        for index, value in enumerate(columns[name]):
            if value is None:
                continue
            if isinstance(value, bool):
                violations.append(f'{name}[{index}] is a bool masquerading as a number')
            elif not isinstance(value, (int, float)):
                violations.append(
                    f'{name}[{index}] must be a number or None, got {type(value).__name__}'
                )
            elif not _is_finite_number(value):
                violations.append(f'{name}[{index}] must be finite, got {value!r}')

    pairs = zip(columns['answers'], columns['evidence_start'], columns['evidence_end'])
    for index, (answer, start, end) in enumerate(pairs):
        if (start is None) != (end is None):
            violations.append(
                f'evidence[{index}] is half-filled: start={start!r}, end={end!r}'
            )
            continue
        if start is None:
            continue
        if answer is False:
            violations.append(
                f'evidence[{index}] must be null for a False answer, got ({start!r}, {end!r})'
            )
        if not (_is_finite_number(start) and _is_finite_number(end)):
            continue  # type problems were reported above
        if start < 0:
            violations.append(f'evidence_start[{index}] is negative ({start!r})')
        if end <= start:
            violations.append(
                f'evidence_end[{index}] ({end!r}) is not after evidence_start[{index}] ({start!r})'
            )
        if duration is not None and end > duration + END_TOLERANCE_S + _EPSILON:
            violations.append(
                f'evidence_end[{index}] ({end!r}) is beyond the audio length '
                f'{duration!r} + {END_TOLERANCE_S}'
            )

    # Parity: whatever the reference validator refuses, we refuse too. It
    # raises ValueError by design, but a DTO built with ``model_construct`` can
    # be missing a field, which surfaces as AttributeError/TypeError there;
    # callers are promised a single exception type, so those count as
    # violations as well.
    try:
        _reference_validate(response, n)
    except (ValueError, TypeError, AttributeError, OverflowError) as exc:
        violations.append(f'utils.validate_response: {type(exc).__name__}: {exc}')

    if violations:
        raise ResponseCheckError(violations)


def emergency_response(n_questions: int, answer: bool = True) -> ASRQuestionResponseDto:
    """A constant, always-valid response for when nothing else is available.

    ``answer`` for every question and null evidence everywhere; worth roughly
    half the accuracy marks, versus nothing for an exception.
    """
    n = _check_n_questions(n_questions)
    value = _coerce_fallback(answer)
    return ASRQuestionResponseDto(
        answers=[value] * n, evidence_start=[None] * n, evidence_end=[None] * n,
    )


def to_wire_json(response: ASRQuestionResponseDto) -> str:
    """The exact JSON text the API will send (pydantic's serialiser)."""
    if not isinstance(response, ASRQuestionResponseDto):
        raise ResponseCheckError(
            f'response must be an ASRQuestionResponseDto, got {type(response).__name__}'
        )
    return response.model_dump_json()


def _reject_constant(name: str) -> Any:
    """``json.loads`` hook: Python would happily parse ``NaN``/``Infinity``,
    which are not JSON and which the service cannot read."""
    raise ResponseCheckError(f'non-finite JSON constant {name} is not valid JSON')


def _strict_object(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """``json.loads`` hook: a duplicate key means two values compete for one
    slot and the service picks whichever its parser keeps; refuse instead."""
    payload: Dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ResponseCheckError(f'duplicate key {_short_repr(key)} in JSON object')
        payload[key] = value
    return payload


def _describe(value: Any) -> str:
    """JSON-flavoured type name plus a bounded repr, for error messages."""
    if value is None:
        return 'null'
    if isinstance(value, bool):
        return f'boolean {value!r}'
    if isinstance(value, (int, float)):
        return f'number {value!r}'
    if isinstance(value, str):
        return f'string {_short_repr(value)}'
    if isinstance(value, list):
        return 'array'
    if isinstance(value, dict):
        return 'object'
    return type(value).__name__


def check_wire_json(text: Union[str, bytes], n_questions: int) -> None:
    """Raise ``ResponseCheckError`` unless ``text`` is exactly what the service
    expects: a JSON object with only the three keys (each once), ``n_questions``
    entries each, real JSON booleans for answers and real JSON numbers or nulls
    for timestamps, every interval either fully null or ``0 <= start < end``,
    and null intervals on every ``false``. ``NaN``/``Infinity`` tokens are
    refused at parse time. This is the same contract ``strict_validate``
    enforces on the DTO, re-checked on the bytes, because the serialiser is one
    more place a value can change (pydantic writes NaN as ``null``, which turns
    a bad span into a half-filled one).
    """
    n = _check_n_questions(n_questions)
    if not isinstance(text, (str, bytes, bytearray)):
        raise ResponseCheckError(f'wire payload must be str or bytes, got {type(text).__name__}')
    try:
        payload = json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_strict_object)
    except ResponseCheckError:
        raise
    except json.JSONDecodeError as exc:
        raise ResponseCheckError(
            f'wire payload is not valid JSON: {exc.msg} at position {exc.pos}'
        ) from exc
    except (ValueError, RecursionError) as exc:
        # UnicodeDecodeError (bytes that are not UTF-8) is a ValueError; absurd
        # nesting exhausts the parser's recursion. Neither is a body we sent.
        raise ResponseCheckError(
            f'wire payload is not valid JSON: {type(exc).__name__}: {_short_repr(str(exc))}'
        ) from exc
    if not isinstance(payload, dict):
        raise ResponseCheckError(f'wire payload must be a JSON object, got {_describe(payload)}')

    violations: List[str] = []
    keys = set(payload)
    missing = sorted(WIRE_KEYS - keys)
    extra = sorted(str(k) for k in keys - WIRE_KEYS)
    if missing:
        violations.append(f'missing keys: {missing}')
    if extra:
        violations.append(f'unexpected keys: {_short_repr(extra)}')

    for name in sorted(WIRE_KEYS & keys):
        values = payload[name]
        if not isinstance(values, list):
            violations.append(f'{name} must be a JSON array, got {_describe(values)}')
            continue
        if len(values) != n:
            violations.append(f'{name} must have {n} entries, got {len(values)}')
        for index, value in enumerate(values):
            if name == 'answers':
                if not isinstance(value, bool):
                    violations.append(
                        f'answers[{index}] must be a JSON boolean, got {_describe(value)}'
                    )
            elif value is not None and not _is_finite_number(value):
                violations.append(
                    f'{name}[{index}] must be a finite JSON number or null, got {_describe(value)}'
                )

    columns = [payload.get(name) for name in ('answers', 'evidence_start', 'evidence_end')]
    if all(isinstance(column, list) for column in columns):
        for index, (answer, start, end) in enumerate(zip(*columns)):
            if (start is None) != (end is None):
                violations.append(
                    f'evidence[{index}] is half-filled on the wire: start={start!r}, end={end!r}'
                )
                continue
            if start is None:
                continue
            if answer is False:
                violations.append(
                    f'evidence[{index}] must be null for a false answer, got ({start!r}, {end!r})'
                )
            if not (_is_finite_number(start) and _is_finite_number(end)):
                continue  # type problems were reported above
            if start < 0:
                violations.append(f'evidence_start[{index}] is negative ({start!r})')
            if end <= start:
                violations.append(
                    f'evidence_end[{index}] ({end!r}) is not after evidence_start[{index}] ({start!r})'
                )

    if violations:
        raise ResponseCheckError(violations)


def summarize(response: ASRQuestionResponseDto) -> Dict[str, Any]:
    """Counts for a one-line log: how many yes answers, how many carry a span.

    Tolerant of a malformed response (only well-formed spans are measured), so
    it can be logged right before ``strict_validate`` rejects something.
    """
    def column(name: str) -> list:
        values = getattr(response, name, None)
        return list(values) if isinstance(values, (list, tuple)) else []

    answers, starts, ends = column('answers'), column('evidence_start'), column('evidence_end')

    def has_span(start: Any, end: Any) -> bool:
        return _is_finite_number(start) and _is_finite_number(end)

    spans = [(s, e) for s, e in zip(starts, ends) if has_span(s, e)]
    return {
        'n_questions': len(answers),
        'n_true': sum(1 for a in answers if a is True),
        'n_false': sum(1 for a in answers if a is False),
        'n_with_span': len(spans),
        'n_true_without_span': sum(
            1 for a, s, e in zip(answers, starts, ends) if a is True and not has_span(s, e)
        ),
        'span_total_s': round(sum(e - s for s, e in spans), TIMESTAMP_DECIMALS),
    }
