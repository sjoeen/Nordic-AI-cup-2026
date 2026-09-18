"""Timed speech recognition: faster-whisper behind the ``ASRBackend`` protocol,
a scripted fake for tests, and a JSON cache for offline evaluation.

Why the transcription loop is lazy and deadline-aware: faster-whisper decodes
one 30-second window per pull on its segment generator, so stopping the
iteration early genuinely stops the compute. A request that is running out of
its 60-second budget therefore gets the segments decoded so far (a partial
transcript still answers some questions) instead of an exception or a
timeout that loses every question about the conversation.

Trust model: transcript text is untrusted data. This module only stores and
converts it; nothing here interprets it, and cache paths are built from the
config hash and a caller-supplied key that must pass a strict character
whitelist, never from text.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
)

import numpy as np

from core_types import (
    ASRBackend,
    AudioContext,
    Deadline,
    TranscriptUnit,
    Word,
    is_finite_number,
    stable_hash,
)

__all__ = [
    'ASRConfig',
    'CachingASRBackend',
    'FakeASRBackend',
    'FakeClock',
    'FasterWhisperBackend',
    'REQUIRED_SAMPLE_RATE',
    'build_asr_backend',
    'resolve_compute_type',
    'resolve_device',
    'segments_to_units',
    'units_from_dicts',
    'units_to_dicts',
]

logger = logging.getLogger(__name__)

# Whisper models are trained on 16 kHz audio and faster-whisper's ndarray input
# path does no resampling, so anything else would silently mis-time every word.
REQUIRED_SAMPLE_RATE = 16000

# Bumped whenever the on-disk cache layout or the segment conversion changes
# meaning, so stale files are ignored instead of misread.
CACHE_FORMAT_VERSION = 1

# One path component: no separators, no leading dot, bounded length. Applied to
# the config hash and the cache key before they touch the filesystem.
_SAFE_PATH_COMPONENT = re.compile(r'[A-Za-z0-9_-][A-Za-z0-9_.-]{0,127}')

_T = TypeVar('_T')


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def resolve_device(device: str, cuda_device_count: Optional[Callable[[], int]] = None) -> str:
    """Turn ``'auto'`` into ``'cuda'`` or ``'cpu'``; other values pass through.

    ``cuda_device_count`` is injectable so the choice is testable without a
    GPU. Any failure while probing means "no usable GPU", never an exception,
    because a serving process must come up on whatever hardware it has.
    """
    if device != 'auto':
        return device
    if cuda_device_count is None:
        try:
            import ctranslate2
        except ImportError:
            return 'cpu'
        cuda_device_count = ctranslate2.get_cuda_device_count
    try:
        return 'cuda' if cuda_device_count() > 0 else 'cpu'
    except Exception:  # noqa: BLE001 - a broken CUDA runtime must degrade to CPU
        return 'cpu'


def resolve_compute_type(compute_type: str, device: str) -> str:
    """Turn ``'auto'`` into the fastest sane type for the device.

    int8 on CPU is the measured sweet spot on this project's dev machine
    (``eval_tools/bench_asr.py``); float16 is the native type on CUDA.
    """
    if compute_type != 'auto':
        return compute_type
    return 'float16' if device == 'cuda' else 'int8'


def _parse_bool(raw: str) -> bool:
    """Strict: an empty value is an error, because ``MA_ASR_WORD_TIMESTAMPS=``
    silently meaning ``False`` would switch word timings off for every
    request without anyone asking for it."""
    value = raw.strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError('expected a boolean (1/0, true/false, yes/no, on/off)')


def _parse_int(raw: str) -> int:
    return int(raw.strip())


def _parse_float(raw: str) -> float:
    value = float(raw.strip())
    if not math.isfinite(value):
        raise ValueError('expected a finite number')
    return value


def _parse_str(raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError('must not be empty')
    return value


def _parse_optional_str(raw: str) -> Optional[str]:
    """An empty variable means "unset", so an operator can clear an optional
    value from a systemd unit or a shell without editing code."""
    return raw.strip() or None


# Explicit per-field parsers: a field that is missing here is not readable
# from the environment, and the test-suite checks the map stays complete.
_ENV_PARSERS: Dict[str, Callable[[str], Any]] = {
    'model_size': _parse_str,
    'device': _parse_str,
    'compute_type': _parse_str,
    'beam_size': _parse_int,
    'language': _parse_optional_str,
    'word_timestamps': _parse_bool,
    'vad_filter': _parse_bool,
    'condition_on_previous_text': _parse_bool,
    'cpu_threads': _parse_int,
    'num_workers': _parse_int,
    'temperature': _parse_float,
    'initial_prompt': _parse_optional_str,
    'download_root': _parse_optional_str,
    'local_files_only': _parse_bool,
    'stop_reserve_s': _parse_float,
}

# Fields that change what the model outputs. Operational fields (threads,
# workers, download location, deadline reserve) are deliberately left out of
# the hash so changing them never invalidates cached transcripts.
_OUTPUT_FIELDS: Tuple[str, ...] = (
    'model_size',
    'beam_size',
    'language',
    'word_timestamps',
    'vad_filter',
    'condition_on_previous_text',
    'temperature',
    'initial_prompt',
)


@dataclass(frozen=True)
class ASRConfig:
    """Everything that decides which model runs and how it decodes.

    Defaults are the measured serving choice for a 4-core CPU: ``small`` at
    int8 keeps the real-time factor near 0.1-0.16, greedy decoding
    (``beam_size=1``, ``temperature=0.0``) is deterministic and avoids
    temperature fallbacks, and ``condition_on_previous_text=False`` stops one
    hallucinated window from poisoning the next.
    """

    model_size: str = 'small'
    device: str = 'auto'
    compute_type: str = 'auto'
    beam_size: int = 1
    language: Optional[str] = 'en'
    word_timestamps: bool = True
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    cpu_threads: int = 4
    num_workers: int = 1
    temperature: float = 0.0
    initial_prompt: Optional[str] = None
    download_root: Optional[str] = None
    local_files_only: bool = False
    stop_reserve_s: float = 8.0

    def __post_init__(self) -> None:
        """Type and range checks, so a JSON config with ``"cpu_threads": 2.5``
        or ``"beam_size": true`` fails at start-up instead of inside the first
        model load, where the failure is contained and every request would
        then silently run without a transcript."""
        for name in ('model_size', 'device', 'compute_type'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a non-empty string, got {value!r}')
        for name in ('language', 'initial_prompt', 'download_root'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{name} must be a non-empty string or None, got {value!r}')
        for name in ('word_timestamps', 'vad_filter', 'condition_on_previous_text', 'local_files_only'):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise ValueError(f'{name} must be a bool, got {value!r}')
        for name, minimum in (('beam_size', 1), ('cpu_threads', 0), ('num_workers', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}, got {value!r}')
        for name in ('temperature', 'stop_reserve_s'):
            value = getattr(self, name)
            if not is_finite_number(value) or value < 0.0:
                raise ValueError(f'{name} must be a finite number >= 0, got {value!r}')

    @property
    def model_id(self) -> str:
        return f'faster-whisper:{self.model_size}'

    def resolved_runtime(self) -> Tuple[str, str]:
        """``(device, compute_type)`` with ``'auto'`` resolved for this host."""
        device = resolve_device(self.device)
        return device, resolve_compute_type(self.compute_type, device)

    @property
    def config_hash(self) -> str:
        """Identity of the transcripts this config produces.

        The resolved device and compute type are part of it because int8 on
        CPU and float16 on CUDA do not decode identically, and a cache
        written on one host must not be read as if it came from the other.
        """
        device, compute_type = self.resolved_runtime()
        payload = {name: getattr(self, name) for name in _OUTPUT_FIELDS}
        payload.update(device=device, compute_type=compute_type, format=CACHE_FORMAT_VERSION)
        return stable_hash(payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_env(
        cls,
        prefix: str = 'MA_ASR_',
        environ: Optional[Mapping[str, str]] = None,
        base: Optional['ASRConfig'] = None,
    ) -> 'ASRConfig':
        """Overlay ``<prefix><FIELD_NAME>`` variables on ``base`` (or the defaults).

        Parsing errors raise ``ValueError`` naming the variable: a
        misconfigured server should fail at start-up, not silently serve with
        a default it was told not to use.
        """
        environ = os.environ if environ is None else environ
        base = cls() if base is None else base
        overrides: Dict[str, Any] = {}
        seen: List[str] = []
        for name, parser in _ENV_PARSERS.items():
            key = f'{prefix}{name.upper()}'
            if key not in environ:
                continue
            raw = environ[key]
            seen.append(key)
            try:
                overrides[name] = parser(raw)
            except ValueError as exc:
                raise ValueError(f'{key}={raw!r}: {exc}') from None
        try:
            return replace(base, **overrides)
        except ValueError as exc:
            # Semantic checks live in __post_init__; point the operator at the
            # variables that were read so the message is actionable.
            raise ValueError(f'invalid ASR configuration from {", ".join(seen)}: {exc}') from None


# --------------------------------------------------------------------------- #
# Segment conversion (pure)
# --------------------------------------------------------------------------- #

def _finite(value: Any) -> Optional[float]:
    """``float(value)`` when it is a finite real number, otherwise ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def segments_to_units(
    segments: Iterable[Any],
    first_unit_id: int = 0,
    first_word_id: int = 0,
) -> List[TranscriptUnit]:
    """Convert faster-whisper segments into ``TranscriptUnit`` objects.

    Duck-typed on purpose: a segment needs ``.start .end .text`` and may carry
    ``.words`` (``None`` when word timestamps were off), ``.avg_logprob`` and
    ``.no_speech_prob``; a word needs ``.start .end .word`` and may carry
    ``.probability``. Word text is kept raw (whisper's leading space included)
    so later stages can rebuild the exact segment text.

    Ids are consecutive from the given starts, in iteration order. Sanity
    rules, applied so nothing downstream has to defend against them:

    - a word with a non-finite timestamp is dropped from ``words`` (its text
      survives in the unit text);
    - a segment with non-finite bounds takes its bounds from its words, or is
      dropped when it has none;
    - an end before its start is clamped to the start;
    - a segment with blank text and no words is dropped.
    """
    units: List[TranscriptUnit] = []
    unit_id = first_unit_id
    word_id = first_word_id
    for segment in segments:
        text = str(getattr(segment, 'text', '') or '')
        words: List[Word] = []
        for raw_word in getattr(segment, 'words', None) or []:
            start = _finite(getattr(raw_word, 'start', None))
            end = _finite(getattr(raw_word, 'end', None))
            if start is None or end is None:
                logger.debug('dropping word without finite timing: %r', getattr(raw_word, 'word', ''))
                continue
            words.append(Word(
                word_id=word_id,
                start_s=start,
                end_s=max(start, end),
                text=str(getattr(raw_word, 'word', '') or ''),
                probability=_finite(getattr(raw_word, 'probability', None)),
            ))
            word_id += 1

        start = _finite(getattr(segment, 'start', None))
        end = _finite(getattr(segment, 'end', None))
        if start is None or end is None:
            if not words:
                logger.debug('dropping segment without finite bounds: %r', text)
                continue
            start = min(word.start_s for word in words)
            end = max(word.end_s for word in words)
        if not text.strip() and not words:
            continue

        units.append(TranscriptUnit(
            unit_id=unit_id,
            start_s=start,
            end_s=max(start, end),
            text=text,
            words=words,
            avg_logprob=_finite(getattr(segment, 'avg_logprob', None)),
            no_speech_prob=_finite(getattr(segment, 'no_speech_prob', None)),
        ))
        unit_id += 1
    return units


def units_to_dicts(units: Iterable[TranscriptUnit]) -> List[Dict[str, Any]]:
    return [asdict(unit) for unit in units]


def units_from_dicts(raw_units: Any) -> List[TranscriptUnit]:
    """Inverse of ``units_to_dicts``; raises on anything malformed.

    Reuses ``AudioContext.from_dict`` so there is exactly one parser for
    serialised units in the codebase.
    """
    if not isinstance(raw_units, list):
        raise TypeError(f'units must be a list, got {type(raw_units).__name__}')
    for raw_unit in raw_units:
        if not isinstance(raw_unit, dict):
            raise TypeError(f'unit must be a mapping, got {type(raw_unit).__name__}')
        raw_words = raw_unit.get('words', [])
        if not isinstance(raw_words, list):
            raise TypeError(f'unit words must be a list, got {type(raw_words).__name__}')
        for item in (raw_unit, *raw_words):
            # ``from_dict`` would stringify a missing/None text into 'None'.
            if not isinstance(item, dict) or not isinstance(item.get('text'), str):
                raise TypeError('unit and word text must be strings')
    context = AudioContext.from_dict({'audio_sha256': '', 'duration_s': 0.0, 'units': raw_units})
    for unit in context.units:
        _check_unit_numbers(unit)
    return context.units


def _check_unit_numbers(unit: TranscriptUnit) -> None:
    """Reject non-finite or mistyped numbers that ``AudioContext.from_dict``
    lets through (``1e400`` parses to ``inf`` without touching
    ``parse_constant``; ``avg_logprob`` is not converted at all)."""
    label = f'unit {unit.unit_id}'
    if not (is_finite_number(unit.start_s) and is_finite_number(unit.end_s)):
        raise ValueError(f'{label}: non-finite bounds')
    for name in ('avg_logprob', 'no_speech_prob'):
        value = getattr(unit, name)
        if value is not None and not is_finite_number(value):
            raise ValueError(f'{label}: {name} must be a finite number or null, got {value!r}')
    if unit.speaker is not None and not isinstance(unit.speaker, str):
        raise ValueError(f'{label}: speaker must be a string or null, got {unit.speaker!r}')
    for word in unit.words:
        if not (is_finite_number(word.start_s) and is_finite_number(word.end_s)):
            raise ValueError(f'word {word.word_id}: non-finite timing')
        if word.probability is not None and not is_finite_number(word.probability):
            raise ValueError(f'word {word.word_id}: probability must be a finite number or null')


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _validate_waveform(waveform: Any, sample_rate: int) -> np.ndarray:
    """Return a contiguous float32 mono array or raise ``ValueError``.

    NaN/inf samples are zeroed rather than rejected: they come from a decoder
    hiccup, and a few silent samples cost far less than a lost conversation.
    """
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise ValueError(
            f'ASR expects {REQUIRED_SAMPLE_RATE} Hz mono audio, got sample_rate={sample_rate!r}'
        )
    audio = np.asarray(waveform, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError(f'waveform must be 1-D mono, got shape {audio.shape}')
    if audio.size and not np.all(np.isfinite(audio)):
        logger.warning('waveform contains non-finite samples; zeroing them')
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(audio)


def _drain(
    iterator: Iterator[_T],
    deadline: Optional[Deadline],
    stop_reserve_s: float,
) -> Tuple[List[_T], bool, Optional[str]]:
    """Pull from a lazy iterator until it ends, the deadline gets close, or it fails.

    The deadline is checked *before* each pull because the pull is where the
    compute happens; an expired deadline always stops the loop, even with a
    zero reserve, so no pull ever starts past the budget. ``truncated`` means
    "stopped consuming with the iterator possibly not exhausted"; ``error`` is
    the failure that ended the loop, if any, with whatever was gathered before
    it still returned. The iterator is closed on the way out so a suspended
    generator (and the encoder output it holds) is released now rather than
    whenever the garbage collector gets to it.
    """
    items: List[_T] = []
    truncated = False
    error: Optional[str] = None
    try:
        while True:
            if deadline is not None and (deadline.expired() or deadline.remaining() < stop_reserve_s):
                truncated = True
                break
            try:
                item = next(iterator)
            except StopIteration:
                break
            items.append(item)
    except Exception as exc:  # noqa: BLE001 - partial output beats no output
        error = f'{type(exc).__name__}: {exc}'
        logger.exception('ASR stopped after %d segments', len(items))
    finally:
        close = getattr(iterator, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001 - clean-up must not mask the result
                logger.exception('closing the ASR segment iterator failed')
    return items, truncated, error


def _base_run_info(model_id: str, n_samples: int) -> Dict[str, Any]:
    return {
        'model_id': model_id,
        'audio_s': n_samples / REQUIRED_SAMPLE_RATE,
        'segments': 0,
        'units': 0,
        'truncated': False,
        'error': None,
        'seconds': 0.0,
    }


class FakeClock:
    """Manual clock for tests: calling it reads the time, ``advance`` moves it.

    Plugs into ``Deadline(clock=...)`` and ``FakeASRBackend(clock=...)`` so
    deadline behaviour is tested exactly, without sleeping.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

ModelFactory = Callable[[ASRConfig, str, str], Any]


def _default_model_factory(config: ASRConfig, device: str, compute_type: str) -> Any:
    # Imported here so the module (and every test using the fake) stays cheap
    # to import and usable where faster-whisper is not installed.
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=config.cpu_threads,
        num_workers=config.num_workers,
        download_root=config.download_root,
        local_files_only=config.local_files_only,
    )


class FasterWhisperBackend:
    """``ASRBackend`` over faster-whisper with lazy loading and deadline stops.

    ``transcribe`` never raises for model or decoding failures: it returns the
    units gathered so far and records the failure in ``last_run_info['error']``.
    It does raise ``ValueError`` for caller mistakes (wrong sample rate, non-mono
    audio), because those are bugs to fix, not conditions to survive.
    ``warm_up`` raises when the model cannot load so start-up fails loudly.

    Not safe for concurrent ``transcribe`` calls on one instance:
    ``last_run_info`` is per instance and ctranslate2 serialises the work anyway.
    """

    def __init__(
        self,
        config: ASRConfig,
        model_factory: Optional[ModelFactory] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.model_id = config.model_id
        self.config_hash = config.config_hash
        self.device, self.compute_type = config.resolved_runtime()
        self._model_factory = model_factory or _default_model_factory
        self._clock = clock
        self._model: Any = None
        self.load_seconds: Optional[float] = None
        self.last_run_info: Dict[str, Any] = {}

    # -- model lifecycle -------------------------------------------------- #

    def _ensure_model(self) -> Any:
        if self._model is None:
            started = self._clock()
            logger.info(
                'loading %s on %s/%s (cpu_threads=%d)',
                self.model_id, self.device, self.compute_type, self.config.cpu_threads,
            )
            self._model = self._model_factory(self.config, self.device, self.compute_type)
            self.load_seconds = self._clock() - started
            logger.info('loaded %s in %.1fs', self.model_id, self.load_seconds)
        return self._model

    def warm_up(self) -> None:
        """Load the model and decode one second of silence.

        The first call pays for weight loading and thread-pool spin-up; doing
        it here keeps that cost out of the first scored request.
        """
        model = self._ensure_model()
        silence = np.zeros(REQUIRED_SAMPLE_RATE, dtype=np.float32)
        segments, _ = model.transcribe(silence, **self._transcribe_kwargs())
        for _ in segments:
            pass

    def _transcribe_kwargs(self) -> Dict[str, Any]:
        config = self.config
        return {
            'language': config.language,
            'beam_size': config.beam_size,
            'word_timestamps': config.word_timestamps,
            'vad_filter': config.vad_filter,
            'condition_on_previous_text': config.condition_on_previous_text,
            'temperature': config.temperature,
            'initial_prompt': config.initial_prompt,
        }

    def _segment_iter(self, waveform: np.ndarray) -> Iterator[Any]:
        """Yield raw segments lazily; every bit of compute happens inside a pull.

        A generator so that model loading and the initial feature extraction
        also run inside the deadline-checked, exception-guarded loop.
        """
        model = self._ensure_model()
        segments, info = model.transcribe(waveform, **self._transcribe_kwargs())
        self.last_run_info['language'] = getattr(info, 'language', None)
        self.last_run_info['language_probability'] = _finite(getattr(info, 'language_probability', None))
        yield from segments

    # -- protocol --------------------------------------------------------- #

    def transcribe(
        self,
        waveform: Any,
        sample_rate: int,
        deadline: Optional[Deadline] = None,
        cache_key: Optional[str] = None,
    ) -> List[TranscriptUnit]:
        audio = _validate_waveform(waveform, sample_rate)
        started = self._clock()
        self.last_run_info = _base_run_info(self.model_id, len(audio))
        # Seeded here so the keys exist even when no segment is ever pulled.
        self.last_run_info.update(
            device=self.device, compute_type=self.compute_type,
            language=None, language_probability=None,
        )
        if audio.size == 0:
            return []

        segments, truncated, error = _drain(
            self._segment_iter(audio), deadline, self.config.stop_reserve_s
        )
        units = segments_to_units(segments)
        self.last_run_info.update(
            segments=len(segments),
            units=len(units),
            truncated=truncated,
            error=error,
            seconds=self._clock() - started,
        )
        if truncated:
            logger.warning(
                'ASR stopped early at %d segments (%.1fs of %.1fs audio) to respect the deadline',
                len(segments), units[-1].end_s if units else 0.0, self.last_run_info['audio_s'],
            )
        return units


class FakeASRBackend:
    """Scripted ``ASRBackend`` for tests and offline replays.

    Returns deep copies of ``units`` so a caller that mutates its result cannot
    corrupt the script. With ``delay_s`` it charges that much per unit: by
    advancing the clock when it is a ``FakeClock`` (instant, exact deadline
    tests), by really sleeping otherwise. ``error`` is raised from
    ``transcribe`` to exercise fallback paths; ``last_run_info`` records it
    first so diagnostics never show a stale earlier run.
    """

    def __init__(
        self,
        units: Iterable[TranscriptUnit],
        delay_s: float = 0.0,
        clock: Optional[Callable[[], float]] = None,
        stop_reserve_s: float = 0.0,
        error: Optional[BaseException] = None,
        model_id: str = 'fake',
        config_hash: str = 'fake',
    ) -> None:
        self.units = list(units)
        self.delay_s = float(delay_s)
        self.clock = clock
        self.stop_reserve_s = float(stop_reserve_s)
        self.error = error
        self.model_id = model_id
        self.config_hash = config_hash
        self.calls: List[Dict[str, Any]] = []
        self.warmed_up = False
        self.last_run_info: Dict[str, Any] = {}

    def warm_up(self) -> None:
        self.warmed_up = True

    def _now(self) -> float:
        return self.clock() if self.clock is not None else time.monotonic()

    def _spend(self) -> None:
        if self.delay_s <= 0.0:
            return
        advance = getattr(self.clock, 'advance', None)
        if advance is None:
            time.sleep(self.delay_s)
        else:
            advance(self.delay_s)

    def _unit_iter(self) -> Iterator[TranscriptUnit]:
        for unit in self.units:
            self._spend()
            yield copy.deepcopy(unit)

    def transcribe(
        self,
        waveform: Any,
        sample_rate: int,
        deadline: Optional[Deadline] = None,
        cache_key: Optional[str] = None,
    ) -> List[TranscriptUnit]:
        if sample_rate != REQUIRED_SAMPLE_RATE:
            raise ValueError(
                f'ASR expects {REQUIRED_SAMPLE_RATE} Hz mono audio, got sample_rate={sample_rate!r}'
            )
        n_samples = len(waveform) if hasattr(waveform, '__len__') else 0
        self.calls.append({'n_samples': n_samples, 'sample_rate': sample_rate, 'cache_key': cache_key})
        if self.error is not None:
            self.last_run_info = _base_run_info(self.model_id, n_samples)
            self.last_run_info['error'] = f'{type(self.error).__name__}: {self.error}'
            raise self.error
        started = self._now()
        units, truncated, error = _drain(self._unit_iter(), deadline, self.stop_reserve_s)
        self.last_run_info = _base_run_info(self.model_id, n_samples)
        self.last_run_info.update(
            segments=len(units),
            units=len(units),
            truncated=truncated,
            error=error,
            seconds=self._now() - started,
        )
        return units


class CachingASRBackend:
    """Wrap a backend with a JSON cache at ``cache_dir/<config_hash>/<cache_key>.json``.

    Made for offline evaluation, where the same 39 recordings are transcribed
    over and over; at serve time every conversation is new and ``cache_key``
    may simply be left ``None`` to pass straight through. Truncated or failed
    runs are never cached, because a cache hit must mean a complete transcript.
    A corrupt file is treated as a miss and overwritten.
    """

    def __init__(self, inner: ASRBackend, cache_dir: os.PathLike | str) -> None:
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.model_id = inner.model_id
        self.config_hash = inner.config_hash
        self.hits = 0
        self.misses = 0
        self.last_run_info: Dict[str, Any] = {}

    def warm_up(self) -> None:
        self.inner.warm_up()

    def cache_path(self, cache_key: str) -> Path:
        """Path for a key; raises ``ValueError`` for anything unsafe as a filename."""
        for name, part in (('config_hash', self.config_hash), ('cache_key', cache_key)):
            if not isinstance(part, str) or not _SAFE_PATH_COMPONENT.fullmatch(part):
                raise ValueError(f'{name} is not a safe path component: {part!r}')
        return self.cache_dir / self.config_hash / f'{cache_key}.json'

    def _inner_info(self) -> Dict[str, Any]:
        return dict(getattr(self.inner, 'last_run_info', None) or {})

    def transcribe(
        self,
        waveform: Any,
        sample_rate: int,
        deadline: Optional[Deadline] = None,
        cache_key: Optional[str] = None,
    ) -> List[TranscriptUnit]:
        if cache_key is None:
            self.last_run_info = {'cache': 'bypass'}
            units = self.inner.transcribe(waveform, sample_rate, deadline, cache_key)
            self.last_run_info = {**self._inner_info(), 'cache': 'bypass'}
            return units

        path = self.cache_path(cache_key)
        cached = _read_cache(path)
        if cached is not None:
            self.hits += 1
            self.last_run_info = {
                **_base_run_info(self.model_id, 0),
                'units': len(cached),
                'segments': len(cached),
                'cache': 'hit',
                'cache_path': str(path),
            }
            return cached

        self.misses += 1
        # Set before delegating: if the inner backend raises, the caller must
        # not read the previous request's numbers as this one's.
        self.last_run_info = {'cache': 'miss', 'cache_path': str(path)}
        units = self.inner.transcribe(waveform, sample_rate, deadline, cache_key)
        info = self._inner_info()
        self.last_run_info = {**info, 'cache': 'miss', 'cache_path': str(path)}
        if info.get('truncated') or info.get('error'):
            logger.info('not caching incomplete ASR run for %s', cache_key)
            return units
        _write_cache(path, {
            'format': CACHE_FORMAT_VERSION,
            'model_id': self.model_id,
            'config_hash': self.config_hash,
            'cache_key': cache_key,
            'units': units_to_dicts(units),
        })
        return units


def _reject_json_constant(name: str) -> None:
    raise ValueError(f'non-finite number {name} in cache file')


def _read_cache(path: Path) -> Optional[List[TranscriptUnit]]:
    """Units from a cache file, or ``None`` when it is missing or unusable."""
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning('could not read ASR cache %s: %s', path, exc)
        return None
    try:
        payload = json.loads(text, parse_constant=_reject_json_constant)
        if payload.get('format') != CACHE_FORMAT_VERSION:
            raise ValueError(f'unsupported cache format {payload.get("format")!r}')
        return units_from_dicts(payload['units'])
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        logger.warning('ignoring corrupt ASR cache file %s: %s', path, exc)
        return None


def _write_cache(path: Path, payload: Dict[str, Any]) -> bool:
    """Atomically write a cache file; a failure is logged, never raised, and
    leaves no half-written temp file behind."""
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        tmp.write_text(text, encoding='utf-8')
        os.replace(tmp, path)
        return True
    except (OSError, ValueError, TypeError) as exc:
        logger.warning('could not write ASR cache %s: %s', path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def build_asr_backend(config: ASRConfig, cache_dir: Optional[os.PathLike | str] = None) -> ASRBackend:
    """The serving backend: faster-whisper, wrapped in a cache when asked."""
    backend: ASRBackend = FasterWhisperBackend(config)
    if cache_dir is not None:
        backend = CachingASRBackend(backend, cache_dir)
    return backend
