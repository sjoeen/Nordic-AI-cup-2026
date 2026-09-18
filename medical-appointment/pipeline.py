"""Request orchestration: model lifecycle, deadline, stage sequencing, fallback.

This is the only module that knows about every stage. Each stage is an
injectable object so the whole pipeline can be exercised with fakes (see
``tests/test_pipeline.py``) and debugged one stage at a time through
``predict_with_trace``.

Fallback ladder for a valid request:

1. QA backend raises or returns the wrong count → ``LexicalBackend``.
2. ASR raises or returns nothing → constant ``fallback_answer`` with null spans.
3. Audio cannot be decoded → same constant response.

Anything that still escapes is caught in ``example.predict``. The response is
always exactly one entry per question.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core_types import (
    ASRBackend,
    AudioContext,
    Deadline,
    Prediction,
    QABackend,
    QAResult,
)
from dtos import ASRQuestionResponseDto

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'
DEFAULT_CONFIG_PATH = CONFIG_DIR / 'default.json'


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _env_override(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``<prefix><FIELD>`` environment variables on top of ``values``.

    Booleans accept 1/0/true/false/yes/no; numbers are parsed by the type of
    the existing value; ``null`` clears a value. Unknown keys are ignored so a
    stale variable cannot break start-up.
    """
    out = dict(values)
    for key in list(out.keys()):
        raw = os.environ.get(f'{prefix}{key.upper()}')
        if raw is None:
            continue
        current = out[key]
        if raw.lower() == 'null':
            out[key] = None
        elif isinstance(current, bool):
            out[key] = raw.strip().lower() in ('1', 'true', 'yes', 'on')
        elif isinstance(current, int) and not isinstance(current, bool):
            out[key] = int(raw)
        elif isinstance(current, float):
            out[key] = float(raw)
        else:
            out[key] = raw
    return out


@dataclass
class PipelineConfig:
    """Top-level configuration; sub-configs are plain dicts handed to the
    stage constructors so this module does not depend on their field lists."""

    request_budget_s: float = 52.0
    finalize_reserve_s: float = 1.5
    fallback_answer: bool = True
    strict_checks: bool = False
    warm_up_on_init: bool = True
    asr_cache_dir: Optional[str] = None
    trace_log_path: Optional[str] = None
    resegment: Dict[str, Any] = field(default_factory=dict)
    asr: Dict[str, Any] = field(default_factory=dict)
    qa: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ''

    @classmethod
    def load(cls, path: Optional[os.PathLike] = None) -> 'PipelineConfig':
        """Read ``configs/<name>.json`` (or ``MA_CONFIG``) and apply env overrides."""
        chosen = path or os.environ.get('MA_CONFIG') or DEFAULT_CONFIG_PATH
        chosen = Path(chosen)
        if not chosen.is_absolute() and not chosen.exists():
            chosen = CONFIG_DIR / chosen
        data = json.loads(Path(chosen).read_text(encoding='utf-8'))
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        known = {f.name for f in fields(cls)}
        top = {k: v for k, v in data.items() if k in known and k not in ('asr', 'qa', 'evidence', 'resegment')}
        top = _env_override('MA_PIPELINE_', top)
        cfg = cls(
            resegment=_env_override('MA_RESEGMENT_', data.get('resegment', {})),
            asr=_env_override('MA_ASR_', data.get('asr', {})),
            qa=_env_override('MA_QA_', data.get('qa', {})),
            evidence=_env_override('MA_EVIDENCE_', data.get('evidence', {})),
            source_path=str(chosen),
            **top,
        )
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


# --------------------------------------------------------------------------- #
# Trace: everything a debugger wants to see for one request
# --------------------------------------------------------------------------- #

@dataclass
class RequestTrace:
    audio_filename: str = ''
    n_questions: int = 0
    audio_sha256: str = ''
    duration_s: Optional[float] = None
    stage_times: Dict[str, float] = field(default_factory=dict)
    context: Optional[AudioContext] = None
    qa_results: List[QAResult] = field(default_factory=list)
    predictions: List[Prediction] = field(default_factory=list)
    fallbacks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    asr_info: Dict[str, Any] = field(default_factory=dict)
    total_s: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            'audio_filename': self.audio_filename,
            'n_questions': self.n_questions,
            'duration_s': self.duration_s,
            'n_units': len(self.context.units) if self.context else 0,
            'stage_times': {k: round(v, 3) for k, v in self.stage_times.items()},
            'total_s': round(self.total_s, 3),
            'n_true': sum(1 for p in self.predictions if p.answer),
            'n_spans': sum(1 for p in self.predictions if p.span is not None),
            'fallbacks': self.fallbacks,
            'errors': self.errors,
            'asr_info': self.asr_info,
        }


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class Pipeline:
    """One resident instance serves every request sequentially."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        asr: Optional[ASRBackend] = None,
        qa: Optional[QABackend] = None,
        fallback_qa: Optional[QABackend] = None,
        evidence_policy: Any = None,
    ) -> None:
        self.config = config or PipelineConfig.load()
        self._validate_resegment_config()
        self.asr = asr if asr is not None else self._build_asr()
        self.qa = qa if qa is not None else self._build_qa()
        self.fallback_qa = fallback_qa if fallback_qa is not None else self._build_fallback_qa()
        self.evidence_policy = evidence_policy if evidence_policy is not None else self._build_evidence_policy()
        self.requests_served = 0
        self.warmed_up = False
        # What warm-up actually managed to do. Every stage imports its heavy
        # dependency lazily, so a process with no ML stack installed still
        # CONSTRUCTS cleanly and then answers every question with the constant
        # fallback. Construction succeeding proves nothing; these two fields are
        # the evidence that the models are really there.
        self.warm_up_failures: Dict[str, str] = {}
        self.warm_up_trace: Dict[str, Any] = {}
        if self.config.warm_up_on_init:
            self.warm_up()

    # -- construction ----------------------------------------------------- #

    def _validate_resegment_config(self) -> None:
        """Fail at boot, not per request, when the resegment settings are
        unusable (e.g. an env override typed as ``null`` for a boolean)."""
        try:
            from transcript import resegment

            resegment([], **(self.config.resegment or {}))
        except Exception as exc:
            raise ValueError(f'invalid resegment config {self.config.resegment!r}: {exc}') from exc

    def _build_asr(self) -> ASRBackend:
        from asr_backend import ASRConfig, build_asr_backend

        return build_asr_backend(ASRConfig(**self.config.asr), cache_dir=self.config.asr_cache_dir)

    def _build_qa(self) -> QABackend:
        from qa_backend import QAConfig, build_qa_backend

        return build_qa_backend(QAConfig(**self.config.qa))

    def _build_fallback_qa(self) -> QABackend:
        from qa_backend import LexicalBackend, QAConfig

        qa_kwargs = dict(self.config.qa)
        qa_kwargs['backend'] = 'lexical'
        return LexicalBackend(QAConfig(**qa_kwargs))

    def _build_evidence_policy(self) -> Any:
        from evidence import EvidencePolicy

        return EvidencePolicy(**self.config.evidence)

    # -- lifecycle -------------------------------------------------------- #

    def warm_up(self) -> None:
        """Load every model and push one synthetic request through the whole
        path so the first real request pays no lazy cost."""
        started = time.monotonic()
        self.warm_up_failures = {}
        self.warm_up_trace = {}
        for name, stage in (('asr', self.asr), ('qa', self.qa), ('fallback_qa', self.fallback_qa)):
            try:
                stage.warm_up()
            except Exception as exc:
                logger.exception('warm-up of %s failed; continuing', name)
                self.warm_up_failures[name] = f'{type(exc).__name__}: {exc}'
        try:
            import base64
            import io
            import wave

            import numpy as np

            samples = (0.1 * np.sin(np.linspace(0, 2 * np.pi * 440 * 2, 32000))).astype(np.float32)
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes((samples * 32767).astype(np.int16).tobytes())
            payload = base64.b64encode(buf.getvalue()).decode('ascii')
            _, warm_trace = self.predict_with_trace(
                payload, ['Is this a warm-up question?'], audio_filename='warmup.wav',
            )
            self.warm_up_trace = warm_trace.summary()
            # A pure tone transcribes to nothing, so 'no_transcript' is the
            # healthy outcome here. An *error*, or the emergency response, means
            # a stage is genuinely missing.
            if warm_trace.errors:
                self.warm_up_failures['synthetic_request'] = '; '.join(warm_trace.errors)
            elif 'emergency_response' in warm_trace.fallbacks:
                self.warm_up_failures['synthetic_request'] = 'fell through to the emergency response'
        except Exception as exc:
            logger.exception('synthetic warm-up request failed; continuing')
            self.warm_up_failures['synthetic_request'] = f'{type(exc).__name__}: {exc}'
        self.warmed_up = True
        if self.warm_up_failures:
            logger.error(
                'DEGRADED: warm-up failed for %s. Every answer will be the constant '
                'fallback with no evidence span. Do not start an attempt against this process.',
                ', '.join(sorted(self.warm_up_failures)),
            )
        logger.info('pipeline warm-up finished in %.1f s', time.monotonic() - started)

    @property
    def healthy(self) -> bool:
        """True only on positive evidence that the real models are serving.

        Fails closed: a pipeline that never warmed up cannot be vouched for, so
        it reports unhealthy rather than unknown. This is read before an
        attempt that cannot be retried.
        """
        return self.warmed_up and not self.warm_up_failures

    # -- serving ---------------------------------------------------------- #

    def predict(self, audio_base64: str, questions: List[str], audio_filename: str = '') -> ASRQuestionResponseDto:
        response, _ = self.predict_with_trace(audio_base64, questions, audio_filename)
        return response

    def predict_with_trace(
        self,
        audio_base64: str,
        questions: List[str],
        audio_filename: str = '',
    ) -> tuple:
        """Run every stage and return ``(response, trace)``.

        ``audio_filename`` is metadata for logs only. It is never used as a
        path, a cache key or a feature.
        """
        from response_checks import emergency_response, finalize, strict_validate

        n = len(questions)
        deadline = Deadline(self.config.request_budget_s, reserve_s=self.config.finalize_reserve_s)
        trace = RequestTrace(audio_filename=str(audio_filename)[:200], n_questions=n)
        started = time.monotonic()

        # 1. Decode -------------------------------------------------------- #
        try:
            from audio_io import decode_request_audio

            decoded = decode_request_audio(audio_base64)
        except Exception as exc:
            logger.exception('audio decode failed for %s', trace.audio_filename)
            trace.errors.append(f'decode: {type(exc).__name__}: {exc}')
            trace.fallbacks.append('emergency_response')
            trace.total_s = time.monotonic() - started
            return emergency_response(n, self.config.fallback_answer), trace
        trace.audio_sha256 = decoded.audio_sha256
        trace.duration_s = decoded.duration_s
        trace.stage_times['decode'] = deadline.checkpoint('decode')

        # 2. ASR ------------------------------------------------------------ #
        raw_units: List[Any] = []
        try:
            raw_units = self.asr.transcribe(
                decoded.waveform, decoded.sample_rate, deadline, cache_key=decoded.audio_sha256,
            )
            trace.asr_info = dict(getattr(self.asr, 'last_run_info', {}) or {})
        except Exception as exc:
            logger.exception('ASR failed for %s', trace.audio_filename)
            trace.errors.append(f'asr: {type(exc).__name__}: {exc}')
        trace.stage_times['asr'] = deadline.checkpoint('asr')

        try:
            from transcript import build_context

            context = build_context(
                raw_units,
                audio_sha256=decoded.audio_sha256,
                duration_s=decoded.duration_s,
                asr_model_id=getattr(self.asr, 'model_id', ''),
                asr_config_hash=getattr(self.asr, 'config_hash', ''),
                resegment_kwargs=self.config.resegment or None,
            )
        except Exception as exc:
            logger.exception('context build failed for %s', trace.audio_filename)
            trace.errors.append(f'context: {type(exc).__name__}: {exc}')
            context = AudioContext(decoded.audio_sha256, decoded.duration_s, units=[])
        trace.context = context
        trace.stage_times['context'] = deadline.checkpoint('context')

        if not context.units:
            trace.fallbacks.append('no_transcript')
            trace.predictions = [Prediction(self.config.fallback_answer, None, source='no_transcript') for _ in questions]
            response = finalize(trace.predictions, n, decoded.duration_s, self.config.fallback_answer)
            trace.total_s = time.monotonic() - started
            self._log_trace(trace)
            return response, trace

        # 3. QA ------------------------------------------------------------- #
        results = self._answer(context, questions, deadline, trace)
        trace.qa_results = results
        trace.stage_times['qa'] = deadline.checkpoint('qa')

        # 4. Evidence ------------------------------------------------------- #
        predictions: List[Prediction] = []
        try:
            from evidence import select_span
        except Exception as exc:  # pragma: no cover - import failure is fatal in dev
            logger.exception('evidence module unavailable')
            trace.errors.append(f'evidence import: {exc}')
            select_span = None  # type: ignore
        for question, result in zip(questions, results):
            span = None
            if result.answer and select_span is not None:
                try:
                    span = select_span(context, result, self.evidence_policy, question=question)
                except Exception as exc:
                    logger.exception('evidence selection failed for question %d', result.question_index)
                    trace.errors.append(f'evidence[{result.question_index}]: {type(exc).__name__}: {exc}')
            predictions.append(Prediction(
                answer=bool(result.answer), span=span, source=result.backend,
                diagnostics={'confidence': result.confidence, 'unit_ids': list(result.evidence_unit_ids)},
            ))
        trace.predictions = predictions
        trace.stage_times['evidence'] = deadline.checkpoint('evidence')

        # 5. Finalise ------------------------------------------------------- #
        response = finalize(predictions, n, decoded.duration_s, self.config.fallback_answer)
        response = self._guard_response(response, n, decoded.duration_s, trace)
        trace.stage_times['finalize'] = deadline.checkpoint('finalize')
        trace.total_s = time.monotonic() - started
        self.requests_served += 1
        self._log_trace(trace)
        return response, trace

    # -- helpers ------------------------------------------------------------ #

    def _guard_response(self, response: ASRQuestionResponseDto, n: int, duration_s: Optional[float], trace: RequestTrace) -> ASRQuestionResponseDto:
        """Last check before the wire: the semantic validator and a parse of
        the serialised bytes. A body that would lose all ten questions is
        replaced by the constant emergency response (worth half a mark each);
        with ``strict_checks`` the violation raises instead, for development.
        """
        from response_checks import check_wire_json, emergency_response, strict_validate, to_wire_json

        try:
            strict_validate(response, n, duration_s)
            check_wire_json(to_wire_json(response), n)
            return response
        except Exception as exc:
            if self.config.strict_checks:
                raise
            logger.error('final response failed validation (%s); sending emergency response', exc)
            trace.errors.append(f'response_check: {type(exc).__name__}: {exc}')
            trace.fallbacks.append('emergency_response')
            return emergency_response(n, self.config.fallback_answer)

    def _answer(self, context: AudioContext, questions: List[str], deadline: Deadline, trace: RequestTrace) -> List[QAResult]:
        n = len(questions)
        for name, backend in (('qa', self.qa), ('fallback_qa', self.fallback_qa)):
            if backend is None:
                continue
            try:
                results = backend.answer(context, questions, deadline)
                if len(results) != n:
                    raise ValueError(f'{name} returned {len(results)} results for {n} questions')
                for i, r in enumerate(results):
                    r.question_index = i
                    if not r.backend:
                        r.backend = getattr(backend, 'name', name)
                return list(results)
            except Exception as exc:
                logger.exception('%s backend failed', name)
                trace.errors.append(f'{name}: {type(exc).__name__}: {exc}')
                trace.fallbacks.append(name)
        return [
            QAResult(question_index=i, answer=self.config.fallback_answer, backend='constant', confidence=0.0)
            for i in range(n)
        ]

    def _log_trace(self, trace: RequestTrace) -> None:
        record = trace.summary()
        logger.info('request %s: %s', trace.audio_filename, json.dumps(record, default=str))
        self._persist_trace(record)

    def _persist_trace(self, record: Dict[str, Any]) -> None:
        """Append one request's trace to ``trace_log_path`` as a JSONL line.

        Diagnostics only, and unable to fail a request: an unwritable path or a
        full disk costs a log line, not ten questions. The record says what the
        request *did* - timings, ASR info, fallbacks, errors, how many answers
        were yes and how many carried a span - and holds neither the audio nor
        the transcript text.
        """
        path = self.config.trace_log_path
        if not path:
            return
        try:
            record = dict(
                record,
                logged_at=datetime.now(timezone.utc).isoformat(),
                request_index=self.requests_served,
                config_source=self.config.source_path,
            )
            target = Path(path)
            if target.parent != Path(''):
                target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, default=str) + '\n')
        except Exception:
            logger.warning('could not persist trace to %s', path, exc_info=True)


# --------------------------------------------------------------------------- #
# Process-wide instance
# --------------------------------------------------------------------------- #

_PIPELINE: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """The resident pipeline, built on first use (``example.py`` calls this at
    import time so the models are warm before the first request)."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


def set_pipeline(pipeline: Optional[Pipeline]) -> None:
    """Replace the resident pipeline (tests inject fakes through this)."""
    global _PIPELINE
    _PIPELINE = pipeline
