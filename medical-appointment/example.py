"""The model entry point the API calls. Thin adapter around ``pipeline``.

The real work lives in the stage modules (``audio_io``, ``asr_backend``,
``transcript``, ``qa_backend``, ``evidence``, ``response_checks``) and is
orchestrated by ``pipeline.Pipeline``. This file only guarantees two things:

* the models are loaded and warm at import time, before the first request;
* ``predict`` never raises. A response with a guess for every question is
  worth half a mark on average; an exception is worth nothing and costs all
  ten questions of the conversation.
"""

import logging

from dtos import ASRQuestionRequestDto, ASRQuestionResponseDto
from response_checks import emergency_response

logger = logging.getLogger(__name__)

# Answer used when everything else has failed. Both label sets are balanced,
# so the choice is arithmetically neutral; ``True`` keeps the evidence path
# alive in partial failures (a ``False`` can never carry a span).
FALLBACK_ANSWER = True

# Why the pipeline is not there, when it is not. A degraded process still
# answers every request with a valid body, so nothing downstream notices: the
# service scores it as ten confident wrong answers rather than an outage. This
# string is what ``/health`` reports, and it is the difference between finding
# that out in a second and finding it out from the score.
INIT_ERROR = None

try:
    from pipeline import get_pipeline

    PIPELINE = get_pipeline()
except Exception as exc:  # pragma: no cover - only hit when models cannot load
    logger.exception('pipeline failed to initialise; serving emergency responses')
    INIT_ERROR = f'{type(exc).__name__}: {exc}'
    PIPELINE = None


def pipeline_status() -> dict:
    """What ``/health`` reports: is the real pipeline serving, or the fallback?

    ``degraded`` is the field to alert on. When it is true every answer is the
    constant ``FALLBACK_ANSWER`` with no evidence span, which scores about half
    the accuracy marks and none of the evidence ones.
    """
    pipeline = PIPELINE
    if pipeline is None:
        return {'pipeline_live': False, 'degraded': True, 'init_error': INIT_ERROR}

    # Constructing the pipeline proves nothing: every stage imports its heavy
    # dependency lazily, so a host with no ML stack builds a complete object
    # that answers every question with the constant fallback. Health is whether
    # warm-up actually loaded the models.
    failures = dict(getattr(pipeline, 'warm_up_failures', {}) or {})
    return {
        'pipeline_live': True,
        'degraded': not getattr(pipeline, 'healthy', False),
        'init_error': INIT_ERROR,
        'warm_up_failures': failures,
        'warmed_up': getattr(pipeline, 'warmed_up', False),
        'config_source': getattr(pipeline.config, 'source_path', ''),
        'requests_served': getattr(pipeline, 'requests_served', 0),
        'asr_model_size': (pipeline.config.asr or {}).get('model_size'),
        'qa_backend': (pipeline.config.qa or {}).get('backend'),
        'trace_log_path': getattr(pipeline.config, 'trace_log_path', None),
        'warm_up_asr_info': (getattr(pipeline, 'warm_up_trace', {}) or {}).get('asr_info', {}),
    }


### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###

def predict(request: ASRQuestionRequestDto) -> ASRQuestionResponseDto:
    """Answer every question about one conversation. Never raises."""
    n_questions = len(request.questions)
    try:
        pipeline = PIPELINE if PIPELINE is not None else get_pipeline()
        return pipeline.predict(
            request.audio_base64, list(request.questions), request.audio_filename,
        )
    except Exception:
        logger.exception(
            'predict failed for %s; returning emergency response', request.audio_filename,
        )
        return emergency_response(n_questions, FALLBACK_ANSWER)
