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

try:
    from pipeline import get_pipeline

    PIPELINE = get_pipeline()
except Exception:  # pragma: no cover - only hit when models cannot load
    logger.exception('pipeline failed to initialise; serving emergency responses')
    PIPELINE = None


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
