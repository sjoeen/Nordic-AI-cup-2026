"""Shared fixtures. Every module's tests must run without real models or
network; real-model tests carry ``@pytest.mark.slow`` and only run when
``RUN_SLOW=1`` is set."""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core_types import AudioContext, TranscriptUnit, Word  # noqa: E402


def pytest_collection_modifyitems(config, items):
    if os.environ.get('RUN_SLOW') == '1':
        return
    skip = pytest.mark.skip(reason='set RUN_SLOW=1 to run real-model tests')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)


def make_context(
    sentences,
    words_per_second: float = 2.5,
    gap_s: float = 0.4,
    start_s: float = 0.0,
    audio_sha256: str = 'test',
    duration_s=None,
):
    """Build an AudioContext from plain sentences with synthetic timings.

    Each sentence becomes one unit; each whitespace token becomes a word with
    consecutive global ids. Deterministic, so tests can assert exact spans.
    """
    units = []
    word_id = 0
    cursor = start_s
    for unit_id, sentence in enumerate(sentences):
        tokens = sentence.split()
        words = []
        unit_start = cursor
        for token in tokens:
            w_end = cursor + 1.0 / words_per_second
            words.append(Word(word_id=word_id, start_s=round(cursor, 3),
                              end_s=round(w_end, 3), text=token, probability=0.9))
            word_id += 1
            cursor = w_end
        units.append(TranscriptUnit(
            unit_id=unit_id, start_s=round(unit_start, 3), end_s=round(cursor, 3),
            text=sentence, words=words,
        ))
        cursor += gap_s
    return AudioContext(
        audio_sha256=audio_sha256,
        duration_s=float(duration_s if duration_s is not None else cursor + 1.0),
        units=units,
        asr_model_id='fake',
        asr_config_hash='fake',
    )


@pytest.fixture
def context_factory():
    return make_context


@pytest.fixture
def clinic_context():
    """A small consultation with a dose, a duration and a negation in it."""
    return make_context([
        "Good morning, what brings you in today?",
        "I've had a sore throat for about a week and the swab came back positive.",
        "Okay, so we'll start you on penicillin, 100 milligrams daily.",
        "Take it after a meal, not on an empty stomach.",
        "The treatment will last two weeks.",
        "Your blood pressure was 130 over 85, which is fine.",
        "No, we won't need to refer you to a specialist.",
        "Come back in two weeks if it hasn't settled.",
    ])
