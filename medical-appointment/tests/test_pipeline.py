"""End-to-end tests of ``pipeline.Pipeline`` and ``example.predict`` with fake
stages. No real model is loaded here; the point is the plumbing: ordering,
fallbacks, never raising, schema validity, and that the filename is metadata.
"""

import base64
import json

import pytest

from core_types import AudioContext, Deadline, QAResult
from dtos import ASRQuestionRequestDto
from tests.audio_fixtures import wav_base64
from tests.conftest import make_context

pytestmark = pytest.mark.integration

SENTENCES = [
    "Good morning, what brings you in today?",
    "I've had a sore throat for about a week and the swab came back positive.",
    "Okay, so we'll start you on penicillin, 100 milligrams daily.",
    "Take it after a meal, not on an empty stomach.",
    "The treatment will last two weeks.",
    "Your blood pressure was 130 over 85, which is fine.",
    "No, we won't need to refer you to a specialist.",
    "Come back in two weeks if it hasn't settled.",
]
QUESTIONS = [
    "Should the daily dose be 100 mg?",
    "Was the prescribed dose 200 mg daily?",
    "Will the treatment last two weeks?",
    "Is there any mention of attending a concert?",
]


class ScriptedQA:
    """Returns fixed verdicts so evidence mapping can be asserted exactly."""

    name = 'scripted'

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def warm_up(self):
        pass

    def answer(self, context, questions, deadline=None):
        self.calls += 1
        return [
            QAResult(question_index=i, answer=a, evidence_unit_ids=list(ids), backend=self.name)
            for i, (a, ids) in enumerate(self.script[: len(questions)])
        ]


class ExplodingQA:
    name = 'exploding'

    def warm_up(self):
        pass

    def answer(self, context, questions, deadline=None):
        raise RuntimeError('boom')


class WrongCountQA(ScriptedQA):
    name = 'wrong_count'

    def answer(self, context, questions, deadline=None):
        return super().answer(context, questions, deadline)[:-1]


class ExplodingASR:
    model_id = 'exploding'
    config_hash = 'x'
    last_run_info = {}

    def warm_up(self):
        pass

    def transcribe(self, waveform, sample_rate, deadline=None, cache_key=None):
        raise RuntimeError('asr boom')


@pytest.fixture
def context():
    return make_context(SENTENCES)


@pytest.fixture
def fake_asr(context):
    from asr_backend import FakeASRBackend

    return FakeASRBackend(context.units)


@pytest.fixture
def test_config():
    from pipeline import PipelineConfig

    return PipelineConfig.load('configs/test_fake.json')


@pytest.fixture
def audio_b64(context):
    return wav_base64(context.duration_s)


def _pipeline(test_config, asr, qa, fallback=None):
    from pipeline import Pipeline
    from qa_backend import LexicalBackend, QAConfig

    fb = fallback or LexicalBackend(QAConfig(backend='lexical'))
    return Pipeline(test_config, asr=asr, qa=qa, fallback_qa=fb)


def test_end_to_end_scripted(test_config, fake_asr, context, audio_b64):
    from response_checks import check_wire_json, strict_validate

    qa = ScriptedQA([(True, [2]), (False, []), (True, [4]), (False, [])])
    pipe = _pipeline(test_config, fake_asr, qa)
    response, trace = pipe.predict_with_trace(audio_b64, QUESTIONS, 'conversation_sample_4.mp3')

    strict_validate(response, len(QUESTIONS), context.duration_s)
    check_wire_json(response.model_dump_json(), len(QUESTIONS))
    assert response.answers == [True, False, True, False]
    assert response.evidence_start[1] is None and response.evidence_end[1] is None
    unit2 = context.units[2]
    assert response.evidence_start[0] is not None
    assert unit2.start_s - 0.5 <= response.evidence_start[0] <= unit2.end_s
    assert unit2.start_s <= response.evidence_end[0] <= unit2.end_s + 0.5
    assert trace.fallbacks == []
    assert set(trace.stage_times) >= {'decode', 'asr', 'context', 'qa', 'evidence', 'finalize'}
    assert trace.duration_s == pytest.approx(context.duration_s, abs=0.05)


def test_question_order_is_preserved(test_config, fake_asr, audio_b64):
    qa = ScriptedQA([(True, [2]), (False, []), (True, [4]), (False, [])])
    pipe = _pipeline(test_config, fake_asr, qa)
    response = pipe.predict(audio_b64, QUESTIONS)
    assert response.answers == [True, False, True, False]
    qa2 = ScriptedQA([(False, []), (True, [4]), (True, [2]), (False, [])])
    pipe2 = _pipeline(test_config, fake_asr, qa2)
    response2 = pipe2.predict(audio_b64, [QUESTIONS[1], QUESTIONS[2], QUESTIONS[0], QUESTIONS[3]])
    assert response2.answers == [False, True, True, False]


def test_qa_failure_falls_back_to_lexical(test_config, fake_asr, audio_b64):
    from response_checks import strict_validate

    pipe = _pipeline(test_config, fake_asr, ExplodingQA())
    response, trace = pipe.predict_with_trace(audio_b64, QUESTIONS)
    strict_validate(response, len(QUESTIONS))
    assert 'qa' in trace.fallbacks
    assert all(r.backend for r in trace.qa_results)


def test_wrong_count_from_qa_falls_back(test_config, fake_asr, audio_b64):
    from response_checks import strict_validate

    pipe = _pipeline(test_config, fake_asr, WrongCountQA([(True, [2]), (False, []), (True, [4]), (False, [])]))
    response, trace = pipe.predict_with_trace(audio_b64, QUESTIONS)
    strict_validate(response, len(QUESTIONS))
    assert 'qa' in trace.fallbacks


def test_asr_failure_gives_constant_valid_response(test_config, audio_b64):
    from response_checks import strict_validate

    pipe = _pipeline(test_config, ExplodingASR(), ScriptedQA([]))
    response, trace = pipe.predict_with_trace(audio_b64, QUESTIONS)
    strict_validate(response, len(QUESTIONS))
    assert response.answers == [test_config.fallback_answer] * len(QUESTIONS)
    assert all(s is None for s in response.evidence_start)
    assert 'no_transcript' in trace.fallbacks


def test_undecodable_audio_gives_emergency_response(test_config, fake_asr):
    from response_checks import strict_validate

    pipe = _pipeline(test_config, fake_asr, ScriptedQA([]))
    response, trace = pipe.predict_with_trace('!!!not base64!!!', QUESTIONS)
    strict_validate(response, len(QUESTIONS))
    assert 'emergency_response' in trace.fallbacks


def test_filename_is_metadata_only(test_config, fake_asr, audio_b64):
    qa = ScriptedQA([(True, [2]), (False, []), (True, [4]), (False, [])])
    pipe = _pipeline(test_config, fake_asr, qa)
    response = pipe.predict(audio_b64, QUESTIONS, audio_filename='../../etc/passwd; rm -rf /')
    assert len(response.answers) == len(QUESTIONS)


def test_zero_questions(test_config, fake_asr, audio_b64):
    qa = ScriptedQA([])
    pipe = _pipeline(test_config, fake_asr, qa)
    response = pipe.predict(audio_b64, [])
    assert response.answers == [] and response.evidence_start == [] and response.evidence_end == []


def test_example_predict_never_raises(test_config, fake_asr, audio_b64, monkeypatch):
    import pipeline as pipeline_module

    qa = ScriptedQA([(True, [2]), (False, []), (True, [4]), (False, [])])
    pipe = _pipeline(test_config, fake_asr, qa)
    pipeline_module.set_pipeline(pipe)
    import example

    monkeypatch.setattr(example, 'PIPELINE', pipe)
    request = ASRQuestionRequestDto(audio_base64=audio_b64, audio_filename='x.mp3', questions=QUESTIONS)
    assert example.predict(request).answers == [True, False, True, False]

    def boom(*args, **kwargs):
        raise RuntimeError('total failure')

    monkeypatch.setattr(pipe, 'predict', boom)
    response = example.predict(request)
    assert len(response.answers) == len(QUESTIONS)
    assert all(s is None for s in response.evidence_start)
    pipeline_module.set_pipeline(None)


def test_http_round_trip_through_api(test_config, fake_asr, audio_b64, monkeypatch):
    """The real FastAPI app with the fake pipeline injected: bytes on the wire
    are real JSON booleans, numbers and nulls."""
    from fastapi.testclient import TestClient

    import pipeline as pipeline_module

    qa = ScriptedQA([(True, [2]), (False, []), (True, [4]), (False, [])])
    pipe = _pipeline(test_config, fake_asr, qa)
    pipeline_module.set_pipeline(pipe)
    import api
    import example

    monkeypatch.setattr(example, 'PIPELINE', pipe)
    client = TestClient(api.app)
    body = {'audio_base64': audio_b64, 'audio_filename': 'conversation_sample_4.mp3', 'questions': QUESTIONS}
    resp = client.post('/predict', json=body)
    assert resp.status_code == 200, resp.text
    from response_checks import check_wire_json

    check_wire_json(resp.text, len(QUESTIONS))
    data = json.loads(resp.text)
    assert set(data) == {'answers', 'evidence_start', 'evidence_end'}
    assert data['answers'] == [True, False, True, False]
    assert data['evidence_start'][1] is None
    pipeline_module.set_pipeline(None)


def test_config_env_override(monkeypatch):
    from pipeline import PipelineConfig

    monkeypatch.setenv('MA_ASR_MODEL_SIZE', 'base')
    monkeypatch.setenv('MA_PIPELINE_REQUEST_BUDGET_S', '33.5')
    monkeypatch.setenv('MA_QA_BACKEND', 'lexical')
    cfg = PipelineConfig.load('configs/default.json')
    assert cfg.asr['model_size'] == 'base'
    assert cfg.request_budget_s == 33.5
    assert cfg.qa['backend'] == 'lexical'


def test_invalid_resegment_config_fails_at_boot(test_config, fake_asr):
    from pipeline import Pipeline

    test_config.resegment = {'max_unit_s': 8.0, 'pause_gap_s': 0.7, 'split_on_punct': None}
    with pytest.raises(ValueError):
        Pipeline(test_config, asr=fake_asr, qa=ScriptedQA([]), fallback_qa=ScriptedQA([]))
    test_config.resegment = {'max_unit_s': 8.0, 'no_such_option': 1}
    with pytest.raises(ValueError):
        Pipeline(test_config, asr=fake_asr, qa=ScriptedQA([]), fallback_qa=ScriptedQA([]))


def test_invalid_final_body_is_replaced_by_emergency_response(test_config, fake_asr, audio_b64, monkeypatch):
    """If finalize ever produced a body the service could not score, the
    guard must send a valid constant response rather than that body."""
    import pipeline as pipeline_module
    from dtos import ASRQuestionResponseDto
    from response_checks import strict_validate

    qa = ScriptedQA([(True, [2]), (False, []), (True, [4]), (False, [])])
    test_config.strict_checks = False
    pipe = _pipeline(test_config, fake_asr, qa)

    def broken_finalize(predictions, n, duration_s, fallback_answer=True):
        return ASRQuestionResponseDto(answers=[True] * n, evidence_start=[5.0] * n, evidence_end=[1.0] * n)

    import response_checks

    monkeypatch.setattr(response_checks, 'finalize', broken_finalize)
    response, trace = pipe.predict_with_trace(audio_b64, QUESTIONS)
    strict_validate(response, len(QUESTIONS))
    assert 'emergency_response' in trace.fallbacks
    assert all(s is None for s in response.evidence_start)

    test_config.strict_checks = True
    pipe_strict = _pipeline(test_config, fake_asr, qa)
    with pytest.raises(Exception):
        pipe_strict.predict_with_trace(audio_b64, QUESTIONS)
