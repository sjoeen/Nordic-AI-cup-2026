"""Adversarial fixtures through the pipeline with a fake ASR backend.

Attack text arrives the way ASR would emit it (as unit text) or inside the
question strings. The protected behaviours: schema validity, spans inside the
recording, and answers that are not flipped to yes by instructions. With the
lexical backend only the false-side expectations are asserted; the real NLI
backend is exercised under RUN_SLOW=1.
"""

import json
from pathlib import Path

import pytest

from tests.audio_fixtures import wav_base64
from tests.conftest import make_context

FIXTURE = Path(__file__).resolve().parent.parent / 'security_fixtures' / 'transcript_attacks_v1.json'
CASES = json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']


def _run_case(case, qa):
    from asr_backend import FakeASRBackend
    from pipeline import Pipeline, PipelineConfig
    from qa_backend import LexicalBackend, QAConfig

    context = make_context(case['units'])
    config = PipelineConfig.load('configs/test_fake.json')
    pipe = Pipeline(config, asr=FakeASRBackend(context.units), qa=qa,
                    fallback_qa=LexicalBackend(QAConfig(backend='lexical')))
    response, trace = pipe.predict_with_trace(wav_base64(context.duration_s), case['questions'], 'attack.mp3')
    return context, response, trace


def _assert_expectations(case, context, response, exact_answers):
    from response_checks import check_wire_json, strict_validate

    expected = case['expected']
    n = len(case['questions'])
    if expected.get('schema_valid'):
        strict_validate(response, n, context.duration_s)
        check_wire_json(response.model_dump_json(), n)
        assert set(json.loads(response.model_dump_json())) == {'answers', 'evidence_start', 'evidence_end'}
    if expected.get('not_all_true'):
        assert not all(response.answers), case['id']
    if 'answers_at_most_true' in expected:
        allowed = set(expected['answers_at_most_true'])
        for i, a in enumerate(response.answers):
            if i not in allowed:
                assert a is False, f"{case['id']}: question {i} flipped to true"
    if expected.get('spans_within_duration'):
        for s, e in zip(response.evidence_start, response.evidence_end):
            if s is not None:
                assert 0 <= s < e <= context.duration_s + 0.05
    if 'answers' in expected:
        for i, want in enumerate(expected['answers']):
            if want is False or exact_answers:
                assert response.answers[i] is want, f"{case['id']}: question {i}"


@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_attack_with_lexical_backend(case):
    from qa_backend import LexicalBackend, QAConfig

    context, response, trace = _run_case(case, LexicalBackend(QAConfig(backend='lexical')))
    assert not trace.errors, trace.errors
    _assert_expectations(case, context, response, exact_answers=False)


@pytest.mark.slow
@pytest.mark.parametrize('case', CASES, ids=[c['id'] for c in CASES])
def test_attack_with_retrieval_nli_backend(case):
    from qa_backend import QAConfig, build_qa_backend

    qa = build_qa_backend(QAConfig(backend='retrieval_nli', device='cpu'))
    context, response, trace = _run_case(case, qa)
    assert not trace.errors, trace.errors
    _assert_expectations(case, context, response, exact_answers=True)


def test_oversized_and_malformed_audio_never_raise():
    from asr_backend import FakeASRBackend
    from pipeline import Pipeline, PipelineConfig
    from qa_backend import LexicalBackend, QAConfig
    from response_checks import strict_validate

    context = make_context(['The treatment will last two weeks.'])
    config = PipelineConfig.load('configs/test_fake.json')
    pipe = Pipeline(config, asr=FakeASRBackend(context.units), qa=LexicalBackend(QAConfig(backend='lexical')))
    for payload in ['', 'A' * 10, '%%%%', 'data:audio/mpeg;base64,', 'QUJD' * (50 * 1024 * 1024 // 4)]:
        response = pipe.predict(payload, ['Will the treatment last two weeks?'])
        strict_validate(response, 1)
