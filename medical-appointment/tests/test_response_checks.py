"""Tests for ``response_checks`` (contract section A9).

No models, no audio: every input is a hand-built ``Prediction`` list or DTO.
Malformed DTOs are built with ``model_construct`` because pydantic's lax
validation would otherwise repair them (``1`` -> ``True``, ``True`` -> ``1.0``)
before the validator ever sees the bad value.
"""

import json
import logging
import math
import random

import numpy as np
import pytest

from core_types import Prediction
from dtos import ASRQuestionResponseDto
from utils import validate_response
import response_checks
from response_checks import (
    DEFAULT_FALLBACK_ANSWER,
    END_TOLERANCE_S,
    WIRE_KEYS,
    ResponseCheckError,
    check_wire_json,
    emergency_response,
    finalize,
    strict_validate,
    summarize,
    to_wire_json,
)

DURATION = 30.0


def pred(answer, span=None, source='test'):
    return Prediction(answer=answer, span=span, source=source)


def malformed(answers, starts, ends):
    """A DTO that bypasses pydantic validation, to carry deliberately bad values."""
    return ASRQuestionResponseDto.model_construct(
        answers=answers, evidence_start=starts, evidence_end=ends,
    )


def wire(answers, starts, ends, **extra):
    """Hand-written JSON text, so NaN tokens and bool timestamps survive."""
    payload = {'answers': answers, 'evidence_start': starts, 'evidence_end': ends}
    payload.update(extra)
    return json.dumps(payload)


def assert_all_valid(response, n, duration=None):
    """Every gate the pipeline runs, plus the reference validator."""
    strict_validate(response, n, duration)
    validate_response(response, n)
    check_wire_json(to_wire_json(response), n)


# --------------------------------------------------------------------------- #
# finalize: happy path
# --------------------------------------------------------------------------- #

def test_finalize_happy_path_keeps_values_and_rounds_to_3dp():
    predictions = [
        pred(True, (1.23456, 4.56789)),
        pred(False, None),
        pred(True, None),
        pred(True, (10.0, 12.5)),
    ]
    response = finalize(predictions, 4, DURATION)

    assert isinstance(response, ASRQuestionResponseDto)
    assert response.answers == [True, False, True, True]
    assert response.evidence_start == [1.235, None, None, 10.0]
    assert response.evidence_end == [4.568, None, None, 12.5]
    assert all(isinstance(a, bool) for a in response.answers)
    assert_all_valid(response, 4, DURATION)


def test_finalize_is_deterministic():
    predictions = [pred(True, (0.1 + 0.2, 7 / 3)), pred(False), pred(True, (2, 3))]
    first = finalize(predictions, 3, DURATION)
    second = finalize(predictions, 3, DURATION)
    assert to_wire_json(first) == to_wire_json(second)


def test_finalize_accepts_list_span_and_int_bounds():
    response = finalize([pred(True, [2, 5])], 1, DURATION)
    assert response.evidence_start == [2.0]
    assert response.evidence_end == [5.0]
    assert isinstance(response.evidence_start[0], float)


def test_finalize_accepts_generator_input():
    response = finalize((pred(True, (1.0, 2.0)) for _ in range(2)), 2, DURATION)
    assert response.answers == [True, True]
    assert response.evidence_end == [2.0, 2.0]


def test_finalize_zero_questions_gives_empty_valid_response():
    response = finalize([], 0, DURATION)
    assert response.answers == [] and response.evidence_start == [] and response.evidence_end == []
    assert_all_valid(response, 0, DURATION)


# --------------------------------------------------------------------------- #
# finalize: answers
# --------------------------------------------------------------------------- #

def test_finalize_false_answer_drops_any_span():
    response = finalize([pred(False, (1.0, 2.0))], 1, DURATION)
    assert response.answers == [False]
    assert response.evidence_start == [None] and response.evidence_end == [None]


def test_finalize_true_without_span_gives_nulls():
    response = finalize([pred(True, None)], 1, DURATION)
    assert response.answers == [True]
    assert response.evidence_start == [None] and response.evidence_end == [None]
    assert_all_valid(response, 1, DURATION)


@pytest.mark.parametrize('raw', [1, 0, 'yes', 'True', 'false', None, 1.0, [True], object()])
@pytest.mark.parametrize('fallback', [True, False])
def test_finalize_non_bool_answer_becomes_fallback(raw, fallback, caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(raw, (1.0, 2.0))], 1, DURATION, fallback_answer=fallback)
    assert response.answers == [fallback]
    assert response.answers[0] is fallback
    # The span is only kept when the coerced answer is True.
    expected_span = ([1.0], [2.0]) if fallback else ([None], [None])
    assert (response.evidence_start, response.evidence_end) == expected_span
    assert 'non-bool answer' in caplog.text


def test_finalize_numpy_bool_answer_and_float32_span_are_unwrapped():
    predictions = [pred(np.bool_(True), (np.float32(1.5), np.float32(2.5)))]
    response = finalize(predictions, 1, DURATION)
    assert response.answers == [True] and response.answers[0] is True
    assert response.evidence_start == [1.5] and response.evidence_end == [2.5]
    assert type(response.evidence_start[0]) is float


def test_finalize_items_without_answer_attribute_use_fallback():
    response = finalize([{'answer': True, 'span': (1.0, 2.0)}, (True, (1.0, 2.0))], 2, DURATION,
                        fallback_answer=False)
    assert response.answers == [False, False]
    assert response.evidence_start == [None, None]


def test_finalize_non_bool_fallback_uses_module_default(caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred('maybe')], 1, DURATION, fallback_answer='yes')
    assert response.answers == [DEFAULT_FALLBACK_ANSWER]
    assert 'fallback answer' in caplog.text


# --------------------------------------------------------------------------- #
# finalize: spans
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('span', [
    (math.nan, 2.0), (1.0, math.nan), (math.nan, math.nan),
    (math.inf, 2.0), (1.0, math.inf), (-math.inf, 2.0),
])
def test_finalize_non_finite_span_becomes_nulls(span, caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(True, span)], 1, DURATION)
    assert response.answers == [True]
    assert response.evidence_start == [None] and response.evidence_end == [None]
    assert 'finite' in caplog.text
    assert_all_valid(response, 1, DURATION)


@pytest.mark.parametrize('span', [(5.0, 2.0), (2.0, 2.0)])
def test_finalize_reversed_or_empty_span_becomes_nulls(span):
    response = finalize([pred(True, span)], 1, DURATION)
    assert response.evidence_start == [None] and response.evidence_end == [None]


def test_finalize_span_that_collapses_after_rounding_becomes_nulls():
    response = finalize([pred(True, (1.00001, 1.00002))], 1, DURATION)
    assert response.evidence_start == [None] and response.evidence_end == [None]


def test_finalize_negative_start_is_clamped_to_zero():
    response = finalize([pred(True, (-0.7, 2.0))], 1, DURATION)
    assert response.evidence_start == [0.0] and response.evidence_end == [2.0]
    assert_all_valid(response, 1, DURATION)


def test_finalize_negative_zero_start_is_positive_zero_on_the_wire():
    response = finalize([pred(True, (-0.0, 2.0))], 1, DURATION)
    assert math.copysign(1.0, response.evidence_start[0]) == 1.0
    assert '-0.0' not in to_wire_json(response)


def test_finalize_end_beyond_duration_is_clamped_to_tolerance():
    response = finalize([pred(True, (25.0, 99.0))], 1, DURATION)
    assert response.evidence_end == [round(DURATION + END_TOLERANCE_S, 3)]
    assert_all_valid(response, 1, DURATION)


def test_finalize_clamped_end_never_rounds_past_the_tolerance():
    # duration + 0.05 = 12.3956 -> round() would give 12.396, above the cap.
    duration = 12.3456
    response = finalize([pred(True, (1.0, 50.0))], 1, duration)
    end = response.evidence_end[0]
    assert end == 12.395
    assert end <= duration + END_TOLERANCE_S
    assert_all_valid(response, 1, duration)


def test_finalize_span_entirely_after_the_audio_becomes_nulls():
    response = finalize([pred(True, (31.0, 32.0))], 1, DURATION)
    assert response.evidence_start == [None] and response.evidence_end == [None]


def test_finalize_span_starting_before_zero_and_ending_after_duration_is_clamped_both_ends():
    response = finalize([pred(True, (-5.0, 500.0))], 1, DURATION)
    assert response.evidence_start == [0.0]
    assert response.evidence_end == [round(DURATION + END_TOLERANCE_S, 3)]


def test_finalize_without_duration_does_not_clamp_the_end():
    response = finalize([pred(True, (1.0, 999.0))], 1, None)
    assert response.evidence_end == [999.0]
    assert_all_valid(response, 1)


@pytest.mark.parametrize('duration', [math.nan, math.inf, -1.0, 'thirty', True])
def test_finalize_unusable_duration_is_treated_as_unknown(duration, caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(True, (1.0, 999.0))], 1, duration)
    assert response.evidence_end == [999.0]
    assert 'duration_s' in caplog.text


@pytest.mark.parametrize('span', ['1.0,2.0', {'start': 1.0, 'end': 2.0}, (1.0, 2.0, 3.0), (1.0,), 5.0, b'ab'])
def test_finalize_span_that_is_not_a_pair_becomes_nulls(span, caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(True, span)], 1, DURATION)
    assert response.answers == [True]
    assert response.evidence_start == [None] and response.evidence_end == [None]
    assert 'pair' in caplog.text


@pytest.mark.parametrize('bound', [True, False])
def test_finalize_bool_span_bound_is_not_a_number(bound):
    response = finalize([pred(True, (bound, 2.0))], 1, DURATION)
    assert response.evidence_start == [None] and response.evidence_end == [None]


# --------------------------------------------------------------------------- #
# finalize: length repair
# --------------------------------------------------------------------------- #

def test_finalize_short_list_is_padded_with_fallback_and_logged(caplog):
    with caplog.at_level(logging.ERROR, logger='response_checks'):
        response = finalize([pred(False), pred(True, (1.0, 2.0))], 5, DURATION, fallback_answer=True)
    assert response.answers == [False, True, True, True, True]
    assert response.evidence_start == [None, 1.0, None, None, None]
    assert response.evidence_end == [None, 2.0, None, None, None]
    assert 'expected 5 predictions, got 2' in caplog.text
    assert_all_valid(response, 5, DURATION)


def test_finalize_padding_uses_false_fallback_when_configured():
    response = finalize([], 3, DURATION, fallback_answer=False)
    assert response.answers == [False, False, False]
    assert response.evidence_start == [None] * 3


def test_finalize_long_list_is_truncated_and_logged(caplog):
    predictions = [pred(True, (float(i), float(i) + 1.0)) for i in range(6)]
    with caplog.at_level(logging.ERROR, logger='response_checks'):
        response = finalize(predictions, 4, DURATION)
    assert response.answers == [True] * 4
    assert response.evidence_start == [0.0, 1.0, 2.0, 3.0]
    assert 'truncating' in caplog.text
    assert_all_valid(response, 4, DURATION)


def test_finalize_none_predictions_gives_all_fallback():
    response = finalize(None, 3, DURATION, fallback_answer=True)
    assert response.answers == [True, True, True]
    assert response.evidence_start == [None] * 3 and response.evidence_end == [None] * 3
    assert_all_valid(response, 3, DURATION)


@pytest.mark.parametrize('bad_n', [-1, 2.0, True, '3', None])
def test_finalize_rejects_impossible_n_questions(bad_n):
    with pytest.raises(ResponseCheckError, match='n_questions'):
        finalize([], bad_n, DURATION)


# --------------------------------------------------------------------------- #
# strict_validate
# --------------------------------------------------------------------------- #

def valid_response():
    return ASRQuestionResponseDto(
        answers=[True, False, True],
        evidence_start=[1.0, None, 0.0],
        evidence_end=[2.5, None, DURATION + END_TOLERANCE_S],
    )


def test_strict_validate_accepts_a_clean_response():
    assert strict_validate(valid_response(), 3, DURATION) is None
    assert strict_validate(valid_response(), 3) is None


def test_strict_validate_accepts_end_exactly_at_tolerance_and_rejects_just_past_it():
    strict_validate(valid_response(), 3, DURATION)
    response = malformed([True], [1.0], [DURATION + END_TOLERANCE_S + 0.001])
    with pytest.raises(ResponseCheckError, match='beyond the audio length'):
        strict_validate(response, 1, DURATION)


@pytest.mark.parametrize('answers, starts, ends, n, fragments', [
    ([True, True], [None, None], [None, None], 3,
     ['answers must have 3', 'evidence_start must have 3', 'evidence_end must have 3']),
    ([True], [None, None], [None], 1, ['evidence_start must have 1']),
    ([True], [None], [], 1, ['evidence_end must have 1']),
    (['yes'], [None], [None], 1, ['answers[0] must be a bool']),
    ([1], [None], [None], 1, ['answers[0] must be a bool']),
    ([True], [1.0], [None], 1, ['half-filled']),
    ([True], [None], [2.0], 1, ['half-filled']),
    ([True], [-0.5], [2.0], 1, ['negative']),
    ([True], [2.0], [2.0], 1, ['not after']),
    ([True], [3.0], [2.0], 1, ['not after']),
    ([True], [1.0], [DURATION + 1.0], 1, ['beyond the audio length']),
    ([True], [math.nan], [2.0], 1, ['must be finite']),
    ([True], [1.0], [math.inf], 1, ['must be finite']),
    ([True], [True], [2.0], 1, ['bool masquerading']),
    ([True], [1.0], [False], 1, ['bool masquerading']),
    ([True], ['1.0'], [2.0], 1, ['must be a number or None']),
    ([False], [1.0], [2.0], 1, ['must be null for a False answer']),
    ('yes', [None], [None], 1, ['answers must be a list']),
    ([True], None, [None], 1, ['evidence_start must be a list']),
])
def test_strict_validate_names_each_violation(answers, starts, ends, n, fragments):
    response = malformed(answers, starts, ends)
    with pytest.raises(ResponseCheckError) as info:
        strict_validate(response, n, DURATION)
    message = str(info.value)
    for fragment in fragments:
        assert fragment in message, message
    assert info.value.violations
    assert isinstance(info.value, ValueError)


def test_strict_validate_lists_every_violation_at_once():
    response = malformed([True, 'x', False], [-1.0, 5.0, 1.0], [0.5, math.nan, 2.0])
    with pytest.raises(ResponseCheckError) as info:
        strict_validate(response, 3, DURATION)
    violations = info.value.violations
    joined = '\n'.join(violations)
    assert 'answers[1] must be a bool' in joined
    assert 'evidence_start[0] is negative' in joined
    assert 'evidence_end[1] must be finite' in joined
    assert 'evidence[2] must be null for a False answer' in joined
    assert str(info.value).startswith(f'{len(violations)} response violations:')


def test_strict_validate_rejects_non_dto():
    with pytest.raises(ResponseCheckError, match='ASRQuestionResponseDto'):
        strict_validate({'answers': [True]}, 1)


@pytest.mark.parametrize('duration', [math.nan, math.inf, -1.0, 'x', True])
def test_strict_validate_rejects_unusable_duration(duration):
    with pytest.raises(ResponseCheckError, match='duration_s'):
        strict_validate(valid_response(), 3, duration)


@pytest.mark.parametrize('bad_n', [-1, 1.5, False])
def test_strict_validate_rejects_bad_n_questions(bad_n):
    with pytest.raises(ResponseCheckError, match='n_questions'):
        strict_validate(valid_response(), bad_n)


def test_strict_validate_ignores_span_type_checks_when_lists_are_absent():
    # Missing attribute entirely: reported as "not a list", no crash on zip.
    response = malformed([True], [None], [None])
    del response.evidence_end
    with pytest.raises(ResponseCheckError, match='evidence_end must be a list'):
        strict_validate(response, 1)


# --------------------------------------------------------------------------- #
# Parity with utils.validate_response
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('predictions, n, duration', [
    ([pred(True, (1.0, 2.0)), pred(False), pred(True)], 3, DURATION),
    ([pred(True, (-3.0, 400.0))] * 10, 10, DURATION),
    ([pred(True, (math.nan, 1.0)), pred('bad', (0.0, 1.0))], 2, DURATION),
    ([pred(True, (1.0, 2.0))], 4, None),
    ([pred(True, (float(i), float(i) + 0.5)) for i in range(12)], 10, 7.25),
    ([], 0, DURATION),
])
def test_finalize_output_passes_the_reference_validator(predictions, n, duration):
    response = finalize(predictions, n, duration)
    validate_response(response, n)
    strict_validate(response, n, duration)
    check_wire_json(to_wire_json(response), n)


@pytest.mark.parametrize('answers, starts, ends, n', [
    ([True, True], [None, None], [None, None], 1),
    ([True], [None], [None, None], 1),
    (['true'], [None], [None], 1),
    ([True], [1.0], [None], 1),
    ([True], [3.0], [2.0], 1),
    ([True], [math.nan], [2.0], 1),
    ([True], [True], [2.0], 1),
    ([True], ['1'], [2.0], 1),
    ('nope', [None], [None], 1),
])
def test_strict_validate_rejects_everything_the_reference_validator_rejects(answers, starts, ends, n):
    response = malformed(answers, starts, ends)
    with pytest.raises(ValueError):
        validate_response(response, n)
    with pytest.raises(ResponseCheckError) as info:
        strict_validate(response, n)
    assert any(v.startswith('utils.validate_response:') for v in info.value.violations)


def test_strict_validate_is_stricter_than_the_reference_validator():
    # utils accepts a span on a False answer and a negative start; we do not.
    response = ASRQuestionResponseDto(answers=[False, True], evidence_start=[1.0, -1.0],
                                      evidence_end=[2.0, 2.0])
    validate_response(response, 2)
    with pytest.raises(ResponseCheckError) as info:
        strict_validate(response, 2)
    assert not any(v.startswith('utils.validate_response:') for v in info.value.violations)
    assert len(info.value.violations) == 2


# --------------------------------------------------------------------------- #
# emergency_response
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('n', [0, 1, 10])
def test_emergency_response_is_valid_for_any_length(n):
    response = emergency_response(n)
    assert response.answers == [True] * n
    assert response.evidence_start == [None] * n and response.evidence_end == [None] * n
    assert_all_valid(response, n, DURATION)


def test_emergency_response_honours_false_answer():
    response = emergency_response(4, answer=False)
    assert response.answers == [False] * 4
    assert_all_valid(response, 4)


def test_emergency_response_non_bool_answer_uses_default():
    response = emergency_response(2, answer='yes')
    assert response.answers == [DEFAULT_FALLBACK_ANSWER] * 2


def test_emergency_response_rejects_bad_n():
    with pytest.raises(ResponseCheckError, match='n_questions'):
        emergency_response(-3)


# --------------------------------------------------------------------------- #
# to_wire_json / check_wire_json
# --------------------------------------------------------------------------- #

def test_to_wire_json_matches_pydantic_and_uses_real_json_types():
    response = finalize([pred(True, (1.5, 2.0)), pred(False)], 2, DURATION)
    text = to_wire_json(response)
    assert text == response.model_dump_json()
    assert json.loads(text) == {
        'answers': [True, False], 'evidence_start': [1.5, None], 'evidence_end': [2.0, None],
    }
    assert set(json.loads(text)) == set(WIRE_KEYS)
    check_wire_json(text, 2)
    check_wire_json(text.encode('utf-8'), 2)


def test_to_wire_json_rejects_non_dto():
    with pytest.raises(ResponseCheckError, match='ASRQuestionResponseDto'):
        to_wire_json({'answers': []})


@pytest.mark.parametrize('token', ['NaN', 'Infinity', '-Infinity'])
def test_check_wire_json_rejects_non_finite_tokens(token):
    text = '{"answers": [true], "evidence_start": [%s], "evidence_end": [2.0]}' % token
    with pytest.raises(ResponseCheckError, match='non-finite JSON constant'):
        check_wire_json(text, 1)


@pytest.mark.parametrize('answer', ['true', 'yes', 1, 0, None, 1.0])
def test_check_wire_json_rejects_answers_that_are_not_json_booleans(answer):
    with pytest.raises(ResponseCheckError, match=r'answers\[0\] must be a JSON boolean'):
        check_wire_json(wire([answer], [None], [None]), 1)


def test_check_wire_json_rejects_extra_and_missing_keys():
    with pytest.raises(ResponseCheckError, match='unexpected keys'):
        check_wire_json(wire([True], [None], [None], diagnostics={'x': 1}), 1)
    with pytest.raises(ResponseCheckError, match='missing keys'):
        check_wire_json('{"answers": [true], "evidence_start": [null]}', 1)


@pytest.mark.parametrize('answers, starts, ends, n', [
    ([True, True], [None, None], [None, None], 1),
    ([True], [None, None], [None], 1),
    ([True], [None], [], 1),
    ([], [], [], 3),
])
def test_check_wire_json_rejects_wrong_lengths(answers, starts, ends, n):
    with pytest.raises(ResponseCheckError, match=f'must have {n} entries'):
        check_wire_json(wire(answers, starts, ends), n)


@pytest.mark.parametrize('value', [True, False, '1.0', [1.0], {'s': 1.0}])
def test_check_wire_json_rejects_timestamps_that_are_not_numbers_or_null(value):
    with pytest.raises(ResponseCheckError, match=r'evidence_start\[0\] must be a finite JSON number or null'):
        check_wire_json(wire([True], [value], [2.0]), 1)


def test_check_wire_json_accepts_integer_timestamps():
    check_wire_json('{"answers": [true], "evidence_start": [1], "evidence_end": [2]}', 1)


@pytest.mark.parametrize('text, fragment', [
    ('[true]', 'must be a JSON object'),
    ('{"answers": true, "evidence_start": [], "evidence_end": []}', 'answers must be a JSON array'),
    ('{"answers": [true], "evidence_start": [null], "evidence_end": [null]', 'not valid JSON'),
    ('', 'not valid JSON'),
])
def test_check_wire_json_rejects_malformed_payloads(text, fragment):
    with pytest.raises(ResponseCheckError, match=fragment):
        check_wire_json(text, 1)


def test_check_wire_json_rejects_non_text_payload_and_bad_n():
    with pytest.raises(ResponseCheckError, match='str or bytes'):
        check_wire_json({'answers': [True]}, 1)
    with pytest.raises(ResponseCheckError, match='n_questions'):
        check_wire_json(wire([True], [None], [None]), -1)


def test_check_wire_json_reports_all_violations_together():
    text = wire(['yes', True], [1.0, 'x'], [None], extra=1)
    with pytest.raises(ResponseCheckError) as info:
        check_wire_json(text, 2)
    joined = '\n'.join(info.value.violations)
    assert 'unexpected keys' in joined
    assert 'answers[0] must be a JSON boolean' in joined
    assert 'evidence_start[1] must be a finite JSON number' in joined
    assert 'evidence_end must have 2 entries' in joined


@pytest.mark.parametrize('bad', [math.nan, math.inf, -math.inf])
def test_check_wire_json_catches_nan_that_pydantic_serialises_as_null(bad):
    # pydantic writes NaN/inf as null, which turns a bad span into a
    # half-filled one on the wire. strict_validate catches it upstream; the
    # wire check must catch the *serialised* body too, not just a NaN token.
    response = ASRQuestionResponseDto(answers=[True], evidence_start=[bad], evidence_end=[2.0])
    with pytest.raises(ResponseCheckError):
        strict_validate(response, 1)
    text = to_wire_json(response)
    assert 'NaN' not in text and 'Infinity' not in text  # pydantic wrote null
    with pytest.raises(ResponseCheckError, match=r'evidence\[0\] is half-filled on the wire'):
        check_wire_json(text, 1)
    with pytest.raises(ResponseCheckError):
        check_wire_json(json.dumps({'answers': [True], 'evidence_start': [bad],
                                    'evidence_end': [2.0]}), 1)


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #

def test_summarize_counts_answers_and_spans():
    response = ASRQuestionResponseDto(
        answers=[True, True, False, True],
        evidence_start=[1.0, None, None, 2.0],
        evidence_end=[2.5, None, None, 4.0],
    )
    assert summarize(response) == {
        'n_questions': 4,
        'n_true': 3,
        'n_false': 1,
        'n_with_span': 2,
        'n_true_without_span': 1,
        'span_total_s': 3.5,
    }


def test_summarize_empty_response():
    assert summarize(emergency_response(0)) == {
        'n_questions': 0, 'n_true': 0, 'n_false': 0, 'n_with_span': 0,
        'n_true_without_span': 0, 'span_total_s': 0.0,
    }


def test_summarize_tolerates_malformed_values():
    response = malformed([True, 'yes'], [math.nan, 1.0], [2.0, True])
    summary = summarize(response)
    assert summary['n_questions'] == 2
    assert summary['n_true'] == 1 and summary['n_false'] == 0
    assert summary['n_with_span'] == 0
    assert summary['n_true_without_span'] == 1


# --------------------------------------------------------------------------- #
# ResponseCheckError
# --------------------------------------------------------------------------- #

def test_response_check_error_shapes_its_message():
    single = ResponseCheckError('one thing')
    assert single.violations == ['one thing']
    assert str(single) == '1 response violation: one thing'
    several = ResponseCheckError(['a', 'b'])
    assert str(several) == '2 response violations: a; b'
    empty = ResponseCheckError([])
    assert empty.violations == ['unspecified violation']


def test_module_constants_are_what_the_pipeline_relies_on():
    assert END_TOLERANCE_S == 0.05
    assert response_checks.TIMESTAMP_DECIMALS == 3
    assert WIRE_KEYS == frozenset({'answers', 'evidence_start', 'evidence_end'})


# --------------------------------------------------------------------------- #
# Regressions from the adversarial review: finalize must never raise
# --------------------------------------------------------------------------- #

HUGE_INT = 10 ** 400  # math.isfinite / float() raise OverflowError on this


@pytest.mark.parametrize('span', [(HUGE_INT, HUGE_INT + 1), (HUGE_INT, 2.0), (1.0, HUGE_INT), (-HUGE_INT, 2.0)])
def test_finalize_int_too_large_for_a_float_becomes_nulls_instead_of_raising(span, caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(True, span)], 1, DURATION)
    assert response.answers == [True]
    assert response.evidence_start == [None] and response.evidence_end == [None]
    assert 'finite' in caplog.text
    assert_all_valid(response, 1, DURATION)


def test_finalize_int_duration_too_large_for_a_float_is_treated_as_unknown(caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(True, (1.0, 999.0))], 1, HUGE_INT)
    assert response.evidence_end == [999.0]
    assert 'duration_s' in caplog.text
    assert_all_valid(response, 1)


@pytest.mark.parametrize('predictions', [5, 1.5, object()])
def test_finalize_non_iterable_predictions_gives_all_fallback(predictions, caplog):
    with caplog.at_level(logging.ERROR, logger='response_checks'):
        response = finalize(predictions, 2, DURATION, fallback_answer=False)
    assert response.answers == [False, False]
    assert response.evidence_start == [None, None]
    assert 'could not be iterated' in caplog.text
    assert_all_valid(response, 2, DURATION)


def test_finalize_iterator_that_raises_midway_gives_all_fallback(caplog):
    def broken():
        yield pred(True, (1.0, 2.0))
        raise RuntimeError('stage exploded')

    with caplog.at_level(logging.ERROR, logger='response_checks'):
        response = finalize(broken(), 3, DURATION)
    assert response.answers == [True, True, True]
    assert response.evidence_start == [None] * 3
    assert_all_valid(response, 3, DURATION)


class _RaisingRepr:
    def __repr__(self):
        raise RuntimeError('no repr for you')


class _RaisingIter:
    def __iter__(self):
        raise RuntimeError('no iter for you')


class _RaisingAnswer:
    span = (1.0, 2.0)

    @property
    def answer(self):
        raise RuntimeError('no answer for you')


class _RaisingSpan:
    answer = True

    @property
    def span(self):
        raise RuntimeError('no span for you')


@pytest.mark.parametrize('item, answer_unreadable', [
    pytest.param(pred(_RaisingRepr(), (1.0, 2.0)), True, id='answer-with-raising-repr'),
    pytest.param(_RaisingAnswer(), True, id='raising-answer-property'),
    pytest.param(pred(True, _RaisingRepr()), False, id='span-with-raising-repr'),
    pytest.param(pred(True, _RaisingIter()), False, id='span-with-raising-iter'),
    pytest.param(_RaisingSpan(), False, id='raising-span-property'),
])
@pytest.mark.parametrize('fallback', [True, False])
def test_finalize_hostile_item_never_raises(item, answer_unreadable, fallback, caplog):
    """An unreadable answer becomes the fallback (keeping a good span only when
    that fallback is True); an unreadable span costs the evidence, never the
    answer. Neighbouring predictions are untouched either way."""
    good = pred(True, (5.0, 6.0))
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([good, item, good], 3, DURATION, fallback_answer=fallback)
    if answer_unreadable:
        assert response.answers == [True, fallback, True]
        expected_span = (1.0, 2.0) if fallback else (None, None)  # its span was fine
    else:
        assert response.answers == [True, True, True]
        expected_span = (None, None)
    assert (response.evidence_start[1], response.evidence_end[1]) == expected_span
    assert response.evidence_start[0] == 5.0 and response.evidence_start[2] == 5.0
    assert_all_valid(response, 3, DURATION)


def test_finalize_unwraps_float64_int64_and_0d_arrays():
    predictions = [
        pred(np.array(True), (np.float64(1.5), np.int64(3))),
        pred(np.bool_(False), (np.float64(1.5), np.float64(2.5))),
        pred(True, np.array([2.25, 4.75])),
        pred(True, (np.array(0.5), np.array(1.0))),
    ]
    response = finalize(predictions, 4, DURATION)
    assert response.answers == [True, False, True, True]
    assert all(type(a) is bool for a in response.answers)
    assert response.evidence_start == [1.5, None, 2.25, 0.5]
    assert response.evidence_end == [3.0, None, 4.75, 1.0]
    assert all(type(v) is float for v in response.evidence_start + response.evidence_end if v is not None)
    assert_all_valid(response, 4, DURATION)


@pytest.mark.parametrize('raw', [np.int64(1), np.float64(1.0), np.float32(0.0), np.array([True]), np.array(1)])
def test_finalize_numpy_number_answer_is_not_a_bool(raw, caplog):
    with caplog.at_level(logging.WARNING, logger='response_checks'):
        response = finalize([pred(raw, (1.0, 2.0))], 1, DURATION, fallback_answer=False)
    assert response.answers == [False] and response.answers[0] is False
    assert 'non-bool answer' in caplog.text


@pytest.mark.parametrize('span', [(np.float64('nan'), 2.0), (1.0, np.float32('inf')), (np.array(math.nan), 2.0)])
def test_finalize_numpy_non_finite_span_becomes_nulls(span):
    response = finalize([pred(True, span)], 1, DURATION)
    assert response.evidence_start == [None] and response.evidence_end == [None]
    assert_all_valid(response, 1, DURATION)


def test_finalize_zero_duration_caps_the_end_at_the_tolerance():
    response = finalize([pred(True, (0.0, 1.0)), pred(True, (0.06, 1.0))], 2, 0.0)
    assert response.evidence_start == [0.0, None]
    assert response.evidence_end == [END_TOLERANCE_S, None]
    assert_all_valid(response, 2, 0.0)


@pytest.mark.parametrize('start, expected', [
    (30.049, (30.049, 30.05)),   # inside the tolerance: kept, 1 ms long
    (30.05, (None, None)),       # end == start after clamping: dropped
    (30.0499, (None, None)),     # rounds up onto the cap: dropped
    (29.9996, (30.0, 30.05)),    # rounds to 30.0, still before the cap
])
def test_finalize_clamping_near_the_cap_never_yields_end_at_or_before_start(start, expected):
    response = finalize([pred(True, (start, 99.0))], 1, DURATION)
    assert (response.evidence_start[0], response.evidence_end[0]) == expected
    assert_all_valid(response, 1, DURATION)


def test_finalize_tiny_negative_end_does_not_leak_negative_zero():
    response = finalize([pred(True, (-1e-20, -1e-21))], 1, DURATION)
    assert response.evidence_start == [None] and response.evidence_end == [None]
    assert '-0.0' not in to_wire_json(response)


def test_finalize_output_survives_every_gate_under_seeded_fuzz():
    """Whatever mix of garbage comes in, the output passes every gate."""
    rng = random.Random(2026)
    answer_pool = [True, False, np.bool_(True), np.bool_(False), np.array(True), 1, 0, 1.0,
                   'yes', 'false', None, [True], np.int64(1), np.float64('nan'), _RaisingRepr()]
    bound_pool = [0.0, -0.0, 1e-9, 0.5, 3.14159, 29.99, 30.0, 30.049, 30.05, 30.051, 99.0,
                  -1.0, -1e-20, 1e308, -1e308, HUGE_INT, -HUGE_INT, math.nan, math.inf, -math.inf,
                  np.float32(2.5), np.float64(7.25), np.int64(4), np.array(1.0), True, False,
                  '1.0', None, b'1', [1.0]]
    for _ in range(300):
        n = rng.randint(0, 12)
        duration = rng.choice([DURATION, 0.0, 12.3456, None, math.nan, 1e308, np.float32(30.0)])
        count = max(0, n + rng.choice([-2, -1, 0, 0, 0, 1, 2]))
        items = []
        for _ in range(count):
            kind = rng.random()
            if kind < 0.7:
                span = (rng.choice(bound_pool), rng.choice(bound_pool))
            elif kind < 0.8:
                span = np.array([rng.uniform(-5, 40), rng.uniform(-5, 40)])
            elif kind < 0.9:
                span = None
            else:
                span = rng.choice(['1,2', {'s': 1}, (1.0,), (1.0, 2.0, 3.0), _RaisingIter(), 5.0])
            items.append(pred(rng.choice(answer_pool), span))
        fallback = rng.choice([True, False])
        response = finalize(items, n, duration, fallback_answer=fallback)
        assert len(response.answers) == n
        usable = duration if (duration is not None and math.isfinite(float(duration))) else None
        assert_all_valid(response, n, usable)


# --------------------------------------------------------------------------- #
# Regressions: strict_validate / check_wire_json raise one exception type
# --------------------------------------------------------------------------- #

def test_strict_validate_reports_int_too_large_for_a_float_as_a_violation():
    response = malformed([True], [HUGE_INT], [2.0])
    with pytest.raises(ResponseCheckError, match=r'evidence_start\[0\] must be finite') as info:
        strict_validate(response, 1, DURATION)
    assert any(v.startswith('utils.validate_response:') for v in info.value.violations)
    with pytest.raises(ResponseCheckError, match='duration_s'):
        strict_validate(valid_response(), 3, HUGE_INT)


@pytest.mark.parametrize('answers, starts, ends, fragment', [
    ([True], [None], [2.0], r'evidence\[0\] is half-filled on the wire'),
    ([True], [1.0], [None], r'evidence\[0\] is half-filled on the wire'),
    ([True], [3.0], [2.0], r'evidence_end\[0\] \(2.0\) is not after'),
    ([True], [2.0], [2.0], r'not after'),
    ([True], [-0.5], [2.0], r'evidence_start\[0\] is negative'),
    ([False], [1.0], [2.0], r'evidence\[0\] must be null for a false answer'),
])
def test_check_wire_json_rejects_intervals_that_score_nothing(answers, starts, ends, fragment):
    with pytest.raises(ResponseCheckError, match=fragment):
        check_wire_json(wire(answers, starts, ends), 1)


def test_check_wire_json_pair_checks_do_not_mask_type_violations():
    text = wire([True, False, True], ['x', 1.0, None], [2.0, 2.0, 3.0])
    with pytest.raises(ResponseCheckError) as info:
        check_wire_json(text, 3)
    joined = '\n'.join(info.value.violations)
    assert 'evidence_start[0] must be a finite JSON number' in joined
    assert 'evidence[1] must be null for a false answer' in joined
    assert 'evidence[2] is half-filled' in joined
    assert 'not after' not in joined  # the string bound is not compared


def test_check_wire_json_accepts_any_key_order_and_whitespace():
    check_wire_json('{ "evidence_end" : [2.0, null],\n "answers":[true,false], "evidence_start":[1.0,null] }', 2)


def test_check_wire_json_rejects_duplicate_keys():
    text = '{"answers":[true],"evidence_start":[null],"evidence_end":[null],"answers":[false]}'
    with pytest.raises(ResponseCheckError, match='duplicate key'):
        check_wire_json(text, 1)


@pytest.mark.parametrize('text, fragment', [
    pytest.param(b'{"answers":[true],"evidence_start":[\xff],"evidence_end":[2]}', 'not valid JSON',
                 id='invalid-utf8-bytes'),
    pytest.param('[' * 200_000 + ']' * 200_000, 'not valid JSON', id='nesting-past-recursion-limit'),
    pytest.param('{"answers":[true],"evidence_start":[' + '9' * 400 + '],"evidence_end":[2]}',
                 'must be a finite JSON number', id='int-too-large-for-float'),
    pytest.param('{"answers":[true],"evidence_start":[1e999],"evidence_end":[2]}',
                 'must be a finite JSON number', id='1e999-parses-as-inf'),
    pytest.param('{"answers":[true],"evidence_start":[-1e999],"evidence_end":[2]}',
                 'must be a finite JSON number', id='-1e999-parses-as-neg-inf'),
])
def test_check_wire_json_never_leaks_a_decoder_exception(text, fragment):
    with pytest.raises(ResponseCheckError, match=fragment):
        check_wire_json(text, 1)


def test_check_wire_json_accepts_the_exact_body_finalize_produces_for_edge_spans():
    predictions = [pred(True, (0.0, 0.001)), pred(True, (29.9999, 40.0)), pred(False, (1.0, 2.0)),
                   pred(True, None), pred(True, (1e-9, 5.0))]
    response = finalize(predictions, 5, DURATION)
    text = to_wire_json(response)
    check_wire_json(text, 5)
    payload = json.loads(text)
    assert payload['evidence_start'] == [0.0, 30.0, None, None, 0.0]
    assert payload['evidence_end'] == [0.001, 30.05, None, None, 5.0]


# --------------------------------------------------------------------------- #
# Regressions: summarize really is tolerant
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('response', [
    malformed(None, [1.0], [2.0]),
    malformed([True], None, None),
    malformed([True, HUGE_INT], [HUGE_INT, 1.0], [2.0, HUGE_INT]),
    {'answers': [True]},
])
def test_summarize_never_raises_on_a_broken_response(response):
    summary = summarize(response)
    assert set(summary) == {'n_questions', 'n_true', 'n_false', 'n_with_span',
                            'n_true_without_span', 'span_total_s'}
    assert summary['n_with_span'] >= 0


def test_summarize_missing_column_counts_as_empty():
    response = malformed([True], [None], [None])
    del response.evidence_end
    assert summarize(response)['n_with_span'] == 0
