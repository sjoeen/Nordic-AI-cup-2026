"""Tests for ``evidence.py``: QA unit/word ids -> one audio span.

Everything runs on synthetic contexts from ``tests.conftest.make_context``
(0.4 s per word, 0.4 s gap between units), so every expected span below is
arithmetic on those timings. No models, no network, no files.
"""

import sys

import pytest

from core_types import AudioContext, QAResult, TranscriptUnit, Word
from evidence import (
    DEFAULT_PAD_GRID,
    FILLER,
    MODES,
    EvidencePolicy,
    calibrate_padding,
    focus_tokens,
    group_runs,
    select_span,
    trim_filler_words,
    trim_to_focus,
    valid_positions,
)
from tests.conftest import make_context

# Unit timings of CLINIC (see make_context): u0 0.0-2.0, u1 2.4-5.2, u2 5.6-7.6,
# u3 8.0-10.4, u4 10.8-12.8; duration 14.2. Word ids: u0 0-4, u1 5-11, u2 12-16,
# u3 17-22, u4 23-27.
SENTENCES = [
    'the patient came in today',
    'we will start penicillin 100 milligrams daily',
    'take it after a meal',
    'blood pressure was 130 over 85',
    'come back in two weeks',
]
BP_QUESTION = 'Was the blood pressure 130 over 85?'
PENICILLIN_QUESTION = 'Did the doctor start penicillin 100 milligrams daily?'
OFF_TOPIC_QUESTION = 'Was the weather nice yesterday?'


@pytest.fixture
def clinic():
    return make_context(SENTENCES)


def yes(unit_ids, word_ids=None):
    return QAResult(question_index=0, answer=True, evidence_unit_ids=list(unit_ids),
                    evidence_word_ids=word_ids)


def approx_span(expected):
    return pytest.approx(expected, abs=1e-6)


def unit(unit_id, start, end, text='', words=()):
    return TranscriptUnit(unit_id=unit_id, start_s=start, end_s=end, text=text, words=list(words))


def manual_context(units, duration_s):
    return AudioContext(audio_sha256='manual', duration_s=duration_s, units=list(units))


def words_from(texts, start=0.0, step=0.4):
    """Consecutive ``Word`` objects with fixed spacing, for the filler tests."""
    return [
        Word(word_id=i, start_s=start + i * step, end_s=start + (i + 1) * step, text=text)
        for i, text in enumerate(texts)
    ]


# --------------------------------------------------------------------------- #
# EvidencePolicy
# --------------------------------------------------------------------------- #

def test_policy_defaults_match_contract():
    policy = EvidencePolicy()
    assert (policy.mode, policy.max_span_s, policy.pad_pre_s, policy.pad_post_s) == ('union', 20.0, 0.0, 0.0)
    assert policy.trim_filler is True and policy.min_span_s == 0.2
    assert MODES == ('union', 'first_run', 'longest_run', 'best_run')


def test_policy_normalises_mode_and_coerces_numbers():
    policy = EvidencePolicy(mode=' Best_Run ', pad_pre_s=1, max_span_s=5, trim_filler=0)
    assert policy.mode == 'best_run'
    assert policy.pad_pre_s == 1.0 and isinstance(policy.pad_pre_s, float)
    assert policy.max_span_s == 5.0 and policy.trim_filler is False
    assert EvidencePolicy(max_span_s=None).max_span_s is None


@pytest.mark.parametrize('kwargs', [
    {'mode': 'nope'},
    {'pad_pre_s': float('nan')},
    {'pad_post_s': float('inf')},
    {'pad_post_s': 'x'},
    {'min_span_s': -0.1},
    {'max_span_s': 0.0},
    {'max_span_s': float('nan')},
])
def test_policy_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        EvidencePolicy(**kwargs)


# --------------------------------------------------------------------------- #
# Id validation and run grouping
# --------------------------------------------------------------------------- #

def test_group_runs_groups_adjacent_positions():
    assert group_runs([0, 1, 2, 5, 7, 8]) == [[0, 1, 2], [5], [7, 8]]
    assert group_runs([2, 0, 1, 1]) == [[0, 1, 2]]
    assert group_runs([4]) == [[4]]
    assert group_runs([]) == []


def test_valid_positions_filters_and_sorts(clinic):
    ids = [3, 99, -1, 'x', True, 1.0, None, 1, 1, 0]
    assert valid_positions(clinic, ids) == [0, 1, 3]
    assert valid_positions(clinic, None) == []
    assert valid_positions(clinic, []) == []
    assert valid_positions(make_context([]), [0]) == []


def test_valid_positions_are_indexes_not_ids():
    context = manual_context([unit(10, 0.0, 1.0), unit(20, 2.0, 3.0), unit(30, 4.0, 5.0)], 6.0)
    assert valid_positions(context, [30, 10]) == [0, 2]


def test_valid_positions_drops_units_with_nan_bounds():
    context = manual_context([unit(0, 0.0, 1.0), unit(1, float('nan'), 3.0), unit(2, 4.0, float('inf'))], 6.0)
    assert valid_positions(context, [0, 1, 2]) == [0]


# --------------------------------------------------------------------------- #
# select_span: modes
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('mode, ids, question, expected', [
    ('union', [0, 2, 3], None, (0.0, 10.4)),           # bridges the gap over u1
    ('first_run', [0, 2, 3], None, (0.0, 2.0)),
    ('longest_run', [0, 2, 3], None, (5.6, 10.4)),      # [2,3] lasts 4.8 s, [0] 2.0 s
    ('best_run', [0, 3], BP_QUESTION, (8.0, 10.4)),     # only u3 holds the question's terms
    ('best_run', [0, 3], None, (0.0, 2.0)),             # no terms at all: earliest run
    ('union', [1], None, (2.4, 5.2)),                   # single unit
    ('union', [3, 1, 0], None, (0.0, 10.4)),            # order of ids is irrelevant
])
def test_select_span_modes(clinic, mode, ids, question, expected):
    policy = EvidencePolicy(mode=mode, max_span_s=None)
    assert select_span(clinic, yes(ids), policy, question) == approx_span(expected)


def test_longest_run_and_best_run_tie_break_to_earliest(clinic):
    # u0 (2.0 s) and u2 (2.0 s) have equal duration and equal (zero) density.
    assert select_span(clinic, yes([0, 2]), EvidencePolicy(mode='longest_run')) == approx_span((0.0, 2.0))
    assert select_span(clinic, yes([2, 0]), EvidencePolicy(mode='best_run'), OFF_TOPIC_QUESTION) == approx_span((0.0, 2.0))


def test_consecutive_means_adjacent_positions_not_ids():
    context = manual_context([unit(10, 0.0, 1.0), unit(20, 2.0, 3.0), unit(30, 4.0, 5.0)], 6.0)
    policy = EvidencePolicy(mode='first_run')
    assert select_span(context, yes([10, 20]), policy) == approx_span((0.0, 3.0))
    assert select_span(context, yes([10, 30]), policy) == approx_span((0.0, 1.0))
    assert select_span(context, yes([10, 30]), EvidencePolicy(mode='union')) == approx_span((0.0, 5.0))


# --------------------------------------------------------------------------- #
# select_span: never invents a span
# --------------------------------------------------------------------------- #

def test_invalid_ids_are_ignored(clinic):
    result = yes([99, 1, -1, 'x', True, 1.0, None])
    assert select_span(clinic, result, EvidencePolicy()) == approx_span((2.4, 5.2))
    assert select_span(clinic, yes([1, 1, 1]), EvidencePolicy()) == approx_span((2.4, 5.2))


@pytest.mark.parametrize('ids', [[], [99], [-1, 'x', None], None])
def test_no_valid_ids_gives_none(clinic, ids):
    result = QAResult(question_index=0, answer=True, evidence_unit_ids=ids)
    assert select_span(clinic, result, EvidencePolicy()) is None


def test_answer_false_gives_none_even_with_evidence(clinic):
    result = QAResult(question_index=0, answer=False, evidence_unit_ids=[0, 1], evidence_word_ids=[5, 6])
    assert select_span(clinic, result, EvidencePolicy()) is None


def test_empty_context_gives_none():
    assert select_span(make_context([]), yes([0]), EvidencePolicy()) is None


def test_select_span_does_not_mutate_inputs(clinic):
    result = yes([3, 99, 1], word_ids=[8, 9])
    before = (list(result.evidence_unit_ids), list(result.evidence_word_ids))
    select_span(clinic, result, EvidencePolicy(), BP_QUESTION)
    assert (result.evidence_unit_ids, result.evidence_word_ids) == before
    assert [u.unit_id for u in clinic.units] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# select_span: word-level override
# --------------------------------------------------------------------------- #

def test_word_ids_override_unit_bounds(clinic):
    # "penicillin 100 milligrams" = word ids 8-10 of u1 (3.6-4.8).
    assert select_span(clinic, yes([1], word_ids=[10, 8, 9]), EvidencePolicy()) == approx_span((3.6, 4.8))


@pytest.mark.parametrize('word_ids', [
    [8, 9, 3],      # 3 sits in u0, which is not selected: partial resolution is rejected
    [8, 999],       # unknown id
    [8, 'x'],       # non-integer
    [],             # empty means "no override"
])
def test_word_ids_outside_selected_units_fall_back_to_unit_bounds(clinic, word_ids):
    assert select_span(clinic, yes([1], word_ids=word_ids), EvidencePolicy()) == approx_span((2.4, 5.2))


def test_word_ids_override_ignored_when_units_have_no_words():
    context = manual_context([unit(0, 1.0, 3.0, text='the dose is 100 mg')], 5.0)
    assert select_span(context, yes([0], word_ids=[0, 1]), EvidencePolicy()) == approx_span((1.0, 3.0))


# --------------------------------------------------------------------------- #
# Filler trimming
# --------------------------------------------------------------------------- #

def test_filler_set_matches_contract():
    assert FILLER == frozenset({
        'um', 'uh', 'hmm', 'mm', 'mhm', 'mm-hmm', 'uh-huh', 'so', 'okay', 'ok', 'yeah', 'yes',
        'no', 'well', 'right', 'alright', 'oh', 'ah', 'like', 'you know', 'i mean', 'and', 'but',
    })


def test_trim_filler_moves_span_onto_content_words():
    # um 0.0-0.4, so 0.4-0.8, the 0.8-1.2, ..., mg 2.4-2.8, okay 2.8-3.2
    context = make_context(['um so the dose is 100 mg okay'])
    assert select_span(context, yes([0]), EvidencePolicy(trim_filler=True)) == approx_span((0.8, 2.8))
    assert select_span(context, yes([0]), EvidencePolicy(trim_filler=False)) == approx_span((0.0, 3.2))


def test_trim_filler_keeps_span_when_only_filler_remains():
    context = make_context(['um so okay'])
    assert select_span(context, yes([0]), EvidencePolicy()) == approx_span((0.0, 1.2))


def test_trim_filler_handles_multi_word_fillers_and_punctuation():
    context = make_context(['You know, the dose is 100 mg, I mean.'])
    # you 0.0-0.4, know, the 0.8-1.2, dose, is, 100, mg, 2.4-2.8, I, mean.
    assert select_span(context, yes([0]), EvidencePolicy()) == approx_span((0.8, 2.8))


def test_trim_filler_applies_to_word_override_too(clinic):
    # u3 words: blood(17) pressure(18) was(19) 130(20) over(21) 85(22); nothing to trim.
    assert select_span(clinic, yes([3], word_ids=[17, 18]), EvidencePolicy()) == approx_span((8.0, 8.8))
    context = make_context(['okay the dose is 100 mg yeah'])
    # okay(0) the(1) dose(2) is(3) 100(4) mg(5) yeah(6): override [0, 1, 2, 6] trims to the..dose.
    assert select_span(context, yes([0], word_ids=[0, 1, 2, 6]), EvidencePolicy()) == approx_span((0.4, 1.2))


def test_trim_filler_words_directly():
    texts = ['Um,', 'so', 'the', 'dose', 'is', '100', 'mg', 'okay.']
    assert [w.text for w in trim_filler_words(words_from(texts))] == ['the', 'dose', 'is', '100', 'mg']
    assert [w.text for w in trim_filler_words(words_from(['you', 'know', 'the', 'dose', 'I', 'mean']))] == ['the', 'dose']
    assert [w.text for w in trim_filler_words(words_from(['the', 'um', 'dose']))] == ['the', 'um', 'dose']
    assert [w.text for w in trim_filler_words(words_from(['...', 'the', 'dose']))] == ['the', 'dose']
    assert [w.text for w in trim_filler_words(words_from(['Mm-hmm,', 'yes']))] == ['Mm-hmm,', 'yes']
    assert trim_filler_words([]) == []


def test_units_without_words_use_unit_bounds():
    context = manual_context([unit(0, 1.0, 3.0, text='um the dose is 100 mg okay')], 5.0)
    assert select_span(context, yes([0]), EvidencePolicy(trim_filler=True)) == approx_span((1.0, 3.0))


# --------------------------------------------------------------------------- #
# Focus terms and max_span trimming
# --------------------------------------------------------------------------- #

def test_focus_tokens_uses_fallback_when_sibling_is_missing(monkeypatch):
    question_rewrite = pytest.importorskip('question_rewrite')
    monkeypatch.delattr(question_rewrite, 'focus_terms', raising=False)
    assert focus_tokens('Did the patient take two weeks of penicillin?') == {'take', 'two', 'week', 'penicillin'}
    assert focus_tokens(None) == frozenset() and focus_tokens('   ') == frozenset()


def test_focus_tokens_uses_fallback_when_sibling_fails_to_import(monkeypatch):
    monkeypatch.setitem(sys.modules, 'question_rewrite', None)
    assert focus_tokens('Was the blood pressure 130 over 85?') == {'blood', 'pressure', '130', '85'}


def test_focus_tokens_prefers_sibling_and_survives_its_errors(monkeypatch):
    question_rewrite = pytest.importorskip('question_rewrite')
    monkeypatch.setattr(question_rewrite, 'focus_terms', lambda q: ['Weeks', 'penicillin'], raising=False)
    assert focus_tokens('anything') == {'week', 'penicillin'}

    def boom(question):
        raise RuntimeError('sibling bug')

    monkeypatch.setattr(question_rewrite, 'focus_terms', boom, raising=False)
    assert focus_tokens('Did the patient take two weeks of penicillin?') == {'take', 'two', 'week', 'penicillin'}


def test_max_span_trimming_needs_a_question(clinic):
    everything = yes([0, 1, 2, 3, 4])  # union 0.0-12.8
    policy = EvidencePolicy(max_span_s=5.0)
    assert select_span(clinic, everything, policy, BP_QUESTION) == approx_span((8.0, 10.4))
    assert select_span(clinic, everything, policy, PENICILLIN_QUESTION) == approx_span((2.4, 5.2))
    assert select_span(clinic, everything, policy, None) == approx_span((0.0, 12.8))
    assert select_span(clinic, everything, policy, '  ') == approx_span((0.0, 12.8))
    assert select_span(clinic, everything, EvidencePolicy(max_span_s=None), BP_QUESTION) == approx_span((0.0, 12.8))


def test_max_span_trimming_stays_within_selected_runs(clinic):
    # union of runs [0] and [3,4] is 0.0-12.8; the densest fitting sub-run is u3 alone.
    policy = EvidencePolicy(mode='union', max_span_s=5.0)
    assert select_span(clinic, yes([0, 3, 4]), policy, BP_QUESTION) == approx_span((8.0, 10.4))
    # A short span is left alone even with a question.
    assert select_span(clinic, yes([3]), policy, PENICILLIN_QUESTION) == approx_span((8.0, 10.4))


def test_trim_to_focus_picks_densest_fitting_sub_run(clinic):
    assert trim_to_focus(clinic, [0, 1, 2, 3, 4], BP_QUESTION, 5.0) == [3]
    assert trim_to_focus(clinic, [0, 1, 2, 3, 4], PENICILLIN_QUESTION, 5.0) == [1]
    # Wide enough limit: u3 alone (4 terms / 2.4 s) still beats [2, 3] (4 terms / 4.8 s).
    assert trim_to_focus(clinic, [2, 3], BP_QUESTION, 20.0) == [3]
    # No term matches anywhere: the earliest, longest sub-run that fits.
    assert trim_to_focus(clinic, [0, 1, 2, 3, 4], OFF_TOPIC_QUESTION, 5.0) == [0]
    assert trim_to_focus(clinic, [2, 3, 4], None, 5.0) == [2, 3]


def test_trim_to_focus_keeps_a_single_unit_longer_than_the_limit():
    context = make_context([
        'we will start you on penicillin one hundred milligrams daily',  # 10 words, 0.0-4.0
        'okay thank you very much',                                      # 4.4-6.4
    ])
    assert trim_to_focus(context, [0, 1], PENICILLIN_QUESTION, 3.0) == [0]
    policy = EvidencePolicy(max_span_s=3.0)
    assert select_span(context, yes([0, 1]), policy, PENICILLIN_QUESTION) == approx_span((0.0, 4.0))


def test_trim_to_focus_prefers_a_fitting_unit_with_evidence_over_an_overlong_one():
    context = make_context([
        'we will start you on penicillin one hundred milligrams daily',  # 0.0-4.0, three terms
        'penicillin is fine',                                            # 4.4-5.6, one term
    ])
    assert trim_to_focus(context, [0, 1], PENICILLIN_QUESTION, 3.0) == [1]


def test_trim_to_focus_edge_cases(clinic):
    assert trim_to_focus(clinic, [], BP_QUESTION, 5.0) == []
    assert trim_to_focus(clinic, [99, 'x'], BP_QUESTION, 5.0) == []
    assert trim_to_focus(clinic, [3], BP_QUESTION, 0.1) == [3]
    with pytest.raises(ValueError):
        trim_to_focus(clinic, [3], BP_QUESTION, 0.0)
    with pytest.raises(ValueError):
        trim_to_focus(clinic, [3], BP_QUESTION, float('nan'))


# --------------------------------------------------------------------------- #
# Padding, clamping and minimum length
# --------------------------------------------------------------------------- #

def test_padding_is_clamped_at_zero_and_at_duration(clinic):
    policy = EvidencePolicy(pad_pre_s=1.0, pad_post_s=0.5)
    assert select_span(clinic, yes([0]), policy) == approx_span((0.0, 2.5))
    policy = EvidencePolicy(pad_pre_s=0.3, pad_post_s=5.0)
    assert select_span(clinic, yes([4]), policy) == approx_span((10.5, 14.2))


def test_negative_padding_shrinks_the_span(clinic):
    policy = EvidencePolicy(pad_pre_s=-0.4, pad_post_s=-0.4)
    assert select_span(clinic, yes([1]), policy) == approx_span((2.8, 4.8))


def test_min_span_expands_symmetrically_and_slides_off_the_edges(clinic):
    policy = EvidencePolicy(min_span_s=3.0)
    assert select_span(clinic, yes([2]), policy) == approx_span((5.1, 8.1))   # centred on 6.6
    assert select_span(clinic, yes([0]), policy) == approx_span((0.0, 3.0))   # slid off the start
    short_audio = make_context(SENTENCES, duration_s=13.0)
    assert select_span(short_audio, yes([4]), policy) == approx_span((10.0, 13.0))  # slid off the end


def test_zero_length_unit_needs_min_span_to_become_a_span():
    context = manual_context([unit(0, 2.0, 2.0)], 10.0)
    assert select_span(context, yes([0]), EvidencePolicy(min_span_s=0.0)) is None
    assert select_span(context, yes([0]), EvidencePolicy(min_span_s=0.2)) == approx_span((1.9, 2.1))


def test_unit_beyond_duration_is_clamped_or_dropped():
    context = manual_context([unit(0, 9.5, 12.0)], 10.0)
    assert select_span(context, yes([0]), EvidencePolicy()) == approx_span((9.5, 10.0))
    context = manual_context([unit(0, 11.0, 12.0)], 10.0)
    assert select_span(context, yes([0]), EvidencePolicy(min_span_s=0.0)) is None


def test_unusable_duration_falls_back_to_last_unit_end():
    context = manual_context([unit(0, 0.0, 1.0), unit(1, 2.0, 3.0)], float('nan'))
    assert select_span(context, yes([0]), EvidencePolicy(pad_post_s=5.0)) == approx_span((0.0, 3.0))


def test_result_is_rounded_to_milliseconds():
    context = manual_context([unit(0, 1.23456, 2.34567)], 5.0)
    assert select_span(context, yes([0]), EvidencePolicy()) == (1.235, 2.346)


# --------------------------------------------------------------------------- #
# calibrate_padding
# --------------------------------------------------------------------------- #

def test_calibrate_padding_recovers_a_known_offset():
    gold = [(1.0, 4.0), (10.0, 12.5), (20.0, 21.0), (30.5, 33.0)]
    predicted = [(s + 0.3, e - 0.2) for s, e in gold]
    assert calibrate_padding(list(zip(gold, predicted)), DEFAULT_PAD_GRID) == (0.3, 0.2)
    assert calibrate_padding(list(zip(gold, predicted)), [0.0, 0.3], [0.0, 0.2]) == (0.3, 0.2)


def test_calibrate_padding_prefers_no_padding_on_ties_and_skips_bad_pairs():
    gold = [(1.0, 4.0), (10.0, 12.5)]
    pairs = list(zip(gold, gold)) + [(None, (1.0, 2.0)), ((1.0, 2.0), (float('nan'), 2.0)), ((3.0, 1.0), (1.0, 2.0))]
    assert calibrate_padding(pairs, DEFAULT_PAD_GRID) == (0.0, 0.0)
    assert calibrate_padding([], DEFAULT_PAD_GRID) == (0.0, 0.0)
    assert calibrate_padding([(None, None)], [0.5]) == (0.0, 0.0)


def test_calibrate_padding_rejects_bad_grids():
    pairs = [((1.0, 4.0), (1.0, 4.0))]
    with pytest.raises(ValueError):
        calibrate_padding(pairs, [])
    with pytest.raises(ValueError):
        calibrate_padding(pairs, [0.0, float('nan')])
    with pytest.raises(ValueError):
        calibrate_padding(pairs, [0.0], ['x'])


def test_calibrate_padding_is_deterministic_and_uses_mean_iou():
    # Two pairs pull in opposite directions; the grid point maximising the mean wins.
    pairs = [((0.0, 10.0), (1.0, 10.0)), ((20.0, 30.0), (20.0, 30.0))]
    pre, post = calibrate_padding(pairs, [0.0, 1.0], [0.0])
    assert (pre, post) == (1.0, 0.0)
    assert calibrate_padding(pairs, [0.0, 1.0], [0.0]) == (pre, post)


# --------------------------------------------------------------------------- #
# Regressions from adversarial review
# --------------------------------------------------------------------------- #

def test_numpy_and_non_iterable_ids_do_not_raise(clinic):
    np = pytest.importorskip('numpy')
    # A numpy array has no truth value; a bare int is not iterable. Neither may cost the span.
    assert select_span(clinic, yes(np.array([1, 2]), np.array([8, 9])), EvidencePolicy()) == approx_span((3.6, 4.4))
    assert select_span(clinic, yes(np.array([[1, 2]])), EvidencePolicy()) is None
    assert select_span(clinic, yes(np.array([], dtype=int)), EvidencePolicy()) is None
    assert select_span(clinic, QAResult(0, True, evidence_unit_ids=3), EvidencePolicy()) is None
    assert select_span(clinic, QAResult(0, True, [1], evidence_word_ids=8), EvidencePolicy()) == approx_span((2.4, 5.2))
    assert valid_positions(clinic, 5) == [] and valid_positions(clinic, np.array([3, 0])) == [0, 3]
    assert trim_to_focus(clinic, 5, BP_QUESTION, 5.0) == []
    assert trim_to_focus(clinic, np.array([3, 0]), BP_QUESTION, 5.0) == [3]


def test_word_with_non_string_text_does_not_break_focus_ranking():
    words = [Word(0, 1.0, 1.5, None), Word(1, 1.5, 2.0, 'blood'), Word(2, 2.0, 2.5, 7)]
    context = manual_context([unit(0, 1.0, 2.5, text='', words=words), unit(1, 3.0, 3.5, text='okay')], 10.0)
    assert trim_to_focus(context, [0, 1], BP_QUESTION, 0.4) == [0]
    policy = EvidencePolicy(mode='best_run', max_span_s=0.4, trim_filler=False)
    assert select_span(context, yes([0, 1]), policy, BP_QUESTION) == approx_span((1.0, 2.5))
    # A text-less word normalises to '' and is droppable filler at an edge, like pure punctuation.
    policy = EvidencePolicy(mode='best_run', max_span_s=0.4, trim_filler=True)
    assert select_span(context, yes([0, 1]), policy, BP_QUESTION) == approx_span((1.5, 2.5))


def test_rounding_never_pushes_end_past_the_audio():
    for duration in (14.2006, 3.9996, 1.00051, 10.0004):
        context = manual_context([unit(0, 0.0, duration)], duration)
        start, end = select_span(context, yes([0]), EvidencePolicy())
        assert 0.0 <= start < end <= duration
        assert round(end, 3) == end


def test_evidence_entirely_outside_the_audio_gives_none_even_with_min_span():
    # Before: a unit past the end collapsed onto the edge and min_span_s grew (9.8, 10.0) out of it.
    assert select_span(manual_context([unit(0, 11.0, 12.0)], 10.0), yes([0]), EvidencePolicy()) is None
    assert select_span(manual_context([unit(0, -2.0, -1.0)], 10.0), yes([0]), EvidencePolicy()) is None
    # Touching the audio, or reaching into it, still yields a span.
    assert select_span(manual_context([unit(0, 10.0, 10.0)], 10.0), yes([0]), EvidencePolicy()) == approx_span((9.8, 10.0))
    assert select_span(manual_context([unit(0, -2.0, 1.0)], 10.0), yes([0]), EvidencePolicy()) == approx_span((0.0, 1.0))
    # Padding can bring outside evidence back in; that counts as inside.
    assert select_span(manual_context([unit(0, 11.0, 12.0)], 10.0), yes([0]), EvidencePolicy(pad_pre_s=2.0)) == approx_span((9.0, 10.0))


def test_max_span_narrowing_stays_inside_the_word_override(clinic):
    # Override: penicillin 100 milligrams (3.6-4.8, u1) + take it (5.6-6.4, u2). Selected units u1-u3.
    # Before: narrowing ranked all selected units, picked u3 for the blood-pressure question, found none
    # of the named words there and fell back to u3's own words -> (8.0, 10.4), speech the backend never named.
    result = yes([1, 2, 3], word_ids=[8, 9, 10, 12, 13])
    span = select_span(clinic, result, EvidencePolicy(max_span_s=2.0), BP_QUESTION)
    assert span == approx_span((5.6, 6.4))
    assert 3.6 <= span[0] and span[1] <= 6.4
    # Without a limit the override span is returned as is; with a limit and matching words it narrows to them.
    assert select_span(clinic, result, EvidencePolicy(max_span_s=None), BP_QUESTION) == approx_span((3.6, 6.4))
    result = yes([1, 2, 3], word_ids=[8, 9, 10, 20, 21, 22])  # + 130 over 85 (9.2-10.4, u3)
    assert select_span(clinic, result, EvidencePolicy(max_span_s=5.0), BP_QUESTION) == approx_span((9.2, 10.4))


def test_calibrate_padding_skips_malformed_pairs():
    pairs = [((1.0, 4.0), (1.3, 3.8)), None, ((1.0, 2.0), (1.0, 2.0), 3), 'ab', 5, ((1.0, 2.0),)]
    assert calibrate_padding(pairs, [0.0, 0.3], [0.0, 0.2]) == (0.3, 0.2)
    assert calibrate_padding([None, 5, (1, 2, 3)], [0.0]) == (0.0, 0.0)
