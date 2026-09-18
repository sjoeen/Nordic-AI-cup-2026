"""Tests for transcript.py: pure functions, no models, no network."""

import copy
import json
import math

import pytest

from core_types import AudioContext, TranscriptUnit, Word, check_units
from tests.conftest import make_context
from transcript import (
    Window,
    assign_word_ids,
    build_context,
    ends_sentence,
    load_context,
    normalize_text,
    resegment,
    sanitize_units,
    save_context,
    tokenize,
    unit_index,
    windows,
    word_ids_valid,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def whisper_unit(unit_id, spec, first_word_id=0, start=None, end=None, **extra):
    """Whisper-shaped unit: word texts get the leading space whisper emits.

    ``spec`` is a list of ``(text, start, end)``; the unit spans the words
    unless ``start``/``end`` are given (to simulate words outside a segment).
    """
    words = [
        Word(word_id=first_word_id + i, start_s=s, end_s=e, text=' ' + t, probability=0.9)
        for i, (t, s, e) in enumerate(spec)
    ]
    return TranscriptUnit(
        unit_id=unit_id,
        start_s=spec[0][1] if start is None else start,
        end_s=spec[-1][2] if end is None else end,
        text=''.join(w.text for w in words),
        words=words,
        **extra,
    )


def continuous(texts, start=0.0, step=1.0, first_word_id=0, unit_id=0):
    """Back-to-back words (no gaps), ``step`` seconds each."""
    spec = [(t, start + i * step, start + (i + 1) * step) for i, t in enumerate(texts)]
    return whisper_unit(unit_id, spec, first_word_id=first_word_id)


def all_word_ids(units):
    return [w.word_id for u in units for w in u.words]


# --------------------------------------------------------------------------- #
# normalize_text / tokenize
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('raw, expected', [
    ('Hello World', 'hello world'),
    ('0.3 mg.', '0.3 mg'),
    ('130/85', '130/85'),
    ('blood pressure was 130/85, fine.', 'blood pressure was 130/85 fine'),
    ('1,000 units', '1,000 units'),
    ('at 10:30.', 'at 10:30'),
    ("don't", 'dont'),
    ('Don’t', 'dont'),                     # curly apostrophe
    ("patient's", 'patients'),
    ("5'11\"", '5 11'),                  # apostrophe between digits separates
    ("the '90s", 'the 90s'),
    ('“quoted” text', 'quoted text'),  # curly double quotes
    ('follow-up', 'follow up'),
    ('well—known', 'well known'),          # em dash
    ('ＡＢＣ', 'abc'),              # full-width letters via NFKC
    ('penicillin, 100mg!', 'penicillin 100mg'),
    ('take 2. 3 times', 'take 2 3 times'),      # '.' not between digits
    ('0.3.', '0.3'),
    ('  lots   of\tspace\n', 'lots of space'),
    ('café', 'café'),
    ('', ''),
    (None, ''),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent():
    raw = 'Don’t take 0.3 mg; BP 130/85 — follow-up at 10:30!'
    once = normalize_text(raw)
    assert normalize_text(once) == once


def test_tokenize_keeps_numbers_whole():
    assert tokenize('Take 0.3 mg twice; BP 130/85.') == ['take', '0.3', 'mg', 'twice', 'bp', '130/85']
    assert tokenize('') == []
    assert tokenize(None) == []


# --------------------------------------------------------------------------- #
# ends_sentence
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('word, expected', [
    (' today.', True),
    (' today?', True),
    (' today!', True),
    (' today;', True),
    (' today?"', True),     # closing quote after the mark
    (' ...', True),
    (' well…', True),       # unicode ellipsis, NFKC -> "..."
    (' 好！', True),         # full-width exclamation mark
    (' today', False),
    (' today,', False),
    (' 0.3', False),
    (' Dr.', False),        # abbreviation
    (' e.g.', False),
    (' today:', False),     # colon is opt-in
    ('', False),
    (None, False),
])
def test_ends_sentence(word, expected):
    assert ends_sentence(word) is expected


def test_ends_sentence_with_colon_opt_in():
    assert ends_sentence(' today:', '.?!;:') is True


# --------------------------------------------------------------------------- #
# resegment: cutting rules
# --------------------------------------------------------------------------- #

def test_resegment_splits_on_punctuation():
    unit = whisper_unit(7, [
        ('Good', 0.0, 0.3), ('morning.', 0.3, 0.8),
        ('What', 0.9, 1.1), ('brings', 1.1, 1.4), ('you', 1.4, 1.5), ('in', 1.5, 1.6),
        ('today?', 1.6, 2.0), ('Sore', 2.1, 2.4), ('throat.', 2.4, 2.9),
    ])
    out = resegment([unit])
    assert [u.text for u in out] == ['Good morning.', 'What brings you in today?', 'Sore throat.']
    assert [u.unit_id for u in out] == [0, 1, 2]
    assert [(u.start_s, u.end_s) for u in out] == [(0.0, 0.8), (0.9, 2.0), (2.1, 2.9)]
    assert all_word_ids(out) == list(range(9))


def test_resegment_can_disable_punctuation_splits():
    unit = whisper_unit(0, [('Yes.', 0.0, 0.3), ('Okay.', 0.3, 0.6), ('Fine.', 0.6, 0.9)])
    assert len(resegment([unit], split_on_punct=False)) == 1
    assert len(resegment([unit])) == 3


def test_resegment_does_not_split_after_abbreviation():
    unit = whisper_unit(0, [('Dr.', 0.0, 0.3), ('Smith', 0.3, 0.7), ('is', 0.7, 0.8), ('here.', 0.8, 1.2)])
    out = resegment([unit])
    assert [u.text for u in out] == ['Dr. Smith is here.']


def test_resegment_colon_is_opt_in():
    unit = whisper_unit(0, [('Plan:', 0.0, 0.4), ('rest', 0.5, 0.8), ('and', 0.8, 0.9), ('fluids.', 0.9, 1.4)])
    assert len(resegment([unit])) == 1
    assert [u.text for u in resegment([unit], split_on_colon=True)] == ['Plan:', 'rest and fluids.']


def test_resegment_splits_on_pause():
    unit = whisper_unit(0, [
        ('so', 0.0, 0.2), ('we', 0.2, 0.4), ('wait', 0.4, 0.8),
        ('then', 2.0, 2.3), ('continue', 2.3, 2.9),          # 1.2 s pause before 'then'
    ])
    out = resegment([unit], pause_gap_s=0.7)
    assert [u.text for u in out] == ['so we wait', 'then continue']
    assert [(u.start_s, u.end_s) for u in out] == [(0.0, 0.8), (2.0, 2.9)]


def test_resegment_short_pause_does_not_split():
    unit = whisper_unit(0, [('so', 0.0, 0.2), ('we', 0.2, 0.4), ('wait', 0.9, 1.2)])  # 0.5 s gap
    assert len(resegment([unit], pause_gap_s=0.7)) == 1
    assert len(resegment([unit], pause_gap_s=0.4)) == 2


def test_resegment_pause_splitting_can_be_disabled():
    unit = whisper_unit(0, [('so', 0.0, 0.2), ('wait', 5.0, 5.3)])
    assert len(resegment([unit], pause_gap_s=None)) == 1


# --------------------------------------------------------------------------- #
# resegment: max length
# --------------------------------------------------------------------------- #

def test_resegment_enforces_max_length_with_balanced_ties():
    unit = continuous([f'w{i}' for i in range(10)])  # 10 s, no gaps, no punctuation
    out = resegment([unit], max_unit_s=4.0, pause_gap_s=None)
    assert all(u.duration_s <= 4.0 for u in out)
    # 10 -> 5 + 5 (middle cut) -> (2 + 3) + (2 + 3): the tie rule keeps halves balanced.
    assert [len(u.words) for u in out] == [2, 3, 2, 3]
    assert [u.unit_id for u in out] == [0, 1, 2, 3]
    assert all_word_ids(out) == list(range(10))
    assert [u.text for u in out] == ['w0 w1', 'w2 w3 w4', 'w5 w6', 'w7 w8 w9']


def test_resegment_max_length_cuts_at_widest_gap():
    spec = [(f'w{i}', float(i), float(i + 1)) for i in range(3)]              # 0-3
    spec += [(f'w{i}', i + 0.3, i + 1.3) for i in range(3, 9)]              # 3.3-9.3 (0.3 s gap)
    unit = whisper_unit(0, spec)
    assert unit.duration_s > 8.0
    out = resegment([unit], max_unit_s=8.0, pause_gap_s=0.7)
    assert [len(u.words) for u in out] == [3, 6]
    assert out[0].end_s == 3.0 and out[1].start_s == 3.3


def test_resegment_single_long_word_is_kept():
    unit = whisper_unit(0, [('uhhhhh', 0.0, 12.0)])
    out = resegment([unit], max_unit_s=8.0)
    assert len(out) == 1 and out[0].duration_s == 12.0


def test_resegment_max_length_can_be_disabled():
    unit = continuous([f'w{i}' for i in range(20)])
    assert len(resegment([unit], max_unit_s=None, pause_gap_s=None)) == 1
    assert len(resegment([unit], max_unit_s=0, pause_gap_s=None)) == 1
    assert len(resegment([unit], max_unit_s=float('inf'), pause_gap_s=None)) == 1


@pytest.mark.parametrize('bad', ['8', b'8', True, [8.0], {'s': 8}])
def test_resegment_rejects_non_numeric_limits(bad):
    """A textual limit used to disable the rule silently (a 20 s unit survived
    whole); it must fail loudly instead, like an unknown kwarg does."""
    unit = continuous([f'w{i}' for i in range(20)])
    with pytest.raises(TypeError):
        resegment([unit], max_unit_s=bad)
    with pytest.raises(TypeError):
        resegment([unit], pause_gap_s=bad)
    with pytest.raises(TypeError):
        build_context([unit], 'sha', 20.0, 'm', 'h', resegment_kwargs={'max_unit_s': bad})


@pytest.mark.parametrize('bad', ['false', 0, 1, None])
def test_resegment_rejects_non_bool_flags(bad):
    unit = whisper_unit(0, [('Yes.', 0.0, 0.3), ('Okay.', 0.3, 0.6)])
    with pytest.raises(TypeError):
        resegment([unit], split_on_punct=bad)
    with pytest.raises(TypeError):
        resegment([unit], split_on_colon=bad)


def test_resegment_accepts_numpy_limits():
    np = pytest.importorskip('numpy')
    unit = continuous([f'w{i}' for i in range(10)])
    out = resegment([unit], max_unit_s=np.float32(4.0), pause_gap_s=np.float64(0.7))
    assert [len(u.words) for u in out] == [2, 3, 2, 3]


# --------------------------------------------------------------------------- #
# resegment: ids, order, edge cases
# --------------------------------------------------------------------------- #

def test_resegment_renumbers_units_in_time_order():
    late = whisper_unit(5, [('later.', 10.0, 10.5)], first_word_id=1)
    early = whisper_unit(2, [('first.', 0.0, 0.5)], first_word_id=0)
    out = resegment([late, early])
    assert [(u.unit_id, u.text) for u in out] == [(0, 'first.'), (1, 'later.')]


def test_resegment_preserves_valid_word_ids():
    a = whisper_unit(0, [('one.', 0.0, 0.5), ('two', 0.6, 0.9)], first_word_id=100)
    b = whisper_unit(1, [('three.', 1.0, 1.5)], first_word_id=102)
    out = resegment([a, b])
    assert all_word_ids(out) == [100, 101, 102]
    assert len(out) == 3


def test_resegment_reassigns_duplicated_word_ids():
    a = whisper_unit(0, [('one', 0.0, 0.5), ('two.', 0.6, 0.9)])
    b = whisper_unit(1, [('three', 1.0, 1.5)])  # word ids restart at 0: duplicates
    assert not word_ids_valid([a, b])
    out = resegment([a, b])
    assert all_word_ids(out) == [0, 1, 2]
    assert word_ids_valid(out)


def test_resegment_empty_input():
    assert resegment([]) == []


def test_resegment_keeps_unit_without_words_unchanged():
    bare = TranscriptUnit(unit_id=9, start_s=1.0, end_s=20.0, text='  raw segment text ')
    out = resegment([bare], max_unit_s=8.0)
    assert len(out) == 1
    assert out[0].text == '  raw segment text '
    assert (out[0].start_s, out[0].end_s) == (1.0, 20.0)
    assert out[0].unit_id == 0 and out[0].words == []


def test_resegment_rebuilds_text_from_words_and_copies_metadata():
    unit = whisper_unit(0, [('hello', 0.0, 0.3), ('there.', 0.3, 0.6)],
                        avg_logprob=-0.2, no_speech_prob=0.01, speaker='doctor')
    assert unit.text == ' hello there.'
    out = resegment([unit])
    assert out[0].text == 'hello there.'
    assert (out[0].avg_logprob, out[0].no_speech_prob, out[0].speaker) == (-0.2, 0.01, 'doctor')


def test_resegment_skips_empty_word_texts_in_unit_text():
    unit = whisper_unit(0, [('hello', 0.0, 0.3), ('', 0.3, 0.3), ('there.', 0.3, 0.6)])
    out = resegment([unit])
    assert out[0].text == 'hello there.'
    assert len(out[0].words) == 3  # the empty word keeps its id and timing


def test_resegment_does_not_mutate_input():
    units = [whisper_unit(3, [('a.', 0.0, 0.2), ('b', 0.3, 0.5)], first_word_id=5,
                          start=-0.5, end=9.0)]
    units[0].words.append(Word(word_id=7, start_s=float('nan'), end_s=0.7, text=' c'))
    before = copy.deepcopy(units)
    resegment(units)
    assert units == before


def test_resegment_is_stable_on_clean_clause_units(clinic_context):
    out = resegment(clinic_context.units)
    assert [u.text for u in out] == [u.text for u in clinic_context.units]
    assert [(u.start_s, u.end_s) for u in out] == [(u.start_s, u.end_s) for u in clinic_context.units]
    assert all_word_ids(out) == all_word_ids(clinic_context.units)
    assert check_units(out, clinic_context.duration_s) == []


# --------------------------------------------------------------------------- #
# sanitize_units / assign_word_ids / word_ids_valid
# --------------------------------------------------------------------------- #

def test_sanitize_clamps_words_into_segment_and_reports():
    unit = whisper_unit(0, [('a', -0.2, 0.3), ('b', 2.5, 3.6)], start=0.0, end=3.0)
    clean, problems = sanitize_units([unit])
    assert [(w.start_s, w.end_s) for w in clean[0].words] == [(0.0, 0.3), (2.5, 3.0)]
    assert any('word 0' in p and 'clamped' in p for p in problems)
    assert any('word 1' in p and 'clamped' in p for p in problems)
    assert check_units(clean) == []


def test_sanitize_word_end_before_start_becomes_zero_length():
    unit = whisper_unit(0, [('a', 2.0, 1.0)], start=0.0, end=3.0)
    clean, problems = sanitize_units([unit])
    assert (clean[0].words[0].start_s, clean[0].words[0].end_s) == (2.0, 2.0)
    assert any('end before start' in p for p in problems)


def test_sanitize_non_finite_word_timing():
    unit = whisper_unit(0, [('a', 0.0, 0.5), ('b', float('nan'), 0.9), ('c', 1.0, float('inf'))],
                        start=0.0, end=2.0)
    clean, problems = sanitize_units([unit])
    times = [(w.start_s, w.end_s) for w in clean[0].words]
    assert times[1] == (0.5, 0.9)   # lost start -> previous word end
    assert times[2] == (1.0, 1.0)   # lost end -> own start
    assert all(math.isfinite(t) for pair in times for t in pair)
    assert sum('non-finite timing' in p for p in problems) == 2


def test_sanitize_non_finite_unit_bounds():
    nan = float('nan')
    with_words = TranscriptUnit(0, nan, 5.0, ' x', words=[Word(0, 1.0, 2.0, ' x')])
    bare = TranscriptUnit(1, nan, nan, 'silence')
    reversed_unit = TranscriptUnit(2, 8.0, 6.0, 'flipped')
    clean, problems = sanitize_units([with_words, bare, reversed_unit])
    assert (clean[0].start_s, clean[0].end_s) == (1.0, 5.0)
    assert (clean[1].start_s, clean[1].end_s) == (0.0, 0.0)
    assert (clean[2].start_s, clean[2].end_s) == (6.0, 8.0)
    assert len(problems) == 3
    # Sanitising is per unit: each one is internally sane (ordering between
    # units is resegment's job).
    assert all(check_units([unit]) == [] for unit in clean)


def test_sanitize_coerces_unit_metadata_for_json():
    """numpy scalars and NaN in avg_logprob/no_speech_prob used to survive into
    the context and make save_context raise; they are now plain floats or None."""
    np = pytest.importorskip('numpy')
    unit = TranscriptUnit(
        0, np.float32(0.0), np.float64(1.0), ' hi.',
        words=[Word(np.int64(0), np.float32(0.0), np.float64(1.0), ' hi.', np.float32(0.5))],
        avg_logprob=np.float32(-0.25), no_speech_prob=float('nan'), speaker='doctor',
    )
    clean, problems = sanitize_units([unit])
    assert type(clean[0].avg_logprob) is float and clean[0].avg_logprob == pytest.approx(-0.25)
    assert clean[0].no_speech_prob is None
    assert type(clean[0].start_s) is float and type(clean[0].words[0].word_id) is int
    assert type(clean[0].words[0].probability) is float
    assert problems == ['unit 0: unusable no_speech_prob dropped']
    context = build_context([unit], 'sha', np.float64(2.0), 'm', 'h',
                            resegment_kwargs={'max_unit_s': np.float32(8.0), 'pause_gap_s': float('inf')})
    payload = json.loads(context.to_json(allow_nan=False))  # would raise before the fix
    assert payload['diagnostics']['resegment'] == {
        'max_unit_s': 8.0, 'pause_gap_s': None, 'split_on_punct': True, 'split_on_colon': False}
    assert any('no_speech_prob' in p for p in context.diagnostics['input_anomalies'])


def test_sanitize_keeps_none_metadata_silently():
    unit = whisper_unit(0, [('a.', 0.0, 0.5)])
    clean, problems = sanitize_units([unit])
    assert (clean[0].avg_logprob, clean[0].no_speech_prob, clean[0].speaker) == (None, None, None)
    assert problems == []


def test_sanitize_clean_input_reports_nothing():
    unit = whisper_unit(0, [('a', 0.0, 0.5), ('b.', 0.5, 1.0)])
    clean, problems = sanitize_units([unit])
    assert problems == []
    assert clean == [unit]


def test_word_ids_valid():
    ok = [whisper_unit(0, [('a', 0.0, 0.1)]), whisper_unit(1, [('b', 0.2, 0.3)], first_word_id=1)]
    assert word_ids_valid(ok)
    assert word_ids_valid([])
    dup = [whisper_unit(0, [('a', 0.0, 0.1)]), whisper_unit(1, [('b', 0.2, 0.3)])]
    assert not word_ids_valid(dup)
    missing = [TranscriptUnit(0, 0.0, 0.1, ' a', words=[Word(None, 0.0, 0.1, ' a')])]
    assert not word_ids_valid(missing)
    boolean = [TranscriptUnit(0, 0.0, 0.1, ' a', words=[Word(True, 0.0, 0.1, ' a')])]
    assert not word_ids_valid(boolean)


def test_assign_word_ids_global_in_time_order():
    late = whisper_unit(0, [('c', 5.0, 5.2), ('d', 5.2, 5.4)])
    early = whisper_unit(1, [('a', 0.0, 0.2), ('b', 0.2, 0.4)])
    out = assign_word_ids([late, early])
    assert [u.unit_id for u in out] == [1, 0]  # unit ids untouched, order is by time
    assert all_word_ids(out) == [0, 1, 2, 3]
    assert [w.text for u in out for w in u.words] == [' a', ' b', ' c', ' d']
    assert late.words[0].word_id == 0 and early.words[0].word_id == 0  # input untouched


# --------------------------------------------------------------------------- #
# build_context
# --------------------------------------------------------------------------- #

def test_build_context_diagnostics_and_fields():
    raw = [whisper_unit(0, [('Good', 0.0, 0.3), ('morning.', 0.3, 0.8), ('Hi.', 1.0, 1.4)])]
    context = build_context(raw, 'abc123', 30.0, 'faster-whisper/small', 'cfg-hash')
    assert isinstance(context, AudioContext)
    assert (context.audio_sha256, context.asr_model_id, context.asr_config_hash) == (
        'abc123', 'faster-whisper/small', 'cfg-hash')
    assert context.duration_s == 30.0
    assert [u.text for u in context.units] == ['Good morning.', 'Hi.']
    diag = context.diagnostics
    assert diag['n_raw_units'] == 1 and diag['n_units'] == 2 and diag['n_words'] == 3
    assert diag['unit_anomalies'] == [] and diag['input_anomalies'] == []
    assert diag['resegment'] == {'max_unit_s': 8.0, 'pause_gap_s': 0.7,
                                 'split_on_punct': True, 'split_on_colon': False}


def test_build_context_records_anomalies():
    raw = [
        whisper_unit(0, [('a', 0.0, 0.5), ('b', 0.4, 12.0)], start=0.0, end=10.0),
        whisper_unit(0, [('c.', 9.0, 9.5)]),  # overlaps previous, word id 0 duplicated
    ]
    context = build_context(raw, 'sha', 9.0, 'm', 'h')
    diag = context.diagnostics
    assert any('clamped' in p for p in diag['input_anomalies'])
    assert any('word ids' in p for p in diag['input_anomalies'])
    assert any('overlaps' in p for p in diag['unit_anomalies'])
    assert any('ends after audio duration' in p for p in diag['unit_anomalies'])
    assert all_word_ids(context.units) == [0, 1, 2]


def test_build_context_passes_resegment_kwargs():
    raw = [whisper_unit(0, [('Plan:', 0.0, 0.4), ('rest.', 0.5, 0.8)])]
    context = build_context(raw, 'sha', 1.0, 'm', 'h', resegment_kwargs={'split_on_colon': True})
    assert [u.text for u in context.units] == ['Plan:', 'rest.']
    assert context.diagnostics['resegment']['split_on_colon'] is True


def test_build_context_rejects_unknown_resegment_kwargs():
    with pytest.raises(TypeError):
        build_context([], 'sha', 1.0, 'm', 'h', resegment_kwargs={'bogus': 1})


@pytest.mark.parametrize('bad_duration', [float('nan'), float('inf'), -1.0, None])
def test_build_context_replaces_unusable_duration(bad_duration):
    raw = [whisper_unit(0, [('a.', 0.0, 0.5), ('b.', 1.0, 2.5)])]
    context = build_context(raw, 'sha', bad_duration, 'm', 'h')
    assert context.duration_s == 2.5
    assert any('duration_s' in p for p in context.diagnostics['input_anomalies'])


def test_build_context_empty_input():
    context = build_context([], 'sha', 12.0, 'm', 'h')
    assert context.units == [] and context.duration_s == 12.0
    assert context.diagnostics['n_units'] == 0 and context.diagnostics['n_words'] == 0
    assert windows(context) == []
    assert unit_index(context) == {}


# --------------------------------------------------------------------------- #
# windows / unit_index
# --------------------------------------------------------------------------- #

def test_windows_count_and_order_for_four_units():
    context = make_context(['a b', 'c d', 'e f', 'g h'])
    result = windows(context, max_units=3)
    assert len(result) == 4 + 3 + 2
    assert [w.unit_ids for w in result] == [
        (0,), (0, 1), (0, 1, 2), (1,), (1, 2), (1, 2, 3), (2,), (2, 3), (3,)]
    assert result[1].text == 'a b c d'
    assert result[1].start_s == context.units[0].start_s
    assert result[1].end_s == context.units[1].end_s
    assert result[1].duration_s == pytest.approx(result[1].end_s - result[1].start_s)
    assert all(isinstance(w, Window) for w in result)


def test_windows_max_units_beyond_context_length():
    context = make_context(['a', 'b'])
    assert [w.unit_ids for w in windows(context, max_units=5)] == [(0,), (0, 1), (1,)]
    assert [w.unit_ids for w in windows(context, max_units=1)] == [(0,), (1,)]


def test_windows_skip_empty_unit_text():
    context = AudioContext('sha', 3.0, units=[
        TranscriptUnit(0, 0.0, 1.0, 'hello'),
        TranscriptUnit(1, 1.0, 2.0, '   '),
        TranscriptUnit(2, 2.0, 3.0, 'world'),
    ])
    assert windows(context, max_units=3)[2].text == 'hello world'


def test_windows_tolerates_none_unit_text():
    context = AudioContext('sha', 2.0, units=[
        TranscriptUnit(0, 0.0, 1.0, None),
        TranscriptUnit(1, 1.0, 2.0, 'b'),
    ])
    assert [w.text for w in windows(context, max_units=2)] == ['', 'b', 'b']


def test_windows_rejects_invalid_max_units():
    with pytest.raises(ValueError):
        windows(make_context(['a']), max_units=0)


def test_windows_on_clinic_context(clinic_context):
    assert len(windows(clinic_context, max_units=3)) == 8 + 7 + 6


def test_unit_index_first_occurrence_wins():
    context = AudioContext('sha', 3.0, units=[
        TranscriptUnit(4, 0.0, 1.0, 'a'),
        TranscriptUnit(2, 1.0, 2.0, 'b'),
        TranscriptUnit(4, 2.0, 3.0, 'c'),
    ])
    assert unit_index(context) == {4: 0, 2: 1}


# --------------------------------------------------------------------------- #
# save_context / load_context
# --------------------------------------------------------------------------- #

def test_save_and_load_round_trip(tmp_path):
    raw = [whisper_unit(0, [('Café', 0.0, 0.3), ('open.', 0.3, 0.8), ('Yes', 1.7, 2.0)],
                        avg_logprob=-0.1, no_speech_prob=0.02, speaker='patient')]
    context = build_context(raw, 'deadbeef', 5.5, 'faster-whisper/small', 'cfg')
    path = tmp_path / 'nested' / 'dir' / 'deadbeef.json'
    assert save_context(context, path) == path
    assert path.exists()
    assert json.loads(path.read_text(encoding='utf-8'))['audio_sha256'] == 'deadbeef'
    assert [p.name for p in path.parent.iterdir()] == ['deadbeef.json']  # no temp files left
    loaded = load_context(str(path))
    assert loaded == context
    assert loaded.to_dict() == context.to_dict()
    assert loaded.units[0].words[0].text == ' Café'
    assert loaded.diagnostics == context.diagnostics


def test_save_context_refuses_nan(tmp_path):
    context = AudioContext('sha', float('nan'), units=[])
    path = tmp_path / 'bad.json'
    with pytest.raises(ValueError):
        save_context(context, path)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_load_context_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_context(tmp_path / 'missing.json')


# --------------------------------------------------------------------------- #
# Regression tests from review
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('raw, expected', [
    ('İstanbul', 'i̇stanbul'),        # lowercase İ is "i" + combining dot: one token, not two
    ('नमस्ते', 'नमस्ते'),             # Devanagari vowel signs are combining marks
    ('สวัสดี', 'สวัสดี'),             # Thai tone/vowel marks likewise
    ('x́ y', 'x́ y'),              # a mark with no precomposed form stays attached
])
def test_normalize_text_keeps_combining_marks(raw, expected):
    """Combining marks are not alphanumeric; treating them as punctuation used
    to cut every accented or Indic word into fragments."""
    assert normalize_text(raw) == expected
    assert len(tokenize(raw)) == len(raw.split())


def test_build_context_fallback_duration_is_never_negative():
    """With an unusable duration and all-negative unit timings the fallback
    used to be a negative recording length."""
    raw = [TranscriptUnit(0, -2.0, -1.0, ' a', words=[Word(0, -2.0, -1.0, ' a')])]
    context = build_context(raw, 'sha', float('nan'), 'm', 'h')
    assert context.duration_s == 0.0
    assert any('duration_s' in p for p in context.diagnostics['input_anomalies'])


def test_resegment_negative_pause_gap_disables_pause_rule():
    """A negative gap used to cut after every word (every gap is > -1)."""
    unit = continuous([f'w{i}' for i in range(4)], step=0.25)
    assert len(resegment([unit], pause_gap_s=-1.0)) == 1
    assert len(resegment([unit], max_unit_s=-8.0, pause_gap_s=None)) == 1
    # Zero keeps its meaning: any positive gap is a pause, a zero gap is not.
    spaced = whisper_unit(0, [('a', 0.0, 0.2), ('b', 0.3, 0.5), ('c', 0.5, 0.7)])
    assert [u.text for u in resegment([spaced], pause_gap_s=0.0)] == ['a', 'b c']


def test_build_context_diagnostics_report_disabled_limits_as_none():
    """``max_unit_s=0`` disables the cap; the diagnostics said ``0.0``, which
    reads as "cap at zero seconds"."""
    raw = [continuous([f'w{i}' for i in range(20)])]
    context = build_context(raw, 'sha', 20.0, 'm', 'h',
                            resegment_kwargs={'max_unit_s': 0, 'pause_gap_s': -5.0})
    assert len(context.units) == 1
    assert context.diagnostics['resegment']['max_unit_s'] is None
    assert context.diagnostics['resegment']['pause_gap_s'] is None


@pytest.mark.parametrize('bad', [3.0, True, '3', None])
def test_windows_rejects_non_integer_max_units(bad):
    """``True`` used to mean 1 silently and ``3.0`` failed inside ``range``."""
    with pytest.raises(TypeError):
        windows(make_context(['a', 'b']), max_units=bad)


def test_ends_sentence_ignores_am_pm():
    assert ends_sentence(' a.m.') is False
    assert ends_sentence(' p.m.') is False
    unit = whisper_unit(0, [('at', 0.0, 0.2), ('9', 0.2, 0.4), ('a.m.', 0.4, 0.8), ('tomorrow.', 0.8, 1.3)])
    assert [u.text for u in resegment([unit])] == ['at 9 a.m. tomorrow.']


def test_save_context_respects_umask(tmp_path):
    """The temp file behind the atomic write is created owner-only (0600); the
    cache file must end up with the permissions a plain open() would give,
    or a cache written by an offline tool is unreadable by the service user."""
    import os
    import stat
    old = os.umask(0o022)
    try:
        path = save_context(make_context(['a']), tmp_path / 'ctx.json')
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o644
    finally:
        os.umask(old)


def test_resegment_interleaves_overlapping_segments_and_keeps_word_ids():
    """Whisper occasionally emits overlapping segments; pieces are ordered by
    time across segments while word ids stay the ASR's."""
    a = whisper_unit(0, [('a', 0.0, 1.0), ('b.', 1.0, 2.0), ('c', 8.0, 9.0)], first_word_id=0)
    b = whisper_unit(1, [('d', 3.0, 4.0), ('e', 4.0, 5.0)], first_word_id=3)
    out = resegment([a, b])
    assert [(u.unit_id, u.text, u.start_s) for u in out] == [(0, 'a b.', 0.0), (1, 'd e', 3.0), (2, 'c', 8.0)]
    assert all_word_ids(out) == [0, 1, 3, 4, 2]
    assert check_units(out) == []
