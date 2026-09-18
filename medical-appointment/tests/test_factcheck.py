"""Tests for ``factcheck`` (contract A5): quantity extraction, comparison,
fuzzy name matching and entity-term detection. Pure Python, no models.

The extraction table is the specification: each row is one surface form the
ASR or a question can produce and the (family, value, unit) triples that must
come out of it, in reading order. Values are in the family's base unit (mg,
days, per day, ml, kg, cm, Celsius), which is what ``compare`` uses.
"""

import math
import time
from dataclasses import FrozenInstanceError
from difflib import SequenceMatcher

import pytest

from factcheck import (
    FAMILIES,
    STATUS_PRECEDENCE,
    FactCheckVerdict,
    Quantity,
    compare,
    entity_terms,
    extract_quantities,
    fuzzy_contains,
)

DAY = 1.0
HOUR = 1.0 / 24.0
MINUTE = 1.0 / 1440.0
SECOND = 1.0 / 86400.0


def triples(text):
    """(family, value, unit) for every quantity, rounded for table comparison."""
    return [(q.family, round(q.value, 6), q.unit) for q in extract_quantities(text)]


def expect(rows):
    return [(fam, round(val, 6), unit) for fam, val, unit in rows]


# --------------------------------------------------------------------------- #
# Extraction table: every family, digits, number words, aliases, ASR quirks
# --------------------------------------------------------------------------- #

EXTRACTION_CASES = [
    # dose: mass aliases and ASR spellings; values normalised to mg
    ('100mg', [('dose', 100, 'mg')]),
    ('100 milligram', [('dose', 100, 'mg')]),
    ('one hundred milligrams', [('dose', 100, 'mg')]),
    ('0.15mg', [('dose', 0.15, 'mg')]),
    ('one hundred and fifty micrograms', [('dose', 0.15, 'mcg')]),
    ('two hundred and fifty milligrams', [('dose', 250, 'mg')]),
    ('twenty-five mg', [('dose', 25, 'mg')]),
    ('a hundred mg', [('dose', 100, 'mg')]),
    ('hundred milligrams', [('dose', 100, 'mg')]),
    ('1,000 mg', [('dose', 1000, 'mg')]),
    ('.5 mg', [('dose', 0.5, 'mg')]),
    ('zero point three milligrams', [('dose', 0.3, 'mg')]),
    ('500 mcg', [('dose', 0.5, 'mcg')]),
    ('100µg', [('dose', 0.1, 'mcg')]),
    ('0.5 grams', [('dose', 500, 'g')]),
    ('one thousand two hundred mg', [('dose', 1200, 'mg')]),
    ('1,5 mg', [('dose', 1.5, 'mg')]),                      # decimal comma (Norwegian ASR)
    ('1,50 mg', [('dose', 1.5, 'mg')]),
    ('1,500 mg', [('dose', 1500, 'mg')]),                   # thousands separator
    ('nought point five milligrams', [('dose', 0.5, 'mg')]),
    ('1 g', [('dose', 1000, 'g')]),                          # written apart, still a gram
    ('1½ mg', [('dose', 1.5, 'mg')]),                        # mixed number
    ('one hundred thousand units', [('dose', 100000, 'iu')]),
    ('1000 IU', [('dose', 1000, 'iu')]),
    ('10 mg/kg', [('dose', 10, 'mg/kg')]),
    # volume: ml base
    ('5 mls', [('volume', 5, 'ml')]),
    ('10 cc', [('volume', 10, 'ml')]),
    ('2 litres', [('volume', 2000, 'l')]),
    ('500 millilitres', [('volume', 500, 'ml')]),
    ('five mils', [('volume', 5, 'ml')]),
    # count nouns
    ('2 puffs', [('count', 2, 'puff')]),
    ('half a tablet', [('count', 0.5, 'tablet')]),
    ('a quarter of a tablet', [('count', 0.25, 'tablet')]),
    ('one and a half tablets', [('count', 1.5, 'tablet')]),
    ('1/2 tablet', [('count', 0.5, 'tablet')]),
    ('1½ tablets', [('count', 1.5, 'tablet')]),
    ('a third of a tablet', [('count', 1 / 3, 'tablet')]),
    ('two thirds of a tablet', [('count', 2 / 3, 'tablet')]),
    ('one and a third tablets', [('count', 4 / 3, 'tablet')]),
    ('¾ tablet', [('count', 0.75, 'tablet')]),
    ('two capsules', [('count', 2, 'capsule')]),
    ('three times', [('count', 3, 'times')]),
    ('twice', [('count', 2, 'times')]),
    # percentage
    ('5 percent', [('percentage', 5, 'percent')]),
    ('99.5%', [('percentage', 99.5, 'percent')]),
    ('ninety-nine per cent', [('percentage', 99, 'percent')]),
    # pressure: systolic + diastolic, "over" is not a preposition here
    ('130 over 85', [('pressure', 130, 'mmhg'), ('pressure_dia', 85, 'mmhg')]),
    ('130/85', [('pressure', 130, 'mmhg'), ('pressure_dia', 85, 'mmhg')]),
    ('130 over 85 mmHg', [('pressure', 130, 'mmhg'), ('pressure_dia', 85, 'mmhg')]),
    ('a hundred over sixty', [('pressure', 100, 'mmhg'), ('pressure_dia', 60, 'mmhg')]),
    ('one thirty over eighty five', [('pressure', 130, 'mmhg'), ('pressure_dia', 85, 'mmhg')]),
    ('one twenty over eighty mmHg', [('pressure', 120, 'mmhg'), ('pressure_dia', 80, 'mmhg')]),
    ('one hundred and thirty over eighty five', [('pressure', 130, 'mmhg'), ('pressure_dia', 85, 'mmhg')]),
    # temperature: Celsius base, Fahrenheit converted, bare number promoted
    ('38.5 degrees', [('temperature', 38.5, 'c')]),
    ('38.5°C', [('temperature', 38.5, 'c')]),
    ('101 fahrenheit', [('temperature', (101 - 32) * 5 / 9, 'f')]),
    ('her temperature was 38.5', [('temperature', 38.5, 'c')]),
    ('a fever of 101', [('temperature', (101 - 32) * 5 / 9, 'f')]),
    ('temperature of 38 point 5', [('temperature', 38.5, 'c')]),
    ('temperature 38,5', [('temperature', 38.5, 'c')]),
    # length: cm base
    ('175 cm', [('length', 175, 'cm')]),
    ('5 mm', [('length', 0.5, 'mm')]),
    ('2 metres', [('length', 200, 'm')]),
    ('6 inches', [('length', 15.24, 'inch')]),
    # weight: kg base
    ('70 kg', [('weight', 70, 'kg')]),
    ('150 pounds', [('weight', 150 * 0.45359237, 'lb')]),
    ('150 lbs', [('weight', 150 * 0.45359237, 'lb')]),
    ('11 stone', [('weight', 11 * 6.35029318, 'stone')]),
    # duration: days base, original unit kept
    ('a week', [('duration', 7, 'week')]),
    ('an hour', [('duration', HOUR, 'hour')]),
    ('two weeks', [('duration', 14, 'week')]),
    ('14 days', [('duration', 14, 'day')]),
    ('a fortnight', [('duration', 14, 'fortnight')]),
    ('thirty seconds', [('duration', 30 * SECOND, 'second')]),
    ('45 minutes', [('duration', 45 * MINUTE, 'minute')]),
    ('3 months', [('duration', 90, 'month')]),
    ('2 years', [('duration', 730, 'year')]),
    ('half an hour', [('duration', HOUR / 2, 'hour')]),
    ('an hour and a half', [('duration', 1.5 * HOUR, 'hour')]),
    ('two and a half weeks', [('duration', 17.5, 'week')]),
    ('a two-week course', [('duration', 14, 'week')]),
    ('three quarters of an hour', [('duration', 0.75 * HOUR, 'hour')]),
    ('2½ weeks', [('duration', 17.5, 'week')]),
    ('for 45 years', [('duration', 45 * 365, 'year')]),      # a duration, unlike an age
    # frequency: per day base
    ('twice a day', [('frequency', 2, 'per_day')]),
    ('once daily', [('frequency', 1, 'per_day')]),
    ('three times a day', [('frequency', 3, 'per_day')]),
    ('every 8 hours', [('frequency', 3, 'per_day')]),
    ('every 12 hours', [('frequency', 2, 'per_day')]),
    ('every other day', [('frequency', 0.5, 'per_day')]),
    ('weekly', [('frequency', 1 / 7, 'per_day')]),
    ('twice-daily', [('frequency', 2, 'per_day')]),
    ('2x daily', [('frequency', 2, 'per_day')]),
    ('bd', [('frequency', 2, 'per_day')]),
    ('bid', [('frequency', 2, 'per_day')]),
    ('tds', [('frequency', 3, 'per_day')]),
    ('tid', [('frequency', 3, 'per_day')]),
    ('qds', [('frequency', 4, 'per_day')]),
    ('qid', [('frequency', 4, 'per_day')]),
    ('nightly', [('frequency', 1, 'per_day')]),
    ('three times a week', [('frequency', 3 / 7, 'per_day')]),
    ('once every two weeks', [('frequency', 1 / 14, 'per_day')]),
    ('3 a day', [('frequency', 3, 'per_day')]),
    ('every 2nd day', [('frequency', 0.5, 'per_day')]),
    # concentration (lab values) has its own family and never matches a dose
    ('5.5 mmol/l', [('concentration', 5.5, 'mmol/l')]),
    # plain numbers
    ('BMI of 28', [('plain', 28, None)]),
    ('two million', [('plain', 2000000, None)]),
    ('45 years old', [('plain', 45, None)]),                # ages are bare numbers, never durations
    ('a 45-year-old woman', [('plain', 45, None)]),
    ('aged 45', [('plain', 45, None)]),
    ('1 000 mg', [('plain', 1, None)]),                      # "000" is a code fragment, not 0 mg
    # combined sentences keep reading order
    ('100 mg twice daily', [('dose', 100, 'mg'), ('frequency', 2, 'per_day')]),
    ('penicillin, 100 milligrams daily', [('dose', 100, 'mg'), ('frequency', 1, 'per_day')]),
    ('Take two tablets three times a day for five days',
     [('count', 2, 'tablet'), ('frequency', 3, 'per_day'), ('duration', 5, 'day')]),
    # things that must NOT become quantities
    ('10:30', []),
    ('the first dose', []),
    ('2nd dose', []),
    ('on the 3rd of May', []),
    ('in 2024', []),
    ('covid-19', []),
    ('vitamin D', []),
    ('a second opinion', []),
    ('one of the tablets', []),
    ('no one', []),
    ('once the swelling settles', []),
    ('a tablet', []),
    ('phone 12345678', []),
    ('call 555 0100', [('plain', 555, None)]),               # leading-zero groups are never numbers
    ('a third dose', []),                                    # ordinal, not a fraction
    ('the third dose', []),
]


@pytest.mark.parametrize('text,expected', EXTRACTION_CASES, ids=[c[0] for c in EXTRACTION_CASES])
def test_extraction_table(text, expected):
    assert triples(text) == expect(expected)


def test_every_family_is_covered_by_the_table():
    seen = {fam for _, rows in EXTRACTION_CASES for fam, _, _ in rows}
    assert set(FAMILIES) <= seen


def test_raw_keeps_the_surface_form_and_unit_keeps_the_original_unit():
    (q,) = extract_quantities('for about two weeks')
    assert (q.raw, q.unit, q.value, q.qualifier) == ('two weeks', 'week', 14.0, 'approx')
    (q,) = extract_quantities('one hundred and fifty micrograms')
    assert (q.raw, q.unit) == ('one hundred and fifty micrograms', 'mcg')


def test_pressure_emits_systolic_then_diastolic_sharing_raw():
    sys_q, dia_q = extract_quantities('blood pressure was 130 over 85, which is fine')
    assert (sys_q.family, sys_q.value, sys_q.raw) == ('pressure', 130.0, '130 over 85')
    assert (dia_q.family, dia_q.value, dia_q.raw) == ('pressure_dia', 85.0, '130 over 85')


def test_over_outside_pressure_is_a_lower_bound_qualifier():
    (q,) = extract_quantities('it took over two weeks')
    assert (q.family, q.value, q.qualifier) == ('duration', 14.0, 'ge')


def test_ranges_are_a_single_quantity_with_value_hi():
    (q,) = extract_quantities('2 to 3 weeks')
    assert (q.value, q.value_hi, q.is_range) == (14.0, 21.0, True)
    (q,) = extract_quantities('between one and two weeks')
    assert (q.value, q.value_hi) == (7.0, 14.0)
    (q,) = extract_quantities('one or two tablets')
    assert (q.family, q.value, q.value_hi) == ('count', 1.0, 2.0)
    (q,) = extract_quantities('2-3 days')
    assert (q.value, q.value_hi) == (2.0, 3.0)


def test_spoken_ranges_with_a_comma_or_adjacent_numbers():
    (q,) = extract_quantities('two, three weeks')
    assert (q.value, q.value_hi, q.raw) == (14.0, 21.0, 'two, three weeks')
    (q,) = extract_quantities('three four days')
    assert (q.value, q.value_hi) == (3.0, 4.0)
    (q,) = extract_quantities('a week or two')
    assert (q.unit, q.value, q.value_hi) == ('week', 7.0, 14.0)
    (q,) = extract_quantities('a week or so')
    assert (q.value, q.value_hi, q.qualifier) == (7.0, None, 'approx')
    # not before "times": "take two, three times a day" is two tablets, thrice daily
    fams = [(x.family, x.value) for x in extract_quantities('take two, three times a day')]
    assert fams == [('plain', 2.0), ('frequency', 3.0)]
    # a different unit after "or" is its own duration
    assert [x.raw for x in extract_quantities('two weeks or three months')] == ['two weeks', 'three months']
    # without the comma only the next integer up is a range ("ten fifteen" is not)
    assert [x.family for x in extract_quantities('ten fifteen minutes')] == ['plain', 'duration']


def test_qualifier_survives_a_comma():
    assert extract_quantities('about, two weeks')[0].qualifier == 'approx'
    assert extract_quantities('less than, an hour')[0].qualifier == 'lt'


def test_ages_are_bare_numbers_with_an_age_context_word():
    for text in ['she is 45 years old', 'aged 45', 'a 45-year-old woman', 'at the age of 45', '45 years of age']:
        (q,) = extract_quantities(text)
        assert (q.family, q.value) == ('plain', 45.0), text
        assert 'age' in q.context, text
    (q,) = extract_quantities('six months old')
    assert q.family == 'plain' and abs(q.value - 0.5) < 0.02
    (q,) = extract_quantities('for 45 years')
    assert q.family == 'duration'


def test_bound_qualifiers_are_recorded():
    assert extract_quantities('less than an hour')[0].qualifier == 'lt'
    assert extract_quantities('up to 4 tablets')[0].qualifier == 'le'
    assert extract_quantities('at least three days')[0].qualifier == 'ge'
    assert extract_quantities('exactly 4 tablets')[0].qualifier is None


def test_bare_period_adverb_means_at_least_once():
    (q,) = extract_quantities('take it daily')
    assert (q.family, q.value, q.qualifier) == ('frequency', 1.0, 'ge')
    (q,) = extract_quantities('once daily')
    assert q.qualifier is None


def test_plain_numbers_carry_context_words():
    (q,) = extract_quantities('her BMI was 28')
    assert q.family == 'plain' and 'bmi' in q.context


@pytest.mark.parametrize('bad', [None, '', '   ', float('nan'), 12, 3.5, [], {}])
def test_extract_non_text_input_is_empty(bad):
    assert extract_quantities(bad) == []


def test_extraction_is_deterministic_and_in_reading_order():
    text = 'Take 500 mg twice a day for a week, then 250 mg once daily for two weeks.'
    first = extract_quantities(text)
    assert first == extract_quantities(text)
    assert [q.raw for q in first] == [
        '500 mg', 'twice a day', 'a week', '250 mg', 'once daily', 'two weeks',
    ]


def test_quantity_is_a_frozen_dataclass_with_contract_fields():
    q = Quantity(value=1.0, unit='mg', family='dose', raw='1 mg')
    assert (q.value, q.unit, q.family, q.raw) == (1.0, 'mg', 'dose', '1 mg')
    with pytest.raises(FrozenInstanceError):
        q.value = 2.0  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

COMPARE_CASES = [
    # (question, evidence, status, detail substring)
    ('100 mg', 'take 200 mg', 'contradiction', 'dose 100 mg vs evidence 200 mg'),
    ('two weeks', 'six weeks', 'contradiction', 'duration two weeks vs evidence six weeks'),
    ('two weeks', '14 days', 'consistent', 'duration two weeks matches evidence 14 days'),
    ('twice a day', 'once daily', 'contradiction', 'frequency twice a day vs evidence once daily'),
    ('0.3 mg', '0.15 mg', 'contradiction', 'dose 0.3 mg vs evidence 0.15 mg'),
    ('130/85', '130 over 85', 'consistent', 'pressure 130/85 matches evidence 130 over 85'),
    ('130/85', '140/90', 'contradiction', 'pressure 130/85 vs evidence 140/90'),
    ('130/85', '130 over 90', 'contradiction', 'pressure (diastolic) 130/85 vs evidence 130 over 90'),
    ('100 mg', 'no numbers here', 'unverifiable', 'dose 100 mg: no dose in evidence'),
    ('no numbers', 'still none', 'no_quantities', None),
    ('', '', 'no_quantities', None),
    ('a hundred milligrams', 'penicillin 100mg', 'consistent', None),
    ('150 mcg', '0.15 mg', 'consistent', None),
    ('one hundred and fifty micrograms', '0.15mg', 'consistent', None),
    ('5%', '5 percent', 'consistent', None),
    ('1000 IU', '1000 mg', 'unverifiable', None),  # IU is never a mass
    ('101 fahrenheit', '38.3 degrees', 'consistent', None),
    ('38.5 degrees', 'her temperature was 38.5', 'consistent', None),
    ('39 degrees', 'her temperature was 38.5', 'contradiction', None),
    ('70 kg', '154 pounds', 'consistent', None),
    ('70 kg', '80 kilos', 'contradiction', None),
    ('175 cm', '1.75 metres', 'consistent', None),
    ('500 ml', 'half a litre', 'consistent', None),
    ('two tablets', 'take 2 tabs', 'consistent', None),
    ('two tablets', 'take one tablet', 'contradiction', None),
    ('two tablets', 'two pills', 'consistent', None),        # synonym class
    ('two tablets', 'one capsule', 'contradiction', None),
    ('one tablet', 'two puffs', 'unverifiable', None),       # different things never contradict
    ('twice', 'two tablets', 'unverifiable', None),
    ('twice a day', 'every 12 hours', 'consistent', None),
    ('three times a day', 'every 8 hours', 'consistent', None),
    ('once a day', 'daily', 'consistent', None),
    ('daily', 'twice a day', 'consistent', None),  # bare "daily" is a floor
    ('twice a day', 'take it every day', 'consistent', None),
    ('every other day', 'once every two days', 'consistent', None),
    ('weekly', 'once a week', 'consistent', None),
    ('weekly', 'twice a week', 'consistent', None),  # bare "weekly" is a floor too
    ('once a week', 'twice a week', 'contradiction', None),
    ('an hour', '60 minutes', 'consistent', None),
    ('half an hour', '30 minutes', 'consistent', None),
    ('a fortnight', 'two weeks', 'consistent', None),
    ('three months', '90 days', 'consistent', None),
    ('a year', 'twelve months', 'consistent', None),  # 365 vs 360 days: 2 % cross-unit tolerance
    ('a year', 'eleven months', 'contradiction', None),
    ('70 kg', '70.5 kg', 'contradiction', None),  # same unit: strict 0.1 % tolerance
    # ranges are sets: any endpoint or an interior value matches
    ('two weeks', '2 to 3 weeks', 'consistent', None),
    ('three weeks', '2 to 3 weeks', 'consistent', None),
    ('2.5 weeks', '2 to 3 weeks', 'consistent', None),
    ('four weeks', '2 to 3 weeks', 'contradiction', None),
    ('2 to 3 weeks', 'two weeks', 'consistent', None),
    ('2 to 3 weeks', 'four weeks', 'contradiction', None),
    ('one or two tablets', 'two tablets', 'consistent', None),
    # hedges and bounds
    ('about two weeks', '13 days', 'consistent', None),
    ('two weeks', '13 days', 'contradiction', None),  # 7 % apart: beyond the conversion slack
    ('less than an hour', '45 minutes', 'consistent', None),
    ('less than an hour', 'two hours', 'contradiction', None),
    ('at least three days', 'five days', 'consistent', None),
    # plain numbers need a shared context word
    ('BMI of 28', 'her BMI was 28', 'consistent', None),
    ('BMI of 28', 'her BMI was 31', 'contradiction', None),
    ('BMI of 28', 'she is 28 years old', 'no_quantities', None),
    ('aged 45', 'blood pressure 130 over 85', 'no_quantities', None),
    # ASR quirks in the evidence
    ('100 mg', 'one hundred milligram', 'consistent', None),
    ('0.15 mg', 'zero point one five milligrams', 'consistent', None),
    ('twice a day', 'twice-daily', 'consistent', None),
    # -- regressions: false contradictions on true statements ---------------
    # a second same-family value in the window never overrides an exact match
    ('Will the treatment last two weeks?',
     'Take it for two weeks, and come back in six weeks if it has not settled.', 'consistent', None),
    ('Is the follow-up in six weeks?', 'Take it for two weeks and come back in six weeks.', 'consistent', None),
    # ages are not durations, so "45 years old" never fights "for two weeks"
    ('Is the patient 45 years old?', 'She is 45, and has had the pain for two weeks.', 'no_quantities', None),
    ('Is the patient 45 years old?', 'She is 45 years old and has had the pain for two weeks.', 'consistent', None),
    ('Is the patient 46 years old?', 'She is 45 years old.', 'contradiction', None),
    ('Is the patient aged 45?', 'a 45-year-old woman', 'consistent', None),
    ('Has she had asthma for 20 years?', 'I have had asthma since I was 10 years old.', 'unverifiable', None),
    # decimal comma, spoken decimals, fractions, mixed numbers
    ('1.5 mg', '1,5 mg', 'consistent', None),
    ('1.5 mg', '1½ mg', 'consistent', None),
    ('0.5 mg', 'nought point five milligrams', 'consistent', None),
    ('three quarters of an hour', '45 minutes', 'consistent', None),
    ('45 minutes', 'three quarters of an hour', 'consistent', None),
    ('a third of a tablet', 'she had two doses', 'unverifiable', None),
    ('a third dose', 'she had two doses', 'no_quantities', None),
    # spoken blood pressure
    ('130/85', 'one thirty over eighty five', 'consistent', None),
    ('140/90', 'one thirty over eighty five', 'contradiction', None),
    # months are elastic, hedges wider, spoken ranges
    ('a month', 'four weeks', 'consistent', None),
    ('a month', 'five weeks', 'contradiction', None),
    ('three months', 'twelve weeks', 'consistent', None),
    ('two weeks', 'half a month', 'consistent', None),
    ('about a week', 'eight days', 'consistent', None),
    ('about a week', 'ten days', 'contradiction', None),
    ('two weeks', 'a week or two', 'consistent', None),
    ('three days', 'three, four days', 'consistent', None),
    ('three days', 'three four days', 'consistent', None),
    ('two weeks', 'two, three weeks', 'consistent', None),
    ('2 tablets', 'take two, three times a day', 'unverifiable', None),
    ('3 times a day', 'take two, three times a day', 'consistent', None),
    ('2 weeks', 'two weeks or three months', 'consistent', None),
    ('3 months', 'two weeks or three months', 'consistent', None),
    # units and aliases
    ('every other day', 'every 2nd day', 'consistent', None),
    ('1 g', '1000 mg', 'consistent', None),
    ('5 ml', 'five mils', 'consistent', None),
    ('1000 mg', '1 000 mg', 'unverifiable', None),  # never a contradiction with "0 mg"
]


@pytest.mark.parametrize('question,evidence,status,detail', COMPARE_CASES,
                         ids=[f'{q} | {e}' for q, e, _, _ in COMPARE_CASES])
def test_compare_table(question, evidence, status, detail):
    verdict = compare(question, evidence)
    assert verdict.status == status
    if detail is not None:
        assert any(detail in d for d in verdict.details), verdict.details


def test_compare_returns_verdict_with_status_and_details():
    verdict = compare('100 mg for two weeks', 'take 100 mg for six weeks')
    assert isinstance(verdict, FactCheckVerdict)
    assert verdict.status in STATUS_PRECEDENCE
    assert all(isinstance(d, str) for d in verdict.details)
    assert [q.family for q in verdict.question_quantities] == ['dose', 'duration']
    assert [q.family for q in verdict.evidence_quantities] == ['dose', 'duration']


def test_status_precedence_contradiction_beats_everything():
    assert compare('100 mg for two weeks', 'take 100 mg for six weeks').status == 'contradiction'
    assert compare('100 mg for two weeks and 5 puffs', 'take 200 mg').status == 'contradiction'


def test_status_precedence_unverifiable_beats_consistent():
    assert compare('100 mg for two weeks', 'take 100 mg after meals').status == 'unverifiable'


def test_status_consistent_when_everything_matches():
    assert compare('100 mg for two weeks', 'take 100 mg for 14 days').status == 'consistent'


def test_contradiction_detail_lists_every_candidate_once():
    verdict = compare('100 mg', 'first 200 mg, then 200 mg, then 300 mg')
    assert verdict.details == ['dose 100 mg vs evidence 200 mg, 300 mg']


def test_ignored_plain_numbers_are_explained_but_do_not_change_status():
    verdict = compare('type 2 diabetes for 5 years', 'diabetes for five years')
    assert verdict.status == 'consistent'
    assert any(d.startswith('plain 2: ignored') for d in verdict.details)


def test_range_in_question_does_not_produce_two_contradictions():
    verdict = compare('2 to 3 weeks', 'four weeks')
    assert verdict.status == 'contradiction'
    assert len(verdict.details) == 1


@pytest.mark.parametrize('question,evidence', [
    (None, None), (float('nan'), 'x'), ('x', float('nan')), ('', None), (5, '5 mg'),
])
def test_compare_non_text_input_is_no_quantities(question, evidence):
    assert compare(question, evidence).status == 'no_quantities'


def test_compare_against_clinic_context(clinic_context):
    text = clinic_context.full_text()
    assert compare('Did the doctor prescribe 100 milligrams daily?', text).status == 'consistent'
    assert compare('Did the doctor prescribe 200 milligrams daily?', text).status == 'contradiction'
    assert compare('Will the treatment last two weeks?', text).status == 'consistent'
    assert compare('Will the treatment last six weeks?', text).status == 'contradiction'
    assert compare('Was the blood pressure 130 over 85?', text).status == 'consistent'
    assert compare('Was the blood pressure 140 over 90?', text).status == 'contradiction'
    assert compare('Was the patient given 5 ml of syrup?', text).status == 'unverifiable'
    assert compare('Was the patient referred to a specialist?', text).status == 'no_quantities'


def test_no_arithmetic_is_attempted():
    """Documented limits: a dose is compared as written, never totalled, and a
    negated mention is still a mention. Both err towards 'consistent' or a
    literal reading rather than inventing numbers."""
    assert compare('Is the dose 100 mg?', 'Take 100 mg twice a day.').status == 'consistent'
    # "daily dose 200 mg" against "100 mg twice a day" is a literal mismatch;
    # the checker does not multiply, so the question must quote the dose as said.
    assert compare('Is the daily dose 200 mg?', 'Take 100 mg twice a day.').status == 'contradiction'
    assert compare('Was the dose 200 mg?', 'Not 200 mg but 100 mg.').status == 'consistent'
    assert compare('Was the dose 1000 mg?', 'two 500 mg tablets').status == 'contradiction'


def test_contradiction_detail_is_capped_but_counts_the_rest():
    evidence = ', '.join(f'{v} mg' for v in range(200, 350, 10))  # 15 distinct doses
    verdict = compare('100 mg', evidence)
    assert verdict.status == 'contradiction'
    assert verdict.details[0].startswith('dose 100 mg vs evidence 200 mg, 210 mg')
    assert verdict.details[0].endswith('... (5 more)')


def test_question_quantities_are_capped_for_hostile_input():
    question = ' '.join(['1 mg'] * 70) + ' 5 weeks'
    verdict = compare(question, '1 mg for a week')
    assert verdict.status == 'consistent'  # the 65th quantity onwards is not checked
    assert len(verdict.question_quantities) == 71


def _best_of(fn, n=5):
    best = float('inf')
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def test_compare_scales_linearly_and_is_fast_on_a_3000_word_window(clinic_context):
    unit = clinic_context.full_text() + ' '
    short, long = unit * 4, unit * 40
    assert len(long.split()) >= 3000
    question = 'Did the doctor prescribe 100 milligrams twice a day for two weeks?'
    compare(question, long)  # warm caches
    t_short = _best_of(lambda: compare(question, short))
    t_long = _best_of(lambda: compare(question, long))
    assert t_long < 0.1, f'{t_long * 1000:.1f} ms for a 3000-word window'
    assert t_long < 25 * max(t_short, 1e-4), 'compare is not linear in the evidence length'


def test_degenerate_number_soup_is_bounded():
    """Thousands of quantities on both sides must not go quadratic."""
    soup = '1, ' * 3000
    assert _best_of(lambda: compare(soup, soup), n=2) < 1.0
    soup = 'BMI 28 ' * 2000  # plain numbers that all share a context word
    assert _best_of(lambda: compare(soup, soup), n=2) < 1.0
    long_text = ('penicillin one hundred milligrams twice daily for two weeks ' * 300)
    assert _best_of(lambda: fuzzy_contains('ibumetin', long_text)) < 0.05


def test_compare_never_interprets_text():
    """Hostile strings are just text: nothing is evaluated or formatted."""
    hostile = '__import__("os").system("id") {0} %s 100 mg'
    verdict = compare(hostile, hostile)
    assert verdict.status == 'consistent'
    assert compare('{evil}', '$(rm -rf /) 100mg').status == 'no_quantities'


# --------------------------------------------------------------------------- #
# fuzzy_contains
# --------------------------------------------------------------------------- #

def test_fuzzy_threshold_separates_asr_near_misses_from_other_drugs():
    """The 0.8 default sits between the measured ratios documented in the
    docstring: near-misses score >= 0.94, different drugs <= 0.67."""
    near = [('ibumetin', 'ibumetine'), ('paracetamol', 'paracetamole'),
            ('penicillin', 'penicillins'), ('amoxicillin', 'amoxicilin'), ('sertraline', 'sertralin')]
    different = [('ibumetin', 'ibuprofen'), ('metformin', 'metoprolol'),
                 ('losartan', 'lorazepam'), ('pamol', 'panodil')]
    near_min = min(SequenceMatcher(None, a, b).ratio() for a, b in near)
    diff_max = max(SequenceMatcher(None, a, b).ratio() for a, b in different)
    assert near_min > 0.9 > 0.8 > 0.7 > diff_max


@pytest.mark.parametrize('term,text,expected', [
    ('Ibumetin', 'she takes ibumetine daily', True),
    ('Ibumetin', 'she takes ibuprofen daily', False),
    ('ibumetin', 'Ibumetin, 400 mg', True),
    ('paracetamol', 'we tried paracetamole first', True),
    ('metformin', 'switched to metoprolol', False),
    ('blood pressure', 'your blood-pressure was fine', True),
    ('blood pressure', 'your blood presure was fine', True),
    ('blood pressure', 'your pressure was fine', False),
    ('penicillin', '', False),
    ('', 'penicillin', False),
    (None, 'penicillin', False),
    ('penicillin', None, False),
])
def test_fuzzy_contains(term, text, expected):
    assert fuzzy_contains(term, text) is expected


def test_fuzzy_threshold_is_adjustable():
    assert fuzzy_contains('ibumetin', 'ibuprofen', threshold=0.5) is True
    assert fuzzy_contains('ibumetin', 'ibumetine', threshold=0.99) is False
    assert fuzzy_contains('ibumetin', 'ibumetin', threshold=1.0) is True


# --------------------------------------------------------------------------- #
# entity_terms
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('question,expected', [
    ('Did the doctor prescribe Ibumetin and amoxicillin for the patient?', ['ibumetin', 'amoxicillin']),
    ('Was Metoprolol 50 mg prescribed?', ['metoprolol']),
    ('Did the patient take paracetamol?', ['paracetamol']),
    ('Did she mention the Bergen trip?', ['bergen']),
    ('Was an LDL test ordered?', ['ldl']),
    ('Did the doctor start insulin?', ['insulin']),
    ('Is there any mention of medicine or routine?', []),
    ('Did the patient decide to remain inside?', []),
    ('Did the patient climb a mountain with her cousin?', []),
    ('Was a stool sample requested?', []),
    ('Did the doctor examine the spine?', []),
    ('Oslo is short, so it does not count.', []),
    ("Was Ibumetin's dose changed?", ['ibumetin']),
    ('', []),
    (None, []),
])
def test_entity_terms(question, expected):
    assert entity_terms(question) == expected


def test_entity_terms_are_unique_and_ordered():
    assert entity_terms('Was Ibumetin replaced by Panodil, or was Ibumetin kept?') == ['ibumetin', 'panodil']


def test_entity_terms_feed_fuzzy_contains():
    question = 'Did the doctor prescribe Ibumetin?'
    transcript = 'so I will put you on ibumetine, four hundred milligrams'
    assert all(fuzzy_contains(t, transcript) for t in entity_terms(question))
    assert not all(fuzzy_contains(t, 'so I will put you on ibuprofen') for t in entity_terms(question))


def test_nan_values_never_appear_in_quantities():
    for text in ['1e400 mg', '99999999999999999999 mg', '0.0 mg', 'NaN mg', 'inf mg']:
        for q in extract_quantities(text):
            assert math.isfinite(q.value)
