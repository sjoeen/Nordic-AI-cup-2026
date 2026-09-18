"""Rule-based quantity extraction and consistency checks (contract A5).

WHY this module exists
----------------------
The hard negatives in this task are typically the positive question with one
number swapped: "100 mg" becomes "200 mg", "two weeks" becomes "six weeks",
"130/85" becomes "140/90", "twice a day" becomes "once daily". Small NLI models
are poor at exactly this kind of arithmetic, so the QA backend asks this module
for a deterministic second opinion: does every quantity in the question have a
matching quantity of the same kind in the evidence?

Design
------
* Pure Python, no ML, deterministic. Text is UNTRUSTED data: it is only ever
  tokenised and compared, never evaluated, never used to build paths or
  commands.
* ``extract_quantities`` turns text into ``Quantity`` records whose ``value``
  is normalised to one base unit per family (mg for mass doses, days for
  durations, "per day" for frequencies, ml, kg, cm, Celsius) so that "two
  weeks" and "14 days" compare equal while ``raw``/``unit`` keep the wording.
* ``compare`` matches question quantities against evidence quantities family
  by family. Bare numbers ("BMI of 28", "type 2") are only compared when both
  sides share a context word, because a lone "2" in a transcript is far more
  often noise than a fact; the one exception is a bare number next to a
  temperature word ("temperature was 38.5"), which clinicians say without a
  unit so often that it is promoted to the temperature family.
* Equality is 0.1 % relative (durations in days), widened to 2 % across a
  unit conversion ("70 kg" vs "154 pounds"), 8 % when a month is converted
  ("a month" vs "four weeks") and 25 % under a hedge ("about a week").
* No arithmetic is ever attempted: "100 mg twice a day" is a dose of 100 mg
  and a frequency, never a daily total of 200 mg, and "not 200 mg but 100 mg"
  mentions both doses. Ages ("45 years old", "aged 45") are bare numbers with
  an ``age`` context word, not durations, so they never contradict "for two
  weeks" in the same window.
* ``fuzzy_contains`` / ``entity_terms`` support the name near-miss check: ASR
  writes "Ibumetine" for "Ibumetin", and the QA backend should not treat that
  as a missing drug.

Families produced: dose, volume, duration, frequency, pressure, pressure_dia,
percentage, weight, length, temperature, concentration, count, plain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

__all__ = [
    'Quantity',
    'FactCheckVerdict',
    'FAMILIES',
    'STATUS_PRECEDENCE',
    'extract_quantities',
    'compare',
    'fuzzy_contains',
    'entity_terms',
]

FAMILIES: Tuple[str, ...] = (
    'dose', 'volume', 'duration', 'frequency', 'pressure', 'pressure_dia',
    'percentage', 'weight', 'length', 'temperature', 'concentration', 'count',
    'plain',
)

# Highest first. ``compare`` reports the most severe status it encountered.
STATUS_PRECEDENCE: Tuple[str, ...] = (
    'contradiction', 'unverifiable', 'consistent', 'no_quantities',
)

# Tolerances for "equal": absolute floor for tiny values, relative otherwise.
_ABS_TOL = 1e-6
_REL_TOL = 1e-3
# Relative tolerance when either side is hedged ("about two weeks"): "about a
# week" must accept "eight days".
_APPROX_REL_TOL = 0.25
# Relative tolerance when the two sides use different units of one family:
# spoken conversions are rounded ("70 kg" is "154 pounds", "a year" is
# "twelve months"), so exact equality would call them contradictions.
_CONVERTED_REL_TOL = 0.02
# Months are not a fixed number of days: "a month" is "four weeks" (28 vs 30)
# and "three months" is "twelve weeks" (84 vs 90), so a month against another
# unit gets 8 %. "Five weeks" (35) stays a contradiction.
_MONTH_REL_TOL = 0.08


@dataclass(frozen=True)
class Quantity:
    """One quantity mention.

    ``value`` is the comparison value in the family's base unit (see module
    docstring); ``unit`` is the canonical unit as written (``'week'`` for a
    duration given in weeks, ``'per_day'`` for frequencies, ``'mmhg'`` for
    pressures, ``None`` for bare numbers). ``raw`` is the normalised surface
    text of the mention. ``value_hi`` is set when the mention is a range
    ("2 to 3 weeks": value=14, value_hi=21). ``qualifier`` records hedges and
    bounds ('approx', 'lt', 'le', 'gt', 'ge'). ``group`` narrows comparability
    inside a family (a mass dose is never compared with an IU dose).
    ``context`` holds nearby content words and is what makes two bare numbers
    comparable.
    """

    value: float
    unit: Optional[str]
    family: str
    raw: str
    value_hi: Optional[float] = None
    qualifier: Optional[str] = None
    group: str = ''
    context: Tuple[str, ...] = ()

    @property
    def is_range(self) -> bool:
        return self.value_hi is not None


@dataclass
class FactCheckVerdict:
    """Outcome of ``compare``. ``status`` is one of ``STATUS_PRECEDENCE``."""

    status: str
    details: List[str] = field(default_factory=list)
    question_quantities: Tuple[Quantity, ...] = ()
    evidence_quantities: Tuple[Quantity, ...] = ()


# --------------------------------------------------------------------------- #
# Vocabulary tables
# --------------------------------------------------------------------------- #

_ONES: Dict[str, float] = {
    'zero': 0, 'nought': 0, 'naught': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
    'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
    'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16,
    'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
}
_TENS: Dict[str, float] = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
    'seventy': 70, 'eighty': 80, 'ninety': 90,
}
_BIG_SCALES: Dict[str, float] = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9}
_FRACTIONS: Dict[str, float] = {'half': 0.5, 'quarter': 0.25}
# "a third" is only a fraction before "of" ("a third of a tablet"); elsewhere
# it is the ordinal ("a third dose"), so it is not in _FRACTIONS.
_ARTICLE_FRACTIONS: Dict[str, float] = {'half': 0.5, 'quarter': 0.25, 'third': 1 / 3}
_PLURAL_FRACTIONS: Dict[str, float] = {'halves': 0.5, 'quarters': 0.25, 'thirds': 1 / 3}
_ORDINALS: Dict[str, int] = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6,
    'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
}
_ORDINAL_SUFFIXES = ('st', 'nd', 'rd', 'th')
_MONTHS = frozenset((
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
))

# (canonical unit, family, factor to the family base unit, comparability group)
_UnitInfo = Tuple[str, str, float, str]
_UNIT_ALIASES: Dict[str, _UnitInfo] = {}


def _register(info: _UnitInfo, *aliases: str) -> None:
    for alias in aliases:
        _UNIT_ALIASES[alias] = info


# dose (mass) -> mg
_register(('mg', 'dose', 1.0, 'mass'), 'mg', 'mgs', 'milligram', 'milligrams', 'milligramme', 'milligrammes')
_register(('mcg', 'dose', 1e-3, 'mass'), 'mcg', 'ug', 'microgram', 'micrograms', 'microgramme', 'microgrammes')
_register(('g', 'dose', 1e3, 'mass'), 'g', 'gm', 'gms', 'gram', 'grams', 'gramme', 'grammes')
# dose (international units) -> iu; not convertible to mass, hence its own group
_register(('iu', 'dose', 1.0, 'iu'), 'iu', 'unit', 'units')
# volume -> ml
_register(('ml', 'volume', 1.0, 'volume'), 'ml', 'mls', 'mil', 'mils', 'cc', 'millilitre', 'millilitres', 'milliliter', 'milliliters')
_register(('l', 'volume', 1e3, 'volume'), 'l', 'litre', 'litres', 'liter', 'liters')
_register(('dl', 'volume', 100.0, 'volume'), 'dl', 'decilitre', 'decilitres', 'deciliter', 'deciliters')
# weight -> kg
_register(('kg', 'weight', 1.0, 'weight'), 'kg', 'kgs', 'kilogram', 'kilograms', 'kilo', 'kilos')
_register(('lb', 'weight', 0.45359237, 'weight'), 'lb', 'lbs', 'pound', 'pounds')
_register(('stone', 'weight', 6.35029318, 'weight'), 'stone', 'stones')
# length -> cm
_register(('cm', 'length', 1.0, 'length'), 'cm', 'centimetre', 'centimetres', 'centimeter', 'centimeters')
_register(('mm', 'length', 0.1, 'length'), 'mm', 'millimetre', 'millimetres', 'millimeter', 'millimeters')
_register(('m', 'length', 100.0, 'length'), 'm', 'metre', 'metres', 'meter', 'meters')
_register(('inch', 'length', 2.54, 'length'), 'inch', 'inches')
_register(('ft', 'length', 30.48, 'length'), 'ft', 'foot', 'feet')
# percentage
_register(('percent', 'percentage', 1.0, 'percentage'), 'percent', 'pct')
# count nouns (a "family count" quantity: 2 tablets, 3 puffs, 4 doses). The
# group is a synonym class: "two tablets" is compared with "two pills" but
# never with "two puffs", so a count of one thing cannot contradict a count of
# another ("one tablet" vs "two puffs" is unverifiable, not a contradiction).
_register(('tablet', 'count', 1.0, 'pill'), 'tablet', 'tablets', 'tab', 'tabs')
_register(('capsule', 'count', 1.0, 'pill'), 'capsule', 'capsules', 'caps')
_register(('pill', 'count', 1.0, 'pill'), 'pill', 'pills')
_register(('puff', 'count', 1.0, 'puff'), 'puff', 'puffs')
_register(('spray', 'count', 1.0, 'puff'), 'spray', 'sprays')
_register(('drop', 'count', 1.0, 'drop'), 'drop', 'drops')
_register(('dose', 'count', 1.0, 'dose'), 'dose', 'doses')
_register(('injection', 'count', 1.0, 'injection'), 'injection', 'injections', 'shot', 'shots')
_register(('sachet', 'count', 1.0, 'sachet'), 'sachet', 'sachets')
_register(('patch', 'count', 1.0, 'patch'), 'patch', 'patches')
_register(('lozenge', 'count', 1.0, 'lozenge'), 'lozenge', 'lozenges')
# temperature -> Celsius (Fahrenheit converted in code)
_register(('c', 'temperature', 1.0, 'temperature'), 'celsius', 'centigrade')
_register(('f', 'temperature', 1.0, 'temperature'), 'fahrenheit')
# pressure unit (only meaningful after the 130/85 pattern or a lone systolic)
_register(('mmhg', 'pressure', 1.0, 'pressure'), 'mmhg')

# duration -> days. Singular 'second' is an ordinal unless preceded by digits
# ("a second dose" vs "30 second"), so it is handled in code, not here.
_DURATION_UNITS: Dict[str, Tuple[str, float]] = {
    'seconds': ('second', 1 / 86400), 'sec': ('second', 1 / 86400), 'secs': ('second', 1 / 86400),
    'minute': ('minute', 1 / 1440), 'minutes': ('minute', 1 / 1440), 'min': ('minute', 1 / 1440), 'mins': ('minute', 1 / 1440),
    'hour': ('hour', 1 / 24), 'hours': ('hour', 1 / 24), 'hr': ('hour', 1 / 24), 'hrs': ('hour', 1 / 24), 'h': ('hour', 1 / 24),
    'day': ('day', 1.0), 'days': ('day', 1.0), 'd': ('day', 1.0),
    'night': ('night', 1.0), 'nights': ('night', 1.0),
    'week': ('week', 7.0), 'weeks': ('week', 7.0), 'wk': ('week', 7.0), 'wks': ('week', 7.0),
    'fortnight': ('fortnight', 14.0), 'fortnights': ('fortnight', 14.0),
    'month': ('month', 30.0), 'months': ('month', 30.0),
    'year': ('year', 365.0), 'years': ('year', 365.0), 'yr': ('year', 365.0), 'yrs': ('year', 365.0),
}
# Units that may only be recognised when glued to digits ("5d", "8h"): as free
# words they are far too ambiguous ("vitamin D", "h"). "g" is not in the set:
# "1 g" is how clinicians write a gram dose and the letter has no other use
# after a number.
_GLUED_ONLY = frozenset(('d', 'h', 'm', 'l', 'c', 'f', 'x', 'u'))

# Period words for frequencies ("twice a WEEK"), in days.
_PERIOD_DAYS: Dict[str, float] = {
    'day': 1.0, 'days': 1.0, 'night': 1.0, 'nights': 1.0, 'morning': 1.0, 'mornings': 1.0,
    'evening': 1.0, 'evenings': 1.0, 'bedtime': 1.0,
    'week': 7.0, 'weeks': 7.0, 'fortnight': 14.0, 'fortnights': 14.0,
    'month': 30.0, 'months': 30.0, 'year': 365.0, 'years': 365.0,
    'hour': 1 / 24, 'hours': 1 / 24, 'minute': 1 / 1440, 'minutes': 1 / 1440,
}
# Adverbs meaning "once per <period>": value is the period in days.
_PERIOD_ADVERBS: Dict[str, float] = {
    'daily': 1.0, 'nightly': 1.0, 'weekly': 7.0, 'fortnightly': 14.0,
    'monthly': 30.0, 'yearly': 365.0, 'annually': 365.0, 'hourly': 1 / 24,
}
# Prescription abbreviations: times per day.
_ABBREV_PER_DAY: Dict[str, float] = {'bd': 2, 'bid': 2, 'tds': 3, 'tid': 3, 'qds': 4, 'qid': 4}
_MULTIPLIER_WORDS: Dict[str, float] = {'once': 1, 'twice': 2, 'thrice': 3}
_TIMES_WORDS = frozenset(('times', 'time', 'x'))
_PERIOD_PREPOSITIONS = frozenset(('per', 'each', 'every'))

# Concentration units: numerator aliases and denominator aliases.
_CONC_NUM: Dict[str, str] = {
    'mmol': 'mmol', 'millimole': 'mmol', 'millimoles': 'mmol',
    'umol': 'umol', 'micromole': 'umol', 'micromoles': 'umol',
    'nmol': 'nmol', 'pmol': 'pmol', 'mol': 'mol', 'mole': 'mol', 'moles': 'mol',
    'ng': 'ng', 'nanogram': 'ng', 'nanograms': 'ng',
    'mg': 'mg', 'milligram': 'mg', 'milligrams': 'mg',
    'g': 'g', 'gram': 'g', 'grams': 'g',
    'ug': 'mcg', 'mcg': 'mcg', 'microgram': 'mcg', 'micrograms': 'mcg',
    'iu': 'iu', 'u': 'iu', 'units': 'iu', 'unit': 'iu',
}
_CONC_DEN: Dict[str, str] = {
    'l': 'l', 'litre': 'l', 'liter': 'l', 'litres': 'l', 'liters': 'l',
    'dl': 'dl', 'decilitre': 'dl', 'deciliter': 'dl',
    'ml': 'ml', 'millilitre': 'ml', 'milliliter': 'ml',
    'mol': 'mol', 'kg': 'kg', 'kilogram': 'kg', 'kilo': 'kg',
}

_QUALIFIER_PHRASES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (('less', 'than'), 'lt'), (('fewer', 'than'), 'lt'), (('under',), 'lt'), (('below',), 'lt'),
    (('up', 'to'), 'le'), (('at', 'most'), 'le'), (('no', 'more', 'than'), 'le'), (('maximum',), 'le'),
    (('more', 'than'), 'gt'), (('above',), 'gt'), (('exceeding',), 'gt'),
    (('at', 'least'), 'ge'), (('over',), 'ge'), (('minimum',), 'ge'),
    (('about',), 'approx'), (('around',), 'approx'), (('approximately',), 'approx'),
    (('roughly',), 'approx'), (('nearly',), 'approx'), (('almost',), 'approx'), (('approx',), 'approx'),
)
# The same phrases keyed by their last word, so the scanner only inspects the
# phrases that can end at the token before a number.
_QUALIFIER_BY_LAST: Dict[str, Tuple[Tuple[Tuple[str, ...], str], ...]] = {}
for _phrase, _tag in _QUALIFIER_PHRASES:
    _QUALIFIER_BY_LAST[_phrase[-1]] = _QUALIFIER_BY_LAST.get(_phrase[-1], ()) + ((_phrase, _tag),)
del _phrase, _tag
# Words before a bare number that make it an age ("aged 45", "age of 45").
_AGE_WORDS = frozenset(('aged', 'age'))

# Words that never count as context for a bare number.
_STOPWORDS = frozenset((
    'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'for', 'is', 'was', 'were', 'are', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did', 'it', 'its', 'this', 'that', 'these', 'those', 'and', 'or', 'but',
    'with', 'by', 'from', 'as', 'than', 'about', 'around', 'he', 'she', 'they', 'his', 'her', 'their', 'you',
    'your', 'i', 'we', 'my', 'our', 'me', 'us', 'any', 'some', 'there', 'here', 'so', 'if', 'then', 'not',
    'no', 'yes', 'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'come', 'came', 'out',
    'up', 'down', 'over', 'under', 'into', 'only', 'also', 'still', 'just', 'very', 'more', 'less', 'most',
    'many', 'much', 'few', 'recorded', 'measured', 'noted', 'reported', 'found', 'get', 'got', 'now', 'right',
    'okay', 'ok', 'well', 'like', 'what', 'which', 'who', 'when', 'where', 'how', 'why', 'one', 'patient',
    'patients', 'doctor', 'today', 'again', 'back', 'go', 'going', 'went', 'take', 'taking', 'took', 'give',
    'given', 'said', 'say', 'tell', 'told', 'see', 'seen', 'think', 'thought', 'know', 'want', 'need',
))
# Context words that make a bare number a body temperature, and the plausible
# ranges used to tell Celsius from Fahrenheit (no overlap: 30-45 C vs 86-113 F).
_TEMPERATURE_CONTEXT = frozenset(('temperature', 'temp', 'fever', 'febrile'))
_BODY_TEMP_C = (30.0, 45.0)
_BODY_TEMP_F = (86.0, 113.0)


def _f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


# Determiners that turn "one" into a pronoun ("the one", "which one", "no one").
_ONE_DETERMINERS = frozenset(('the', 'this', 'that', 'which', 'another', 'each', 'no', 'any', 'every', 'other'))
_RANGE_CONNECTORS = frozenset(('to', 'or', 'through', '-'))

# --------------------------------------------------------------------------- #
# Normalisation and tokenisation
# --------------------------------------------------------------------------- #

_CHAR_MAP = {
    'µ': 'u', 'μ': 'u',           # micro sign, Greek mu
    '°': ' degrees ', '%': ' percent ',
    '½': ' half ', '¼': ' quarter ', '¾': ' three quarters ',
    '–': '-', '—': '-', '’': "'", '‘': "'",
}
# A vulgar fraction glued to a digit is a mixed number: "1½ mg" is 1.5 mg.
_MIXED_FRACTION_RE = re.compile(r'(?<=\d)\s?([½¼¾])')
_MIXED_FRACTION_WORDS = {'½': ' and a half ', '¼': ' and a quarter ', '¾': ' and three quarters '}
_TOKEN_RE = re.compile(
    r"""
    (?P<time>(?<![\d.])\d{1,2}:\d{2}(?::\d{2})?(?![\d]))
  | (?P<ident>[a-z]+\d[a-z0-9]*)
  | (?P<num>(?<![\d.])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+,\d{1,2}(?![\d,])|\d+(?:\.\d+)?)|(?<![\d])\.\d+)(?P<suffix>[a-z]+)?
  | (?P<word>[a-z]+(?:'[a-z]+)?)
  | (?P<sym>[/\-,])
    """,
    re.VERBOSE,
)
# "1,5" is a decimal comma (1.5); "1,000" is a thousands separator (1000).
_DECIMAL_COMMA_RE = re.compile(r'\d+,\d{1,2}')
_LEADING_DIGITS_RE = re.compile(r'\d+')


def _normalize(text: str) -> str:
    """Lower-case and rewrite symbols into words so one regex can tokenise.

    ``covid-19`` and ``b-12`` are collapsed to ``covid19``/``b12`` so the digit
    can never be mistaken for a quantity. A hyphen between letters becomes a
    space ("twice-daily", "two-week course", "twenty-five") so that the word
    grammar below never has to know about hyphenation; digit ranges ("2-3")
    keep their hyphen.
    """
    out = text.lower()
    out = _MIXED_FRACTION_RE.sub(lambda m: _MIXED_FRACTION_WORDS[m.group(1)], out)
    for src, dst in _CHAR_MAP.items():
        out = out.replace(src, dst)
    out = re.sub(r'(?<=[a-z])-(?=\d)', '', out)
    out = re.sub(r'(?<=[a-z])-(?=[a-z])', ' ', out)
    out = re.sub(r'\bper\s+cent\b', 'percent', out)
    return out


@dataclass(slots=True)
class _Tok:
    kind: str            # 'num', 'word', 'sym', 'time', 'ident', 'ordinal'
    text: str
    start: int
    end: int
    value: Optional[float] = None
    article: bool = False   # a number that came from the article "a"/"an"
    fraction: bool = False  # a bare fraction word ("half", "a quarter")


def _number_value(numtext: str) -> float:
    if ',' in numtext:
        if _DECIMAL_COMMA_RE.fullmatch(numtext):
            return float(numtext.replace(',', '.'))
        return float(numtext.replace(',', ''))
    return float(numtext)


def _tokenize(norm: str) -> List[_Tok]:
    toks: List[_Tok] = []
    append = toks.append
    for m in _TOKEN_RE.finditer(norm):
        kind = m.lastgroup  # 'suffix' when a num has a glued suffix
        if kind == 'word':
            append(_Tok('word', m.group(), m.start(), m.end()))
        elif kind == 'num' or kind == 'suffix':
            numtext = m.group('num')
            if len(numtext) > 1 and numtext[0] == '0' and numtext[1] not in '.,':
                # "0100", "000", "0047": a code or a phone-number fragment, never a quantity.
                append(_Tok('ident', m.group(), m.start(), m.end()))
                continue
            value = _number_value(numtext)
            if kind == 'num':
                append(_Tok('num', numtext, m.start(), m.end(), value))
                continue
            suffix = m.group('suffix')
            if suffix in _ORDINAL_SUFFIXES:
                append(_Tok('ordinal', m.group(), m.start(), m.end()))
            elif suffix in _UNIT_ALIASES or suffix in _DURATION_UNITS or suffix in _GLUED_ONLY \
                    or suffix in _PERIOD_ADVERBS or suffix in _CONC_NUM or suffix == 'mmhg':
                append(_Tok('num', numtext, m.start('num'), m.end('num'), value))
                append(_Tok('word', suffix, m.start('suffix'), m.end('suffix')))
            else:
                # "5htp", "3d-printed": digits glued to an unknown word are an identifier.
                append(_Tok('ident', m.group(), m.start(), m.end()))
        elif kind == 'sym':
            append(_Tok('sym', m.group(), m.start(), m.end()))
        else:  # 'time', 'ident'
            append(_Tok(kind, m.group(), m.start(), m.end()))
    return toks


def _is_word(tok: Optional[_Tok], *texts: str) -> bool:
    return tok is not None and tok.kind == 'word' and (not texts or tok.text in texts)


def _is_sym(tok: Optional[_Tok], text: str) -> bool:
    return tok is not None and tok.kind == 'sym' and tok.text == text


def _is_article_unit_word(word: str) -> bool:
    """Units after which a bare article means one: durations only.

    "a tablet"/"a dose" are deliberately excluded: they are far too weak to
    justify a count quantity, and a spurious count makes a good match
    'unverifiable'.
    """
    return word in _DURATION_UNITS or word in _PERIOD_DAYS


def _fraction_after_and(toks: Sequence[_Tok], j: int) -> Optional[Tuple[float, int]]:
    """Match 'and a half' / 'and half' / 'and a quarter' starting at toks[j] == 'and'.

    Works on raw tokens (inside ``_parse_number``, "two and a half weeks") and
    on merged tokens (after a unit, "an hour and a half"), where the fraction
    has already been collapsed into one ``num`` token flagged ``fraction``.
    """
    if j >= len(toks) or not _is_word(toks[j], 'and'):
        return None
    k = j + 1
    if k < len(toks) and toks[k].kind == 'num' and toks[k].fraction:
        return float(toks[k].value or 0.0), k + 1
    if k < len(toks) and _is_word(toks[k], 'a', 'an'):
        k += 1
    if k < len(toks) and toks[k].kind == 'word' and toks[k].text in _ARTICLE_FRACTIONS:
        return _ARTICLE_FRACTIONS[toks[k].text], k + 1
    if k + 1 < len(toks) and _is_word(toks[k]) and toks[k].text in _ONES \
            and _is_word(toks[k + 1]) and toks[k + 1].text in _PLURAL_FRACTIONS:
        return _ONES[toks[k].text] * _PLURAL_FRACTIONS[toks[k + 1].text], k + 2  # "and three quarters"
    return None


class _Parsed(NamedTuple):
    """Result of ``_parse_number``: the value, the index after it, and whether
    it came from a bare article ("a week") or is a bare fraction ("half")."""

    value: float
    end: int
    article: bool = False
    fraction: bool = False


def _parse_number(toks: Sequence[_Tok], i: int) -> Optional[_Parsed]:
    """Parse a number (digits, number words or an article) starting at toks[i].

    The grammar is the usual English one: ones/tens/hyphen compounds,
    'hundred' multiplies, 'thousand'/'million' close a group, 'and' is only
    allowed after a scale word ("two hundred and fifty") or before a fraction
    ("one and a half"), so "between one and two weeks" never collapses into 3.
    A bare scale word starts at one ("hundred milligrams" is an ASR habit) and
    'point' followed by digit words spells a decimal ("zero point three").
    """
    n = len(toks)
    tok = toks[i]
    total = 0.0
    current = 0.0
    state = ''
    j = i
    if tok.kind == 'num':
        current, j, state = float(tok.value or 0.0), i + 1, 'digit'
    elif tok.kind == 'word' and tok.text in ('a', 'an'):
        nxt = toks[i + 1] if i + 1 < n else None
        if nxt is None or nxt.kind != 'word':
            return None
        if nxt.text == 'hundred' or nxt.text in _BIG_SCALES:
            current, j, state = 1.0, i + 1, 'article'
        elif nxt.text in _FRACTIONS:
            return _Parsed(_FRACTIONS[nxt.text], i + 2, fraction=True)
        elif nxt.text == 'third' and i + 2 < n and _is_word(toks[i + 2], 'of'):
            return _Parsed(_ARTICLE_FRACTIONS['third'], i + 2, fraction=True)  # "a third of a tablet"
        elif _is_article_unit_word(nxt.text):
            return _Parsed(1.0, i + 1, article=True)
        else:
            return None
    elif tok.kind == 'word' and tok.text in _FRACTIONS:
        return _Parsed(_FRACTIONS[tok.text], i + 1, fraction=True)
    elif tok.kind == 'word' and (tok.text == 'hundred' or tok.text in _BIG_SCALES):
        current, j, state = 1.0, i, 'article'  # re-read the scale word in the loop
    elif tok.kind == 'word' and tok.text in _ONES:
        current, j, state = _ONES[tok.text], i + 1, 'ones'
    elif tok.kind == 'word' and tok.text in _TENS:
        current, j, state = _TENS[tok.text], i + 1, 'tens'
    else:
        return None

    while j < n:
        t = toks[j]
        if t.kind == 'word':
            w = t.text
            if w in _ONES and state in ('tens', 'hyphen', 'and', 'scale', 'bigscale'):
                current += _ONES[w]
                state = 'ones'
            elif w in _TENS and state in ('and', 'scale', 'bigscale'):
                current += _TENS[w]
                state = 'tens'
            elif w == 'hundred' and state in ('ones', 'tens', 'digit', 'article') and current < 100:
                current = (current or 1.0) * 100
                state = 'scale'
            elif w in _BIG_SCALES and state in ('ones', 'tens', 'digit', 'scale', 'article'):
                total += (current or 1.0) * _BIG_SCALES[w]
                current = 0.0
                state = 'bigscale'
            elif w == 'and':
                nxt = toks[j + 1] if j + 1 < n else None
                if state in ('scale', 'bigscale') and _is_word(nxt) and (nxt.text in _ONES or nxt.text in _TENS):
                    state = 'and'
                else:
                    frac = _fraction_after_and(toks, j)
                    if frac is not None and state in ('ones', 'tens', 'digit', 'scale', 'bigscale'):
                        current += frac[0]
                        j = frac[1]
                    break
            else:
                break
            j += 1
        elif _is_sym(t, '-') and state == 'tens' and j + 1 < n and _is_word(toks[j + 1]) \
                and toks[j + 1].text in _ONES:
            state = 'hyphen'
            j += 1
        else:
            break
    if j < n and total == 0 and state in ('ones', 'tens', 'digit') and _is_word(toks[j]) \
            and toks[j].text in _PLURAL_FRACTIONS:
        # "three quarters (of an hour)", "two thirds (of a tablet)"
        return _Parsed(current * _PLURAL_FRACTIONS[toks[j].text], j + 1, fraction=True)
    if j < n and _is_word(toks[j], 'point') and state in ('ones', 'tens', 'digit', 'scale', 'bigscale'):
        digits: List[str] = []
        k = j + 1
        while k < n:
            t = toks[k]
            if _is_word(t) and t.text in _ONES and _ONES[t.text] < 10:
                digits.append(str(int(_ONES[t.text])))
            elif t.kind == 'num' and t.text.isdigit():
                digits.append(t.text)
            else:
                break
            k += 1
        if digits:
            return _Parsed(total + current + float('0.' + ''.join(digits)), k)
    return _Parsed(total + current, j)


_NUMBER_START_WORDS = frozenset(_ONES) | frozenset(_TENS) | frozenset(_FRACTIONS) \
    | frozenset(_BIG_SCALES) | frozenset(('a', 'an', 'hundred'))


def _merge_numbers(toks: List[_Tok], norm: str) -> List[_Tok]:
    """Collapse spelled-out numbers (and digit + scale word) into single tokens."""
    out: List[_Tok] = []
    i = 0
    n = len(toks)
    while i < n:
        tok = toks[i]
        parsed = None
        if tok.kind == 'num' or (tok.kind == 'word' and tok.text in _NUMBER_START_WORDS):
            parsed = _parse_number(toks, i)
        if parsed is None:
            out.append(tok)
            i += 1
            continue
        last = toks[parsed.end - 1]
        out.append(_Tok('num', norm[tok.start:last.end], tok.start, last.end, parsed.value,
                        parsed.article, parsed.fraction))
        i = parsed.end
    return out


# --------------------------------------------------------------------------- #
# Scanner: tokens -> quantities
# --------------------------------------------------------------------------- #

class _Scanner:
    def __init__(self, norm: str, toks: List[_Tok]) -> None:
        self.norm = norm
        self.toks = toks
        self.n = len(toks)
        self.out: List[Quantity] = []

    # -- helpers ---------------------------------------------------------- #

    def tok(self, i: int) -> Optional[_Tok]:
        return self.toks[i] if 0 <= i < self.n else None

    def is_article_tok(self, i: int) -> bool:
        t = self.tok(i)
        return t is not None and t.kind == 'num' and t.article

    def raw(self, start_idx: int, end_idx_exclusive: int) -> str:
        first = self.toks[start_idx]
        last = self.toks[max(start_idx, end_idx_exclusive - 1)]
        return self.norm[first.start:last.end]

    def context(self, start_idx: int, end_idx_exclusive: int) -> Tuple[str, ...]:
        """Up to two content words on each side, lightly stemmed."""
        words: List[str] = []
        toks = self.toks
        for rng in (range(start_idx - 1, max(-1, start_idx - 5), -1),
                    range(end_idx_exclusive, min(self.n, end_idx_exclusive + 4))):
            picked = 0
            for k in rng:
                t = toks[k]
                if t.kind != 'word' and t.kind != 'ident':
                    continue
                text = t.text
                if text in _STOPWORDS:
                    continue
                w = _stem(text)
                if len(w) < 2 or w in _STOPWORDS:
                    continue
                words.append(w)
                picked += 1
                if picked == 2:
                    break
        return tuple(dict.fromkeys(words))

    def qualifier(self, start_idx: int) -> Optional[str]:
        prev = self.tok(start_idx - 1)
        if _is_sym(prev, ','):
            prev, start_idx = self.tok(start_idx - 2), start_idx - 1  # "about, two weeks"
        if prev is None or prev.kind != 'word':
            return None
        for phrase, tag in _QUALIFIER_BY_LAST.get(prev.text, ()):
            k = start_idx - len(phrase)
            if k >= 0 and all(_is_word(self.toks[k + p], phrase[p]) for p in range(len(phrase))):
                return tag
        return None

    def emit(self, start_idx: int, end_idx: int, value: float, unit: Optional[str], family: str,
             group: str, value_hi: Optional[float] = None,
             context: Optional[Tuple[str, ...]] = None, qualifier: Optional[str] = None) -> None:
        if value_hi is not None and value_hi < value:
            value, value_hi = value_hi, value
        if value_hi is not None and abs(value_hi - value) <= _ABS_TOL:
            value_hi = None
        self.out.append(Quantity(
            value=float(value), unit=unit, family=family, raw=self.raw(start_idx, end_idx),
            value_hi=None if value_hi is None else float(value_hi),
            qualifier=self.qualifier(start_idx) if qualifier is None else qualifier, group=group,
            context=self.context(start_idx, end_idx) if context is None else context,
        ))

    # -- main loop -------------------------------------------------------- #

    def run(self) -> List[Quantity]:
        i = 0
        while i < self.n:
            t = self.toks[i]
            if t.kind == 'num':
                i = self.at_number(i)
            elif t.kind == 'word':
                i = self.at_word(i)
            elif t.kind == 'ident':
                m = re.fullmatch(r'q(\d{1,2})h', t.text)
                if m and int(m.group(1)) > 0:
                    self.emit(i, i + 1, 24.0 / int(m.group(1)), 'per_day', 'frequency', 'frequency')
                i += 1
            else:
                i += 1
        return self.out

    # -- number-led patterns --------------------------------------------- #

    def at_number(self, i: int) -> int:
        j = self.try_pressure(i)
        if j is not None:
            return j
        t = self.toks[i]
        value = float(t.value or 0.0)
        j = self.try_range(i)
        if j is not None:
            hi_tok = self.toks[j - 1]
            return self.attach(i, value, float(hi_tok.value or 0.0), j)
        return self.attach(i, value, None, i + 1)

    def try_pressure(self, i: int) -> Optional[int]:
        """'130/85', '130 over 85', 'one thirty over eighty five' (optionally + mmHg)."""
        a = self.tok(i)
        if a is None or a.article:
            return None
        sys_v = float(a.value or 0)
        j = i + 1
        spoken = self.tok(j)
        if a.text == 'one' and spoken is not None and spoken.kind == 'num' and not spoken.article \
                and spoken.text[:1].isalpha() and 10 <= float(spoken.value or 0) <= 99 \
                and float(spoken.value or 0).is_integer() \
                and (_is_word(self.tok(j + 1), 'over') or _is_sym(self.tok(j + 1), '/')):
            sys_v, j = 100 + float(spoken.value or 0), j + 1  # "one thirty over eighty five"
        sep, b = self.tok(j), self.tok(j + 1)
        if b is None or b.kind != 'num' or b.article:
            return None
        if not (_is_sym(sep, '/') or _is_word(sep, 'over')):
            return None
        dia_v = float(b.value or 0)
        plausible = (50 <= sys_v <= 300 and 20 <= dia_v <= 200 and sys_v > dia_v
                     and sys_v.is_integer() and dia_v.is_integer())
        if not plausible:
            if _is_sym(sep, '/') and 0 < sys_v < dia_v <= 16 and sys_v.is_integer() and dia_v.is_integer():
                # "1/2 tablet": a vulgar fraction, handed to the unit attacher.
                return self.attach(i, sys_v / dia_v, None, j + 2)
            return None
        end = j + 2
        if _is_word(self.tok(end), 'mmhg'):
            end += 1
        self.emit(i, end, sys_v, 'mmhg', 'pressure', 'pressure')
        self.emit(i, end, dia_v, 'mmhg', 'pressure_dia', 'pressure_dia')
        return end

    def try_range(self, i: int) -> Optional[int]:
        """'2 to 3', '2-3', 'one or two', 'between 5 and 10', and the spoken
        'two, three weeks' / 'three four days' -> index after the 2nd number."""
        first = self.toks[i]
        if first.article:
            return None
        conn, second = self.tok(i + 1), self.tok(i + 2)
        if second is not None and second.kind == 'num' and not second.article:
            if _is_sym(conn, '-') or _is_word(conn, *(_RANGE_CONNECTORS - {'-'})):
                return i + 3
            if _is_word(conn, 'and') and _is_word(self.tok(i - 1), 'between'):
                return i + 3
            if _is_sym(conn, ',') and self.spoken_range(first, second, i + 3, adjacent=False):
                return i + 3
        if conn is not None and conn.kind == 'num' and not conn.article \
                and self.spoken_range(first, conn, i + 2, adjacent=True):
            return i + 2
        return None

    def spoken_range(self, a: _Tok, b: _Tok, after: int, adjacent: bool) -> bool:
        """'two, three weeks' (a < b <= 2a) or, without the comma, only the
        next integer up ('three four days'). Never before 'times', so that
        'take two, three times a day' keeps its two quantities."""
        av, bv = float(a.value or 0), float(b.value or 0)
        if a.fraction or b.fraction or av < 1 or not (av.is_integer() and bv.is_integer()):
            return False
        if adjacent:
            if bv != av + 1:
                return False
        elif not av < bv <= 2 * av:
            return False
        nxt = self.tok(after)
        return not (_is_word(nxt) and nxt.text in _TIMES_WORDS)

    def period_marker(self, j: int) -> Optional[Tuple[float, int]]:
        """Match '(a|per|each|every|/) PERIOD', 'every N PERIOD', 'every other PERIOD',
        'daily'-style adverbs. Returns (period in days, index after)."""
        t = self.tok(j)
        if t is None:
            return None
        if t.kind == 'word' and t.text in _PERIOD_ADVERBS:
            return _PERIOD_ADVERBS[t.text], j + 1
        if (t.kind == 'num' and t.article) or _is_word(t, 'a', 'an'):
            nxt = self.tok(j + 1)
            if _is_word(nxt) and nxt.text in _PERIOD_DAYS:
                return _PERIOD_DAYS[nxt.text], j + 2
            return None
        if _is_sym(t, '/') or (t.kind == 'word' and t.text in _PERIOD_PREPOSITIONS):
            k = j + 1
            mult = 1.0
            nxt = self.tok(k)
            if _is_word(t, 'every') and nxt is not None:
                if _is_word(nxt, 'other'):
                    mult, k = 2.0, k + 1
                elif nxt.kind == 'word' and nxt.text in _ORDINALS:
                    mult, k = float(_ORDINALS[nxt.text]), k + 1
                elif nxt.kind == 'num' and not nxt.article:
                    mult, k = float(nxt.value or 0), k + 1
                elif nxt.kind == 'ordinal':  # "every 2nd day"
                    digits = _LEADING_DIGITS_RE.match(nxt.text)
                    mult, k = float(digits.group() if digits else 0), k + 1
            nxt = self.tok(k)
            if _is_word(nxt) and nxt.text in _PERIOD_DAYS and mult > 0:
                return mult * _PERIOD_DAYS[nxt.text], k + 1
        return None

    def attach(self, start: int, value: float, value_hi: Optional[float], j: int) -> int:
        """Attach a unit/frequency to the number(s) spanning toks[start:j]."""
        t = self.tok(j)
        article = self.toks[start].article

        # "3 times a day", "2x daily", "3 times" (count)
        if _is_word(t) and t.text in _TIMES_WORDS and not article:
            marker = self.period_marker(j + 1)
            if marker is not None:
                period, end = marker
                self.emit(start, end, value / period, 'per_day', 'frequency', 'frequency',
                          None if value_hi is None else value_hi / period)
                return end
            self.emit(start, j + 1, value, 'times', 'count', 'times', value_hi)
            return j + 1

        # "four daily doses", "3 a day", "2 per week". A fraction never counts
        # events, so "half an hour" falls through to the duration units.
        if not article and value >= 1.0 and float(value).is_integer():
            marker = self.period_marker(j)
            if marker is not None:
                period, end = marker
                if _is_word(self.tok(end)) and _UNIT_ALIASES.get(self.tok(end).text, ('', ''))[1] == 'count':
                    end += 1
                self.emit(start, end, value / period, 'per_day', 'frequency', 'frequency',
                          None if value_hi is None else value_hi / period)
                return end

        # Skip fillers between number and unit: "half a tablet", "one million
        # of units", "half an hour" (where "an" was merged into an article token).
        k = j
        skipped = 0
        while skipped < 2 and (_is_word(self.tok(k), 'of', 'a', 'an', 'international')
                               or _is_sym(self.tok(k), '-') or self.is_article_tok(k)):
            k += 1
            skipped += 1
        u = self.tok(k)
        if _is_word(u):
            handled = self.attach_unit(start, value, value_hi, k, u.text)
            if handled is not None:
                return handled
        if article:
            return j  # a bare "a"/"an" that found no unit is not a quantity
        self.emit_plain(start, j, value, value_hi)
        return j

    def attach_unit(self, start: int, value: float, value_hi: Optional[float], k: int, word: str) -> Optional[int]:
        """Try every unit family at toks[k]. Returns index after the mention or None."""
        nxt = self.tok(k + 1)
        scale = lambda v, f: None if v is None else v * f  # noqa: E731

        if word == 'percent':
            self.emit(start, k + 1, value, 'percent', 'percentage', 'percentage', value_hi)
            return k + 1

        if word == 'degrees' or word in ('celsius', 'centigrade', 'fahrenheit') \
                or (word in ('c', 'f') and self.toks[start].end == self.toks[k].start):
            end, unit = k + 1, 'c'
            if word == 'degrees' and _is_word(nxt, 'celsius', 'centigrade', 'fahrenheit', 'c', 'f'):
                end = k + 2
                unit = 'f' if nxt.text in ('fahrenheit', 'f') else 'c'
            elif word in ('fahrenheit', 'f'):
                unit = 'f'
            conv = _f_to_c if unit == 'f' else (lambda v: v)
            self.emit(start, end, conv(value), unit, 'temperature', 'temperature',
                      None if value_hi is None else conv(value_hi))
            return end

        # Concentration "5.0 mmol/l", "47 mmol per mol"; per-weight dose "2 mg/kg".
        if word in _CONC_NUM and (_is_sym(nxt, '/') or _is_word(nxt, 'per')):
            den = self.tok(k + 2)
            if _is_word(den) and den.text in _CONC_DEN:
                unit = f'{_CONC_NUM[word]}/{_CONC_DEN[den.text]}'
                family = 'dose' if _CONC_DEN[den.text] == 'kg' else 'concentration'
                self.emit(start, k + 3, value, unit, family, unit, value_hi)
                return k + 3

        if word in _DURATION_UNITS and not (word == 'd' and self.toks[start].kind == 'num' and self.toks[start].article):
            name, factor = _DURATION_UNITS[word]
            if word in _GLUED_ONLY and self.toks[start].end != self.toks[k].start:
                return None
            end = k + 1
            frac = _fraction_after_and(self.toks, end) if end < self.n else None
            if frac is not None:
                value, end = value + frac[0], frac[1]
            age_end = self.age_end(end)
            if age_end is not None:
                # "45 years old", "six months of age": an age, never a duration.
                context = tuple(dict.fromkeys(self.context(start, age_end) + ('age',)))
                self.emit(start, age_end, value * factor / 365.0, None, 'plain', 'plain',
                          scale(value_hi, factor / 365.0), context)
                return age_end
            qualifier = None
            if value_hi is None and _is_word(self.tok(end), 'or'):
                after = self.tok(end + 1)
                if _is_word(after, 'so'):
                    qualifier, end = 'approx', end + 2  # "an hour or so"
                elif after is not None and after.kind == 'num' and not after.article and not after.fraction \
                        and value < float(after.value or 0) <= value + 2 \
                        and not (_is_word(self.tok(end + 2)) and self.tok(end + 2).text in _DURATION_UNITS):
                    value_hi, end = float(after.value or 0), end + 2  # "a week or two"
            self.emit(start, end, value * factor, name, 'duration', 'duration', scale(value_hi, factor),
                      qualifier=qualifier)
            return end

        if word == 'second' and self.toks[start].kind == 'num' and self.toks[start].text[:1].isdigit():
            self.emit(start, k + 1, value / 86400, 'second', 'duration', 'duration', scale(value_hi, 1 / 86400))
            return k + 1

        info = _UNIT_ALIASES.get(word)
        if info is None:
            return None
        canonical, family, factor, group = info
        if word in _GLUED_ONLY and self.toks[start].end != self.toks[k].start:
            return None  # "2 g" written apart is left alone; "2g" is a dose
        end = k + 1
        frac = _fraction_after_and(self.toks, end) if end < self.n else None
        if frac is not None:
            value, end = value + frac[0], frac[1]
        if family == 'temperature':
            conv = _f_to_c if canonical == 'f' else (lambda v: v)
            self.emit(start, end, conv(value), canonical, family, group,
                      None if value_hi is None else conv(value_hi))
            return end
        self.emit(start, end, value * factor, canonical, family, group, scale(value_hi, factor))
        if family in ('dose', 'volume', 'count'):
            # "100 mg a day", "100 mg daily", "2 tablets per week" -> also a frequency of 1 per period.
            marker = self.period_marker(end)
            if marker is not None:
                period, fend = marker
                self.emit(end, fend, 1.0 / period, 'per_day', 'frequency', 'frequency')
                return fend
        return end

    def age_end(self, end: int) -> Optional[int]:
        """Index after 'old' / 'of age' following a duration, else None."""
        if _is_word(self.tok(end), 'old'):
            return end + 1
        if _is_word(self.tok(end), 'of') and _is_word(self.tok(end + 1), 'age'):
            return end + 2
        return None

    def emit_plain(self, start: int, end: int, value: float, value_hi: Optional[float]) -> None:
        """A bare number, unless it is clearly not a quantity (year, ordinal, pronoun 'one').

        A bare number next to a temperature word ("her temperature was 38.5",
        "a fever of 101") is promoted to the temperature family, with the scale
        inferred from the magnitude, because clinicians routinely drop the
        unit and the question ("38.5 degrees") would otherwise be unverifiable.
        """
        first = self.toks[start]
        prev, nxt = self.tok(start - 1), self.tok(end)
        if value_hi is None and first.text.isdigit() and 1900 <= value <= 2099:
            return  # a year
        if _is_word(prev) and prev.text in _MONTHS or _is_word(nxt) and nxt.text in _MONTHS:
            return  # a date
        if first.text == 'one' and (_is_word(nxt, 'of') or (_is_word(prev) and prev.text in _ONE_DETERMINERS)):
            return  # pronoun: "one of the medicines", "the one"
        if value >= 1e7:
            return  # phone numbers, identifiers
        context = self.context(start, end)
        if _is_word(prev) and (prev.text in _AGE_WORDS or (prev.text == 'of' and _is_word(self.tok(start - 2), 'age'))):
            context = tuple(dict.fromkeys(context + ('age',)))  # "aged 45", "the age of 45"
        if not _TEMPERATURE_CONTEXT.isdisjoint(context) and _BODY_TEMP_C[0] <= value <= _BODY_TEMP_F[1]:
            fahrenheit = value > _BODY_TEMP_C[1]
            conv = _f_to_c if fahrenheit else (lambda v: v)
            self.emit(start, end, conv(value), 'f' if fahrenheit else 'c', 'temperature', 'temperature',
                      None if value_hi is None else conv(value_hi), context)
            return
        self.emit(start, end, value, None, 'plain', 'plain', value_hi, context)

    # -- word-led patterns ------------------------------------------------ #

    def at_word(self, i: int) -> int:
        t = self.toks[i]
        w = t.text
        if w in _MULTIPLIER_WORDS:
            k = _MULTIPLIER_WORDS[w]
            marker = self.period_marker(i + 1)
            if marker is not None:
                period, end = marker
                self.emit(i, end, k / period, 'per_day', 'frequency', 'frequency')
                return end
            if w != 'once':  # bare "once" is usually the conjunction ("once the swelling settles")
                self.emit(i, i + 1, k, 'times', 'count', 'times')
            return i + 1
        if w in ('every', 'each'):
            marker = self.period_marker(i)
            if marker is not None:
                period, end = marker
                explicit = not (end == i + 2)  # "every day" is a bare period, "every 8 hours" is explicit
                self.emit_frequency(i, end, 1.0 / period, bare=not explicit)
                return end
            return i + 1
        if w in _PERIOD_ADVERBS:
            self.emit_frequency(i, i + 1, 1.0 / _PERIOD_ADVERBS[w], bare=True)
            return i + 1
        if w in _ABBREV_PER_DAY:
            self.emit(i, i + 1, _ABBREV_PER_DAY[w], 'per_day', 'frequency', 'frequency')
            return i + 1
        return i + 1

    def emit_frequency(self, start: int, end: int, per_day: float, bare: bool) -> None:
        """Bare 'daily'/'every week' means *at least* once per period: "take it
        daily" is not contradicted by "twice a day", whereas "once daily" is."""
        self.emit(start, end, per_day, 'per_day', 'frequency', 'frequency')
        if bare:
            last = self.out[-1]
            self.out[-1] = Quantity(last.value, last.unit, last.family, last.raw, last.value_hi,
                                    last.qualifier or 'ge', last.group, last.context)


@lru_cache(maxsize=8192)
def _stem(word: str) -> str:
    """Tiny normaliser for context words: drop possessives and a plural -s."""
    w = word.split("'")[0]
    if len(w) > 4 and w.endswith('s') and not w.endswith('ss'):
        w = w[:-1]
    return w


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def extract_quantities(text: str) -> List[Quantity]:
    """Extract every quantity mention from ``text`` in reading order.

    Non-string input (None, NaN from a CSV) yields an empty list rather than
    an exception because the caller is an inference path that must not fail.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    norm = _normalize(text)
    toks = _merge_numbers(_tokenize(norm), norm)
    return _Scanner(norm, toks).run()


def _tolerance(a: float, b: float, rel: float) -> float:
    return max(_ABS_TOL, rel * max(abs(a), abs(b)))


def _values_match(q: Quantity, e: Quantity) -> bool:
    """True when the evidence quantity satisfies the question quantity.

    Ranges are sets: the two intervals must overlap. Bounds from qualifiers
    ("less than an hour", "at least two weeks") are honoured on either side;
    hedges ("about") widen the tolerance to 10 %, and a unit conversion
    ("70 kg" vs "154 pounds") to 2 %; otherwise equality is 0.1 % relative.
    """
    if 'approx' in (q.qualifier, e.qualifier):
        rel = _APPROX_REL_TOL
    elif q.unit != e.unit:
        rel = _MONTH_REL_TOL if 'month' in (q.unit, e.unit) else _CONVERTED_REL_TOL
    else:
        rel = _REL_TOL
    q_lo, q_hi = q.value, q.value if q.value_hi is None else q.value_hi
    e_lo, e_hi = e.value, e.value if e.value_hi is None else e.value_hi
    tol = _tolerance(max(abs(q_hi), abs(e_hi)), max(abs(q_lo), abs(e_lo)), rel)
    for bound_holder, other_lo, other_hi, ref_lo, ref_hi in (
            (q, e_lo, e_hi, q_lo, q_hi), (e, q_lo, q_hi, e_lo, e_hi)):
        qual = bound_holder.qualifier
        if qual in ('lt', 'le'):
            return other_lo <= ref_hi + tol
        if qual in ('gt', 'ge'):
            return other_hi >= ref_lo - tol
    return q_lo <= e_hi + tol and e_lo <= q_hi + tol


# A question is one sentence; anything beyond this many quantities is not a
# question but a hostile payload, and is simply not checked further.
_MAX_QUESTION_QUANTITIES = 64
# Distinct evidence values listed in a contradiction detail line.
_MAX_DETAIL_CANDIDATES = 10


def _family_label(family: str) -> str:
    return 'pressure (diastolic)' if family == 'pressure_dia' else family


def _comparable(q: Quantity, e: Quantity) -> bool:
    if q.family != e.family or q.group != e.group:
        return False
    if q.family == 'plain':
        return bool(set(q.context) & set(e.context))
    return True


def compare(question_text: str, evidence_text: str) -> FactCheckVerdict:
    """Check every quantity in the question against the evidence.

    Per question quantity: a same-family evidence quantity that matches ->
    consistent; same-family evidence present but none match -> contradiction;
    no same-family evidence -> unverifiable. Bare numbers are only compared
    when both sides share a context word and are otherwise ignored. The
    verdict status is the most severe outcome (contradiction > unverifiable >
    consistent > no_quantities).
    """
    q_quants = extract_quantities(question_text)
    e_quants = extract_quantities(evidence_text)
    # Evidence indexed by (family, group), and bare numbers by context word, so
    # the work is linear in the evidence however many numbers a window holds.
    by_key: Dict[Tuple[str, str], List[Quantity]] = {}
    plain_by_word: Dict[str, List[Tuple[int, Quantity]]] = {}
    for pos, e in enumerate(e_quants):
        if e.family == 'plain':
            for w in e.context:
                plain_by_word.setdefault(w, []).append((pos, e))
        else:
            by_key.setdefault((e.family, e.group), []).append(e)
    statuses: List[str] = []
    details: List[str] = []
    for q in q_quants[:_MAX_QUESTION_QUANTITIES]:
        label = _family_label(q.family)
        if q.family == 'plain':
            found = {pos: e for w in q.context for pos, e in plain_by_word.get(w, ())}
            candidates = [found[pos] for pos in sorted(found)]
        else:
            candidates = by_key.get((q.family, q.group), [])
        if not candidates:
            if q.family == 'plain':
                details.append(f'plain {q.raw}: ignored (no comparable number in evidence)')
            else:
                statuses.append('unverifiable')
                details.append(f'{label} {q.raw}: no {label} in evidence')
            continue
        match = next((e for e in candidates if _values_match(q, e)), None)
        if match is not None:
            statuses.append('consistent')
            details.append(f'{label} {q.raw} matches evidence {match.raw}')
        else:
            statuses.append('contradiction')
            raws = list(dict.fromkeys(e.raw for e in candidates))
            seen = ', '.join(raws[:_MAX_DETAIL_CANDIDATES])
            if len(raws) > _MAX_DETAIL_CANDIDATES:
                seen += f', ... ({len(raws) - _MAX_DETAIL_CANDIDATES} more)'
            details.append(f'{label} {q.raw} vs evidence {seen}')
    status = next((s for s in STATUS_PRECEDENCE if s in statuses), 'no_quantities')
    return FactCheckVerdict(status, list(dict.fromkeys(details)), tuple(q_quants), tuple(e_quants))


_FUZZY_CLEAN_RE = re.compile(r"[^a-z0-9 ]+")


def _fuzzy_norm(text: str) -> str:
    out = text.lower().replace('-', ' ').replace('_', ' ')
    out = _FUZZY_CLEAN_RE.sub('', out)
    return ' '.join(out.split())


def fuzzy_contains(term: str, text: str, threshold: float = 0.8) -> bool:
    """True if ``term`` matches a token or bigram of ``text`` at >= ``threshold``.

    Similarity is ``difflib.SequenceMatcher.ratio`` on normalised strings.
    Measured on this task's names the default 0.8 separates ASR near-misses
    from different drugs with a wide margin: ibumetin/ibumetine 0.94,
    paracetamol/paracetamole 0.96, penicillin/penicillins 0.95 versus
    ibumetin/ibuprofen 0.59, metformin/metoprolol 0.53, losartan/lorazepam
    0.47, pamol/panodil 0.67. Candidates whose length alone caps the ratio
    below the threshold are skipped, which keeps whole-transcript scans cheap.
    """
    if not isinstance(term, str) or not isinstance(text, str):
        return False
    t = _fuzzy_norm(term)
    tokens = _fuzzy_norm(text).split()
    if not t or not tokens:
        return False
    candidates = dict.fromkeys(tokens + [f'{a} {b}' for a, b in zip(tokens, tokens[1:])])
    if t in candidates:
        return True
    lt = len(t)
    matcher = SequenceMatcher(None)
    matcher.set_seq1(t)
    for cand in candidates:
        lc = len(cand)
        if 2.0 * min(lt, lc) / (lt + lc) < threshold:
            continue
        matcher.set_seq2(cand)
        # quick_ratio is an upper bound on ratio and far cheaper; most of a
        # transcript's words fail it, so the real alignment runs rarely.
        if matcher.quick_ratio() < threshold:
            continue
        if matcher.ratio() >= threshold:
            return True
    return False


_DRUG_SUFFIXES = (
    'ol', 'in', 'ine', 'ide', 'one', 'pril', 'sartan', 'statin', 'mycin', 'cillin', 'azole',
    'pam', 'lam', 'mab', 'zumab', 'tinib', 'olol', 'dipine', 'prazole', 'tidine', 'floxacin', 'cycline',
)
# Ordinary English words (>= 5 letters) that happen to end in a drug suffix.
_ENTITY_STOPWORDS = frozenset((
    'within', 'again', 'begin', 'alone', 'phone', 'aside', 'inside', 'outside', 'decide', 'provide',
    'beside', 'guide', 'divide', 'slide', 'pride', 'bride', 'glide', 'stride', 'worldwide', 'nationwide',
    'alongside', 'override', 'reside', 'coincide', 'subside', 'routine', 'machine', 'examine', 'determine',
    'genuine', 'engine', 'online', 'discipline', 'deadline', 'timeline', 'guideline', 'baseline',
    'magazine', 'vaccine', 'medicine', 'combine', 'decline', 'define', 'imagine', 'outline', 'spine',
    'urine', 'school', 'control', 'protocol', 'alcohol', 'symbol', 'patrol', 'explain', 'certain',
    'captain', 'contain', 'maintain', 'remain', 'obtain', 'complain', 'terrain', 'brain', 'train', 'chain',
    'plain', 'grain', 'strain', 'cabin', 'margin', 'origin', 'basin', 'robin', 'resin', 'latin', 'herein',
    'therein', 'wherein', 'someone', 'anyone', 'everyone', 'ozone', 'stone', 'prone', 'throne',
    'telephone', 'headphone', 'microphone', 'postpone', 'undone', 'cyclone', 'milestone', 'backbone',
    'april', 'while', 'whole', 'those', 'these', 'there', 'where', 'could', 'should', 'would', 'about',
    'after', 'before', 'being', 'having', 'doing', 'going', 'until', 'since', 'during', 'patient',
    'doctor', 'nurse', 'clinic', 'hospital', 'appointment', 'question', 'mention', 'discussion',
    'stool', 'petrol', 'pistol', 'cousin', 'domain', 'retain', 'sustain', 'entertain', 'bargain',
    'curtain', 'fountain', 'mountain', 'villain', 'drone', 'shine', 'sunshine', 'refine', 'confine',
    'incline', 'pipeline', 'headline', 'marine', 'divine', 'cuisine', 'pristine', 'preside', 'confide',
    'collide', 'abide', 'upside', 'downside', 'bedside', 'roadside', 'seaside', 'countryside',
    'condone',
))
_ENTITY_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']*")
_SENTENCE_END = frozenset('.?!;:')


def entity_terms(question: str) -> List[str]:
    """Tokens that look like drug or proper names, lower-cased, in order.

    A token qualifies when it is capitalised mid-sentence (Ibumetin, Pamol,
    COVID-19), is an all-caps acronym of three or more letters (LDL, BMI,
    ECG), or is at least five letters and ends in a common drug suffix
    (penicillin, fluconazole, metoprolol), unless it is an everyday English
    word from ``_ENTITY_STOPWORDS``. Sentence-initial capitals are ignored
    because every question starts with one.
    """
    if not isinstance(question, str):
        return []
    found: List[str] = []
    for m in _ENTITY_TOKEN_RE.finditer(question):
        tok = m.group(0)
        if tok.lower().endswith("'s"):
            tok = tok[:-2]
        tok = tok.rstrip("-'")
        if not tok:
            continue
        low = tok.lower()
        if low in _ENTITY_STOPWORDS:
            continue
        before = question[:m.start()].rstrip()
        sentence_initial = not before or before[-1] in _SENTENCE_END or before.endswith('"')
        alpha = low.replace('-', '')
        is_acronym = tok.isupper() and len(alpha) >= 3 and alpha.isalnum()
        is_capitalised = tok[0].isupper() and not sentence_initial and len(alpha) >= 5
        is_druglike = len(alpha) >= 5 and alpha.isalpha() and low.endswith(_DRUG_SUFFIXES)
        if (is_acronym or is_capitalised or is_druglike) and low not in found:
            found.append(low)
    return found
