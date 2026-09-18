"""Rule-based rewriting of yes/no questions into declarative statements.

WHY this module exists: the NLI cross-encoder used by ``qa_backend`` was
trained on premise/hypothesis pairs whose hypothesis is a statement. Fed
"Did the patient attend the follow-up?" as a hypothesis it scores noticeably
worse than with "The patient did attend the follow-up." The rewrite has to be
cheap, deterministic and safe on untrusted text: no ML, no eval, no shelling
out, just tokens, tables and anchored regular expressions.

WHY rules and not a model: the challenge questions come from a handful of
templates (auxiliary inversion, tag questions, existence questions such as
"Is there any mention of X?"), so a small rule set covers them, and every
rule can be inspected when the pipeline answers something wrongly.

Trust model: question text is untrusted data. It is only split, matched
against fixed patterns and re-joined; it never chooses a code path by name,
never touches the file system and never becomes a message role.

Public API
----------
``strip_tag(question)``             remove ", didn't it?"-style tags and "?"
``to_statement(question)``          yes/no question -> declarative statement
``is_existence_question(question)`` "does X come up at all?" style question
``focus_terms(question)``           content tokens for lexical retrieval

Rewrite strategy of ``to_statement`` (in this order)
-----------------------------------------------------
1. ``strip_tag``: drop a trailing tag (", right?", ", didn't it?") and the
   terminal "?".
2. A conversation-referencing frame such as "At any point" or "According to
   the conversation," is dropped; a content-bearing frame such as
   "Alongside asthma," is kept and re-attached in front of the statement.
3. If the remainder does not start with an auxiliary it is already
   declarative (tag questions, statements) and is only re-punctuated.
4. Existence frames become "<topic> is mentioned." / "<topic> is discussed."
   ("mentioned" for mention/reference/indication cues, "discussed" for
   discuss/talk cues; present tense after is/are/does/do, past after
   was/were/did): "Is there any mention of X", "Does the conversation /
   patient / doctor mention X", "Did they talk about X", "Are the speakers
   discussing X", "Is X mentioned", "Was X discussed", "Did X come up".
   Clause-taking cues ("Does the conversation indicate (that) <clause>",
   "Does the patient say that <clause>", "Is it true that <clause>") yield
   the clause itself.
5. Auxiliary inversion: the subject is everything between the auxiliary and
   the first *predicate start* (``_find_predicate_start``); the auxiliary
   moves after the subject and a contracted "n't" becomes "not":
   "Isn't the dose 100 mg?" -> "The dose is not 100 mg."
6. If no predicate start is found the subject is the first noun phrase
   (determiner run plus head noun); if even that leaves no predicate, the
   question minus "?" is returned unchanged.

Only ``re`` and, when importable, ``transcript.normalize_text`` are used; a
local normaliser with the same contract takes over when ``transcript`` is
absent, so the module can be tested in isolation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable, FrozenSet, List, Optional, Sequence, Tuple

__all__ = ['strip_tag', 'to_statement', 'is_existence_question', 'focus_terms']


# --------------------------------------------------------------------------- #
# Text normalisation (shared with transcript.py when available)
# --------------------------------------------------------------------------- #

_QUOTE_TRANSLATION = str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'})
_HYPHENS = '‐‑‒–—−-'
_NUMERIC_JOINERS = frozenset('.,/:')


def _fallback_normalize(text: Optional[str]) -> str:
    """Same contract as ``transcript.normalize_text``: lowercase NFKC text,
    hyphens to spaces, apostrophes dropped inside words, ``. , / :`` kept only
    between digits, other punctuation to spaces, whitespace collapsed.

    Kept in sync with ``transcript.normalize_text`` by a test, so that focus
    terms match transcript tokens whichever normaliser is active.
    """
    if not text:
        return ''
    text = unicodedata.normalize('NFKC', str(text)).translate(_QUOTE_TRANSLATION).lower()
    text = text.translate(str.maketrans(_HYPHENS, ' ' * len(_HYPHENS)))
    chars: List[str] = []
    last = len(text) - 1
    for i, ch in enumerate(text):
        between_digits = 0 < i < last and text[i - 1].isdigit() and text[i + 1].isdigit()
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        elif ch in _NUMERIC_JOINERS and between_digits:
            chars.append(ch)
        elif ch == "'":
            if between_digits:
                chars.append(' ')
        else:
            chars.append(' ')
    return ' '.join(''.join(chars).split())


def _resolve_normalizer() -> Callable[[Optional[str]], str]:
    """``transcript.normalize_text`` when the sibling imports cleanly, else
    the local fallback. Any failure counts (a half-written sibling raises
    more than ``ImportError``), because focus terms must never be the reason
    a request fails."""
    try:
        from transcript import normalize_text
    except Exception:  # pragma: no cover - depends on the sibling being present
        return _fallback_normalize
    return normalize_text


_normalize = _resolve_normalizer()


# --------------------------------------------------------------------------- #
# Lexicon
#
# Curated for the medical-consultation domain and deliberately explicit rather
# than derived from a tagger: a wrong split in a rare case is acceptable, an
# unexplainable one is not.
# --------------------------------------------------------------------------- #

def _fs(words: str) -> FrozenSet[str]:
    """Frozenset from a whitespace-separated word list (readability only)."""
    return frozenset(words.split())


COPULA_AUXILIARIES = _fs('is are was were am')
PERFECT_AUXILIARIES = _fs('has have had')
DO_AUXILIARIES = _fs('do does did')
MODAL_AUXILIARIES = _fs('will would should can could may might must shall')
AUXILIARIES = COPULA_AUXILIARIES | PERFECT_AUXILIARIES | DO_AUXILIARIES | MODAL_AUXILIARIES

# Contracted negative auxiliary -> positive auxiliary; the "not" is
# re-inserted after the moved auxiliary.
NEGATED_AUXILIARIES = {
    "isn't": 'is', "aren't": 'are', "wasn't": 'was', "weren't": 'were',
    "doesn't": 'does', "don't": 'do', "didn't": 'did', "hasn't": 'has',
    "haven't": 'have', "hadn't": 'had', "won't": 'will', "wouldn't": 'would',
    "shouldn't": 'should', "can't": 'can', "couldn't": 'could',
    "mustn't": 'must', "mightn't": 'might', "shan't": 'shall',
}

# Tokens that make the following token part of the subject noun phrase.
DETERMINERS = _fs(
    'the a an my your his her its our their no any some each every both '
    'either neither all another such which what this that these those'
)
PREDICATE_DETERMINERS = _fs('a an the another')
DEMONSTRATIVES = _fs('this that')
# Attributive modifiers likewise protect the next token ("heavy breathing
# reported": the predicate is "reported", not "breathing").
ATTRIBUTIVE_MODIFIERS = _fs(
    'heavy mild severe acute chronic daily weekly monthly nightly morning '
    'evening night current previous recent new old first second third last '
    'next main major minor left right upper lower big small large little '
    'long short same other further additional specific certain particular '
    'general physical mental annual regular routine full whole entire initial '
    'final total average high low normal abnormal painful red hot cold warm '
    'dry wet itchy sore swollen tender viral bacterial fungal local systemic '
    'oral topical possible likely usual standard correct wrong various '
    'several many few more most less least own only early late intense open '
    'prescribed planned reduced increased renewed continued elevated raised '
    'impaired persistent constant mysterious unusual known unknown suspected '
    'confirmed'
)
INTENSIFIERS = _fs(
    'very quite rather so too extremely fairly pretty slightly somewhat '
    'highly relatively really particularly especially'
)
PERSONAL_PRONOUNS = _fs('it he she they you we i there')
INDEFINITE_PRONOUNS = _fs(
    'anyone someone everyone anybody somebody everybody anything something '
    'everything nothing nobody none'
)
PARTITIVE_HEADS = _fs('one part some none all both most many much each either neither half')
CONJUNCTIONS = _fs('and or nor plus')

# Prepositions open a prepositional phrase whose noun phrase is skipped; the
# first one is remembered as a *deferred* predicate start ("Is the pain in
# the lower back" -> "The pain is in the lower back") used only when nothing
# stronger follows. "of" only ever post-modifies the subject.
PREPOSITIONS = _fs(
    'with without on in at for to about over under after before from during '
    'since among between into onto through throughout within despite regarding '
    'concerning towards toward against around near beside behind above below '
    'of like because instead than as via per upon across along off out'
)
NON_DEFERRABLE_PREPOSITIONS = _fs('of')

# Adverbs that can only start the predicate ("Is the patient still in pain").
PREDICATE_ADVERBS = _fs(
    'still also already not no ever never currently now only just otherwise '
    'actually really simply completely entirely fully partly partially mostly '
    'mainly largely generally usually normally typically initially previously '
    'recently later again definitely probably possibly perhaps clearly '
    'explicitly specifically indeed eventually finally immediately soon '
    'presently primarily solely merely even always often sometimes rarely '
    'regularly routinely briefly temporarily permanently gradually suddenly '
    'successfully then thus therefore essentially basically ultimately'
)

# Adjectives used predicatively after a copula ("Is the blood pressure
# elevated"). The suffix heuristic in ``_looks_adjective`` backs this up.
PREDICATIVE_ADJECTIVES = _fs(
    'normal abnormal stable unstable fine clear unclear positive negative good '
    'bad high low elevated raised reduced lowered increased decreased present '
    'absent unchanged changed necessary unnecessary needed planned scheduled '
    'ongoing complete incomplete painful painless tender swollen sore regular '
    'irregular allergic pregnant aware unaware happy unhappy concerned worried '
    'able unable likely unlikely possible impossible probable available ready '
    'well better worse best worst unwell sick ill tired dizzy nauseous feverish '
    'febrile afebrile pale red hot warm cold dry moist intact healthy numb '
    'stiff weak strong constant persistent intermittent chronic acute mild '
    'moderate severe benign malignant suspicious harmless serious urgent viral '
    'bacterial fungal overweight obese underweight enlarged unremarkable '
    'remarkable improved ok okay poor adequate inadequate sufficient '
    'insufficient enough certain uncertain mandatory optional satisfied '
    'engaged motivated compliant adherent tolerated effective ineffective due '
    'free safe unsafe sensitive resistant contagious infectious relevant '
    'related unrelated responsible hereditary genetic long-standing occasional '
    'brief long short sharp dull localised localized widespread unexplained '
    'undetermined unaffected affected controlled uncontrolled dependent '
    'independent consistent inconsistent typical atypical appropriate '
    'inappropriate correct incorrect right wrong true false different similar '
    'unusual usual common rare empty full active inactive alert oriented '
    'confused conscious unconscious alive dead single married retired employed '
    'unemployed busy comfortable uncomfortable symptomatic asymptomatic over '
    'gone done back away here there together alone'
)
# Nouns that the adjective suffix heuristic would otherwise mistake.
_SUFFIX_NOUN_EXCEPTIONS = _fs(
    'patient patients student parent parents president resident client agent '
    'event moment accident incident ingredient nutrient component continent '
    'percent content extent intent consent tent rent cent scent dent segment '
    'assistant consultant infant implant plant restaurant participant '
    'applicant attendant tenant grant pant chant hydrant deodorant '
    'antidepressant antiperspirant decongestant expectorant stimulant '
    'lubricant coolant disinfectant irritant sealant variant covenant '
    'relative objective alternative initiative representative perspective '
    'incentive hive five drive olive native table cable vegetable timetable '
    'bible handful spoonful cupful unless'
)
_ADJECTIVE_SUFFIXES = ('ive', 'ous', 'able', 'ible', 'ful', 'less', 'ent', 'ant')

# Base-form verbs: the main verb after do/does/did and modals.
VERB_BASE = _fs(
    'attend take give prescribe start continue need refer come go receive '
    'mention agree get have complain describe show find remain recommend '
    'advise suggest ask say tell apply affect involve include require occur '
    'happen feel hurt smoke drink live seem appear look sound become contain '
    'cover follow pass own exist lead keep hold bring put set send allow mean '
    'decide understand know think believe suspect worry hope insist deny '
    'refuse accept avoid hear see notice conclude regard consider judge assess '
    'express undergo carry leave indicate confirm develop wish want prefer '
    'intend expect examine treat diagnose discuss talk speak explain discover '
    'detect identify rule respond react improve worsen resolve persist spread '
    'swell bleed itch ache cough vomit faint fall break injure drive eat cook '
    'bake travel play learn study read write buy sell pay cost wear wash meet '
    'arrive begin finish complete manage maintain monitor track book arrange '
    'cancel postpone delay miss skip forget remember lose gain weigh grow '
    'shrink help make do try let ensure provide offer warn inform notify remind '
    'encourage urge promise admit claim argue request demand choose pick select '
    'opt switch adjust reduce decrease raise double halve restart resume renew '
    'repeat attribute relate stem originate radiate tolerate lie sit die reveal '
    'turn wake stand walk move breathe swallow lift bend reach mind reflect '
    'note record measure struggle suffer benefit tend prompt stop use '
    'experience last return stay resemble'
)
# Verbs in "Did the doctor plan X" but nouns in "the treatment plan": accepted
# as predicate start only when the next token is not itself verb-like, which
# would make this one the tail of a compound noun ("the pain level change").
VERB_WEAK = _fs(
    'plan report test check change level increase run sleep work schedule '
    'visit rest order exercise concern interest issue call name point state '
    'matter count end lower sign form cause result function support care '
    'control focus risk trial review transfer discharge'
)
# Irregular past participles (regular ones are caught by the -ed suffix).
IRREGULAR_PARTICIPLES = _fs(
    'been had done gone come become begun seen found felt made said told '
    'brought thought kept left lost met put read run set shown spoken stood '
    'understood worn written grown known drawn fallen forgotten gotten got '
    'risen taken given eaten driven chosen frozen torn sworn thrown blown flown '
    'hidden bitten beaten proven woken stolen swollen broken sent spent built '
    'bought caught taught fought sought held led fed bled paid laid heard meant '
    'dealt learnt burnt hurt cut hit let shut split spread cost quit sat sung '
    'struck stuck won sewn'
)
_ED_NOUN_EXCEPTIONS = _fs(
    'need feed seed speed bleed reed shed weed hundred naked wicked sacred '
    'rugged infrared wretched jagged crooked ragged'
)
_ING_NOUN_EXCEPTIONS = _fs(
    'morning evening thing nothing something anything everything during ring '
    'king sibling clothing wedding building ceiling string spring lightning'
)

NUMBER_WORDS = _fs(
    'one two three four five six seven eight nine ten eleven twelve thirteen '
    'fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty '
    'fifty sixty seventy eighty ninety hundred thousand million half quarter '
    'dozen'
)
NUMBER_QUALIFIERS = _fs('about around roughly approximately nearly almost over under at exactly')
UNITS = _fs(
    'mg g kg mcg µg ug ml l dl mmol mmol/l mmol/mol iu units unit tablets '
    'tablet capsules capsule pills pill drops drop puffs puff doses dose times '
    'time days day weeks week months month years year hours hour minutes minute '
    'seconds second cm mm m km kilos kilo kilograms kilogram grams gram '
    'milligrams milligram micrograms microgram litres liters litre liter '
    'millilitres milliliters percent % degrees beats bpm sessions session '
    'visits visit courses course injections injection prescriptions '
    'prescription hundred thousand million'
)

# Question scaffolding and function words removed from focus terms. Written
# in normalised form (apostrophes vanish, so "didn't" is "didnt").
STOPWORDS = _fs(
    'is are was were am do does did has have had will would should can could '
    'may might must shall the a an any there mention mentioned mentions '
    'mentioning discussed discuss discusses discussing discussion conversation '
    'transcript recording dialogue patient patients doctor doctors physician '
    'clinician nurse gp speakers speaker talk talks talked talking about at '
    'point of to in on for with and or it its this that these those be been '
    'being get got still also right correct didnt isnt wasnt doesnt dont arent '
    'werent hasnt havent hadnt wont wouldnt shouldnt cant couldnt mustnt not '
    'no yes true they their them he she his her him we our you your i my me '
    'who whom whose what which when where why how from by as into onto than '
    'then so if whether either neither both all some each every such only '
    'just very quite rather ever never now here up out off over under during '
    'since after before while because though although but nor yet anything '
    'something nothing anyone someone said say says come came comes indicate '
    'indicates indicated reference references according otherwise really '
    'actually already too own same other another more most much many few '
    'less least'
)


# --------------------------------------------------------------------------- #
# Patterns
# --------------------------------------------------------------------------- #

_AUX_ALTERNATION = '|'.join(sorted(AUXILIARIES | set(NEGATED_AUXILIARIES), key=len, reverse=True))
_AUX_ALTERNATION = _AUX_ALTERNATION.replace("'", "'")

# Tag questions: the tag must be preceded by a comma or a dash so that a
# question ending in "...the right?" or "...said no?" keeps its last word.
_TAG_RE = re.compile(
    r"""
    (?:\s*,\s*|\s+[-–—]\s*)
    (?:
        (?:is|isn't|was|wasn't|does|doesn't|did|didn't|has|hasn't|have|haven't
          |had|hadn't|will|won't|would|wouldn't|can|can't|could|couldn't
          |should|shouldn't|must|mustn't|are|aren't|were|weren't|do|don't)
        \s+(?:it|that|this|there|he|she|they|you|we|i)
      | isn't\s+that\s+(?:right|correct|so|true|the\s+case)
      | is\s+that\s+(?:right|correct|so|true|the\s+case)
      | am\s+i\s+right
      | yes\s+or\s+no
      | or\s+not
      | right | correct | yes | no | true | okay | ok | eh
    )
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CONVERSATION_NOUNS = (
    r'(?:conversation|transcript|recording|dialogue|audio|consultation|discussion|exchange|call)'
)
_SPEAKER_NOUNS = (
    r'(?:speakers?|patient|doctor|physician|clinician|nurse|gp|they|anyone|anybody|either\s+speaker)'
)
_EXISTENCE_ADVERBS = r'(?:ever|also|actually|explicitly|specifically|briefly|at\s+any\s+point|at\s+some\s+point|in\s+passing)'
_CONVERSATION_TAIL = (
    r'\s+(?:in|during|throughout|within|anywhere\s+in|at\s+any\s+point\s+(?:in|during))'
    r'\s+(?:the|this)\s+(?:conversation|transcript|recording|consultation|dialogue|discussion|audio|call|visit|appointment)$'
)

# Frames that only point at the conversation are dropped ("At any point does
# the patient..."); any other leading phrase before a comma is content and is
# re-attached ("Alongside asthma, is the patient treated for diabetes?").
_DROPPED_FRAME_RE = re.compile(
    r'^(?:at\s+any\s+(?:point|time|stage)|at\s+some\s+point|anywhere'
    r'|(?:according\s+to|in|during|from|based\s+on|throughout|within|per)\s+(?:the|this)\s+'
    + _CONVERSATION_NOUNS + r')\s*,?\s*',
    re.IGNORECASE,
)

_EXISTENCE_VERBS = _fs('mention mentions discuss discusses talk_about talks_about bring_up brings_up '
                       'refer_to refers_to touch_on touches_on')
_CLAUSE_VERBS = _fs('indicate indicates suggest suggests show shows reveal reveals confirm confirms '
                    'imply implies state states say says note notes report reports claim claims '
                    'explain explains')
_DISCUSSED_CUES = re.compile(r'^(?:discuss|talk|discussion)')

_THERE_RE = re.compile(
    r'^(is|was|are|were)\s+there\s+(?:(?:any|some|a|an)\s+)?(?:(?:explicit|specific|direct|clear|brief|passing)\s+)?'
    r'(mention|mentions|talk|reference|references|discussion|discussions|indication|indications|hint|hints'
    r'|evidence|suggestion|sign|signs)\s+(of|about|to|that|regarding|concerning)\s+(.+)$',
    re.IGNORECASE,
)
_SUBJECT_VERB_RE = re.compile(
    r'^(does|did|do)\s+(?:the\s+)?(' + _CONVERSATION_NOUNS + '|' + _SPEAKER_NOUNS + r')'
    r'(?:\s+' + _EXISTENCE_ADVERBS + r')*\s+'
    r'(mention|mentions|discuss|discusses|talk\s+about|talks\s+about|bring\s+up|brings\s+up|refer\s+to|refers\s+to'
    r'|touch\s+on|touches\s+on|indicate|indicates|suggest|suggests|show|shows|reveal|reveals|confirm|confirms'
    r'|imply|implies|state|states|say|says|note|notes|report|reports|claim|claims|explain|explains)\s+(.+)$',
    re.IGNORECASE,
)
_PROGRESSIVE_RE = re.compile(
    r'^(is|are|was|were)\s+(?:the\s+)?(?:' + _SPEAKER_NOUNS + r'|conversation|two\s+of\s+them|both)'
    r'(?:\s+' + _EXISTENCE_ADVERBS + r')*\s+'
    r'(discussing|talking\s+about|mentioning|referring\s+to|bringing\s+up|touching\s+on)\s+(.+)$',
    re.IGNORECASE,
)
_PASSIVE_RE = re.compile(
    r'^(is|are|was|were)\s+(.+?)\s+(?:' + _EXISTENCE_ADVERBS + r'\s+)?'
    r'(mentioned|discussed|brought\s+up|talked\s+about|touched\s+on)'
    r'(\s+(?:at\s+all|anywhere|at\s+any\s+point|in|during|throughout|within)\b.*)?$',
    re.IGNORECASE,
)
_COME_UP_RE = re.compile(
    r'^(did|does|do)\s+(.+?)\s+(?:ever\s+)?(?:come|comes)\s+up'
    r'(\s+(?:at\s+all|anywhere|at\s+any\s+point|in|during|throughout|within)\b.*)?$',
    re.IGNORECASE,
)
_IT_TRUE_RE = re.compile(
    r'^(?:is|was)\s+it\s+(?:the\s+case|true|correct|right|accurate|fair\s+to\s+say|confirmed|stated|mentioned|said)'
    r'\s+that\s+(.+)$',
    re.IGNORECASE,
)
_OBJECT_PREFIX_RE = re.compile(
    r'^(?:(?:anything|something|things|the\s+fact)\s+(about|that)|(that))\s+(.+)$', re.IGNORECASE
)
_PRONOUN_CLAUSE_RE = re.compile(r'^(?:he|she|they|it|we|i|you|there|this)\b', re.IGNORECASE)

_EXISTENCE_CUE_RE = re.compile(
    r'\b(?:mention(?:s|ed|ing)?|discuss(?:es|ed|ing)?|discussions?|talk(?:s|ed|ing)?\s+(?:about|of)'
    r'|(?:come|comes|came|coming)\s+up|(?:bring|brings|brought|bringing)\s+up|references?\s+to'
    r'|touch(?:es|ed)?\s+on|any\s+(?:indication|hint|evidence|sign|signs|suggestion)\b)',
    re.IGNORECASE,
)
_CONVERSATION_SUBJECT_RE = re.compile(
    r'^(?:does|did|do|is|was)\s+(?:the\s+)?' + _CONVERSATION_NOUNS + r'\b', re.IGNORECASE
)

_TOKEN_PUNCTUATION = '"\'()[]{},;:.!?'
_TRAILING_PUNCTUATION = ' .!?,;:'


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _prepare_text(text: str) -> str:
    """Straight quotes and single spaces; nothing else changes."""
    return ' '.join(text.translate(_QUOTE_TRANSLATION).split())


def _clean_token(token: str) -> str:
    """Lowercased token without surrounding punctuation, for table lookups.

    Internal punctuation stays ("patient's", "check-up", "155/98") because
    it carries the word's identity; only the wrapping is noise.
    """
    return token.strip(_TOKEN_PUNCTUATION).lower()


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lower_first_function_word(text: str) -> str:
    """Lowercase the first word when it is a function word, so a kept frame
    can be put in front of it without producing "Alongside asthma, The ..."
    while a proper noun keeps its capital."""
    first = _clean_token(text.split(' ', 1)[0]) if text else ''
    if first in DETERMINERS or first in PERSONAL_PRONOUNS:
        return text[:1].lower() + text[1:]
    return text


def _finish(text: str) -> str:
    """A statement ends with exactly one period and starts with a capital."""
    text = text.strip().rstrip(_TRAILING_PUNCTUATION).strip()
    return _capitalize_first(text) + '.' if text else ''


def _aux_of(token: str) -> Tuple[Optional[str], bool]:
    """``(auxiliary, negated)`` for a leading token, ``(None, False)`` otherwise."""
    if token in NEGATED_AUXILIARIES:
        return NEGATED_AUXILIARIES[token], True
    if token in AUXILIARIES:
        return token, False
    return None, False


def _aux_class(aux: str) -> str:
    if aux in COPULA_AUXILIARIES:
        return 'copula'
    if aux in PERFECT_AUXILIARIES:
        return 'perfect'
    if aux in DO_AUXILIARIES:
        return 'do'
    return 'modal'


def _be_form(aux: str) -> str:
    """Tense of the existence statement follows the question's auxiliary."""
    return 'was' if aux in ('was', 'were', 'did') else 'is'


def _existence_verb(cue: str) -> str:
    return 'discussed' if _DISCUSSED_CUES.match(cue) else 'mentioned'


def _strip_conversation_tail(topic: str) -> str:
    """"baking a cake in the conversation" -> "baking a cake"."""
    return re.sub(_CONVERSATION_TAIL, '', topic, flags=re.IGNORECASE).strip()


def _contains_auxiliary(text: str) -> bool:
    return any(_clean_token(t) in AUXILIARIES for t in text.split())


# --------------------------------------------------------------------------- #
# Token classification for auxiliary inversion
# --------------------------------------------------------------------------- #

def _is_possessive(token: str) -> bool:
    return token.endswith("'s") or (token.endswith("s'") and len(token) > 2)


def _is_digit_token(token: str) -> bool:
    return bool(token) and token[0].isdigit()


def _protects(token: str) -> bool:
    """Whether ``token`` binds the following token into the same noun phrase."""
    return (
        token in DETERMINERS
        or token in ATTRIBUTIVE_MODIFIERS
        or token in INTENSIFIERS
        or token in NUMBER_WORDS
        or token in INDEFINITE_PRONOUNS
        or _is_possessive(token)
        or _is_digit_token(token)
    )


def _looks_participle(token: str) -> bool:
    if token in IRREGULAR_PARTICIPLES:
        return True
    return len(token) >= 4 and token.endswith('ed') and token not in _ED_NOUN_EXCEPTIONS


def _looks_gerund(token: str) -> bool:
    return len(token) > 4 and token.endswith('ing') and token not in _ING_NOUN_EXCEPTIONS


def _looks_adjective(token: str) -> bool:
    """Suffix heuristic for adjectives outside ``PREDICATIVE_ADJECTIVES``.

    Only suffixes with few noun collisions are used and the known collisions
    are listed, because a false positive splits the subject in the middle.
    """
    if token in PREDICATIVE_ADJECTIVES:
        return True
    if len(token) < 5 or token in _SUFFIX_NOUN_EXCEPTIONS or token.endswith('ment'):
        return False
    return token.endswith(_ADJECTIVE_SUFFIXES)


def _is_number(token: str, following: Optional[str]) -> bool:
    """A digit token, or a number word that is followed by a unit or another
    number word ("two weeks", "one million")."""
    if _is_digit_token(token):
        return True
    return token in NUMBER_WORDS and following is not None and (following in UNITS or following in NUMBER_WORDS)


def _verb_like(token: Optional[str]) -> bool:
    if token is None:
        return False
    return (
        token in VERB_BASE or token in VERB_WEAK or token in AUXILIARIES
        or token in ('be', 'been', 'being') or _looks_participle(token) or _looks_gerund(token)
    )


def _skip_noun_phrase(clean: Sequence[str], start: int) -> int:
    """Index just past a noun phrase beginning at ``start``: a run of
    determiners / modifiers / numbers followed by one head token."""
    n = len(clean)
    j = start
    while j < n and _protects(clean[j]):
        j += 1
    if j < n:
        j += 1
    return j


def _strong_predicate_start(clean: Sequence[str], i: int, aux_class: str) -> bool:
    """Whether token ``i`` (never the first token) certainly opens the predicate."""
    tok = clean[i]
    following = clean[i + 1] if i + 1 < len(clean) else None
    if tok in PREDICATE_ADVERBS or tok in ('be', 'been', 'being'):
        return True
    if tok == 'to' and following is not None and (
        following in ('be', 'have') or following in VERB_BASE or following in VERB_WEAK
    ):
        return True
    if aux_class in ('copula', 'perfect') and (_looks_participle(tok) or _looks_gerund(tok)):
        return True
    if aux_class == 'copula':
        if _looks_adjective(tok) or _is_number(tok, following):
            return True
        if tok in NUMBER_QUALIFIERS and following is not None and _is_number(following, None):
            return True
        if tok in PREDICATE_DETERMINERS or tok in INDEFINITE_PRONOUNS:
            return True
        if tok in PARTITIVE_HEADS and following == 'of':
            return True
    if aux_class == 'perfect' and tok in VERB_BASE:
        return True
    if aux_class in ('do', 'modal'):
        if tok in VERB_BASE or (aux_class == 'modal' and tok == 'have'):
            return True
        if tok in VERB_WEAK and not _verb_like(following):
            return True
    return False


def _find_predicate_start(clean: Sequence[str], aux_class: str) -> Optional[int]:
    """Index of the first predicate token in the words after the auxiliary.

    The first token is always part of the subject. Tokens bound to a
    determiner or modifier are skipped, prepositional phrases and
    conjunction-joined noun phrases are stepped over, and the first
    preposition is remembered as a fallback start ("Is the pain in the
    back" has no stronger predicate). ``None`` when nothing qualifies.
    """
    n = len(clean)
    if n == 0:
        return None
    first = clean[0]
    if n > 1 and (first in PERSONAL_PRONOUNS or (first in DEMONSTRATIVES and _demonstrative_is_pronoun(clean))):
        return 1
    deferred: Optional[int] = None
    i = 1
    while i < n:
        prev = clean[i - 1]
        tok = clean[i]
        if prev in CONJUNCTIONS or prev in PREPOSITIONS:
            i = _skip_noun_phrase(clean, i)
            continue
        protected = _protects(prev) and not (prev in INDEFINITE_PRONOUNS and i == n - 1)
        if protected or tok in CONJUNCTIONS or not tok:
            i += 1
            continue
        if _strong_predicate_start(clean, i, aux_class):
            return i
        if tok in PREPOSITIONS:
            if deferred is None and tok not in NON_DEFERRABLE_PREPOSITIONS:
                deferred = i
        i += 1
    return deferred


def _demonstrative_is_pronoun(clean: Sequence[str]) -> bool:
    """"Is that correct" (pronoun) versus "Did that grandfather die" (determiner)."""
    if len(clean) == 2:
        return True
    following = clean[1]
    return (
        following in DETERMINERS or following in PREDICATE_ADVERBS or following in PREDICATIVE_ADJECTIVES
        or following in ('be', 'been', 'being') or _is_digit_token(following) or _looks_participle(following)
    )


def _noun_phrase_end(clean: Sequence[str]) -> Optional[int]:
    """Fallback subject boundary: determiner run plus head noun, provided a
    predicate remains after it."""
    end = _skip_noun_phrase(clean, 0)
    return end if 0 < end < len(clean) else None


# --------------------------------------------------------------------------- #
# Existence rewrites
# --------------------------------------------------------------------------- #

def _existence_statement(topic: str, aux: str, cue: str) -> Optional[str]:
    topic = _strip_conversation_tail(topic).strip(_TRAILING_PUNCTUATION)
    if not topic:
        return None
    return f'{_capitalize_first(topic)} {_be_form(aux)} {_existence_verb(cue)}'


def _clause_statement(clause: str) -> Optional[str]:
    clause = re.sub(r'^that\s+', '', clause.strip(), flags=re.IGNORECASE)
    clause = _strip_conversation_tail(clause).strip(_TRAILING_PUNCTUATION)
    return _capitalize_first(clause) if clause else None


def _rewrite_existence(core: str) -> Optional[str]:
    """Statement for an existence-shaped question, else ``None``."""
    match = _IT_TRUE_RE.match(core)
    if match:
        return _clause_statement(match.group(1))

    match = _THERE_RE.match(core)
    if match:
        aux, cue, connector, rest = match.group(1).lower(), match.group(2).lower(), match.group(3).lower(), match.group(4)
        if connector == 'that':
            return _clause_statement(rest)
        return _existence_statement(rest, aux, cue)

    match = _SUBJECT_VERB_RE.match(core)
    if match:
        aux, subject, verb, rest = match.group(1).lower(), match.group(2).lower(), match.group(3).lower(), match.group(4)
        verb = '_'.join(verb.split())
        if verb in _EXISTENCE_VERBS:
            prefix = _OBJECT_PREFIX_RE.match(rest)
            if prefix:
                connector = (prefix.group(1) or prefix.group(2)).lower()
                if connector == 'that':
                    return _clause_statement(prefix.group(3))
                rest = prefix.group(3)
            return _existence_statement(rest, aux, verb)
        conversation_subject = re.fullmatch(_CONVERSATION_NOUNS, subject) is not None
        if conversation_subject or rest.lower().startswith('that ') or _PRONOUN_CLAUSE_RE.match(rest):
            return _clause_statement(rest)
        return None

    match = _PROGRESSIVE_RE.match(core)
    if match:
        return _existence_statement(match.group(3), match.group(1).lower(), match.group(2).lower())

    match = _PASSIVE_RE.match(core)
    if match and not _contains_auxiliary(match.group(2)):
        return _existence_statement(match.group(2), match.group(1).lower(), match.group(3).lower())

    match = _COME_UP_RE.match(core)
    if match and not _contains_auxiliary(match.group(2)):
        return _existence_statement(match.group(2), match.group(1).lower(), 'mention')
    return None


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #

def _split_frame(core: str) -> Tuple[Optional[str], str]:
    """``(kept_frame, remainder)``; conversation-only frames are dropped.

    A frame is a leading phrase before a comma when the text after the comma
    starts with an auxiliary and the phrase itself does not.
    """
    stripped = _DROPPED_FRAME_RE.sub('', core, count=1).strip()
    if stripped and stripped != core:
        core = stripped
    head, sep, tail = core.partition(',')
    if not sep:
        return None, core
    head, tail = head.strip(), tail.strip()
    if not head or not tail or _aux_of(_clean_token(head.split()[0]))[0] is not None:
        return None, core
    if _aux_of(_clean_token(tail.split()[0]))[0] is None:
        return None, core
    return head, tail


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def strip_tag(question: object) -> str:
    """Question without a trailing tag (", didn't it?", ", right?") and
    without its terminal "?"; whitespace collapsed. Non-strings give ''.

    The tag must follow a comma or dash: "...found under the right breast?"
    keeps "breast", "...the plan involves no treatment, right?" loses "right".
    """
    if not isinstance(question, str):
        return ''
    text = _prepare_text(question).rstrip(_TRAILING_PUNCTUATION)
    match = _TAG_RE.search(text)
    if match:
        text = text[:match.start()]
    return text.rstrip(_TRAILING_PUNCTUATION).strip()


def to_statement(question: object) -> str:
    """Declarative form of a yes/no question (see the module docstring for
    the rule order). Returns '' for empty or non-string input, and a
    statement is returned unchanged apart from punctuation, so the function
    is idempotent.
    """
    core = strip_tag(question)
    if not core:
        return ''
    frame, core = _split_frame(core)
    if not core:
        return _finish(frame or '')
    statement = _rewrite_core(core)
    if frame:
        statement = f'{frame}, {_lower_first_function_word(statement)}'
    return _finish(statement)


def _rewrite_core(core: str) -> str:
    tokens = core.split()
    clean = [_clean_token(t) for t in tokens]
    aux, negated = _aux_of(clean[0])
    if aux is None:
        return core  # already declarative
    existence = _rewrite_existence(core)
    if existence:
        return existence
    rest_tokens, rest_clean = tokens[1:], clean[1:]
    if not rest_tokens:
        return core
    split = _find_predicate_start(rest_clean, _aux_class(aux))
    if split is None:
        split = _noun_phrase_end(rest_clean)
    if split is None:
        return core
    subject, predicate = rest_tokens[:split], rest_tokens[split:]
    if rest_clean[:split] == ['there'] and predicate and _clean_token(predicate[0]) == 'any':
        predicate = predicate[1:]  # "Were there any signs" -> "There were signs"
    parts = subject + [aux] + (['not'] if negated else []) + predicate
    return ' '.join(parts)


def is_existence_question(question: object) -> bool:
    """Whether the question only asks if a topic came up at all ("any
    mention of", "discussed", "talk about", "come up", questions about what
    the conversation itself indicates). Content questions such as "Was the
    patient referred to a dietitian?" are not existence questions."""
    core = strip_tag(question)
    if not core:
        return False
    _, core = _split_frame(core)
    return bool(_EXISTENCE_CUE_RE.search(core) or _CONVERSATION_SUBJECT_RE.match(core))


def focus_terms(question: object) -> List[str]:
    """Content tokens of the question in order of first appearance, without
    stopwords or question scaffolding, normalised like transcript text so
    they can be matched against transcript tokens. Empty for empty or
    non-string input."""
    core = strip_tag(question)
    if not core:
        return []
    terms: List[str] = []
    seen = set()
    for token in _normalize(core).split():
        if token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return terms
