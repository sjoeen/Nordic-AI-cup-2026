"""Tests for qa_backend with fake models only (one real-model test is slow).

The fakes are deliberately simple so that the assertions test the backend's
decision logic, not the models: the embedder is a hashed bag of content
words, the NLI entails when the hypothesis' content words are in the premise
and contradicts when the numbers disagree.
"""

import hashlib
import os

import numpy as np
import pytest

import qa_backend
from core_types import Deadline, QAResult
from qa_backend import (
    FakeQABackend,
    LexicalBackend,
    LocalVerdict,
    QAConfig,
    RetrievalNLIBackend,
    build_qa_backend,
    focus_tokens,
    lexical_score,
    resolve_device,
    resolve_nli_label_order,
    softmax,
    sorted_windows,
    tokenize_for_match,
)
from tests.conftest import make_context

SIBLING_HOOKS = {
    'transcript_windows': ('transcript', 'windows'),
    'rewrite_to_statement': ('question_rewrite', 'to_statement'),
    'rewrite_focus_terms': ('question_rewrite', 'focus_terms'),
    'rewrite_is_existence_question': ('question_rewrite', 'is_existence_question'),
    'factcheck_compare': ('factcheck', 'compare'),
}


@pytest.fixture(autouse=True)
def isolate_from_siblings(monkeypatch):
    """Pin qa_backend to its local fallbacks so these unit tests are
    deterministic whether or not the sibling modules exist yet. The smoke
    test below re-attaches whatever siblings are importable."""
    for hook in SIBLING_HOOKS:
        monkeypatch.setattr(qa_backend, hook, None)


def _attach_real_siblings(monkeypatch):
    import importlib
    attached = []
    for hook, (module_name, attr) in SIBLING_HOOKS.items():
        try:
            fn = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            continue
        monkeypatch.setattr(qa_backend, hook, fn)
        attached.append(hook)
    return attached


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

DIM = 512


def _content_tokens(text):
    return [t for t in tokenize_for_match(text) if t not in qa_backend._FALLBACK_STOPWORDS]


def _slot(token):
    return int(hashlib.sha1(token.encode('utf-8')).hexdigest(), 16) % DIM


class FakeEmbedder:
    """Hashed bag of content words, L2-normalised; zero vector for no words."""

    def __init__(self):
        self.calls = 0

    def encode(self, texts, batch_size=32):
        self.calls += 1
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _content_tokens(text):
                out[i, _slot(tok)] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


ENTAIL = np.array([-2.0, 3.0, -1.0])
CONTRA = np.array([3.0, -2.0, -1.0])
NEUTRAL = np.array([-1.0, -1.0, 2.0])


def _fake_nli_logits(premise, hypothesis):
    prem = set(tokenize_for_match(premise))
    hyp = _content_tokens(hypothesis)
    hyp_numbers = {t for t in hyp if qa_backend.is_number_token(t)}
    prem_numbers = {t for t in prem if qa_backend.is_number_token(t)}
    if hyp_numbers and prem_numbers and not (hyp_numbers & prem_numbers):
        return CONTRA
    if hyp and sum(1 for t in hyp if t in prem) / len(hyp) >= 0.6:
        return ENTAIL
    return NEUTRAL


class FakeNLI:
    """Logits in the canonical [contradiction, entailment, neutral] order."""

    label_names = ['contradiction', 'entailment', 'neutral']

    def __init__(self):
        self.calls = 0
        self.pairs_seen = 0

    def predict(self, pairs, batch_size=32):
        self.calls += 1
        self.pairs_seen += len(pairs)
        return np.stack([_fake_nli_logits(p, h) for p, h in pairs])


class _Config:
    def __init__(self, id2label):
        self.id2label = id2label


class _Model:
    def __init__(self, id2label):
        self.config = _Config(id2label)


class SwappedFakeNLI(FakeNLI):
    """Same model, but reports and emits columns as [entailment, neutral, contradiction]
    through a CrossEncoder-shaped ``model.config.id2label``."""

    label_names = None

    def __init__(self):
        super().__init__()
        self.model = _Model({'0': 'ENTAILMENT', '1': 'NEUTRAL', '2': 'CONTRADICTION'})

    def predict(self, pairs, batch_size=32):
        canonical = super().predict(pairs, batch_size)
        return canonical[:, [1, 2, 0]]


class AlwaysEntailsNLI(FakeNLI):
    def predict(self, pairs, batch_size=32):
        self.calls += 1
        return np.tile(ENTAIL, (len(pairs), 1))


QUESTIONS = [
    'Did the doctor start the patient on penicillin 100 milligrams daily?',   # 0 yes, unit 2
    'Did the doctor start the patient on penicillin 200 mg daily?',          # 1 no: wrong dose
    'Will the treatment last six weeks?',                                    # 2 no: wrong duration
    'Should the medication be taken on an empty stomach?',                   # 3 no: negated
    'Did they discuss the concert next Friday?',                             # 4 no: off topic
    'Was there any mention of blood pressure?',                              # 5 yes, unit 5
]


def _backend(config=None, nli=None, embedder=None):
    return RetrievalNLIBackend(config or QAConfig(), embedder=embedder or FakeEmbedder(), nli=nli or FakeNLI())


# --------------------------------------------------------------------------- #
# Tokenisation and scoring helpers
# --------------------------------------------------------------------------- #

def test_tokenize_keeps_numbers_and_canonicalises_units_and_plurals():
    assert tokenize_for_match('Take 0.3 ml, BP 130/85, two weeks, 100 Milligrams!') == \
        ['take', '0.3', 'ml', 'bp', '130/85', '2', 'week', '100', 'mg']
    assert tokenize_for_match('') == []
    assert tokenize_for_match(None) == []


def test_focus_tokens_drop_scaffolding_and_duplicates():
    terms = focus_tokens('Was there any mention of blood pressure, the blood pressure?')
    assert terms == ['blood', 'pressure']
    assert focus_tokens('') == []


def test_lexical_score_fraction_bonus_and_cap():
    window = set(tokenize_for_match('penicillin 100 milligrams daily'))
    assert lexical_score([], window) == 0.0
    assert lexical_score(['penicillin', 'daily'], window) == 1.0
    assert lexical_score(['penicillin', 'tablet'], window) == pytest.approx(0.5)
    # exact number match adds a bonus on top of the fraction, capped at 1.0
    assert lexical_score(['penicillin', '100', 'tablet', 'weekly'], window) == pytest.approx(0.6)
    assert lexical_score(['100'], window) == 1.0
    # a window with a different number than the question is penalised, never below 0
    assert lexical_score(['penicillin', '200', 'mg', 'daily'], window) == pytest.approx(0.75 - 0.3)
    assert lexical_score(['200'], window) == 0.0
    # ...but not when the window states no number at all (it simply lacks the fact)
    assert lexical_score(['penicillin', '200'], {'penicillin', 'daily'}) == pytest.approx(0.5)


def test_softmax_rows_sum_to_one_and_survive_nan():
    probs = softmax(np.array([[1.0, 2.0, 3.0], [np.nan, np.inf, -np.inf]]))
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.argmax(probs[0]) == 2
    assert np.allclose(probs[1], 1 / 3)


def test_sorted_windows_enumerates_runs_in_order(clinic_context):
    wins = sorted_windows(clinic_context, 3)
    n = len(clinic_context.units)
    assert len(wins) == n + (n - 1) + (n - 2)
    assert [w.unit_ids for w in wins[:3]] == [(0,), (0, 1), (0, 1, 2)]
    assert wins[-1].unit_ids == (7,)
    assert wins[1].text.startswith('Good morning') and wins[1].end_s == clinic_context.units[1].end_s
    assert sorted_windows(make_context([]), 3) == []
    single = sorted_windows(make_context(['Only one line.']), 3)
    assert [w.unit_ids for w in single] == [(0,)]


def test_resolve_device_maps_auto_to_library_default():
    assert resolve_device('auto') is None and resolve_device(' AUTO ') is None
    assert resolve_device('') is None and resolve_device(None) is None
    assert resolve_device('cpu') == 'cpu' and resolve_device('CUDA:0') == 'cuda:0'


def test_resolve_nli_label_order_from_names_config_and_fallback():
    assert resolve_nli_label_order(FakeNLI()) == ([0, 1, 2], 'label_names')
    assert resolve_nli_label_order(SwappedFakeNLI()) == ([2, 0, 1], 'id2label')
    assert resolve_nli_label_order(object()) == ([0, 1, 2], 'assumed')
    assert resolve_nli_label_order(_Model({0: 'LABEL_0', 1: 'LABEL_1', 2: 'LABEL_2'})) == ([0, 1, 2], 'assumed')


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def test_config_from_env_parses_every_type_and_ignores_unset():
    env = {
        'MA_QA_BACKEND': 'lexical', 'MA_QA_TOP_K': '3', 'MA_QA_YES_THRESHOLD': '0.7',
        'MA_QA_USE_FACTCHECK': 'off', 'MA_QA_UNVERIFIABLE_MEANS_NO': 'true',
        'MA_QA_DEVICE': 'cpu', 'MA_QA_EMBED_MODEL': 'x/y', 'MA_QA_NLI_MODEL': 'x/z',
        'MA_QA_MAX_WINDOW_UNITS': '2', 'MA_QA_OFF_TOPIC_SIM_THRESHOLD': '0.1',
        'MA_QA_LEXICAL_WEIGHT': '0.5', 'MA_QA_BATCH_SIZE': '8', 'OTHER': 'ignored',
    }
    cfg = QAConfig.from_env('MA_QA_', env=env)
    assert (cfg.backend, cfg.top_k, cfg.yes_threshold, cfg.use_factcheck) == ('lexical', 3, 0.7, False)
    assert cfg.unverifiable_means_no is True and cfg.batch_size == 8 and cfg.lexical_weight == 0.5
    assert cfg.embed_model == 'x/y' and cfg.nli_model == 'x/z' and cfg.max_window_units == 2
    assert cfg.min_time_for_nli_s == 3.0  # untouched default
    assert QAConfig.from_env('MA_QA_', env={}) == QAConfig()
    assert cfg.config_hash != QAConfig().config_hash


@pytest.mark.parametrize('env', [
    {'MA_QA_TOP_K': 'six'}, {'MA_QA_USE_FACTCHECK': 'maybe'}, {'MA_QA_YES_THRESHOLD': '1.5'},
    {'MA_QA_TOP_K': '0'}, {'MA_QA_LEXICAL_WEIGHT': 'nan'},
])
def test_config_rejects_bad_values(env):
    with pytest.raises(ValueError):
        QAConfig.from_env('MA_QA_', env=env)


# --------------------------------------------------------------------------- #
# Lexical backend
# --------------------------------------------------------------------------- #

def test_lexical_backend_baseline(clinic_context):
    results = LexicalBackend(QAConfig(backend='lexical')).answer(clinic_context, QUESTIONS)
    assert [r.question_index for r in results] == list(range(len(QUESTIONS)))
    assert all(r.backend == 'lexical' for r in results)
    assert results[0].answer is True and results[0].evidence_unit_ids == [2]
    assert results[5].answer is True and results[5].evidence_unit_ids == [5]
    assert results[4].answer is False and results[4].confidence == 0.0
    # "200 mg": 4 of 5 terms match but the window's number disagrees -> penalised below 0.6
    assert results[1].answer is False and results[1].diagnostics['score'] == pytest.approx(0.5, abs=0.01)
    assert results[2].answer is False  # "six weeks" vs "two weeks"
    assert 0.0 <= results[0].confidence <= 1.0


def test_lexical_backend_factcheck_gate_and_edge_inputs(clinic_context, monkeypatch):
    monkeypatch.setattr(qa_backend, 'factcheck_compare',
                        lambda q, t: LocalVerdict('contradiction', ['100 mg != 200 mg']))
    gated = LexicalBackend().answer(clinic_context, [QUESTIONS[0]])
    assert gated[0].answer is False and gated[0].diagnostics['verdict'] == 'contradiction'
    assert gated[0].evidence_unit_ids == [2]  # still reports what it looked at

    assert LexicalBackend().answer(clinic_context, []) == []
    empty = LexicalBackend().answer(make_context([]), QUESTIONS[:2])
    assert [r.answer for r in empty] == [False, False]
    assert all(r.evidence_unit_ids == [] and r.diagnostics['reason'] == 'no_transcript' for r in empty)
    odd = LexicalBackend().answer(clinic_context, ['', '   ', 'Is it?'])
    assert [r.diagnostics['reason'] for r in odd] == ['empty_question', 'empty_question', 'no_focus_terms']
    assert all(r.answer is False for r in odd)


def test_lexical_backend_unverifiable_gate_is_configurable(clinic_context, monkeypatch):
    monkeypatch.setattr(qa_backend, 'factcheck_compare', lambda q, t: LocalVerdict('unverifiable', []))
    assert LexicalBackend(QAConfig(unverifiable_means_no=False)).answer(clinic_context, [QUESTIONS[0]])[0].answer is True
    assert LexicalBackend(QAConfig(unverifiable_means_no=True)).answer(clinic_context, [QUESTIONS[0]])[0].answer is False
    disabled = LexicalBackend(QAConfig(use_factcheck=False, unverifiable_means_no=True))
    assert disabled.answer(clinic_context, [QUESTIONS[0]])[0].diagnostics['verdict'] == 'no_quantities'


# --------------------------------------------------------------------------- #
# Retrieval + NLI backend
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('nli_cls', [FakeNLI, SwappedFakeNLI])
def test_retrieval_nli_decisions_on_clinic(clinic_context, nli_cls):
    nli = nli_cls()
    backend = _backend(nli=nli)
    results = backend.answer(clinic_context, QUESTIONS)

    assert len(results) == len(QUESTIONS)
    assert [r.question_index for r in results] == list(range(len(QUESTIONS)))
    assert all(isinstance(r, QAResult) and r.backend == 'retrieval_nli' for r in results)
    assert nli.calls == 1  # one batch across all questions
    assert nli.pairs_seen == len(QUESTIONS) * QAConfig().top_k

    dose, wrong_dose, six_weeks, stomach, concert, bp = results
    assert dose.answer is True and dose.evidence_unit_ids == [2] and dose.confidence > 0.9
    assert wrong_dose.answer is False and wrong_dose.diagnostics['p_con'] > wrong_dose.diagnostics['p_ent']
    assert six_weeks.answer is False
    assert stomach.answer is False
    assert concert.answer is False and concert.diagnostics['existence'] is True
    assert concert.diagnostics['max_similarity'] < QAConfig().off_topic_sim_threshold
    assert bp.answer is True and bp.evidence_unit_ids == [5] and bp.diagnostics['existence'] is True
    for r in results:
        assert set(r.evidence_unit_ids) <= {u.unit_id for u in clinic_context.units}
        assert 0.0 <= r.confidence <= 1.0
        assert {'statement', 'top_windows', 'verdict', 'p_ent', 'p_con'} <= set(r.diagnostics)
        assert len(r.diagnostics['top_windows']) == QAConfig().top_k


def test_wrong_number_is_a_contradiction_not_a_yes(clinic_context):
    results = _backend().answer(clinic_context, [QUESTIONS[1], QUESTIONS[2]])
    for r in results:
        assert r.answer is False
        # the best window still points at the near-miss passage, useful for analysis
        assert r.evidence_unit_ids
        assert r.diagnostics['p_con'] > r.diagnostics['p_ent']


def test_factcheck_contradiction_overrides_entailment(clinic_context, monkeypatch):
    def compare(question, evidence_text):
        return LocalVerdict('contradiction' if '200' in question else 'consistent', [])
    monkeypatch.setattr(qa_backend, 'factcheck_compare', compare)
    results = _backend(nli=AlwaysEntailsNLI()).answer(clinic_context, [QUESTIONS[0], QUESTIONS[1]])
    assert results[0].answer is True and results[0].diagnostics['verdict'] == 'consistent'
    assert results[1].answer is False and results[1].diagnostics['verdict'] == 'contradiction'
    assert results[1].diagnostics['p_ent'] > 0.9  # the NLI said yes; the gate said no


def test_existence_question_also_respects_contradiction(clinic_context, monkeypatch):
    monkeypatch.setattr(qa_backend, 'factcheck_compare', lambda q, t: LocalVerdict('contradiction', []))
    r = _backend(nli=AlwaysEntailsNLI()).answer(clinic_context, [QUESTIONS[5]])[0]
    assert r.diagnostics['existence'] is True and r.answer is False


def test_unverifiable_means_no_flag(clinic_context, monkeypatch):
    monkeypatch.setattr(qa_backend, 'factcheck_compare', lambda q, t: LocalVerdict('unverifiable', []))
    assert _backend(QAConfig(unverifiable_means_no=False)).answer(clinic_context, [QUESTIONS[0]])[0].answer is True
    assert _backend(QAConfig(unverifiable_means_no=True)).answer(clinic_context, [QUESTIONS[0]])[0].answer is False


def test_factcheck_raising_does_not_lose_the_question(clinic_context, monkeypatch):
    def boom(question, evidence_text):
        raise RuntimeError('factcheck exploded')
    monkeypatch.setattr(qa_backend, 'factcheck_compare', boom)
    results = _backend().answer(clinic_context, QUESTIONS)
    assert len(results) == len(QUESTIONS)
    assert results[0].answer is True and results[0].diagnostics['verdict'] == 'error'
    assert 'factcheck exploded' in results[0].diagnostics['verdict_details'][0]
    lexical = LexicalBackend().answer(clinic_context, QUESTIONS[:1])
    assert lexical[0].answer is True and lexical[0].diagnostics['verdict'] == 'error'


def test_per_question_exception_isolation(clinic_context, monkeypatch):
    backend = _backend()
    original = backend._decide_question

    def flaky(index, *args, **kwargs):
        if index == 1:
            raise KeyError('boom')
        return original(index, *args, **kwargs)
    monkeypatch.setattr(backend, '_decide_question', flaky)
    results = backend.answer(clinic_context, QUESTIONS[:3])
    assert [r.question_index for r in results] == [0, 1, 2]
    assert results[1].answer is False and results[1].evidence_unit_ids == []
    assert 'KeyError' in results[1].diagnostics['error']
    assert results[0].answer is True and results[2].answer is False

    lex = LexicalBackend()
    monkeypatch.setattr(lex, '_decide', lambda *a, **k: (_ for _ in ()).throw(ValueError('lex boom')))
    lex_results = lex.answer(clinic_context, QUESTIONS[:2])
    assert len(lex_results) == 2 and all('lex boom' in r.diagnostics['error'] for r in lex_results)


def test_deadline_degrades_to_lexical_without_running_nli(clinic_context):
    nli, embedder = FakeNLI(), FakeEmbedder()
    backend = _backend(nli=nli, embedder=embedder)
    tight = Deadline(budget_s=2.0, clock=lambda: 0.0)  # remaining 2.0 < min_time_for_nli_s 3.0
    results = backend.answer(clinic_context, QUESTIONS)
    assert nli.calls == 1
    degraded = backend.answer(clinic_context, QUESTIONS, deadline=tight)
    assert nli.calls == 1  # NLI batch skipped
    assert len(degraded) == len(QUESTIONS)
    assert all(r.diagnostics['degraded'] is True and r.diagnostics['degrade_reason'] == 'deadline'
               for r in degraded)
    assert all(r.backend == 'lexical' for r in degraded)
    assert degraded[0].answer is True and degraded[0].evidence_unit_ids == [2]
    assert [r.answer for r in results] != [] and results[0].answer is True

    expired = Deadline(budget_s=1.0, clock=iter([0.0, 5.0, 5.0, 5.0, 5.0]).__next__)
    embed_calls = embedder.calls
    gone = backend.answer(clinic_context, QUESTIONS[:2], deadline=expired)
    assert embedder.calls == embed_calls  # no embedding either
    assert all(r.diagnostics['degrade_reason'] == 'deadline_expired' for r in gone)

    roomy = Deadline(budget_s=60.0, clock=lambda: 0.0)
    fine = backend.answer(clinic_context, QUESTIONS[:2], deadline=roomy)
    assert nli.calls == 2 and 'degraded' not in fine[0].diagnostics


def test_model_failures_degrade_instead_of_raising(clinic_context):
    class BrokenNLI(FakeNLI):
        def predict(self, pairs, batch_size=32):
            raise RuntimeError('nli down')

    class WrongShapeNLI(FakeNLI):
        def predict(self, pairs, batch_size=32):
            return np.zeros((len(pairs), 2))

    class BrokenEmbedder(FakeEmbedder):
        def encode(self, texts, batch_size=32):
            raise RuntimeError('embedder down')

    for nli, embedder, reason in [
        (BrokenNLI(), FakeEmbedder(), 'nli_error'),
        (WrongShapeNLI(), FakeEmbedder(), 'nli_error'),
        (FakeNLI(), BrokenEmbedder(), 'retrieval_error'),
    ]:
        results = _backend(nli=nli, embedder=embedder).answer(clinic_context, QUESTIONS[:3])
        assert len(results) == 3
        assert all(r.diagnostics['degraded'] is True and r.diagnostics['degrade_reason'] == reason
                   for r in results)
        assert 'down' in results[0].diagnostics['error'] or 'shape' in results[0].diagnostics['error']
        assert results[0].answer is True and results[0].evidence_unit_ids == [2]


def test_lazy_model_loading_failure_degrades(clinic_context, monkeypatch):
    def cannot_load(*args, **kwargs):
        raise OSError('no model on disk')
    monkeypatch.setattr(qa_backend, 'SentenceTransformerEmbedder', cannot_load)
    backend = RetrievalNLIBackend(QAConfig())  # nothing injected: would load on first use
    results = backend.answer(clinic_context, QUESTIONS[:2])
    assert len(results) == 2 and all(r.diagnostics['degrade_reason'] == 'retrieval_error' for r in results)
    with pytest.raises(OSError):
        backend.warm_up()


def test_empty_and_odd_inputs_for_retrieval(clinic_context):
    backend = _backend()
    assert backend.answer(clinic_context, []) == []
    no_units = backend.answer(make_context([]), QUESTIONS[:2])
    assert [r.diagnostics['reason'] for r in no_units] == ['no_transcript', 'no_transcript']
    results = backend.answer(clinic_context, ['', QUESTIONS[0], QUESTIONS[0]])
    assert results[0].answer is False and results[0].diagnostics['reason'] == 'empty_question'
    assert results[1].answer is True and results[2].answer is True  # duplicates are independent
    single = make_context(['We will start penicillin 100 milligrams daily.'])
    one = _backend(QAConfig(top_k=6)).answer(single, [QUESTIONS[0]])[0]
    assert one.answer is True and one.evidence_unit_ids == [0] and len(one.diagnostics['top_windows']) == 1


def test_top_k_and_window_units_are_honoured(clinic_context):
    nli = FakeNLI()
    cfg = QAConfig(top_k=2, max_window_units=1)
    results = _backend(cfg, nli=nli).answer(clinic_context, QUESTIONS[:2])
    assert nli.pairs_seen == 4
    assert all(len(r.evidence_unit_ids) == 1 for r in results)


def test_warm_up_with_injected_fakes_runs_one_tiny_batch():
    nli, embedder = FakeNLI(), FakeEmbedder()
    backend = _backend(nli=nli, embedder=embedder)
    backend.warm_up()
    assert nli.calls == 1 and embedder.calls == 1
    assert backend.nli_label_permutation == [0, 1, 2] and backend.nli_label_source == 'label_names'


# --------------------------------------------------------------------------- #
# Fallback wrappers around the optional siblings
# --------------------------------------------------------------------------- #

def test_sibling_failures_fall_back_locally(monkeypatch):
    monkeypatch.setattr(qa_backend, 'rewrite_to_statement', lambda q: (_ for _ in ()).throw(ValueError('x')))
    monkeypatch.setattr(qa_backend, 'rewrite_focus_terms', lambda q: (_ for _ in ()).throw(ValueError('x')))
    monkeypatch.setattr(qa_backend, 'rewrite_is_existence_question', lambda q: (_ for _ in ()).throw(ValueError('x')))
    monkeypatch.setattr(qa_backend, 'transcript_windows', lambda c, m: (_ for _ in ()).throw(ValueError('x')))
    assert qa_backend.statement_for('Was the dose 100 mg?') == 'Was the dose 100 mg'
    assert qa_backend.statement_for('Was there any mention of blood pressure?') == 'blood pressure is mentioned.'
    assert qa_backend.statement_for('Did they discuss the concert next Friday?') == 'the concert next Friday is mentioned.'
    assert qa_backend.statement_for('Did diet come up?') == 'diet is mentioned.'
    assert qa_backend.statement_for('Was smoking discussed?') == 'smoking is mentioned.'
    assert qa_backend.statement_for('?') == '?' and qa_backend.statement_for('') == ''
    assert focus_tokens('Was the dose 100 mg?') == ['dose', '100', 'mg']
    assert qa_backend.is_existence_question('Was there any mention of it?') is True
    assert qa_backend.is_existence_question('Was the dose 100 mg?') is False
    assert len(sorted_windows(make_context(['a b', 'c d']), 2)) == 3

    monkeypatch.setattr(qa_backend, 'rewrite_to_statement', lambda q: 'The dose was 100 mg.')
    monkeypatch.setattr(qa_backend, 'rewrite_focus_terms', lambda q: ['Dose', '100 mg'])
    assert qa_backend.statement_for('Was the dose 100 mg?') == 'The dose was 100 mg.'
    assert focus_tokens('Was the dose 100 mg?') == ['dose', '100', 'mg']

    monkeypatch.setattr(qa_backend, 'factcheck_compare', None)
    assert qa_backend.factcheck_verdict('q', 't').status == 'no_quantities'


def test_smoke_with_real_siblings_when_present(clinic_context, monkeypatch):
    """Runs against whichever of transcript/question_rewrite/factcheck exist;
    asserts only the invariants, since their exact behaviour is theirs."""
    attached = _attach_real_siblings(monkeypatch)
    unit_ids = {u.unit_id for u in clinic_context.units}
    for backend in (_backend(), LexicalBackend()):
        results = backend.answer(clinic_context, QUESTIONS)
        assert [r.question_index for r in results] == list(range(len(QUESTIONS)))
        for r in results:
            assert isinstance(r.answer, bool) and 'error' not in r.diagnostics
            assert set(r.evidence_unit_ids) <= unit_ids and 0.0 <= r.confidence <= 1.0
        assert results[4].answer is False  # off-topic stays no with any implementation
    if 'factcheck_compare' in attached:
        verdict = qa_backend.factcheck_verdict(QUESTIONS[1], clinic_context.units[2].text)
        assert verdict.status in {'consistent', 'contradiction', 'unverifiable', 'no_quantities', 'error'}


# --------------------------------------------------------------------------- #
# Factory and test double
# --------------------------------------------------------------------------- #

def test_build_qa_backend_maps_names(monkeypatch):
    assert isinstance(build_qa_backend(QAConfig(backend='lexical')), LexicalBackend)
    assert isinstance(build_qa_backend(QAConfig(backend='Retrieval_NLI')), RetrievalNLIBackend)
    with pytest.raises(ValueError, match='unknown QA backend'):
        build_qa_backend(QAConfig(backend='oracle'))
    import builtins
    real_import = builtins.__import__

    def no_llm(name, *args, **kwargs):
        if name == 'qa_llm_backend':
            raise ImportError('missing')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_llm)
    with pytest.raises(RuntimeError, match='qa_llm_backend'):
        build_qa_backend(QAConfig(backend='llm'))


def test_build_qa_backend_llm_reads_its_own_env(monkeypatch):
    """The LLM branch is configured from MA_LLM_*, not from QAConfig; with the
    fake engine nothing is loaded, so this stays a fast test."""
    qa_llm_backend = pytest.importorskip('qa_llm_backend')
    monkeypatch.setenv('MA_LLM_ENGINE', 'fake')
    backend = build_qa_backend(QAConfig(backend='llm'))
    assert isinstance(backend, qa_llm_backend.LLMBackend)
    assert backend.name == 'llm' and backend.config.engine == 'fake'


def test_fake_qa_backend_keeps_invariant(clinic_context):
    scripted = [QAResult(question_index=7, answer=True, evidence_unit_ids=[2], confidence=0.9)]
    fake = FakeQABackend(scripted)
    out = fake.answer(clinic_context, ['a', 'b'])
    assert [r.question_index for r in out] == [0, 1]
    assert out[0].answer is True and out[0].evidence_unit_ids == [2] and out[0].backend == 'fake'
    assert out[1].answer is False and out[1].diagnostics['reason'] == 'unscripted'
    assert fake.answer(clinic_context, []) == [] and fake.calls == [['a', 'b'], []]


# --------------------------------------------------------------------------- #
# Real models (slow)
# --------------------------------------------------------------------------- #

@pytest.mark.slow
def test_real_models_on_clinic_context(clinic_context):
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    cfg = QAConfig(local_files_only=True)
    backend = RetrievalNLIBackend(cfg)
    backend.warm_up()
    assert backend.nli_label_source == 'id2label' and backend.nli_label_permutation == [0, 1, 2]
    results = backend.answer(clinic_context, [QUESTIONS[0], QUESTIONS[1], QUESTIONS[4], QUESTIONS[5]],
                             deadline=Deadline(budget_s=120.0))
    dose, wrong_dose, concert, bp = results
    assert dose.answer is True and 2 in dose.evidence_unit_ids
    assert wrong_dose.answer is False
    assert concert.answer is False
    assert bp.answer is True and 5 in bp.evidence_unit_ids
    assert all(0.0 <= r.confidence <= 1.0 for r in results)


# --------------------------------------------------------------------------- #
# Learned calibration layer wiring (calibration.py)
# --------------------------------------------------------------------------- #

def _calibration_that_prefers(prob_yes: float, prefer_short: bool):
    import numpy as np

    from calibration import QUESTION_FEATURES, WINDOW_FEATURES, Calibration, LinearModel

    dw = np.zeros(len(QUESTION_FEATURES))
    intercept = 60.0 if prob_yes >= 0.5 else -60.0
    ww = np.zeros(len(WINDOW_FEATURES))
    ww[WINDOW_FEATURES.index('n_units')] = -1.0 if prefer_short else 1.0
    return Calibration(LinearModel(QUESTION_FEATURES, dw, intercept), LinearModel(WINDOW_FEATURES, ww, 0.0, 'linear'))


def _make_calibrated_backend(clinic_context, prob_yes, prefer_short, hard_gate=True):
    """A backend with fake models whose decision is fully determined by the
    injected calibration object, so the wiring itself is what is tested."""
    import numpy as np

    from qa_backend import QAConfig, RetrievalNLIBackend

    class Embedder:
        # Hashed bag of words: a fixed dimension whatever the call, so question
        # and window vectors (embedded in separate calls) are comparable.
        def encode(self, texts, batch_size=32):
            out = np.zeros((len(texts), 64))
            for i, text in enumerate(texts):
                for t in text.lower().split():
                    out[i, sum(ord(c) for c in t) % 64] += 1.0
            return out

    class NLI:
        def predict(self, pairs, batch_size=32):
            return np.array([[0.0, 3.0, 0.0] for _ in pairs])

    config = QAConfig(backend='retrieval_nli', use_factcheck=True, hard_contradiction_gate=hard_gate)
    backend = RetrievalNLIBackend(config, embedder=Embedder(), nli=NLI())
    backend.set_calibration(_calibration_that_prefers(prob_yes, prefer_short))
    return backend


def test_calibration_decides_and_picks_short_window(clinic_context):
    backend = _make_calibrated_backend(clinic_context, prob_yes=0.99, prefer_short=True)
    results = backend.answer(clinic_context, ["Will the treatment last two weeks?"])
    assert results[0].answer is True
    assert results[0].diagnostics['calibrated'] is True
    assert results[0].diagnostics['p_yes'] > 0.99
    assert len(results[0].evidence_unit_ids) == 1
    assert results[0].confidence == results[0].diagnostics['p_yes']


def test_calibration_can_say_no_and_prefers_long_window(clinic_context):
    backend = _make_calibrated_backend(clinic_context, prob_yes=0.01, prefer_short=False)
    results = backend.answer(clinic_context, ["Will the treatment last two weeks?"])
    assert results[0].answer is False
    assert len(results[0].evidence_unit_ids) == 3


def test_hard_contradiction_gate_overrides_calibration(clinic_context, monkeypatch):
    import qa_backend

    monkeypatch.setattr(qa_backend, 'factcheck_verdict',
                        lambda question, text: qa_backend.LocalVerdict('contradiction', ['forced by test']))
    backend = _make_calibrated_backend(clinic_context, prob_yes=0.99, prefer_short=True, hard_gate=True)
    results = backend.answer(clinic_context, ["Was the prescribed dose 200 mg daily?"])
    assert results[0].diagnostics['verdict'] == 'contradiction'
    assert results[0].answer is False
    backend2 = _make_calibrated_backend(clinic_context, prob_yes=0.99, prefer_short=True, hard_gate=False)
    results2 = backend2.answer(clinic_context, ["Was the prescribed dose 200 mg daily?"])
    assert results2[0].answer is True


def test_missing_calibration_file_falls_back_to_rules(clinic_context, tmp_path):
    from qa_backend import QAConfig, RetrievalNLIBackend

    backend = RetrievalNLIBackend(QAConfig(calibration_path=str(tmp_path / 'nope.json')))
    backend._ensure_calibration()
    assert backend._calibration is None and backend._calibration_state == 'missing'
    (tmp_path / 'bad.json').write_text('{"version": 1}')
    backend2 = RetrievalNLIBackend(QAConfig(calibration_path=str(tmp_path / 'bad.json')))
    backend2._ensure_calibration()
    assert backend2._calibration is None and backend2._calibration_state == 'invalid'


def test_real_calibration_file_loads_relative_to_project():
    from qa_backend import QAConfig, RetrievalNLIBackend

    backend = RetrievalNLIBackend(QAConfig(calibration_path='configs/calibration_v1.json'))
    backend._ensure_calibration()
    assert backend._calibration_state == 'loaded'
