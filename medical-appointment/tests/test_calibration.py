import json
import math

import numpy as np
import pytest

from calibration import (
    QUESTION_FEATURES,
    WINDOW_FEATURES,
    Calibration,
    Candidate,
    DecisionExample,
    LinearModel,
    WindowExample,
    fit_calibration,
    question_features,
    window_features,
)


def cand(ids, p_ent, cosine=0.5, lexical=0.5, p_con=0.0, rank=0, start=0.0, end=2.0):
    return Candidate(tuple(ids), start, end, p_ent, p_con, cosine, lexical, rank)


def test_question_features_shape_and_order():
    cands = [cand([1], 0.9, rank=0), cand([1, 2], 0.4, rank=1)]
    x = question_features(cands, existence=False, verdict_status='consistent', question='Is the dose 100 mg?')
    assert x.shape == (len(QUESTION_FEATURES),)
    assert x[0] == 0.9 and x[2] == 0.4
    assert x[QUESTION_FEATURES.index('verdict_consistent')] == 1.0
    assert x[QUESTION_FEATURES.index('existence')] == 0.0
    assert x[QUESTION_FEATURES.index('question_len')] == pytest.approx(5 / 20)


def test_question_features_empty_and_nan():
    x = question_features([], existence=True, verdict_status='weird', question='')
    assert x[0] == 0.0 and x[QUESTION_FEATURES.index('verdict_no_quantities')] == 1.0
    x = question_features([cand([0], float('nan'))], False, 'contradiction', 'q')
    assert math.isfinite(x[0]) and x[QUESTION_FEATURES.index('verdict_contradiction')] == 1.0


def test_window_features_relative_columns():
    cands = [cand([0], 0.5, cosine=0.8, rank=0, end=1.0), cand([0, 1], 1.0, cosine=0.4, rank=1, end=4.0)]
    W = window_features(cands)
    assert W.shape == (2, len(WINDOW_FEATURES))
    assert W[1, WINDOW_FEATURES.index('p_ent_rel')] == pytest.approx(1.0, abs=1e-5)
    assert W[0, WINDOW_FEATURES.index('cosine_rel')] == pytest.approx(1.0, abs=1e-5)
    assert W[1, WINDOW_FEATURES.index('n_units')] == 2.0
    assert W[1, WINDOW_FEATURES.index('duration')] == pytest.approx(0.4)
    assert window_features([]).shape == (0, len(WINDOW_FEATURES))


def test_linear_model_predict_and_validation():
    m = LinearModel(('a', 'b'), np.array([1.0, -1.0]), 0.0, 'logistic')
    assert m.predict(np.array([[0.0, 0.0]]))[0] == pytest.approx(0.5)
    assert m.predict(np.array([5.0, 0.0]))[0] > 0.99
    lin = LinearModel(('a', 'b'), [2.0, 0.5], 1.0, 'linear')
    assert lin.predict([[1.0, 2.0]])[0] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        LinearModel(('a',), [1.0, 2.0], 0.0)
    with pytest.raises(ValueError):
        m.predict(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        LinearModel(('a',), [1.0], 0.0, kind='tree')


def _calibration(threshold=0.5):
    dw = np.zeros(len(QUESTION_FEATURES))
    dw[QUESTION_FEATURES.index('p_ent_max')] = 10.0
    ww = np.zeros(len(WINDOW_FEATURES))
    ww[WINDOW_FEATURES.index('cosine')] = 1.0
    return Calibration(LinearModel(QUESTION_FEATURES, dw, -5.0), LinearModel(WINDOW_FEATURES, ww, 0.0, 'linear'), threshold)


def test_calibration_decide_and_pick():
    cal = _calibration()
    yes, prob = cal.decide([cand([0], 0.9)], False, 'no_quantities', 'q')
    assert yes and prob > 0.9
    no, prob = cal.decide([cand([0], 0.1)], False, 'no_quantities', 'q')
    assert not no and prob < 0.1
    cands = [cand([0], 0.9, cosine=0.2), cand([1], 0.2, cosine=0.9)]
    assert cal.pick_window(cands) is cands[1]
    assert cal.pick_window([]) is None


def test_calibration_round_trip(tmp_path):
    cal = _calibration(0.4)
    cal.provenance = {'run': 'r001'}
    path = cal.save(tmp_path / 'cal.json')
    loaded = Calibration.load(path)
    assert loaded.decision_threshold == 0.4
    assert loaded.provenance == {'run': 'r001'}
    assert np.allclose(loaded.decision.weights, cal.decision.weights)
    data = json.loads(path.read_text())
    data['version'] = 99
    with pytest.raises(ValueError):
        Calibration.from_dict(data)


def test_calibration_rejects_feature_mismatch():
    with pytest.raises(ValueError):
        Calibration(LinearModel(('x',), [1.0], 0.0), LinearModel(WINDOW_FEATURES, np.zeros(len(WINDOW_FEATURES)), 0.0, 'linear'))


def test_fit_recovers_separable_rule():
    rng = np.random.default_rng(0)
    dec = []
    for i in range(200):
        p = rng.uniform(0, 1)
        label = int(p > 0.5)
        x = question_features([cand([0], p, rank=0), cand([1], p * 0.5, rank=1)], False, 'no_quantities', 'is it so')
        dec.append(DecisionExample(x, label, f'g{i % 10}'))
    win = []
    for i in range(60):
        cands = [cand([0], 0.3, cosine=0.9, rank=0, end=1.0), cand([0, 1], 0.9, cosine=0.3, rank=1, end=4.0)]
        win.append(WindowExample(window_features(cands), np.array([0.9, 0.2]), f'g{i % 10}'))
    cal, report = fit_calibration(dec, win)
    assert report['cv_accuracy'] > 0.95
    assert report['cv_window_tiou'] > 0.85
    yes, _ = cal.decide([cand([0], 0.95), cand([1], 0.4, rank=1)], False, 'no_quantities', 'is it so')
    assert yes
    picked = cal.pick_window([cand([0], 0.3, cosine=0.9, rank=0, end=1.0), cand([0, 1], 0.9, cosine=0.3, rank=1, end=4.0)])
    assert picked.unit_ids == (0,)
