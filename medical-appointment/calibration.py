"""Learned decision and window-ranking layer for the retrieval + NLI backend.

The rule-based decision in ``qa_backend`` thresholds the entailment
probability of the best candidate window. Measured on the dev split that
gives 0.73 accuracy, while a logistic regression over a handful of
question-level features reaches ~0.87 under grouped cross-validation, and a
linear ranker over window-level features lifts the localisation of the chosen
window from 0.37 to ~0.51 tIoU. This module holds the shared feature
extraction (so training and serving cannot drift apart), the tiny linear
models, and their JSON persistence.

Serving needs only numpy. Fitting (``fit_calibration``) imports scikit-learn
lazily and is only ever run offline on the development split.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CALIBRATION_VERSION = 1

VERDICT_ORDER = ('contradiction', 'unverifiable', 'consistent', 'no_quantities')

QUESTION_FEATURES = (
    'p_ent_max', 'log_p_ent_max', 'p_ent_second', 'p_con_max', 'cosine_max',
    'lexical_max', 'existence', 'verdict_contradiction', 'verdict_unverifiable',
    'verdict_consistent', 'verdict_no_quantities', 'question_len',
)

WINDOW_FEATURES = (
    'p_ent', 'p_ent_rel', 'p_con', 'cosine', 'cosine_rel', 'lexical',
    'n_units', 'duration', 'rank',
)


@dataclass(frozen=True)
class Candidate:
    """One retrieved window with its scores, as the NLI backend produces it."""

    unit_ids: Tuple[int, ...]
    start_s: float
    end_s: float
    p_ent: float
    p_con: float
    cosine: float
    lexical: float
    rank: int

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def question_features(
    candidates: Sequence[Candidate],
    existence: bool,
    verdict_status: str,
    question: str,
) -> np.ndarray:
    """Fixed-order feature vector for the yes/no decision (see QUESTION_FEATURES)."""
    p_ent = sorted((_finite(c.p_ent) for c in candidates), reverse=True) or [0.0]
    p_con = max((_finite(c.p_con) for c in candidates), default=0.0)
    cosine = max((_finite(c.cosine) for c in candidates), default=0.0)
    lexical = max((_finite(c.lexical) for c in candidates), default=0.0)
    verdict = [1.0 if verdict_status == v else 0.0 for v in VERDICT_ORDER]
    if not any(verdict):
        verdict[-1] = 1.0
    return np.array([
        p_ent[0],
        math.log(p_ent[0] + 1e-4),
        p_ent[1] if len(p_ent) > 1 else 0.0,
        p_con,
        cosine,
        lexical,
        1.0 if existence else 0.0,
        *verdict,
        len(question.split()) / 20.0,
    ], dtype=np.float64)


def window_features(candidates: Sequence[Candidate]) -> np.ndarray:
    """One row per candidate (see WINDOW_FEATURES); relative features are
    normalised by the best value among the candidates so the ranker sees
    'how close to the best' rather than absolute scale."""
    if not candidates:
        return np.zeros((0, len(WINDOW_FEATURES)), dtype=np.float64)
    p_ent_max = max(_finite(c.p_ent) for c in candidates) + 1e-6
    cos_max = max(_finite(c.cosine) for c in candidates) + 1e-6
    rows = []
    for c in candidates:
        rows.append([
            _finite(c.p_ent),
            _finite(c.p_ent) / p_ent_max,
            _finite(c.p_con),
            _finite(c.cosine),
            _finite(c.cosine) / cos_max,
            _finite(c.lexical),
            float(len(c.unit_ids)),
            c.duration_s / 10.0,
            c.rank / 6.0,
        ])
    return np.array(rows, dtype=np.float64)


@dataclass
class LinearModel:
    """``sigmoid(w.x + b)`` (logistic) or ``w.x + b`` (linear)."""

    feature_names: Tuple[str, ...]
    weights: np.ndarray
    intercept: float
    kind: str = 'logistic'

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if self.weights.shape[0] != len(self.feature_names):
            raise ValueError('weights and feature_names length differ')
        if self.kind not in ('logistic', 'linear'):
            raise ValueError(f'unknown model kind {self.kind!r}')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.weights.shape[0]:
            raise ValueError(f'expected {self.weights.shape[0]} features, got {X.shape[1]}')
        z = X @ self.weights + self.intercept
        if self.kind == 'logistic':
            return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return z

    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_names': list(self.feature_names),
            'weights': [float(w) for w in self.weights],
            'intercept': float(self.intercept),
            'kind': self.kind,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LinearModel':
        return cls(
            feature_names=tuple(data['feature_names']),
            weights=np.array(data['weights'], dtype=np.float64),
            intercept=float(data['intercept']),
            kind=str(data.get('kind', 'logistic')),
        )


@dataclass
class Calibration:
    """The two fitted models plus the decision threshold and provenance."""

    decision: LinearModel
    window: LinearModel
    decision_threshold: float = 0.5
    version: int = CALIBRATION_VERSION
    provenance: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if tuple(self.decision.feature_names) != QUESTION_FEATURES:
            raise ValueError('decision model features do not match QUESTION_FEATURES')
        if tuple(self.window.feature_names) != WINDOW_FEATURES:
            raise ValueError('window model features do not match WINDOW_FEATURES')
        if not (0.0 < self.decision_threshold < 1.0):
            raise ValueError('decision_threshold must be in (0, 1)')

    def decide(
        self,
        candidates: Sequence[Candidate],
        existence: bool,
        verdict_status: str,
        question: str,
    ) -> Tuple[bool, float]:
        """Return ``(answer, probability_of_yes)``."""
        prob = float(self.decision.predict(question_features(candidates, existence, verdict_status, question))[0])
        return prob >= self.decision_threshold, prob

    def rank_windows(self, candidates: Sequence[Candidate]) -> List[float]:
        if not candidates:
            return []
        return [float(v) for v in self.window.predict(window_features(candidates))]

    def pick_window(self, candidates: Sequence[Candidate]) -> Optional[Candidate]:
        scores = self.rank_windows(candidates)
        if not scores:
            return None
        return candidates[int(np.argmax(scores))]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'decision_threshold': self.decision_threshold,
            'decision': self.decision.to_dict(),
            'window': self.window.to_dict(),
            'provenance': self.provenance,
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding='utf-8')
        return path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Calibration':
        if int(data.get('version', -1)) != CALIBRATION_VERSION:
            raise ValueError(f'calibration version {data.get("version")} != {CALIBRATION_VERSION}')
        return cls(
            decision=LinearModel.from_dict(data['decision']),
            window=LinearModel.from_dict(data['window']),
            decision_threshold=float(data.get('decision_threshold', 0.5)),
            version=int(data['version']),
            provenance=dict(data.get('provenance', {})),
        )

    @classmethod
    def load(cls, path: Path) -> 'Calibration':
        return cls.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


# --------------------------------------------------------------------------- #
# Fitting (offline, development data only)
# --------------------------------------------------------------------------- #

@dataclass
class DecisionExample:
    features: np.ndarray
    label: int
    group: str


@dataclass
class WindowExample:
    features: np.ndarray       # (n_candidates, len(WINDOW_FEATURES))
    tious: np.ndarray          # (n_candidates,)
    group: str


def fit_calibration(
    decision_examples: Sequence[DecisionExample],
    window_examples: Sequence[WindowExample],
    decision_threshold: float = 0.5,
    n_folds: int = 5,
    C: float = 1.0,
    ridge_alpha: float = 1.0,
    provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[Calibration, Dict[str, Any]]:
    """Fit both models on all examples and report grouped-CV estimates.

    The CV numbers are what to believe; the final models are refit on
    everything. Groups are recording ids so no recording is in both sides.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold

    X = np.stack([e.features for e in decision_examples])
    y = np.array([e.label for e in decision_examples])
    g = np.array([e.group for e in decision_examples])
    folds = min(n_folds, len(set(g)))
    report: Dict[str, Any] = {'n_decision': int(len(y)), 'n_window_questions': int(len(window_examples)), 'folds': folds}
    if folds >= 2:
        prob = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=folds).split(X, y, g):
            model = LogisticRegression(C=C, max_iter=5000).fit(X[tr], y[tr])
            prob[te] = model.predict_proba(X[te])[:, 1]
        pred = (prob >= decision_threshold).astype(int)
        report['cv_accuracy'] = float((pred == y).mean())
        report['cv_positive_recall'] = float((pred[y == 1] == 1).mean()) if (y == 1).any() else None
        report['cv_negative_accuracy'] = float((pred[y == 0] == 0).mean()) if (y == 0).any() else None
    lr = LogisticRegression(C=C, max_iter=5000).fit(X, y)
    decision = LinearModel(QUESTION_FEATURES, lr.coef_[0], float(lr.intercept_[0]), 'logistic')

    if window_examples:
        W = np.concatenate([e.features for e in window_examples])
        T = np.concatenate([e.tious for e in window_examples])
        owner = np.concatenate([np.full(len(e.tious), i) for i, e in enumerate(window_examples)])
        gw = np.concatenate([np.full(len(e.tious), e.group) for e in window_examples])
        wfolds = min(n_folds, len(set(gw)))
        if wfolds >= 2:
            predT = np.zeros(len(T))
            for tr, te in GroupKFold(n_splits=wfolds).split(W, T, gw):
                predT[te] = Ridge(alpha=ridge_alpha).fit(W[tr], T[tr]).predict(W[te])
            picked = [T[owner == i][int(np.argmax(predT[owner == i]))] for i in range(len(window_examples))]
            report['cv_window_tiou'] = float(np.mean(picked))
            report['oracle_window_tiou'] = float(np.mean([T[owner == i].max() for i in range(len(window_examples))]))
            report['p_ent_window_tiou'] = float(np.mean([T[owner == i][int(np.argmax(W[owner == i][:, 0]))] for i in range(len(window_examples))]))
        ridge = Ridge(alpha=ridge_alpha).fit(W, T)
        window = LinearModel(WINDOW_FEATURES, ridge.coef_, float(ridge.intercept_), 'linear')
    else:
        # Fall back to "highest entailment wins" when no window data exists.
        weights = np.zeros(len(WINDOW_FEATURES))
        weights[WINDOW_FEATURES.index('p_ent')] = 1.0
        window = LinearModel(WINDOW_FEATURES, weights, 0.0, 'linear')

    calibration = Calibration(decision, window, decision_threshold, provenance=dict(provenance or {}))
    return calibration, report
