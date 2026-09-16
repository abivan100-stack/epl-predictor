"""Machine learning models module for Premier League outcome and scoreline forecasting.

Implements Random Forest, XGBoost, and multinomial logistic-regression
classifiers (Win / Draw / Loss), goal regressors (Home Goals / Away
Goals), a pure-statistical Elo-Poisson member, and a stacked ensemble
with a meta-learner, with side-by-side benchmarking.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


def apply_temperature_scaling(probas: np.ndarray, t: float) -> np.ndarray:
    """Applies temperature scaling: softmax(log(p)/T) row-wise."""
    t = max(0.05, float(t))
    logp = np.log(np.clip(np.asarray(probas, dtype=float), 1e-15, 1.0))
    scaled = logp / t
    scaled -= scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _rps_loss(probas: np.ndarray, y: np.ndarray) -> float:
    """Computes multiclass Ranked Probability Score for [Away, Draw, Home]."""
    cumulative = np.cumsum(np.asarray(probas, dtype=float), axis=1)[:, :2]
    truth = np.eye(3, dtype=float)[np.asarray(y, dtype=int)][:, :2]
    return float(np.mean(np.sum((cumulative - np.cumsum(truth, axis=1)) ** 2, axis=1) / 2.0))


def blend_market_probabilities(
    model_probas: np.ndarray,
    market_home_draw_away: np.ndarray,
    odds_missing: np.ndarray,
    market_weight: np.ndarray,
) -> np.ndarray:
    """Blends model [Away, Draw, Home] with market [Home, Draw, Away]."""
    model = np.asarray(model_probas, dtype=float)
    market = np.asarray(market_home_draw_away, dtype=float)[:, [2, 1, 0]]
    missing = np.asarray(odds_missing, dtype=float).reshape(-1)
    weights = np.asarray(market_weight, dtype=float).reshape(-1)
    if model.shape != market.shape or model.shape[1] != 3 or len(missing) != len(model) or len(weights) != len(model):
        raise ValueError("Model, market, missing, and weight arrays must have matching rows and three probability columns.")
    weights = np.clip(weights, 0.0, 1.0) * (missing < 0.5)
    blended = (1.0 - weights[:, None]) * model + weights[:, None] * market
    return blended / np.maximum(blended.sum(axis=1, keepdims=True), 1e-12)


def fit_goal_calibration(
    predicted_home: np.ndarray,
    predicted_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
) -> Tuple[float, float]:
    """Fits bounded multiplicative goal corrections from held-out predictions."""
    pred_h = np.maximum(0.05, np.asarray(predicted_home, dtype=float))
    pred_a = np.maximum(0.05, np.asarray(predicted_away, dtype=float))
    true_h = np.asarray(actual_home, dtype=float)
    true_a = np.asarray(actual_away, dtype=float)
    if not (len(pred_h) == len(pred_a) == len(true_h) == len(true_a)) or len(pred_h) == 0:
        return 1.0, 1.0
    home = float(np.clip(np.mean(true_h) / max(1e-6, np.mean(pred_h)), 0.8, 1.25))
    away = float(np.clip(np.mean(true_a) / max(1e-6, np.mean(pred_a)), 0.8, 1.25))
    return home, away


def align_probas(classes, probas: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Aligns predict_proba output to the full [Away, Draw, Home] columns.

    Estimators fit on degenerate slices may have seen only a subset of
    classes and return fewer columns; without alignment every downstream
    consumer (blend, RPS, draw rule) breaks. Missing classes get zero
    mass and rows renormalize.
    """
    probas = np.asarray(probas, dtype=float)
    class_list = [int(c) for c in np.asarray(classes).ravel().tolist()]
    if probas.shape[1] == n_classes and class_list == list(range(n_classes)):
        return probas
    aligned = np.zeros((probas.shape[0], n_classes))
    for col, cls in enumerate(class_list):
        if 0 <= cls < n_classes and col < probas.shape[1]:
            aligned[:, cls] = probas[:, col]
    sums = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(sums, 1e-12)

# Outcome decision thresholds. Pure argmax on sharp 3-way probabilities almost
# never selects draws (0/380 in practice), which is indefensible for the EPL
# (~22% historical draw rate). A close home/away race with a robust draw
# probability is therefore called a draw. This rule is the SINGLE source of
# truth: predict_scoreline, pipeline forecasts, and validation metrics all
# use favor_outcome_from_proba so scorelines always agree with outcomes.
# Retuned 2026-09-14 per market regime for the 8-season + 365d-decay
# production forest (full margin x min grid on the disjoint eval slice
# + 350-row forecast-slate grid; objective eval accuracy subject to
# 18-27% draws on BOTH). Market-present: 0.12/0.24 (47.3% acc, 19.7%
# draws, n=402). No-market: 0.12/0.25 (eval ~52% acc, ~21% draws;
# ~22-23% on the no-market slate). Phase 7 receipt (2026-09-15): values moved
# to config.yaml and retained after the disjoint eval/forecast RPS grid with
# 18-27% draw-share constraints.
DRAW_MARGIN = 0.12
DRAW_MIN_PROB = 0.24
DRAW_MARGIN_NO_MARKET = 0.12
DRAW_MIN_PROB_NO_MARKET = 0.25


def get_draw_rule() -> Dict[str, float]:
    """Returns config-driven draw thresholds with backward-compatible defaults."""
    from src.config import get_config

    configured = get_config().get("model", {}).get("draw_rule", {})
    return {
        "market_margin": float(configured.get("market_margin", DRAW_MARGIN)),
        "market_min_prob": float(configured.get("market_min_prob", DRAW_MIN_PROB)),
        "no_market_margin": float(configured.get("no_market_margin", DRAW_MARGIN_NO_MARKET)),
        "no_market_min_prob": float(configured.get("no_market_min_prob", DRAW_MIN_PROB_NO_MARKET)),
    }


def favor_outcome_from_proba(probas, odds_missing: float = 0.0) -> int:
    """Maps a [p_away, p_draw, p_home] vector to 0 (Away), 1 (Draw), 2 (Home).

    A close home/away race with a robust draw probability is called a
    draw; thresholds are regime-aware because market-present and
    no-market proba distributions differ sharply.
    """
    rule = get_draw_rule()
    if float(odds_missing) >= 0.5:
        margin, min_prob = rule["no_market_margin"], rule["no_market_min_prob"]
    else:
        margin, min_prob = rule["market_margin"], rule["market_min_prob"]
    p_a, p_d, p_h = float(probas[0]), float(probas[1]), float(probas[2])
    if abs(p_h - p_a) <= margin and p_d >= min_prob:
        return 1
    return int(np.argmax(probas))


def tune_draw_rule(
    eval_probas: np.ndarray,
    eval_y: np.ndarray,
    eval_odds_missing: np.ndarray,
    forecast_probas: np.ndarray | None = None,
    forecast_missing: np.ndarray | None = None,
    draw_band: Tuple[float, float] = (0.18, 0.27),
) -> Dict[str, float]:
    """Grid-searches regime thresholds under the draw-share constraint.

    RPS is computed from hard decisions on the calibration-independent eval
    slice; forecast shares are an additional plausibility constraint.
    """
    eval_probas = np.asarray(eval_probas, dtype=float)
    eval_y = np.asarray(eval_y, dtype=int)
    eval_missing = np.asarray(eval_odds_missing, dtype=float)
    forecast_probas = eval_probas if forecast_probas is None else np.asarray(forecast_probas, dtype=float)
    forecast_missing = eval_missing if forecast_missing is None else np.asarray(forecast_missing, dtype=float)
    candidates = np.arange(0.08, 0.181, 0.01)
    minimum, maximum = draw_band

    def _decision_share(probas, missing, margin, minimum_prob):
        decisions = np.asarray([
            1 if abs(float(p[2]) - float(p[0])) <= margin and float(p[1]) >= minimum_prob
            else int(np.argmax(p))
            for p in probas
        ])
        return decisions, float(np.mean(decisions == 1))

    def _rps(decisions, truth):
        one_hot = np.eye(3, dtype=float)[decisions]
        return float(np.mean(np.sum((np.cumsum(one_hot, axis=1)[:, :2] - np.cumsum(np.eye(3)[truth], axis=1)[:, :2]) ** 2, axis=1) / 2.0))

    result: Dict[str, float] = {}
    for prefix, is_missing in (("market", False), ("no_market", True)):
        eval_mask = (eval_missing >= 0.5) if is_missing else (eval_missing < 0.5)
        forecast_mask = (forecast_missing >= 0.5) if is_missing else (forecast_missing < 0.5)
        best = None
        for margin in candidates:
            for minimum_prob in np.arange(0.20, 0.301, 0.01):
                eval_decisions, eval_share = _decision_share(eval_probas[eval_mask], eval_missing[eval_mask], margin, minimum_prob)
                forecast_decisions, forecast_share = _decision_share(forecast_probas[forecast_mask], forecast_missing[forecast_mask], margin, minimum_prob)
                if minimum <= eval_share <= maximum and minimum <= forecast_share <= maximum:
                    score = _rps(eval_decisions, eval_y[eval_mask]) if len(eval_decisions) else float("inf")
                    candidate = (score, float(margin), float(minimum_prob), eval_share, forecast_share)
                    if best is None or candidate < best:
                        best = candidate
        if best is None:
            best = (float("inf"), DRAW_MARGIN_NO_MARKET if is_missing else DRAW_MARGIN, DRAW_MIN_PROB_NO_MARKET if is_missing else DRAW_MIN_PROB, 0.0, 0.0)
        result[f"{prefix}_margin"] = best[1]
        result[f"{prefix}_min_prob"] = best[2]
        result[f"{prefix}_draw_share"] = best[3]
        result[f"{prefix}_forecast_draw_share"] = best[4]
    forecast_decisions = []
    for p, missing in zip(forecast_probas, forecast_missing):
        prefix = "no_market" if missing >= 0.5 else "market"
        draw = (
            abs(float(p[2]) - float(p[0])) <= result[f"{prefix}_margin"]
            and float(p[1]) >= result[f"{prefix}_min_prob"]
        )
        forecast_decisions.append(1 if draw else int(np.argmax(p)))
    result["forecast_draw_share"] = float(np.mean(np.asarray(forecast_decisions) == 1))
    return result


def blend_weights() -> Tuple[float, float, float]:
    """Normalized (classifier, hg/ag-Poisson, supremacy/totals) blend weights.

    Read from config.yaml (keys blend_classifier, blend_poisson,
    blend_supremacy); normalized to sum to 1 so partial configs stay valid.
    """
    from src.config import get_config

    cfg = get_config()["model"]
    raw = (
        float(cfg.get("blend_classifier", 0.6)),
        float(cfg.get("blend_poisson", 0.0)),
        float(cfg.get("blend_supremacy", 0.4)),
    )
    total = sum(raw)
    if total <= 0:
        return (0.6, 0.0, 0.4)
    return (raw[0] / total, raw[1] / total, raw[2] / total)


def supremacy_to_means(
    supremacy: np.ndarray, totals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Splits supremacy/total goal expectations into home/away means.

    lam_h = (total + supremacy) / 2, lam_a = (total - supremacy) / 2,
    floored at 0.05 so Poisson grids stay valid.
    """
    sup = np.asarray(supremacy, dtype=float)
    tot = np.maximum(0.0, np.asarray(totals, dtype=float))
    lam_h = np.maximum(0.05, (tot + sup) / 2.0)
    lam_a = np.maximum(0.05, (tot - sup) / 2.0)
    return lam_h, lam_a


# Outcome label mapping: 0 -> Away Win (A), 1 -> Draw (D), 2 -> Home Win (H)
OUTCOME_NAMES = {0: "Away Win", 1: "Draw", 2: "Home Win"}
OUTCOME_CODES = {0: "A", 1: "D", 2: "H"}


def _fit_with_weights(estimator, X, y, sample_weight) -> None:
    """Fits one estimator, routing sample weights through Pipelines.

    Plain estimators take ``sample_weight`` directly; sklearn Pipelines
    need the ``<step>__sample_weight`` prefix (here ``clf__``). Falls
    back to unweighted fit when neither is accepted.
    """
    if sample_weight is None:
        estimator.fit(X, y)
        return
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
        return
    except (TypeError, ValueError):
        pass
    try:
        estimator.fit(X, y, clf__sample_weight=sample_weight)
    except (TypeError, ValueError):
        estimator.fit(X, y)


class MatchPredictorModel:
    """Wrapper encapsulating an outcome classifier and home/away goal regressors."""

    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type.lower()
        if self.model_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            self.home_regressor = RandomForestRegressor(
                n_estimators=250,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
            self.away_regressor = RandomForestRegressor(
                n_estimators=250,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
            # Supremacy (signed goal difference) and totals heads. Supremacy
            # uses squared error (it can go negative); totals are counts.
            self.supremacy_regressor = RandomForestRegressor(
                n_estimators=250,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
            self.totals_regressor = RandomForestRegressor(
                n_estimators=250,
                max_depth=6,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "logreg":
            # Linear member for the stack: standardized multinomial logistic
            # regression (C fixed; the meta-learner, not this member, is what
            # gets tuned). Poisson regressors shared with the xgb branch.
            self.classifier = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    solver="lbfgs",
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                )),
            ])
            self.home_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                n_jobs=-1,
            )
            self.away_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                n_jobs=-1,
            )
            self.supremacy_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
            self.totals_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                n_jobs=-1,
            )
        else:  # xgboost
            self.classifier = XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )
            self.home_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                n_jobs=-1,
            )
            self.away_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                n_jobs=-1,
            )
            self.supremacy_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
            self.totals_regressor = XGBRegressor(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="count:poisson",
                random_state=42,
                n_jobs=-1,
            )

        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        # Temperature scaling for calibrated probabilities (fit on validation).
        # T > 1 softens overconfident peaks; T == 1.0 means uncalibrated.
        self.calibration_temperature: float = 1.0
        self.calibration_method: str = "temperature"
        self.calibration_scores: Dict[str, float] = {}
        self.calibrated_classifier = None
        # Multiplicative bias correction for expected goals (fit on training
        # means): counters systematic under/over-prediction of Poisson means.
        self.home_goal_correction: float = 1.0
        self.away_goal_correction: float = 1.0
        self.market_blend_weight: float = 0.0
        self.no_market_blend_weight: float = 0.0

    def apply_params(self, params: Dict[str, Any]) -> MatchPredictorModel:
        """Applies hyperparameter overrides to classifier + regressors.

        Each key is routed only to sub-estimators that accept it
        (e.g. min_samples_leaf reaches the forests, learning_rate the
        boosters). Must be called before fit; unknown keys are ignored.
        """
        for est in (self.classifier, self.home_regressor, self.away_regressor,
                    self.supremacy_regressor, self.totals_regressor):
            try:
                valid = est.get_params()
            except Exception:
                continue
            routed = {k: v for k, v in (params or {}).items() if k in valid}
            if routed:
                est.set_params(**routed)
        return self

    def fit(self, X: pd.DataFrame, y_outcome: pd.Series, y_hg: pd.Series, y_ag: pd.Series, sample_weight=None) -> MatchPredictorModel:
        """Fits the outcome classifier and all four goal regressors."""
        self.feature_names = list(X.columns)
        self.market_blend_weight = 0.0
        self.no_market_blend_weight = 0.0
        self.calibrated_classifier = None
        self.calibration_method = "temperature"
        self.calibration_scores = {}
        _fit_with_weights(self.classifier, X, y_outcome, sample_weight)
        self.home_regressor.fit(X, y_hg, sample_weight=sample_weight)
        self.away_regressor.fit(X, y_ag, sample_weight=sample_weight)
        # Supremacy/total heads derive from the same goal targets (no new
        # plumbing): supremacy is signed, totals are counts.
        y_sup = np.asarray(y_hg, dtype=float) - np.asarray(y_ag, dtype=float)
        y_tot = np.asarray(y_hg, dtype=float) + np.asarray(y_ag, dtype=float)
        self.supremacy_regressor.fit(X, y_sup, sample_weight=sample_weight)
        self.totals_regressor.fit(X, y_tot, sample_weight=sample_weight)
        # Fit goal bias correction on training means (guarded, bounded).
        try:
            pred_h = np.maximum(0.05, np.asarray(self.home_regressor.predict(X), dtype=float))
            pred_a = np.maximum(0.05, np.asarray(self.away_regressor.predict(X), dtype=float))
            true_h = np.asarray(y_hg, dtype=float)
            true_a = np.asarray(y_ag, dtype=float)
            self.home_goal_correction, self.away_goal_correction = fit_goal_calibration(
                pred_h, pred_a, true_h, true_a
            )
        except Exception:
            self.home_goal_correction = 1.0
            self.away_goal_correction = 1.0
        self.is_fitted = True
        return self

    def calibrate_goals(self, X_cal: pd.DataFrame, y_hg, y_ag) -> Tuple[float, float]:
        """Fits bounded goal corrections on a calibration slice only."""
        if len(X_cal) == 0:
            return self.home_goal_correction, self.away_goal_correction
        if self.home_regressor is None or self.away_regressor is None:
            pred_h, pred_a = self.predict_expected_goals(X_cal)
        else:
            pred_h = np.maximum(0.05, np.asarray(self.home_regressor.predict(X_cal), dtype=float))
            pred_a = np.maximum(0.05, np.asarray(self.away_regressor.predict(X_cal), dtype=float))
        self.home_goal_correction, self.away_goal_correction = fit_goal_calibration(
            pred_h, pred_a, np.asarray(y_hg, dtype=float), np.asarray(y_ag, dtype=float)
        )
        return self.home_goal_correction, self.away_goal_correction

    def calibrate_market_blend(self, X_cal: pd.DataFrame, y_cal: pd.Series) -> Tuple[float, float]:
        """Learns market blend weight on calibration rows using RPS only."""
        required = {"odds_implied_home", "odds_implied_draw", "odds_implied_away", "odds_missing"}
        if len(X_cal) == 0 or not required.issubset(X_cal.columns):
            self.market_blend_weight = 0.0
            self.no_market_blend_weight = 0.0
            return 0.0, 0.0
        base = self.predict_outcome_proba(X_cal, apply_temperature=False)
        market = X_cal[["odds_implied_home", "odds_implied_draw", "odds_implied_away"]].to_numpy(dtype=float)
        missing = X_cal["odds_missing"].to_numpy(dtype=float)
        y = np.asarray(y_cal, dtype=int)
        from src.config import get_config
        cfg = get_config().get("model", {})
        step = max(0.01, float(cfg.get("market_blend_grid_step", 0.05)))
        maximum = float(np.clip(cfg.get("market_blend_max_weight", 0.50), 0.0, 1.0))
        present = missing < 0.5
        best_weight, best_score = 0.0, float("inf")
        for weight in np.arange(0.0, maximum + step / 2.0, step):
            candidate = blend_market_probabilities(
                base, market, missing, np.full(len(base), float(weight))
            )
            score = _rps_loss(candidate[present], y[present]) if present.any() else float("inf")
            if score < best_score:
                best_score, best_weight = score, float(round(weight, 4))
        self.market_blend_weight = best_weight
        self.no_market_blend_weight = 0.0
        return self.market_blend_weight, self.no_market_blend_weight

    def _apply_market_blend(self, probas: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        required = {"odds_implied_home", "odds_implied_draw", "odds_implied_away", "odds_missing"}
        if not required.issubset(X.columns):
            return probas
        missing = X["odds_missing"].to_numpy(dtype=float)
        weights = np.where(missing < 0.5, self.market_blend_weight, self.no_market_blend_weight)
        market = X[["odds_implied_home", "odds_implied_draw", "odds_implied_away"]].to_numpy(dtype=float)
        return blend_market_probabilities(probas, market, missing, weights)

    def calibrate_temperature(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Fits temperature scaling on validation blended probabilities.

        Grid-searches T over the configured calibration grid to minimize
        multi-class log-loss. Returns the fitted temperature and stores
        it for inference.
        """
        probas = self.predict_outcome_proba(X_val, apply_temperature=False)
        y = np.asarray(y_val.values if hasattr(y_val, "values") else y_val, dtype=int)
        eps = 1e-15
        from src.config import get_config

        grid_cfg = get_config()["model"]
        gmin = float(grid_cfg.get("calibration_grid_min", 0.5))
        gmax = float(grid_cfg.get("calibration_grid_max", 3.0))
        gstep = float(grid_cfg.get("calibration_grid_step", 0.05))
        best_t, best_rps = 1.0, float("inf")
        grid = np.arange(gmin, gmax + gstep / 2, gstep)
        for t in [round(float(x), 2) for x in grid]:
            scaled = self._apply_temperature(probas, t)
            clipped = np.clip(scaled, eps, 1 - eps)
            rps = _rps_loss(clipped, y)
            if rps < best_rps:
                best_rps, best_t = rps, t
        self.calibration_temperature = float(best_t)
        self.calibration_method = "temperature"
        self.calibrated_classifier = None
        return self.calibration_temperature

    def calibrate_outcome(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        X_eval: pd.DataFrame | None = None,
        y_eval: pd.Series | None = None,
    ) -> Dict[str, Any]:
        """Compares temperature, sigmoid, and isotonic on disjoint slices."""
        if self.classifier is None:
            self.calibrate_temperature(X_cal, y_cal)
            self.calibration_method = "temperature"
            self.calibration_scores = {"temperature": 0.0, "sigmoid": float("inf"), "isotonic": float("inf")}
            return {"winner": "temperature", "scores": self.calibration_scores}

        # Keep candidate fitting and selection disjoint. The outer evaluation
        # slice remains reserved for headline scoring by the caller.
        fit_slice, score_slice = split_calibration_evaluation(len(X_cal))
        fit_X = X_cal.iloc[fit_slice] if fit_slice.stop else X_cal
        fit_y = y_cal.iloc[fit_slice] if fit_slice.stop else y_cal
        score_X = X_cal.iloc[score_slice] if score_slice.start else X_cal
        score_y = np.asarray(y_cal.iloc[score_slice] if score_slice.start else y_cal, dtype=int)
        raw_sources = self._source_probas(score_X)
        w_clf, w_poiss, w_sup = blend_weights()

        def _blend(clf):
            blended = w_clf * clf + w_poiss * raw_sources["poisson"] + w_sup * raw_sources["supremacy"]
            return blended / blended.sum(axis=1, keepdims=True)

        self.calibrate_temperature(fit_X, fit_y)
        temperature_base = self._apply_market_blend(_blend(raw_sources["clf"]), score_X)
        scores = {"temperature": _rps_loss(self._apply_temperature(temperature_base, self.calibration_temperature), score_y)}
        candidates = {}
        for method in ("sigmoid", "isotonic"):
            try:
                candidate = CalibratedClassifierCV(self.classifier, method=method, cv="prefit")
                candidate.fit(fit_X, fit_y)
                calibrated = align_probas(candidate.classes_, candidate.predict_proba(score_X))
                scores[method] = _rps_loss(self._apply_market_blend(_blend(calibrated), score_X), score_y)
                candidates[method] = candidate
            except (ValueError, TypeError, RuntimeError):
                scores[method] = float("inf")

        winner = min(("temperature", "sigmoid", "isotonic"), key=lambda name: scores[name])
        self.calibration_method = winner
        self.calibration_scores = {key: float(value) for key, value in scores.items()}
        self.calibrated_classifier = candidates.get(winner)
        return {"winner": winner, "scores": self.calibration_scores}

    @staticmethod
    def _apply_temperature(probas: np.ndarray, t: float) -> np.ndarray:
        """Applies temperature scaling: softmax(log(p)/T) row-wise."""
        return apply_temperature_scaling(probas, t)

    @staticmethod
    def compute_poisson_grid(h_exp: float, a_exp: float, max_goals: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates normalized bivariate Poisson probability grid and outcome probabilities.

        Applies Dixon-Coles adjustment for low scores (0-0, 1-0, 0-1, 1-1).
        """
        lh = max(0.2, float(h_exp))
        la = max(0.2, float(a_exp))
        # Dixon-Coles low-score dependence. rho must stay small and positive
        # (literature ~0.1); a negative value inverts the correction and
        # inflates draws while suppressing 1-0/0-1.
        rho = 0.11
        grid = np.zeros((max_goals + 1, max_goals + 1))

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p_h = (lh ** h) * math.exp(-lh) / math.factorial(h)
                p_a = (la ** a) * math.exp(-la) / math.factorial(a)
                tau = 1.0
                if h == 0 and a == 0:
                    tau = 1.0 - (lh * la * rho)
                elif h == 0 and a == 1:
                    tau = 1.0 + (lh * rho)
                elif h == 1 and a == 0:
                    tau = 1.0 + (la * rho)
                elif h == 1 and a == 1:
                    tau = 1.0 - rho
                grid[h, a] = max(0.0, tau) * p_h * p_a

        tot = grid.sum()
        if tot > 0:
            grid /= tot

        p_home = float(np.sum(np.tril(grid, -1)))
        p_draw = float(np.sum(np.diag(grid)))
        p_away = float(np.sum(np.triu(grid, 1)))
        return grid, np.array([p_away, p_draw, p_home])

    def predict_outcome_proba(self, X: pd.DataFrame, apply_temperature: bool = True) -> np.ndarray:
        """Returns calibrated probability matrix of shape (N, 3): [p_away, p_draw, p_home].

        Blends multi-class tree probabilities with count Poisson probabilities,
        then applies fitted temperature scaling (if calibrated).
        """
    def _source_probas(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Untempered per-source probability matrices (each (N, 3)).

        Sources: classifier, Poisson grid from hg/ag regressors, Poisson
        grid from the supremacy/totals decomposition. Shared by blending
        and by blend-weight tuning.
        """
        classifier = self.calibrated_classifier if self.calibration_method in {"sigmoid", "isotonic"} and self.calibrated_classifier is not None else self.classifier
        clf_probas = align_probas(classifier.classes_, classifier.predict_proba(X))
        exp_hg, exp_ag = self.predict_expected_goals(X)
        sup = np.asarray(self.supremacy_regressor.predict(X), dtype=float)
        tot = np.asarray(self.totals_regressor.predict(X), dtype=float)
        lam_h, lam_a = supremacy_to_means(sup, tot)
        poisson = np.zeros_like(clf_probas)
        supremacy = np.zeros_like(clf_probas)
        for i in range(len(X)):
            _, poisson[i] = self.compute_poisson_grid(float(exp_hg[i]), float(exp_ag[i]))
            _, supremacy[i] = self.compute_poisson_grid(float(lam_h[i]), float(lam_a[i]))
        return {"clf": clf_probas, "poisson": poisson, "supremacy": supremacy}

    def predict_outcome_proba(self, X: pd.DataFrame, apply_temperature: bool = True) -> np.ndarray:
        """Returns calibrated probability matrix of shape (N, 3): [p_away, p_draw, p_home].

        Blends classifier, hg/ag-Poisson, and supremacy/totals-Poisson
        sources with configured weights, then applies fitted temperature
        scaling (if calibrated).
        """
        sources = self._source_probas(X)
        w_clf, w_poiss, w_sup = blend_weights()

        blended = (
            w_clf * sources["clf"]
            + w_poiss * sources["poisson"]
            + w_sup * sources["supremacy"]
        )
        blended /= blended.sum(axis=1, keepdims=True)

        blended = self._apply_market_blend(blended, X)
        if apply_temperature and self.calibration_method == "temperature" and self.calibration_temperature != 1.0:
            blended = self._apply_temperature(blended, self.calibration_temperature)
        return blended

    def predict_expected_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns bias-corrected expected float goals (expected_hg, expected_ag)."""
        exp_hg = np.maximum(0.0, self.home_regressor.predict(X)) * self.home_goal_correction
        exp_ag = np.maximum(0.0, self.away_regressor.predict(X)) * self.away_goal_correction
        return exp_hg, exp_ag

    def predict_scoreline(self, X: pd.DataFrame) -> List[Tuple[int, int]]:
        """Forecasts integer scoreline for each match in X using Poisson mode argmax.

        Selects the highest-probability joint scoreline (h, a) consistent with
        the blended outcome argmax, so the scoreline always agrees with
        ``predict_outcome_proba``. This eliminates forced 2-1 mode collapse
        while keeping probabilities and predicted outcomes consistent.
        """
        exp_hg, exp_ag = self.predict_expected_goals(X)
        probas = self.predict_outcome_proba(X)

        scorelines: List[Tuple[int, int]] = []
        # 0-10 covers >99.9% of Poisson mass for EPL means (<3.0); 0-6
        # truncated high-scoring tails and biased outcome probs after renorm.
        max_goals = 10

        for i in range(len(X)):
            grid, _ = self.compute_poisson_grid(exp_hg[i], exp_ag[i], max_goals=max_goals)
            # Favored outcome uses the shared draw-aware rule (see
            # favor_outcome_from_proba) so the scoreline always agrees with
            # the pipeline's predicted_outcome: 0: Away, 1: Draw, 2: Home.
            no_market = float(X["odds_missing"].iloc[i]) if "odds_missing" in X.columns else 0.0
            fav_outcome = favor_outcome_from_proba(probas[i], odds_missing=no_market)

            # Select best scoreline matching favored outcome
            best_s = (1, 1)
            best_p = -1.0

            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    cond = (h > a) if fav_outcome == 2 else ((h == a) if fav_outcome == 1 else (h < a))
                    if cond and grid[h, a] > best_p:
                        best_p = grid[h, a]
                        best_s = (h, a)

            scorelines.append(best_s)

        return scorelines

    def get_feature_importances(self) -> pd.Series:
        """Extracts normalized feature importances from the outcome classifier."""
        if hasattr(self.classifier, "feature_importances_"):
            fi = self.classifier.feature_importances_
        else:
            fi = np.zeros(len(self.feature_names))
        return pd.Series(fi, index=self.feature_names).sort_values(ascending=False)

    def save(self, filepath: str) -> str:
        """Serializes fitted model artifacts, regressors, and metadata to disk using joblib."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted MatchPredictorModel.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_type": self.model_type,
            "classifier": self.classifier,
            "home_regressor": self.home_regressor,
            "away_regressor": self.away_regressor,
            "supremacy_regressor": self.supremacy_regressor,
            "totals_regressor": self.totals_regressor,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "calibration_temperature": self.calibration_temperature,
            "calibration_method": self.calibration_method,
            "calibration_scores": self.calibration_scores,
            "calibrated_classifier": self.calibrated_classifier,
            "home_goal_correction": self.home_goal_correction,
            "away_goal_correction": self.away_goal_correction,
            "market_blend_weight": self.market_blend_weight,
            "no_market_blend_weight": self.no_market_blend_weight,
        }
        joblib.dump(payload, filepath, compress=3)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> MatchPredictorModel:
        """Loads a serialized checkpoint from disk (dispatches stacked)."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")
        payload = joblib.load(filepath)
        if payload.get("model_type") == "stacked":
            return StackedEnsembleModel.load_stacked(filepath)
        instance = cls(payload["model_type"])
        instance.classifier = payload["classifier"]
        instance.home_regressor = payload["home_regressor"]
        instance.away_regressor = payload["away_regressor"]
        if "supremacy_regressor" not in payload or "totals_regressor" not in payload:
            raise RuntimeError(
                "Checkpoint predates the supremacy/totals head; retrain with "
                "`python run_pipeline.py` or `predict.py --retrain`."
            )
        instance.supremacy_regressor = payload["supremacy_regressor"]
        instance.totals_regressor = payload["totals_regressor"]
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = payload["is_fitted"]
        # Backward-compatible: checkpoints saved before calibration default to 1.0.
        instance.calibration_temperature = float(payload.get("calibration_temperature", 1.0))
        instance.calibration_method = payload.get("calibration_method", "temperature")
        instance.calibration_scores = dict(payload.get("calibration_scores", {}))
        instance.calibrated_classifier = payload.get("calibrated_classifier")
        instance.home_goal_correction = float(payload.get("home_goal_correction", 1.0))
        instance.away_goal_correction = float(payload.get("away_goal_correction", 1.0))
        instance.market_blend_weight = float(payload.get("market_blend_weight", 0.0))
        instance.no_market_blend_weight = float(payload.get("no_market_blend_weight", 0.0))
        return instance


def split_calibration_evaluation(
    n_rows: int,
) -> Tuple[slice, slice]:
    """Splits a validation slice into disjoint calibration/eval halves.

    Fitting a temperature on the same rows used for headline metrics makes
    reported log-loss optimistic, so calibration takes the first half and
    metrics the second. Single source of truth shared by benchmarking and
    hyperparameter tuning.
    """
    calibration_n = int(n_rows * 0.5)
    if n_rows >= 8 and calibration_n >= 3 and n_rows - calibration_n >= 3:
        return slice(0, calibration_n), slice(calibration_n, None)
    return slice(0, 0), slice(0, None)


def _score_benchmark_model(
    model,
    X_val: pd.DataFrame,
    y_val_outcome: pd.Series,
    y_val_hg: pd.Series,
    y_val_ag: pd.Series,
) -> Dict[str, Any]:
    """Calibrates one model and scores it on the disjoint evaluation slice.

    Shared by every benchmarked model (trees and stacked alike) so the
    numbers stay comparable. Returns the metrics entry, including RPS.
    """
    from src.evaluate import ranked_probability_score

    calibration_slice, evaluation_slice = split_calibration_evaluation(len(X_val))
    if calibration_slice.stop:
        model.calibrate_goals(
            X_val.iloc[calibration_slice],
            y_val_hg.iloc[calibration_slice],
            y_val_ag.iloc[calibration_slice],
        )
        model.calibrate_market_blend(
            X_val.iloc[calibration_slice], y_val_outcome.iloc[calibration_slice]
        )
    if calibration_slice.stop:
        try:
            model.calibrate_outcome(
                X_val.iloc[calibration_slice], y_val_outcome.iloc[calibration_slice]
            )
        except Exception:
            model.calibration_temperature = 1.0

    # Predictions on validation (calibrated)
    eval_X = X_val.iloc[evaluation_slice]
    eval_y_outcome = y_val_outcome.iloc[evaluation_slice]
    eval_y_hg = y_val_hg.iloc[evaluation_slice]
    eval_y_ag = y_val_ag.iloc[evaluation_slice]
    val_proba = model.predict_outcome_proba(eval_X)
    # Operational decision rule (draw-aware, regime-aware), so reported
    # accuracy/F1 reflect what the pipeline actually publishes.
    _missing = eval_X["odds_missing"].values if "odds_missing" in eval_X.columns else np.zeros(len(eval_X))
    val_preds = np.array([favor_outcome_from_proba(p, odds_missing=m)
                          for p, m in zip(val_proba, _missing)])
    exp_hg, exp_ag = model.predict_expected_goals(eval_X)
    pred_scores = model.predict_scoreline(eval_X)
    all_pred_scores = model.predict_scoreline(X_val)
    pred_hg = np.array([s[0] for s in pred_scores])
    pred_ag = np.array([s[1] for s in pred_scores])

    # Classification metrics
    acc = float(np.mean(val_preds == eval_y_outcome.values))

    # Multi-class log loss
    eps = 1e-15
    clipped_proba = np.clip(val_proba, eps, 1 - eps)
    # One-hot true outcomes
    y_val_onehot = np.zeros_like(val_proba)
    for row_idx, true_cls in enumerate(eval_y_outcome.values):
        y_val_onehot[row_idx, int(true_cls)] = 1.0
    log_loss = float(-np.mean(np.sum(y_val_onehot * np.log(clipped_proba), axis=1)))

    # Macro F1
    f1_scores = []
    for c in [0, 1, 2]:
        tp = np.sum((val_preds == c) & (eval_y_outcome.values == c))
        fp = np.sum((val_preds == c) & (eval_y_outcome.values != c))
        fn = np.sum((val_preds != c) & (eval_y_outcome.values == c))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores))

    # Goal prediction metrics: MAE on continuous expected goals
    # (regressor quality). Integer scorelines are evaluated separately via
    # exact-score / within-1-goal accuracy below.
    mae_hg = float(np.mean(np.abs(exp_hg - eval_y_hg.values)))
    mae_ag = float(np.mean(np.abs(exp_ag - eval_y_ag.values)))
    avg_mae = (mae_hg + mae_ag) / 2.0

    exact_score_acc = float(np.mean((pred_hg == eval_y_hg.values) & (pred_ag == eval_y_ag.values)))
    within_1_goal = float(
        np.mean((np.abs(pred_hg - eval_y_hg.values) <= 1) & (np.abs(pred_ag - eval_y_ag.values) <= 1))
    )
    rps = float(ranked_probability_score(eval_y_outcome, val_proba))

    return {
        "accuracy": acc,
        "log_loss": log_loss,
        "macro_f1": macro_f1,
        "rps": rps,
        "mae_home_goals": mae_hg,
        "mae_away_goals": mae_ag,
        "avg_goal_mae": avg_mae,
        "exact_score_acc": exact_score_acc,
        "within_1_goal_acc": within_1_goal,
        "val_preds": val_preds,
        "val_proba": val_proba,
        # Keep the historical public shape for callers that use this as a
        # validation-length diagnostic; metrics themselves use the
        # calibration-independent evaluation tail below.
        "pred_scores": all_pred_scores,
        "eval_pred_scores": pred_scores,
        "feature_importances": model.get_feature_importances(),
        "calibration_temperature": model.calibration_temperature,
        "calibration_method": getattr(model, "calibration_method", "temperature"),
        "calibration_scores": getattr(model, "calibration_scores", {}),
        "home_goal_correction": model.home_goal_correction,
        "away_goal_correction": model.away_goal_correction,
        "market_blend_weight": getattr(model, "market_blend_weight", 0.0),
        "no_market_blend_weight": getattr(model, "no_market_blend_weight", 0.0),
        "eval_y_outcome": eval_y_outcome.values,
        "eval_y_home_goals": eval_y_hg.values,
        "eval_y_away_goals": eval_y_ag.values,
    }


def train_and_benchmark_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[Dict[str, MatchPredictorModel], Dict[str, Dict[str, Any]]]:
    """Trains Random Forest, XGBoost, and the stacked ensemble on train_df and benchmarks on val_df."""
    X_train = train_df[feature_cols]
    y_train_outcome = train_df["target_outcome"]
    y_train_hg = train_df["target_home_goals"]
    y_train_ag = train_df["target_away_goals"]

    X_val = val_df[feature_cols]
    y_val_outcome = val_df["target_outcome"]
    y_val_hg = val_df["target_home_goals"]
    y_val_ag = val_df["target_away_goals"]

    models: Dict[str, MatchPredictorModel] = {
        "Random Forest": MatchPredictorModel("rf"),
        "XGBoost": MatchPredictorModel("xgboost"),
    }
    # Tuned hyperparameters from models/tuning.json when present;
    # missing file degrades to the built-in defaults above.
    from src.tuning import get_tuned_params

    models["Random Forest"].apply_params(get_tuned_params("rf"))
    models["XGBoost"].apply_params(get_tuned_params("xgboost"))

    metrics: Dict[str, Dict[str, Any]] = {}

    # Recency decay: recent seasons teach current strength (None = uniform).
    from src.config import get_config as _get_cfg_weights
    from src.feature_engineering import recency_weights as _recency_weights

    _half_life = float(_get_cfg_weights()["model"].get("recency_half_life_days", 730))
    _train_weights = _recency_weights(
        train_df["date"] if "date" in train_df.columns else None,
        half_life_days=_half_life,
    )

    for name, model in models.items():
        model.fit(X_train, y_train_outcome, y_train_hg, y_train_ag,
                  sample_weight=_train_weights)
        metrics[name] = _score_benchmark_model(
            model, X_val, y_val_outcome, y_val_hg, y_val_ag
        )

    # Stacked ensemble: level-0 members fit on train rows, meta-learner
    # fit on the calibration slice only; scored on the same eval slice.
    stacked = train_stacked_ensemble(train_df, val_df, feature_cols,
                                     sample_weight=_train_weights)
    models["Stacked"] = stacked
    metrics["Stacked"] = _score_benchmark_model(
        stacked, X_val, y_val_outcome, y_val_hg, y_val_ag
    )

    return models, metrics


# Level-0 member order everywhere (meta dims, determinism, docs).
STACK_MEMBER_ORDER: List[str] = ["rf", "xgb", "logreg", "elopoisson"]


class EloPoissonModel:
    """Pure-statistical member: Elo strength gap -> Poisson goal means.

    Reads only ``home_elo``/``away_elo`` at inference (plus
    ``odds_missing`` for the shared decision regime, never as a goal
    predictor). League means and the Elo slope are fit on TRAINING rows
    only; never validation means, priors, or substitutes.
    """

    def __init__(self):
        self.mu_h: float = 1.4
        self.mu_a: float = 1.1
        self.beta: float = 1.0
        self.calibration_temperature: float = 1.0
        self.calibration_method: str = "temperature"
        self.calibration_scores: Dict[str, float] = {}
        self.home_goal_correction: float = 1.0
        self.away_goal_correction: float = 1.0
        self.market_blend_weight: float = 0.0
        self.no_market_blend_weight: float = 0.0
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self, X: pd.DataFrame, y_outcome: pd.Series, y_hg: pd.Series, y_ag: pd.Series,
        sample_weight=None,
    ) -> EloPoissonModel:
        """Fits league means (train only) and the Elo slope via Poisson NLL."""
        self.feature_names = list(X.columns)
        true_h = np.asarray(y_hg, dtype=float)
        true_a = np.asarray(y_ag, dtype=float)
        if sample_weight is None:
            weights = np.ones(len(true_h))
        else:
            weights = np.asarray(sample_weight, dtype=float)
            weights = weights / max(1e-12, weights.mean())
        self.mu_h = float(np.sum(weights * true_h) / max(1e-12, weights.sum()))
        self.mu_a = float(np.sum(weights * true_a) / max(1e-12, weights.sum()))
        gap = (X["home_elo"].to_numpy(dtype=float) - X["away_elo"].to_numpy(dtype=float)) / 400.0
        best_beta, best_nll = 1.0, float("inf")
        for beta in np.arange(0.0, 2.0 + 1e-9, 0.05):
            lam_h = np.maximum(0.05, self.mu_h * np.exp(beta * gap))
            lam_a = np.maximum(0.05, self.mu_a * np.exp(-beta * gap))
            nll = float(
                np.sum(weights * (lam_h - true_h * np.log(lam_h)))
                + np.sum(weights * (lam_a - true_a * np.log(lam_a)))
            )
            if nll < best_nll:
                best_nll, best_beta = nll, float(beta)
        self.beta = best_beta
        self.is_fitted = True
        return self

    def _lambdas(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Expected (home, away) goals from the Elo gap (no home-advantage double count)."""
        gap = (X["home_elo"].to_numpy(dtype=float) - X["away_elo"].to_numpy(dtype=float)) / 400.0
        lam_h = np.maximum(0.05, self.mu_h * np.exp(self.beta * gap))
        lam_a = np.maximum(0.05, self.mu_a * np.exp(-self.beta * gap))
        return lam_h, lam_a

    def predict_expected_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns expected float goals (expected_hg, expected_ag)."""
        return self._lambdas(X)

    def predict_outcome_proba(self, X: pd.DataFrame, apply_temperature: bool = True) -> np.ndarray:
        """Dixon-Coles outcome probabilities from Elo-implied goal means."""
        rows = []
        for lam_h, lam_a in zip(*self._lambdas(X)):
            _, p_poiss = MatchPredictorModel.compute_poisson_grid(
                float(lam_h), float(lam_a), max_goals=10
            )
            rows.append(p_poiss)
        blended = np.array(rows)
        if apply_temperature and self.calibration_method == "temperature" and self.calibration_temperature != 1.0:
            blended = apply_temperature_scaling(blended, self.calibration_temperature)
        return blended

    def calibrate_temperature(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Fits RPS-optimized temperature scaling on validation probabilities."""
        probas = self.predict_outcome_proba(X_val, apply_temperature=False)
        y = np.asarray(y_val.values if hasattr(y_val, "values") else y_val, dtype=int)
        eps = 1e-15
        from src.config import get_config

        grid_cfg = get_config()["model"]
        gmin = float(grid_cfg.get("calibration_grid_min", 1.0))
        gmax = float(grid_cfg.get("calibration_grid_max", 3.0))
        gstep = float(grid_cfg.get("calibration_grid_step", 0.05))
        best_t, best_rps = 1.0, float("inf")
        for t in [round(float(x), 2) for x in np.arange(gmin, gmax + gstep / 2, gstep)]:
            scaled = apply_temperature_scaling(probas, t)
            clipped = np.clip(scaled, eps, 1 - eps)
            rps = _rps_loss(clipped, y)
            if rps < best_rps:
                best_rps, best_t = rps, t
        self.calibration_temperature = float(best_t)
        self.calibration_method = "temperature"
        return self.calibration_temperature

    def calibrate_outcome(self, X_cal, y_cal, X_eval=None, y_eval=None):
        """Uses the temperature fallback for this classifier-free member."""
        self.calibrate_temperature(X_cal, y_cal)
        score = _rps_loss(self.predict_outcome_proba(X_cal), np.asarray(y_cal, dtype=int))
        self.calibration_scores = {"temperature": float(score), "sigmoid": float("inf"), "isotonic": float("inf")}
        return {"winner": "temperature", "scores": self.calibration_scores}

    def predict_scoreline(self, X: pd.DataFrame) -> List[Tuple[int, int]]:
        """Constrained-argmax scorelines consistent with the shared rule."""
        lam_h, lam_a = self.predict_expected_goals(X)
        probas = self.predict_outcome_proba(X)
        missing = X["odds_missing"].values if "odds_missing" in X.columns else np.zeros(len(X))
        scorelines: List[Tuple[int, int]] = []
        for i in range(len(X)):
            grid, _ = MatchPredictorModel.compute_poisson_grid(
                float(lam_h[i]), float(lam_a[i]), max_goals=10
            )
            fav_outcome = favor_outcome_from_proba(probas[i], odds_missing=float(missing[i]))
            best_s, best_p = (1, 1), -1.0
            for h in range(11):
                for a in range(11):
                    cond = (h > a) if fav_outcome == 2 else ((h == a) if fav_outcome == 1 else (h < a))
                    if cond and grid[h, a] > best_p:
                        best_p, best_s = grid[h, a], (h, a)
            scorelines.append(best_s)
        return scorelines

    def get_feature_importances(self) -> pd.Series:
        """No learned feature weights; returns zeros (tree chart uses forests)."""
        return pd.Series(np.zeros(len(self.feature_names)), index=self.feature_names)


class StackedEnsembleModel(MatchPredictorModel):
    """Stacked ensemble: meta-learner over level-0 member probabilities.

    Members (rf, xgb, logreg, elopoisson) fit on train rows; the
    multinomial logistic meta-learner fits on the calibration slice only.
    Expected goals are the mean of member means. ``fit()`` refits members
    on new data while keeping meta weights frozen (no held-out slice
    exists at refit time); temperature stays benchmark-fitted like the
    tree models.
    """

    def __init__(
        self,
        members: Dict[str, Any],
        meta: Any,
        member_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.model_type = "stacked"
        self.members = dict(members)
        self.meta = meta
        self.member_specs = dict(member_specs or {})
        self.classifier = None
        self.home_regressor = None
        self.away_regressor = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.calibration_temperature: float = 1.0
        self.calibration_method: str = "temperature"
        self.calibration_scores: Dict[str, float] = {}
        self.home_goal_correction: float = 1.0
        self.away_goal_correction: float = 1.0
        self.market_blend_weight: float = 0.0
        self.no_market_blend_weight: float = 0.0

    def _stack_probas(self, X: pd.DataFrame) -> np.ndarray:
        """Concatenates calibrated member probas in STACK_MEMBER_ORDER."""
        parts = [np.asarray(self.members[key].predict_outcome_proba(X)) for key in STACK_MEMBER_ORDER]
        return np.concatenate(parts, axis=1)

    def apply_params(self, params: Dict[str, Any]) -> StackedEnsembleModel:
        """No-op: tuned params live on the members (applied at build)."""
        return self

    def fit(
        self, X: pd.DataFrame, y_outcome: pd.Series, y_hg: pd.Series, y_ag: pd.Series,
        sample_weight=None,
    ) -> StackedEnsembleModel:
        """Refits members on new data; meta weights stay frozen."""
        self.feature_names = list(X.columns)
        self.market_blend_weight = 0.0
        self.no_market_blend_weight = 0.0
        for key in STACK_MEMBER_ORDER:
            member = self.members[key]
            spec = self.member_specs.get(key, {})
            if hasattr(member, "apply_params"):
                member.apply_params(spec)
            member.fit(X, y_outcome, y_hg, y_ag, sample_weight=sample_weight)
        self.is_fitted = True
        return self

    def predict_outcome_proba(self, X: pd.DataFrame, apply_temperature: bool = True) -> np.ndarray:
        """Meta-learner probabilities over member probabilities."""
        stacked = self._stack_probas(X)
        blended = align_probas(self.meta.classes_, self.meta.predict_proba(stacked))
        blended = self._apply_market_blend(blended, X)
        if apply_temperature and self.calibration_method == "temperature" and self.calibration_temperature != 1.0:
            blended = self._apply_temperature(blended, self.calibration_temperature)
        return blended

    def predict_expected_goals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Mean of member expected-goal means."""
        homes, aways = [], []
        for key in STACK_MEMBER_ORDER:
            exp_hg, exp_ag = self.members[key].predict_expected_goals(X)
            homes.append(np.asarray(exp_hg, dtype=float))
            aways.append(np.asarray(exp_ag, dtype=float))
        return np.mean(homes, axis=0), np.mean(aways, axis=0)

    def get_feature_importances(self) -> pd.Series:
        """Mean importance over members that report any (the forests)."""
        series = [
            self.members[key].get_feature_importances()
            for key in STACK_MEMBER_ORDER
            if float(np.sum(np.abs(self.members[key].get_feature_importances().values))) > 0
        ]
        if not series:
            return pd.Series(np.zeros(len(self.feature_names)), index=self.feature_names)
        return pd.concat(series, axis=1).mean(axis=1).sort_values(ascending=False)

    def save(self, filepath: str) -> str:
        """Serializes members, meta-learner, and metadata via joblib."""
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted StackedEnsembleModel.")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "model_type": "stacked",
            "members": self.members,
            "meta": self.meta,
            "member_specs": self.member_specs,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "calibration_temperature": self.calibration_temperature,
            "calibration_method": self.calibration_method,
            "calibration_scores": self.calibration_scores,
            "home_goal_correction": self.home_goal_correction,
            "away_goal_correction": self.away_goal_correction,
            "market_blend_weight": self.market_blend_weight,
            "no_market_blend_weight": self.no_market_blend_weight,
        }
        joblib.dump(payload, filepath, compress=3)
        return filepath

    @classmethod
    def load_stacked(cls, filepath: str) -> StackedEnsembleModel:
        """Loads a serialized stacked checkpoint from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")
        payload = joblib.load(filepath)
        if payload.get("model_type") != "stacked":
            raise ValueError(f"Not a stacked checkpoint: {filepath}")
        instance = cls(payload["members"], payload["meta"], payload.get("member_specs"))
        instance.feature_names = payload["feature_names"]
        instance.is_fitted = payload["is_fitted"]
        instance.calibration_temperature = float(payload.get("calibration_temperature", 1.0))
        instance.calibration_method = payload.get("calibration_method", "temperature")
        instance.calibration_scores = dict(payload.get("calibration_scores", {}))
        instance.home_goal_correction = float(payload.get("home_goal_correction", 1.0))
        instance.away_goal_correction = float(payload.get("away_goal_correction", 1.0))
        instance.market_blend_weight = float(payload.get("market_blend_weight", 0.0))
        instance.no_market_blend_weight = float(payload.get("no_market_blend_weight", 0.0))
        return instance


def _build_level_zero_members() -> Dict[str, Any]:
    """Constructs fresh level-0 members (tuned params where available)."""
    from src.tuning import get_tuned_params

    return {
        "rf": MatchPredictorModel("rf").apply_params(get_tuned_params("rf")),
        "xgb": MatchPredictorModel("xgboost").apply_params(get_tuned_params("xgboost")),
        "logreg": MatchPredictorModel("logreg"),
        "elopoisson": EloPoissonModel(),
    }


def _fit_and_calibrate_member(
    member: Any,
    X_train: pd.DataFrame,
    y_train_outcome: pd.Series,
    y_train_hg: pd.Series,
    y_train_ag: pd.Series,
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    sample_weight=None,
) -> Any:
    """Fits one member on train rows and calibrates it on the cal slice."""
    member.fit(X_train, y_train_outcome, y_train_hg, y_train_ag,
               sample_weight=sample_weight)
    if len(X_cal) > 0:
        try:
            if hasattr(member, "calibrate_outcome"):
                member.calibrate_outcome(X_cal, y_cal)
            else:
                member.calibrate_temperature(X_cal, y_cal)
        except Exception:
            member.calibration_temperature = 1.0
    return member


def train_stacked_ensemble(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    n_oof_splits: int = 5,
    sample_weight=None,
) -> StackedEnsembleModel:
    """Builds the stacked ensemble with honest out-of-fold meta training.

    Level-0 members are fit under TimeSeriesSplit over TRAIN rows only;
    their out-of-fold probabilities train the meta-learner, so the meta
    never sees in-sample member outputs. Members are then refit on all
    train rows (calibrated on the fixed cal slice) for serving. The final
    evaluation slice is touched only at scoring time.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train_outcome = train_df["target_outcome"].reset_index(drop=True)
    y_train_hg = train_df["target_home_goals"].reset_index(drop=True)
    y_train_ag = train_df["target_away_goals"].reset_index(drop=True)

    calibration_slice, _ = split_calibration_evaluation(len(val_df))
    X_cal = val_df[feature_cols].iloc[calibration_slice] if calibration_slice.stop else val_df[feature_cols].iloc[0:0]
    y_cal = val_df["target_outcome"].iloc[calibration_slice] if calibration_slice.stop else val_df["target_outcome"].iloc[0:0]

    n_members = len(STACK_MEMBER_ORDER)
    sw_full = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    # Production members fit first: they serve degenerate OOF folds below
    # (micro-frames with < 3 classes, synthetic-only) and are refit-free.
    members = _build_level_zero_members()
    for key in STACK_MEMBER_ORDER:
        _fit_and_calibrate_member(
            members[key], X_train, y_train_outcome, y_train_hg, y_train_ag, X_cal, y_cal,
            sample_weight=sw_full,
        )

    oof_probas = np.zeros((len(X_train), 3 * n_members))
    splitter = TimeSeriesSplit(n_splits=max(2, n_oof_splits))
    for train_idx, held_idx in splitter.split(X_train):
        if len(np.unique(y_train_outcome.iloc[train_idx].values)) < 3:
            # Degenerate fold: reuse full-train members for these rows.
            oof_probas[held_idx] = np.concatenate(
                [np.asarray(members[key].predict_outcome_proba(X_train.iloc[held_idx]))
                 for key in STACK_MEMBER_ORDER],
                axis=1,
            )
            continue
        fold_members = _build_level_zero_members()
        sw_fold = None if sw_full is None else sw_full[np.asarray(train_idx)]
        for key in STACK_MEMBER_ORDER:
            fold_members[key].fit(
                X_train.iloc[train_idx], y_train_outcome.iloc[train_idx],
                y_train_hg.iloc[train_idx], y_train_ag.iloc[train_idx],
                sample_weight=sw_fold,
            )
        oof_probas[held_idx] = np.concatenate(
            [np.asarray(fold_members[key].predict_outcome_proba(X_train.iloc[held_idx]))
             for key in STACK_MEMBER_ORDER],
            axis=1,
        )

    from src.tuning import get_tuned_params

    meta = LogisticRegression(solver="lbfgs", C=1.0, max_iter=2000, random_state=42)
    meta.fit(oof_probas, np.asarray(y_train_outcome.values, dtype=int))

    stacked = StackedEnsembleModel(
        members, meta,
        member_specs={"rf": get_tuned_params("rf"), "xgb": get_tuned_params("xgboost"),
                      "logreg": {}, "elopoisson": {}},
    )
    stacked.feature_names = list(feature_cols)
    stacked.is_fitted = True
    return stacked
