"""Unit tests for the Elo-Poisson member and the stacked ensemble."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import get_feature_column_names
from src.models import (
    EloPoissonModel,
    MatchPredictorModel,
    StackedEnsembleModel,
    blend_market_probabilities,
    fit_goal_calibration,
    favor_outcome_from_proba,
    train_stacked_ensemble,
)


def test_market_blend_maps_bookmaker_home_draw_away_to_model_order():
    model = np.array([[0.30, 0.20, 0.50]])
    market_hda = np.array([[0.60, 0.25, 0.15]])

    blended = blend_market_probabilities(model, market_hda, np.array([0.0]), np.array([0.5]))

    np.testing.assert_allclose(blended, [[0.225, 0.225, 0.55]])


def test_goal_calibration_is_bounded_and_uses_observed_to_predicted_ratio():
    home, away = fit_goal_calibration(
        np.array([1.0, 2.0]), np.array([1.0, 1.0]),
        np.array([2.0, 2.0]), np.array([0.5, 1.0]),
    )

    assert home == 1.25
    assert away == 0.8


def _frames(n_train=120, n_val=40, seed=5):
    rng = np.random.RandomState(seed)
    cols = get_feature_column_names()
    n_feat = len(cols)
    elos = rng.uniform(1600, 1900, size=(n_train + n_val, 2))

    def _mk(n, start):
        X = pd.DataFrame(rng.randn(n, n_feat), columns=cols)
        X["home_elo"] = elos[start:start + n, 0]
        X["away_elo"] = elos[start:start + n, 1]
        X["elo_diff"] = (X["home_elo"] + 65.0) - X["away_elo"]
        X["odds_missing"] = 0.0
        # Goals carry a real Elo signal so slope fitting is meaningful.
        strength = (X["home_elo"].values - X["away_elo"].values) / 400.0
        hg = rng.poisson(np.maximum(0.3, 1.4 + strength))
        ag = rng.poisson(np.maximum(0.3, 1.1 - strength))
        res = np.where(hg > ag, 2, np.where(hg < ag, 0, 1))
        df = X.copy()
        df["target_outcome"] = res
        df["target_home_goals"] = hg
        df["target_away_goals"] = ag
        return df

    return _mk(n_train, 0), _mk(n_val, n_train), cols


def test_elo_member_uses_train_means_only():
    train, val, cols = _frames()
    model = EloPoissonModel().fit(
        train[cols], train["target_outcome"],
        train["target_home_goals"], train["target_away_goals"],
    )
    assert model.mu_h == pytest.approx(float(train["target_home_goals"].mean()))
    assert model.mu_a == pytest.approx(float(train["target_away_goals"].mean()))
    assert 0.0 <= model.beta <= 2.0
    # A big home Elo edge must raise home expectancy above the mean.
    strong = train[cols].iloc[:5].copy()
    strong["home_elo"] = 1950.0
    strong["away_elo"] = 1600.0
    hg, ag = model.predict_expected_goals(strong)
    assert bool((hg > model.mu_h).all())
    assert bool((ag < model.mu_a).all())


def test_elo_member_probas_and_scoreline_contract():
    train, val, cols = _frames()
    model = EloPoissonModel().fit(
        train[cols], train["target_outcome"],
        train["target_home_goals"], train["target_away_goals"],
    )
    assert model.calibrate_temperature(val[cols], val["target_outcome"]) >= 1.0
    probas = model.predict_outcome_proba(val[cols])
    assert probas.shape == (len(val), 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
    for i, (h, a) in enumerate(model.predict_scoreline(val[cols])):
        fav = favor_outcome_from_proba(probas[i])
        assert (h > a) if fav == 2 else ((h == a) if fav == 1 else (h < a))


def test_calibration_upgrade_compares_methods_and_persists_winner():
    train, val, cols = _frames()
    model = MatchPredictorModel("rf").fit(
        train[cols], train["target_outcome"], train["target_home_goals"], train["target_away_goals"]
    )

    result = model.calibrate_outcome(val[cols], val["target_outcome"])

    assert result["winner"] in {"temperature", "sigmoid", "isotonic"}
    assert set(result["scores"]) == {"temperature", "sigmoid", "isotonic"}
    assert model.calibration_method == result["winner"]
    assert np.allclose(model.predict_outcome_proba(val[cols]).sum(axis=1), 1.0)


def test_calibration_method_round_trips_in_checkpoint(tmp_path):
    train, val, cols = _frames()
    model = MatchPredictorModel("rf").fit(
        train[cols], train["target_outcome"], train["target_home_goals"], train["target_away_goals"]
    )
    model.calibrate_outcome(val[cols], val["target_outcome"])
    path = tmp_path / "model.joblib"
    model.save(str(path))

    loaded = MatchPredictorModel.load(str(path))
    assert loaded.calibration_method == model.calibration_method


def test_reliability_curves_render(tmp_path):
    from src.evaluate import plot_reliability_curves

    path = plot_reliability_curves(
        np.array([0, 1, 2, 0, 1, 2]),
        {"RF": np.full((6, 3), 1 / 3)},
        output_path=str(tmp_path / "reliability.png"),
    )
    assert path.endswith("reliability.png")


def test_rps_panel_renders(tmp_path):
    from src.evaluate import plot_rps_comparison

    path = plot_rps_comparison(
        {"RF": 0.21, "Stacked": 0.22},
        output_path=str(tmp_path / "rps.png"),
    )
    assert path.endswith("rps.png")


def test_stacked_meta_shapes_and_determinism():
    train, val, cols = _frames()
    first = train_stacked_ensemble(train, val, cols)
    second = train_stacked_ensemble(train, val, cols)
    p1 = first.predict_outcome_proba(val[cols])
    p2 = second.predict_outcome_proba(val[cols])
    assert p1.shape == (len(val), 3)
    assert np.allclose(p1.sum(axis=1), 1.0)
    assert np.allclose(p1, p2)  # fixed seeds everywhere
    assert first.meta.coef_.shape[1] == 12  # 4 members x 3 classes
    hg, ag = first.predict_expected_goals(val[cols])
    assert hg.shape == ag.shape == (len(val),)
    assert len(first.predict_scoreline(val[cols])) == len(val)


def test_stacked_save_load_dispatch_and_refit(tmp_path):
    train, val, cols = _frames()
    stacked = train_stacked_ensemble(train, val, cols)
    path = str(tmp_path / "stacked.joblib")
    stacked.save(path)
    loaded = MatchPredictorModel.load(path)
    assert isinstance(loaded, StackedEnsembleModel)
    assert loaded.model_type == "stacked"
    assert np.allclose(
        loaded.predict_outcome_proba(val[cols]),
        stacked.predict_outcome_proba(val[cols]),
    )
    # Refit keeps meta weights frozen while members re-learn.
    before = loaded.meta.coef_.copy()
    refit = loaded.fit(val[cols], val["target_outcome"],
                       val["target_home_goals"], val["target_away_goals"])
    assert np.array_equal(refit.meta.coef_, before)
    assert refit.is_fitted


def test_probas_stay_three_columns_on_degenerate_slices():
    # Members train on realistic 3-class data, but the calibration slice
    # carries only two classes: the meta-learner then emits 2 columns and
    # every consumer still requires (N, 3) in [Away, Draw, Home] order.
    rng = np.random.RandomState(0)
    cols = get_feature_column_names()
    n_feat = len(cols)

    def _mk(n, labels):
        X = pd.DataFrame(rng.randn(n, n_feat), columns=cols)
        X["home_elo"] = 1800.0
        X["away_elo"] = 1700.0
        X["elo_diff"] = 165.0
        X["odds_missing"] = 0.0
        df = X.copy()
        df["target_outcome"] = np.array(labels)
        df["target_home_goals"] = rng.poisson(1.4, size=n)
        df["target_away_goals"] = rng.poisson(1.1, size=n)
        return df

    train = _mk(30, [0, 1, 2] * 10)
    val = _mk(12, [0, 2] * 6)
    stacked = train_stacked_ensemble(train, val, cols)
    probas = stacked.predict_outcome_proba(val[cols])
    assert probas.shape == (12, 3)
    assert np.allclose(probas.sum(axis=1), 1.0)
    assert len(stacked.predict_scoreline(val[cols])) == 12
    single = MatchPredictorModel("rf").apply_params({"n_estimators": 10})
    single.fit(train[cols], train["target_outcome"],
               train["target_home_goals"], train["target_away_goals"])
    solo = single.predict_outcome_proba(val[cols])
    assert solo.shape == (12, 3)


def test_export_benchmark_includes_stacked():
    from types import SimpleNamespace

    from export_web_data import build_benchmark

    def _entry(**kw):
        base = {
            "accuracy": 0.45, "macro_f1": 0.4, "log_loss": 1.03,
            "rps": 0.2068,
            "mae_home_goals": 0.95, "mae_away_goals": 0.86,
            "avg_goal_mae": 0.9, "within_1_goal_acc": 0.58,
            "exact_score_acc": 0.09,
            "feature_importances": pd.Series([0.5, 0.5], index=["a", "b"]),
        }
        base.update(kw)
        return base

    fake = SimpleNamespace(
        metrics={"Random Forest": _entry(), "XGBoost": _entry(), "Stacked": _entry()},
        best_model_name="Stacked",
    )
    payload = build_benchmark(fake)
    assert payload["productionModel"] == "Stacked"
    assert [m["name"] for m in payload["models"]] == ["Random Forest", "XGBoost", "Stacked"]
    assert [m["isProduction"] for m in payload["models"]] == [False, False, True]
    assert all(m["rps"] == 0.2068 for m in payload["models"])
