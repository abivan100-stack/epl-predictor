"""Unit tests for travel, congestion, and recency weights (Phase 4)."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.feature_engineering import (
    AWAY_TRAVEL_FALLBACK_KM,
    get_feature_column_names,
    haversine_km,
    build_engineered_dataset,
    build_fixture_features,
    recency_weights,
    compute_team_rolling_features,
)
from src.models import EloPoissonModel, MatchPredictorModel


def test_exponentially_weighted_form_uses_only_prior_matches():
    team_df = pd.DataFrame({
        "team": ["Arsenal"] * 3,
        "date": pd.date_range("2024-01-01", periods=3, freq="7D"),
        "match_id": [1, 2, 3],
        "is_home": [1, 1, 1],
        "goals_for": [1.0, 3.0, 5.0],
        "goals_against": [1.0, 1.0, 1.0],
        "goal_diff": [0.0, 2.0, 4.0],
        "shots_for": [10.0, 12.0, 14.0],
        "shots_target_for": [3.0, 4.0, 5.0],
        "possession": [50.0, 55.0, 60.0],
        "points": [1.0, 3.0, 3.0],
    })

    result = compute_team_rolling_features(team_df)

    assert result.loc[0, "ewm_goals_for_5"] != result.loc[0, "goals_for"]
    assert result.loc[1, "ewm_goals_for_5"] == result.loc[0, "goals_for"]


def _mini_history(n=6, start="2024-01-01", home="Arsenal", away="Chelsea"):
    dates = pd.date_range(start, periods=n, freq="4D")
    return pd.DataFrame([{
        "season": "2023-24", "date": d,
        "home_team": home, "away_team": away,
        "home_goals": 2, "away_goals": 1, "result": "H",
        "home_shots": 15.0, "away_shots": 8.0,
        "home_shots_target": 6.0, "away_shots_target": 2.0,
        "home_corners": 7.0, "away_corners": 3.0,
        "home_possession": 58.0, "away_possession": 42.0,
    } for d in dates])


def test_haversine_known_distances():
    # London derbies are short hops; Newcastle away days are long hauls.
    assert haversine_km("Arsenal", "Tottenham") < 25.0
    assert haversine_km("Arsenal", "Arsenal") == 0.0
    assert haversine_km("Newcastle", "Bournemouth") > 400.0
    assert haversine_km("No Such Club", "Arsenal") == AWAY_TRAVEL_FALLBACK_KM
    # Symmetry for the same pair.
    assert haversine_km("Liverpool", "Everton") == abs(haversine_km("Everton", "Liverpool"))


def test_congestion_counts_prior_window_only():
    # 6 games 4 days apart (Jan 1..21): trailing-14d prior counts grow
    # 0,1,2,3 then saturate (Jan 1 falls out of later windows).
    eng = build_engineered_dataset(_mini_history())
    home_cong = eng["home_congestion_14d"].tolist()
    assert home_cong == [0, 1, 2, 3, 3, 3]
    assert (eng["diff_congestion_14d"] == 0.0).all()  # symmetric fixture list
    # Arsenal hosting Chelsea is a crosstown trip, not a long haul.
    assert (eng["away_travel_km"] < 25.0).all()


def test_congestion_serving_matches_training():
    raw = _mini_history()  # last game Jan 21
    far = datetime(2024, 2, 20)
    feat = build_fixture_features("Arsenal", "Chelsea", far, raw)
    # Window [Feb 6, Feb 20) holds no prior games.
    assert float(feat["home_congestion_14d"].iloc[0]) == 0.0
    near = datetime(2024, 1, 24)
    feat2 = build_fixture_features("Arsenal", "Chelsea", near, raw)
    # Window [Jan 10, Jan 24): Jan 13, Jan 17, Jan 21.
    assert float(feat2["home_congestion_14d"].iloc[0]) == 3.0


def test_travel_unknown_club_falls_back():
    raw = _mini_history(home="Arsenal", away="Atlantis FC")
    eng = build_engineered_dataset(raw)
    assert (eng["away_travel_km"] == AWAY_TRAVEL_FALLBACK_KM).all()
    feat = build_fixture_features("Arsenal", "Atlantis FC", datetime(2024, 3, 1), raw)
    assert float(feat["away_travel_km"].iloc[0]) == AWAY_TRAVEL_FALLBACK_KM


def test_feature_count_includes_phase4():
    assert len(get_feature_column_names()) == 112
    cols = get_feature_column_names()
    for col in ("home_congestion_14d", "away_congestion_14d",
                "diff_congestion_14d", "away_travel_km"):
        assert col in cols


def test_recency_weights_properties():
    dates = pd.date_range("2020-01-01", periods=4, freq="365D")
    w = recency_weights(dates, half_life_days=365.0)
    assert w is not None
    assert bool((np.diff(w) > 0).all())  # newer rows weigh more
    assert w[-1] == 1.0  # reference row
    assert abs(w[-2] / w[-1] - 0.5) < 0.02  # halves per half-life
    assert recency_weights(dates, half_life_days=0) is None
    assert recency_weights(None, half_life_days=365) is None


def test_fit_with_weights_runs_and_uniform_matches():
    rng = np.random.RandomState(4)
    cols = get_feature_column_names()
    X = pd.DataFrame(rng.randn(50, len(cols)), columns=cols)
    y = pd.Series(rng.choice([0, 1, 2], size=50))
    hg = pd.Series(rng.poisson(1.4, size=50))
    ag = pd.Series(rng.poisson(1.1, size=50))
    sw = np.linspace(0.5, 1.5, 50)
    for mt in ("rf", "xgboost", "logreg"):
        model = MatchPredictorModel(mt).apply_params({"n_estimators": 10})
        model.fit(X, y, hg, ag, sample_weight=sw)
        probas = model.predict_outcome_proba(X)
        assert probas.shape == (50, 3)
    elo = EloPoissonModel().fit(X, y, hg, ag, sample_weight=sw)
    assert elo.predict_outcome_proba(X).shape == (50, 3)
    # Uniform weights reproduce the unweighted fit exactly.
    plain = EloPoissonModel().fit(X, y, hg, ag)
    uni = EloPoissonModel().fit(X, y, hg, ag, sample_weight=np.ones(50))
    assert plain.mu_h == uni.mu_h and plain.beta == uni.beta
