"""Unit tests for historical odds ingestion (no network; inline fixtures)."""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    ODDS_FEATURE_COLUMNS,
    build_engineered_dataset,
    build_fixture_features,
    get_feature_column_names,
)
from src.odds_loader import (
    closing_movement,
    implied_probabilities,
    load_odds_frame,
    lookup_odds,
    parse_odds_csv,
    select_odds_triple,
)
from src.validation import assert_odds_coverage, assert_odds_frame_clean


def _clean_odds_frame():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16", "2024-08-17"]),
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Wolves", "Manchester City"],
        "odds_implied_home": [0.70, 0.40],
        "odds_implied_draw": [0.18, 0.27],
        "odds_implied_away": [0.12, 0.33],
        "odds_overround": [0.05, 0.06],
        "odds_move_home": [0.02, -0.01],
        "odds_source": ["avg_close", "avg_close"],
    })


def _write_csv(tmp_path, name, content):
    path = os.path.join(str(tmp_path), name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)
    return path


def test_implied_probabilities_strip_overround():
    # Fair triple 2.0/4.0/4.0 -> raw 0.5/0.25/0.25 sums to 1, overround 0.0
    p_h, p_d, p_a, ov = implied_probabilities(2.0, 4.0, 4.0)
    assert (p_h, p_d, p_a, ov) == pytest.approx((0.5, 0.25, 0.25, 0.0))
    # Book triple sums above 1: stripped back to a distribution
    p_h2, p_d2, p_a2, ov2 = implied_probabilities(1.8, 3.9, 4.5)
    assert p_h2 + p_d2 + p_a2 == pytest.approx(1.0)
    assert ov2 == pytest.approx(1 / 1.8 + 1 / 3.9 + 1 / 4.5 - 1.0)
    assert p_h2 > p_d2 > p_a2


def test_implied_probabilities_reject_bad_legs():
    for bad in (0.0, -2.0, np.nan, None, "abc"):
        out = implied_probabilities(bad, 3.5, 2.0)
        assert all(float(v) != float(v) for v in out), f"leg {bad!r} must yield NaNs"


def test_select_odds_triple_prefers_avg_close():
    row = pd.Series({
        "AvgCH": 2.0, "AvgCD": 3.6, "AvgCA": 4.0,
        "AvgH": 5.0, "AvgD": 5.0, "AvgA": 5.0,
        "B365H": 9.0, "B365D": 9.0, "B365A": 9.0,
    })
    p_h, _, _, _, source = select_odds_triple(row)
    assert source == "avg_close"
    assert p_h == pytest.approx((1 / 2.0) / (1 / 2.0 + 1 / 3.6 + 1 / 4.0))


def test_select_odds_triple_cascade_on_missing():
    only_open = pd.Series({"AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0})
    assert select_odds_triple(only_open)[4] == "avg_open"
    only_b365 = pd.Series({"B365H": 2.0, "B365D": 3.5, "B365A": 4.0})
    assert select_odds_triple(only_b365)[4] == "b365"
    empty = pd.Series({"B365H": np.nan, "B365D": 3.5, "B365A": 4.0})
    out = select_odds_triple(empty)
    assert out[4] == "missing"
    assert all(float(v) != float(v) for v in out[:4])


def test_closing_movement_same_book():
    row = pd.Series({
        "B365H": 2.5, "B365D": 3.4, "B365A": 3.0,      # open: home ~0.370 pre-strip
        "B365CH": 2.0, "B365CD": 3.6, "B365CA": 4.0,   # close: home ~0.5 pre-strip
    })
    move = closing_movement(row)
    open_h = (1 / 2.5) / (1 / 2.5 + 1 / 3.4 + 1 / 3.0)
    close_h = (1 / 2.0) / (1 / 2.0 + 1 / 3.6 + 1 / 4.0)
    assert move == pytest.approx(close_h - open_h)
    assert move > 0  # steam toward home


def test_closing_movement_nan_when_leg_missing():
    row = pd.Series({"B365H": 2.5, "B365D": 3.4, "B365A": 3.0})
    assert float(closing_movement(row)) != float(closing_movement(row))


def test_parse_odds_csv_canonicalizes_and_drops_results(tmp_path):
    content = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,AvgH,AvgD,AvgA,"
        "AvgCH,AvgCD,AvgCA,B365H,B365D,B365A,B365CH,B365CD,B365CA\n"
        "E0,12/09/2020,Arsenal,West Ham,2,1,H,1.8,3.9,4.5,1.7,4.0,5.0,1.85,3.8,4.4,1.72,3.9,5.2\n"
        "E0,28/09/2020,Wolves,Man City,1,3,A,5.5,4.2,1.6,6.0,4.4,1.55,5.25,4.0,1.62,5.75,4.33,1.57\n"
        "E0,30/11/2020,West Brom,Nott'm Forest,0,0,D,3.0,3.1,2.6,2.9,3.2,2.6,3.1,3.0,2.62,2.9,3.2,2.62\n"
    )
    path = _write_csv(tmp_path, "E0_2021.csv", content)
    frame = parse_odds_csv(path, "2021")
    assert list(frame.columns) == [
        "date", "home_team", "away_team", "season",
        "odds_implied_home", "odds_implied_draw", "odds_implied_away",
        "odds_overround", "odds_source", "odds_move_home",
    ]
    # Result columns can never leak: nothing named FTHG/FTAG/FTR survives.
    assert not any(c in frame.columns for c in ("FTHG", "FTAG", "FTR"))
    assert frame["home_team"].tolist() == ["Arsenal", "Wolves", "West Brom"]
    assert frame["away_team"].tolist() == ["West Ham", "Manchester City", "Nottingham Forest"]
    assert frame["odds_source"].tolist() == ["avg_close"] * 3
    probs = frame[["odds_implied_home", "odds_implied_draw", "odds_implied_away"]]
    assert ((probs.sum(axis=1) - 1.0).abs() < 1e-9).all()
    assert (frame["odds_overround"] > 0).all()


def test_parse_odds_csv_rejects_bad_schema(tmp_path):
    path = _write_csv(tmp_path, "bad.csv", "A,B\n1,2\n")
    with pytest.raises(ValueError, match="Unexpected odds schema"):
        parse_odds_csv(path, "2021")


def test_lookup_odds_hit_and_miss():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16", "2024-08-16"]),
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Wolves", "Manchester City"],
        "odds_implied_home": [0.7, 0.4],
        "odds_implied_draw": [0.18, 0.27],
        "odds_implied_away": [0.12, 0.33],
        "odds_overround": [0.05, 0.06],
        "odds_move_home": [0.02, -0.01],
        "odds_source": ["avg_close", "avg_close"],
        "season": ["2425", "2425"],
    })
    hit = lookup_odds(frame, "arsenal", "WOLVES", datetime(2024, 8, 16, 15, 30))
    assert hit is not None
    assert hit["odds_implied_home"] == pytest.approx(0.7)
    assert lookup_odds(frame, "Arsenal", "Chelsea", datetime(2024, 8, 16)) is None
    assert lookup_odds(frame, "Arsenal", "Wolves", datetime(2024, 8, 17)) is None
    assert lookup_odds(pd.DataFrame(), "Arsenal", "Wolves", datetime(2024, 8, 16)) is None


def test_load_odds_frame_offline_missing_cache_raises(tmp_path, monkeypatch):
    import src.odds_loader as odds_mod

    monkeypatch.setattr(odds_mod, "ODDS_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(odds_mod, "COMBINED_CACHE", str(tmp_path / "empty" / "e0_combined.csv"))
    with pytest.raises(FileNotFoundError, match="Offline mode"):
        load_odds_frame(seasons=["2021"], offline=True)


def test_odds_frame_clean_passes():
    assert assert_odds_frame_clean(_clean_odds_frame()) is True


def test_odds_frame_clean_rejects_result_columns():
    dirty = _clean_odds_frame().copy()
    dirty["FTHG"] = [2, 1]
    dirty["FTR"] = ["H", "H"]
    with pytest.raises(ValueError, match="post-kickoff columns"):
        assert_odds_frame_clean(dirty)


def test_odds_frame_clean_rejects_missing_columns():
    thin = _clean_odds_frame().drop(columns=["odds_overround"])
    with pytest.raises(ValueError, match="missing required columns"):
        assert_odds_frame_clean(thin)


def test_odds_frame_clean_rejects_bad_triples():
    bad = _clean_odds_frame().copy()
    bad.loc[0, "odds_implied_home"] = 0.99  # triple sums to 1.29
    with pytest.raises(ValueError, match="do not sum to 1"):
        assert_odds_frame_clean(bad)


def test_odds_frame_clean_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        assert_odds_frame_clean(pd.DataFrame())


def test_odds_coverage_full_pass_returns_one():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16", "2024-08-17"]),
        "home_team": ["Arsenal", "Chelsea"],
        "away_team": ["Wolves", "Manchester City"],
    })
    assert assert_odds_coverage(matches, _clean_odds_frame()) == pytest.approx(1.0)


def test_odds_coverage_fails_loudly_below_minimum():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2024-08-16"] + ["2024-08-18"] * 19),
        "home_team": ["Arsenal"] + ["Everton"] * 19,
        "away_team": ["Wolves"] + ["Fulham"] * 19,
    })
    thin_odds = _clean_odds_frame().iloc[:1]  # only 1 of 20 rows covered
    with pytest.raises(ValueError, match="below minimum"):
        assert_odds_coverage(matches, thin_odds, min_coverage=0.95)


def test_odds_archive_covers_historical_window():
    from src.data_loader import load_historical_stats
    from src.odds_loader import ODDS_SEASONS, load_odds_frame
    from src.validation import assert_odds_coverage

    raw = load_historical_stats(offline=True)
    odds = load_odds_frame(offline=True)
    assert {"1819", "1920"} <= set(odds["season"].astype(str))
    assert {"1819", "1920"} <= set(ODDS_SEASONS)
    assert assert_odds_coverage(raw, odds) >= 0.95


def _mini_history():
    dates = pd.date_range("2024-01-01", periods=6, freq="7D")
    return pd.DataFrame([{
        "season": "2023-24", "date": d,
        "home_team": "Arsenal", "away_team": "Chelsea",
        "home_goals": 2, "away_goals": 1, "result": "H",
        "home_shots": 15.0, "away_shots": 8.0,
        "home_shots_target": 6.0, "away_shots_target": 2.0,
        "home_corners": 7.0, "away_corners": 3.0,
        "home_possession": 58.0, "away_possession": 42.0,
    } for d in dates])


def _mini_odds_frame(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(list(dates)),
        "home_team": ["Arsenal"] * len(dates),
        "away_team": ["Chelsea"] * len(dates),
        "odds_implied_home": [0.60] * len(dates),
        "odds_implied_draw": [0.25] * len(dates),
        "odds_implied_away": [0.15] * len(dates),
        "odds_overround": [0.05] * len(dates),
        "odds_move_home": [0.03] * len(dates),
        "odds_source": ["avg_close"] * len(dates),
        "season": ["2324"] * len(dates),
    })


def test_feature_columns_include_market_signals():
    cols = get_feature_column_names()
    assert cols[-6:] == ODDS_FEATURE_COLUMNS
    assert len(cols) == 112
    assert "away_travel_km" in cols


def test_engineered_dataset_joins_odds():
    raw = _mini_history()
    odds = _mini_odds_frame(raw["date"].dt.normalize().unique())
    eng = build_engineered_dataset(raw, odds_df=odds)
    assert np.allclose(eng["odds_implied_home"].values, 0.60)
    assert (eng["odds_missing"] == 0.0).all()
    for col in ODDS_FEATURE_COLUMNS:
        assert col in eng.columns
        assert not eng[col].isna().any()


def test_engineered_dataset_without_odds_is_neutral():
    eng = build_engineered_dataset(_mini_history())
    assert np.allclose(eng["odds_implied_home"].values, 1 / 3)
    assert (eng["odds_missing"] == 1.0).all()
    assert (eng["odds_move_home"] == 0.0).all()


def test_shared_outcome_rule_is_regime_aware():
    from src.models import favor_outcome_from_proba

    # Close race, strong draw prob: draw under both regimes.
    probas = [0.30, 0.30, 0.40]
    assert favor_outcome_from_proba(probas, odds_missing=0.0) == 1
    assert favor_outcome_from_proba(probas, odds_missing=1.0) == 1
    # |h-a| = 0.11, draw prob 0.24: draw with market present
    # (margin 0.12, min 0.24), home win without market (min 0.25).
    probas = [0.31, 0.24, 0.42]
    assert favor_outcome_from_proba(probas, odds_missing=0.0) == 1
    assert favor_outcome_from_proba(probas, odds_missing=1.0) == 2
    # Clear favorite: argmax everywhere.
    assert favor_outcome_from_proba([0.15, 0.20, 0.65], odds_missing=1.0) == 2


def test_shared_outcome_rule_reads_thresholds_from_config(monkeypatch):
    import src.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: {"model": {"draw_rule": {
            "market_margin": 0.01, "market_min_prob": 0.40,
            "no_market_margin": 0.01, "no_market_min_prob": 0.40,
        }}},
    )
    from src.models import favor_outcome_from_proba

    assert favor_outcome_from_proba([0.30, 0.40, 0.30], odds_missing=0.0) == 1
    assert favor_outcome_from_proba([0.30, 0.40, 0.30], odds_missing=1.0) == 1
    assert favor_outcome_from_proba([0.20, 0.25, 0.30], odds_missing=0.0) == 2


def test_draw_rule_tuning_enforces_regime_draw_band():
    from src.models import tune_draw_rule

    probas = np.vstack([
        np.tile([[0.30, 0.40, 0.30]], (12, 1)),
        np.tile([[0.60, 0.10, 0.30]], (48, 1)),
        np.tile([[0.30, 0.40, 0.30]], (12, 1)),
        np.tile([[0.60, 0.10, 0.30]], (48, 1)),
    ])
    missing = np.repeat([0.0, 1.0], 60)
    y = np.tile([1, 0, 2, 1, 0, 2], 20)
    tuned = tune_draw_rule(probas, y, missing, forecast_probas=probas, forecast_missing=missing)

    assert {"market_margin", "market_min_prob", "no_market_margin", "no_market_min_prob"} <= set(tuned)
    assert 0.18 <= tuned["market_draw_share"] <= 0.27
    assert 0.18 <= tuned["no_market_draw_share"] <= 0.27
    assert 0.18 <= tuned["forecast_draw_share"] <= 0.27


def test_fixture_features_use_history_or_neutral_prior():
    raw = _mini_history()
    future = datetime(2024, 3, 1)
    odds = _mini_odds_frame([future.date()])
    feat_hist = build_fixture_features("Arsenal", "Chelsea", future, raw, odds_df=odds)
    assert float(feat_hist["odds_implied_home"].iloc[0]) == pytest.approx(0.60)
    feat_none = build_fixture_features("Arsenal", "Chelsea", future, raw)
    assert float(feat_none["odds_implied_home"].iloc[0]) == pytest.approx(1 / 3)
    assert float(feat_none["odds_missing"].iloc[0]) == 1.0
