"""Feature engineering pipeline for match outcome and scoreline forecasting.

Computes rolling statistics (goals, shots, possession, points momentum),
venue-specific form, head-to-head metrics, rest days, and pre-kickoff
bookmaker odds signals with strict zero-leakage guarantees.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WINDOWS: List[int] = [3, 5, 10]
EWM_SPANS: List[int] = [5, 10]
EWM_METRICS: List[str] = [
    "goals_for", "goals_against", "goal_diff", "shots_for",
    "shots_target_for", "points",
]


def _chronological(df: pd.DataFrame, *extra_keys: str) -> pd.DataFrame:
    """Return a stable chronological copy.

    Several matches can share a kickoff date.  A plain date sort is not a
    sufficient ordering for recursive features because the sort implementation
    is free to rearrange ties.  Preserve the caller's order as the final key,
    while allowing a real match id to provide a stronger deterministic key.
    """
    out = df.copy()
    if "_source_order" not in out.columns:
        out["_source_order"] = np.arange(len(out), dtype=int)
    keys = ["date", *[key for key in extra_keys if key in out.columns], "_source_order"]
    return out.sort_values(keys, kind="mergesort").reset_index(drop=True)

# Zero-leakage fallbacks for cold-start rolling features. These are fixed
# league-average priors, NOT dataset medians (which would leak future info).
LEAGUE_DEFAULTS: Dict[str, float] = {
    "rest_days": 7.0,
    "congestion_14d": 0.0,
    "roll_goals_for": 1.35,
    "roll_goals_against": 1.35,
    "roll_goal_diff": 0.0,
    "roll_shots_for": 12.0,
    "roll_shots_target_for": 4.0,
    "roll_possession": 50.0,
    "roll_points": 1.35,
    "ewm_goals_for": 1.35,
    "ewm_goals_against": 1.35,
    "ewm_goal_diff": 0.0,
    "ewm_shots_for": 12.0,
    "ewm_shots_target_for": 4.0,
    "ewm_points": 1.35,
    "venue_roll_goals_for": 1.45,
    "venue_roll_goals_against": 1.35,
    "venue_roll_points": 1.45,
    "h2h_home_win_rate": 0.33,
    "h2h_goal_diff": 0.0,
    "h2h_matches_count": 0.0,
}

# Stadium coordinates (decimal degrees, verified against Wikipedia infobox
# data; 2dp is ~1km precision, ample for a travel-fatigue feature).
# Unknown clubs fall back to AWAY_TRAVEL_FALLBACK_KM (league-average trip).
STADIUM_COORDS: Dict[str, Tuple[float, float]] = {
    "Arsenal": (51.56, -0.11),
    "Aston Villa": (52.51, -1.88),
    "Bournemouth": (50.74, -1.84),
    "Brentford": (51.49, -0.29),
    "Brighton": (50.86, -0.08),
    "Burnley": (53.79, -2.23),
    "Cardiff": (51.47, -3.20),
    "Chelsea": (51.48, -0.19),
    "Coventry": (52.45, -1.50),
    "Crystal Palace": (51.40, -0.09),
    "Everton": (53.44, -2.97),
    "Fulham": (51.48, -0.22),
    "Huddersfield": (53.65, -1.77),
    "Hull": (53.75, -0.37),
    "Ipswich": (52.06, 1.15),
    "Leeds": (53.78, -1.57),
    "Leicester": (52.62, -1.14),
    "Liverpool": (53.43, -2.96),
    "Luton": (51.88, -0.43),
    "Manchester City": (53.48, -2.20),
    "Manchester United": (53.46, -2.29),
    "Newcastle": (54.98, -1.62),
    "Norwich": (52.62, 1.31),
    "Nottingham Forest": (52.94, -1.13),
    "Sheffield United": (53.37, -1.47),
    "Southampton": (50.91, -1.39),
    "Sunderland": (54.91, -1.39),
    "Tottenham": (51.60, -0.07),
    "Watford": (51.65, -0.40),
    "West Brom": (52.51, -1.96),
    "West Ham": (51.54, -0.02),
    "Wolves": (52.59, -2.13),
}

AWAY_TRAVEL_FALLBACK_KM = 180.0


def haversine_km(home_team: str, away_team: str) -> float:
    """One-way away-team travel distance between home stadiums (km).

    Static geography: zero leakage by construction. Unknown clubs return
    the league-average fallback instead of crashing.
    """
    home = STADIUM_COORDS.get(home_team)
    away = STADIUM_COORDS.get(away_team)
    if home is None or away is None:
        return AWAY_TRAVEL_FALLBACK_KM
    lat1, lon1 = np.radians(home)
    lat2, lon2 = np.radians(away)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(2.0 * 6371.0 * np.arcsin(np.sqrt(h)))


def recency_weights(dates, half_life_days: float = 730.0, ref=None):
    """Exponential sample weights by age: 0.5 ** (age_days / half_life).

    Anchored at the max date by default (deterministic given data).
    Returns None when disabled (half_life_days falsy/non-positive) or when
    no dates are available (synthetic frames without a date column).
    """
    if not half_life_days or half_life_days <= 0:
        return None
    if dates is None:
        return None
    stamps = pd.to_datetime(pd.Series(list(dates))).reset_index(drop=True)
    ref_stamp = pd.to_datetime(ref) if ref is not None else stamps.max()
    age_days = (ref_stamp - stamps).dt.total_seconds() / (24 * 3600)
    age_days = age_days.clip(lower=0.0).to_numpy(dtype=float)
    return np.power(0.5, age_days / float(half_life_days))


# Historical baseline Elo ratings & power parameters across 2020-2026
BASE_ELO: Dict[str, float] = {
    # Elite Title Contenders
    "Manchester City": 1950.0,
    "Arsenal": 1915.0,
    "Liverpool": 1895.0,
    # European Contenders
    "Chelsea": 1790.0,
    "Newcastle": 1785.0,
    "Aston Villa": 1780.0,
    "Tottenham": 1770.0,
    "Manchester United": 1760.0,
    # Established Mid-Table
    "West Ham": 1690.0,
    "Brighton": 1690.0,
    "Brentford": 1670.0,
    "Bournemouth": 1660.0,
    "Crystal Palace": 1650.0,
    "Fulham": 1645.0,
    "Wolves": 1640.0,
    # Lower Table / Regulars
    "Leicester": 1600.0,
    "Everton": 1590.0,
    "Nottingham Forest": 1575.0,
    "Leeds": 1560.0,
    "Southampton": 1540.0,
    "Burnley": 1520.0,
    # Relegation Battlers & Promoted Sides
    "Watford": 1500.0,
    "Norwich": 1490.0,
    "West Brom": 1490.0,
    "Sheffield United": 1480.0,
    "Luton": 1480.0,
    "Ipswich": 1475.0,
    "Sunderland": 1460.0,
    "Coventry": 1430.0,
    "Hull": 1420.0,
}

CLUB_POWER_INDEX: Dict[str, Dict[str, float]] = {
    "Manchester City": {"elo": 1950.0, "gf_baseline": 2.42, "ga_baseline": 0.86, "points_baseline": 2.30, "shots_baseline": 16.5, "target_baseline": 6.5, "poss_baseline": 65.0},
    "Arsenal": {"elo": 1915.0, "gf_baseline": 2.18, "ga_baseline": 0.99, "points_baseline": 2.20, "shots_baseline": 15.5, "target_baseline": 5.8, "poss_baseline": 60.0},
    "Liverpool": {"elo": 1895.0, "gf_baseline": 2.14, "ga_baseline": 1.08, "points_baseline": 2.15, "shots_baseline": 16.0, "target_baseline": 6.0, "poss_baseline": 61.0},
    "Chelsea": {"elo": 1790.0, "gf_baseline": 1.64, "ga_baseline": 1.22, "points_baseline": 1.70, "shots_baseline": 14.5, "target_baseline": 5.2, "poss_baseline": 57.0},
    "Newcastle": {"elo": 1785.0, "gf_baseline": 1.62, "ga_baseline": 1.32, "points_baseline": 1.68, "shots_baseline": 14.0, "target_baseline": 5.0, "poss_baseline": 53.0},
    "Aston Villa": {"elo": 1780.0, "gf_baseline": 1.56, "ga_baseline": 1.36, "points_baseline": 1.65, "shots_baseline": 13.5, "target_baseline": 4.8, "poss_baseline": 52.0},
    "Tottenham": {"elo": 1770.0, "gf_baseline": 1.76, "ga_baseline": 1.38, "points_baseline": 1.62, "shots_baseline": 14.5, "target_baseline": 5.3, "poss_baseline": 55.0},
    "Manchester United": {"elo": 1760.0, "gf_baseline": 1.58, "ga_baseline": 1.27, "points_baseline": 1.60, "shots_baseline": 14.0, "target_baseline": 5.0, "poss_baseline": 53.0},
    "West Ham": {"elo": 1690.0, "gf_baseline": 1.40, "ga_baseline": 1.45, "points_baseline": 1.35, "shots_baseline": 12.0, "target_baseline": 4.0, "poss_baseline": 45.0},
    "Brighton": {"elo": 1690.0, "gf_baseline": 1.40, "ga_baseline": 1.33, "points_baseline": 1.38, "shots_baseline": 14.0, "target_baseline": 4.6, "poss_baseline": 56.0},
    "Brentford": {"elo": 1670.0, "gf_baseline": 1.45, "ga_baseline": 1.49, "points_baseline": 1.32, "shots_baseline": 12.5, "target_baseline": 4.3, "poss_baseline": 46.0},
    "Bournemouth": {"elo": 1660.0, "gf_baseline": 1.28, "ga_baseline": 1.69, "points_baseline": 1.30, "shots_baseline": 12.5, "target_baseline": 4.2, "poss_baseline": 46.0},
    "Crystal Palace": {"elo": 1650.0, "gf_baseline": 1.22, "ga_baseline": 1.43, "points_baseline": 1.28, "shots_baseline": 11.5, "target_baseline": 3.9, "poss_baseline": 45.0},
    "Fulham": {"elo": 1645.0, "gf_baseline": 1.25, "ga_baseline": 1.46, "points_baseline": 1.25, "shots_baseline": 12.0, "target_baseline": 4.1, "poss_baseline": 48.0},
    "Wolves": {"elo": 1640.0, "gf_baseline": 1.20, "ga_baseline": 1.50, "points_baseline": 1.22, "shots_baseline": 11.0, "target_baseline": 3.8, "poss_baseline": 46.0},
    "Everton": {"elo": 1590.0, "gf_baseline": 1.10, "ga_baseline": 1.44, "points_baseline": 1.05, "shots_baseline": 11.5, "target_baseline": 3.7, "poss_baseline": 42.0},
    "Nottingham Forest": {"elo": 1575.0, "gf_baseline": 1.18, "ga_baseline": 1.74, "points_baseline": 1.00, "shots_baseline": 11.0, "target_baseline": 3.5, "poss_baseline": 41.0},
    "Leeds": {"elo": 1560.0, "gf_baseline": 1.36, "ga_baseline": 1.81, "points_baseline": 0.98, "shots_baseline": 12.0, "target_baseline": 3.8, "poss_baseline": 47.0},
    "Leicester": {"elo": 1600.0, "gf_baseline": 1.25, "ga_baseline": 1.75, "points_baseline": 0.95, "shots_baseline": 11.5, "target_baseline": 3.6, "poss_baseline": 46.0},
    "Southampton": {"elo": 1540.0, "gf_baseline": 1.10, "ga_baseline": 1.85, "points_baseline": 0.90, "shots_baseline": 11.0, "target_baseline": 3.5, "poss_baseline": 44.0},
    "Burnley": {"elo": 1520.0, "gf_baseline": 1.05, "ga_baseline": 1.85, "points_baseline": 0.88, "shots_baseline": 10.5, "target_baseline": 3.2, "poss_baseline": 43.0},
    "Ipswich": {"elo": 1475.0, "gf_baseline": 1.13, "ga_baseline": 1.94, "points_baseline": 0.85, "shots_baseline": 10.0, "target_baseline": 3.1, "poss_baseline": 42.0},
    "Sunderland": {"elo": 1460.0, "gf_baseline": 1.05, "ga_baseline": 1.82, "points_baseline": 0.82, "shots_baseline": 9.8, "target_baseline": 3.0, "poss_baseline": 42.0},
    "Coventry": {"elo": 1430.0, "gf_baseline": 0.97, "ga_baseline": 1.86, "points_baseline": 0.78, "shots_baseline": 9.5, "target_baseline": 2.9, "poss_baseline": 41.0},
    "Hull": {"elo": 1420.0, "gf_baseline": 0.94, "ga_baseline": 1.89, "points_baseline": 0.75, "shots_baseline": 9.2, "target_baseline": 2.8, "poss_baseline": 40.0},
    "Watford": {"elo": 1500.0, "gf_baseline": 1.08, "ga_baseline": 1.80, "points_baseline": 0.88, "shots_baseline": 10.8, "target_baseline": 3.4, "poss_baseline": 43.0},
    "Norwich": {"elo": 1490.0, "gf_baseline": 1.05, "ga_baseline": 1.82, "points_baseline": 0.86, "shots_baseline": 10.5, "target_baseline": 3.3, "poss_baseline": 43.0},
    "West Brom": {"elo": 1490.0, "gf_baseline": 1.04, "ga_baseline": 1.83, "points_baseline": 0.85, "shots_baseline": 10.4, "target_baseline": 3.2, "poss_baseline": 42.0},
    "Sheffield United": {"elo": 1480.0, "gf_baseline": 1.02, "ga_baseline": 1.85, "points_baseline": 0.84, "shots_baseline": 10.2, "target_baseline": 3.1, "poss_baseline": 42.0},
    "Luton": {"elo": 1480.0, "gf_baseline": 1.10, "ga_baseline": 1.88, "points_baseline": 0.84, "shots_baseline": 10.3, "target_baseline": 3.2, "poss_baseline": 42.0},
}


def compute_dynamic_elo(
    matches_df: pd.DataFrame,
    initial_ratings: Optional[Dict[str, float]] = None,
    home_adv: Optional[float] = None,
    k_base: float = 20.0,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    Dict[str, List[float]],
    Dict[str, float],
    Dict[str, List[float]],
]:
    """Calculates chronological pre-match Elo ratings and momentum trajectories with zero leakage.

    Features:
        - Dynamic streak momentum acceleration: Teams on active streaks receive amplified K updates.
        - Inter-season regression to the mean (15% reversion to base Elo, streak reset).
        - Multi-window Elo velocity (3-match and 5-match rolling rate of change).
        - Non-linear goal margin scaling.

    Returns:
        home_elos: Pre-match Elo for home team for each row.
        away_elos: Pre-match Elo for away team for each row.
        elo_diffs: Pre-match (home_elo + home_adv - away_elo) for each row.
        momentum_dict: Pre-match 3-match and 5-match Elo momentum features.
        current_ratings: Final Elo state after all matches played.
        rating_histories: Chronological history of pre-match Elo ratings per club.
    """
    # Always operate chronologically; callers may pass unsorted frames.
    if "date" in matches_df.columns:
        matches_df = _chronological(matches_df, "match_id")
    if home_adv is None:
        from src.config import get_config

        home_adv = float(get_config()["model"].get("home_advantage", 65.0))
    ratings: Dict[str, float] = dict(initial_ratings) if initial_ratings else dict(BASE_ELO)
    rating_histories: Dict[str, List[float]] = {t: [] for t in BASE_ELO.keys()}
    streaks: Dict[str, int] = {t: 0 for t in BASE_ELO.keys()}

    n_matches = len(matches_df)
    if n_matches == 0:
        empty_mom = {
            "home_elo_momentum_3": [],
            "away_elo_momentum_3": [],
            "diff_elo_momentum_3": [],
            "home_elo_momentum_5": [],
            "away_elo_momentum_5": [],
            "diff_elo_momentum_5": [],
        }
        return [], [], [], empty_mom, ratings, rating_histories

    home_elos: List[float] = [0.0] * n_matches
    away_elos: List[float] = [0.0] * n_matches
    elo_diffs: List[float] = [0.0] * n_matches

    h_mom_3: List[float] = [0.0] * n_matches
    a_mom_3: List[float] = [0.0] * n_matches
    diff_mom_3: List[float] = [0.0] * n_matches

    h_mom_5: List[float] = [0.0] * n_matches
    a_mom_5: List[float] = [0.0] * n_matches
    diff_mom_5: List[float] = [0.0] * n_matches

    ht_arr = matches_df["home_team"].values
    at_arr = matches_df["away_team"].values
    season_arr = matches_df["season"].values if "season" in matches_df.columns else None
    has_goals = "home_goals" in matches_df.columns and "away_goals" in matches_df.columns
    hg_arr = matches_df["home_goals"].values if has_goals else None
    ag_arr = matches_df["away_goals"].values if has_goals else None

    last_season = season_arr[0] if season_arr is not None and len(season_arr) > 0 else None

    for i in range(n_matches):
        ht = ht_arr[i]
        at = at_arr[i]

        # Handle off-season mean reversion between seasons
        if season_arr is not None:
            curr_season = season_arr[i]
            if curr_season != last_season:
                for team in ratings:
                    base = BASE_ELO.get(team, 1420.0)
                    ratings[team] = 0.85 * ratings[team] + 0.15 * base
                    streaks[team] = 0
                last_season = curr_season

        rh = ratings.get(ht, BASE_ELO.get(ht, 1420.0))
        ra = ratings.get(at, BASE_ELO.get(at, 1420.0))

        # Record pre-match ratings
        home_elos[i] = rh
        away_elos[i] = ra
        elo_diffs[i] = (rh + home_adv) - ra

        # Compute pre-match Elo momentum (rate of change over past matches)
        h_hist = rating_histories.get(ht, [])
        a_hist = rating_histories.get(at, [])

        h_m3 = (rh - h_hist[-3]) if len(h_hist) >= 3 else (rh - BASE_ELO.get(ht, 1420.0))
        a_m3 = (ra - a_hist[-3]) if len(a_hist) >= 3 else (ra - BASE_ELO.get(at, 1420.0))
        h_mom_3[i] = round(h_m3, 2)
        a_mom_3[i] = round(a_m3, 2)
        diff_mom_3[i] = round(h_m3 - a_m3, 2)

        h_m5 = (rh - h_hist[-5]) if len(h_hist) >= 5 else (rh - BASE_ELO.get(ht, 1420.0))
        a_m5 = (ra - a_hist[-5]) if len(a_hist) >= 5 else (ra - BASE_ELO.get(at, 1420.0))
        h_mom_5[i] = round(h_m5, 2)
        a_mom_5[i] = round(a_m5, 2)
        diff_mom_5[i] = round(h_m5 - a_m5, 2)

        # Update rating history before match is played (zero-leakage)
        if ht not in rating_histories:
            rating_histories[ht] = []
        if at not in rating_histories:
            rating_histories[at] = []
        rating_histories[ht].append(rh)
        rating_histories[at].append(ra)

        # If match was played with valid score, update Elo for subsequent matches
        if has_goals:
            hg = hg_arr[i]
            ag = ag_arr[i]
            if pd.notna(hg) and pd.notna(ag):
                hg_val = float(hg)
                ag_val = float(ag)

                dr = (rh + home_adv) - ra
                eh = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
                ea = 1.0 - eh

                sh = 1.0 if hg_val > ag_val else (0.5 if hg_val == ag_val else 0.0)
                sa = 1.0 - sh

                margin = abs(hg_val - ag_val)
                g_mult = 1.0 if margin <= 1 else (1.5 if margin == 2 else 1.75 + (margin - 3) / 8.0)

                # Streak momentum multiplier
                h_streak = streaks.get(ht, 0)
                a_streak = streaks.get(at, 0)

                h_streak_mult = 1.0
                if sh == 1.0 and h_streak >= 2:
                    h_streak_mult += min(0.35, 0.08 * (h_streak - 1))
                elif sh == 0.0 and h_streak <= -2:
                    h_streak_mult += min(0.35, 0.08 * (abs(h_streak) - 1))

                a_streak_mult = 1.0
                if sa == 1.0 and a_streak >= 2:
                    a_streak_mult += min(0.35, 0.08 * (a_streak - 1))
                elif sa == 0.0 and a_streak <= -2:
                    a_streak_mult += min(0.35, 0.08 * (abs(a_streak) - 1))

                k_h = k_base * g_mult * h_streak_mult
                k_a = k_base * g_mult * a_streak_mult

                ratings[ht] = rh + k_h * (sh - eh)
                ratings[at] = ra + k_a * (sa - ea)

                # Update streaks
                if sh == 1.0:
                    streaks[ht] = h_streak + 1 if h_streak > 0 else 1
                    streaks[at] = a_streak - 1 if a_streak < 0 else -1
                elif sa == 1.0:
                    streaks[at] = a_streak + 1 if a_streak > 0 else 1
                    streaks[ht] = h_streak - 1 if h_streak < 0 else -1
                else:
                    streaks[ht] = 0
                    streaks[at] = 0

    momentum_dict = {
        "home_elo_momentum_3": h_mom_3,
        "away_elo_momentum_3": a_mom_3,
        "diff_elo_momentum_3": diff_mom_3,
        "home_elo_momentum_5": h_mom_5,
        "away_elo_momentum_5": a_mom_5,
        "diff_elo_momentum_5": diff_mom_5,
    }

    return home_elos, away_elos, elo_diffs, momentum_dict, ratings, rating_histories


def transform_matches_to_team_perspective(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms match results into a row-per-team dataset sorted chronologically.

    Each match generates two rows: one for the home team and one for the away team.
    """
    if df.empty:
        return pd.DataFrame()

    df = _chronological(df, "match_id")
    n = len(df)
    m_ids = df.index.values

    # Determine match results if not already present
    if "result" in df.columns:
        res = df["result"]
    else:
        res = np.where(
            df["home_goals"] > df["away_goals"],
            "H",
            np.where(df["home_goals"] < df["away_goals"], "A", "D"),
        )

    h_pts = np.where(res == "H", 3, np.where(res == "D", 1, 0))
    a_pts = np.where(res == "A", 3, np.where(res == "D", 1, 0))

    home_df = pd.DataFrame(
        {
            "match_id": m_ids,
            "date": df["date"],
            "team": df["home_team"],
            "opponent": df["away_team"],
            "is_home": 1,
            "goals_for": df["home_goals"],
            "goals_against": df["away_goals"],
            "goal_diff": df["home_goals"] - df["away_goals"],
            "shots_for": df.get("home_shots", 12.0),
            "shots_against": df.get("away_shots", 10.0),
            "shots_target_for": df.get("home_shots_target", 4.0),
            "shots_target_against": df.get("away_shots_target", 3.0),
            "possession": df.get("home_possession", 50.0),
            "points": h_pts,
            "win": np.where(res == "H", 1, 0),
            "draw": np.where(res == "D", 1, 0),
            "loss": np.where(res == "A", 1, 0),
        }
    )

    away_df = pd.DataFrame(
        {
            "match_id": m_ids,
            "date": df["date"],
            "team": df["away_team"],
            "opponent": df["home_team"],
            "is_home": 0,
            "goals_for": df["away_goals"],
            "goals_against": df["home_goals"],
            "goal_diff": df["away_goals"] - df["home_goals"],
            "shots_for": df.get("away_shots", 10.0),
            "shots_against": df.get("home_shots", 12.0),
            "shots_target_for": df.get("away_shots_target", 3.0),
            "shots_target_against": df.get("home_shots_target", 4.0),
            "possession": df.get("away_possession", 50.0),
            "points": a_pts,
            "win": np.where(res == "A", 1, 0),
            "draw": np.where(res == "D", 1, 0),
            "loss": np.where(res == "H", 1, 0),
        }
    )

    team_df = pd.concat([home_df, away_df], ignore_index=True)
    team_df = team_df.sort_values(by=["date", "match_id", "is_home"], kind="mergesort").reset_index(drop=True)
    return team_df


def compute_team_rolling_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """Computes rolling averages for each team prior to each match (shift 1)."""
    metrics = [
        "goals_for",
        "goals_against",
        "goal_diff",
        "shots_for",
        "shots_target_for",
        "possession",
        "points",
    ]

    team_df = team_df.sort_values(by=["team", "date", "match_id"], kind="mergesort").reset_index(drop=True)

    # Rest days calculation
    team_df["prev_date"] = team_df.groupby("team")["date"].shift(1)
    team_df["rest_days"] = (team_df["date"] - team_df["prev_date"]).dt.total_seconds() / (24 * 3600)
    team_df["rest_days"] = team_df["rest_days"].fillna(7.0).clip(lower=1.0, upper=30.0)

    # Congestion: prior matches in the trailing 14-day window (shift-safe:
    # only rows strictly before the current one are counted).
    team_vals = team_df["team"].values
    date_vals = team_df["date"].values
    congestion = np.zeros(len(team_df), dtype=int)
    start = 0
    for end in range(1, len(team_df) + 1):
        if end == len(team_df) or team_vals[end] != team_vals[start]:
            block = date_vals[start:end]
            left = np.searchsorted(block, block - np.timedelta64(14, "D"))
            congestion[start:end] = np.arange(end - start) - left
            start = end
    team_df["congestion_14d"] = congestion

    # Rolling overall metrics
    for w in WINDOWS:
        for m in metrics:
            col_name = f"roll_{m}_{w}"
            team_df[col_name] = (
                team_df.groupby("team")[m]
                .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            )

    # Exponentially weighted form reacts faster than fixed windows while
    # remaining strictly pre-match: shift before ewm so the current result
    # can never influence its own feature row.
    for span in EWM_SPANS:
        for m in EWM_METRICS:
            team_df[f"ewm_{m}_{span}"] = (
                team_df.groupby("team")[m]
                .transform(lambda s: s.shift(1).ewm(span=span, adjust=False, min_periods=1).mean())
            )

    # Venue-specific rolling metrics (home form for home games, away form for away games)
    team_df = team_df.sort_values(by=["team", "is_home", "date", "match_id"], kind="mergesort").reset_index(drop=True)
    venue_metrics = ["goals_for", "goals_against", "points"]
    for m in venue_metrics:
        team_df[f"venue_roll_{m}_5"] = (
            team_df.groupby(["team", "is_home"])[m]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )

    team_df = team_df.sort_values(by=["match_id", "is_home"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    return team_df


def compute_head_to_head_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Computes historical head-to-head records prior to each match."""
    matches_df = _chronological(matches_df, "match_id")

    h2h_h_win_rate = []
    h2h_goal_diff = []
    h2h_total_matches = []

    # Dictionary mapping frozenset({teamA, teamB}) to list of prior encounters
    # encounter: (home_team, hg, ag)
    h2h_history: Dict[Tuple[str, str], List[Tuple[str, int, int]]] = {}

    for _, row in matches_df.iterrows():
        ht = row["home_team"]
        at = row["away_team"]
        pair_key = (ht, at) if ht < at else (at, ht)

        prior_encounters = h2h_history.get(pair_key, [])
        if prior_encounters:
            # Filter up to last 5 encounters
            recent = prior_encounters[-5:]
            h_wins = 0
            gd_sum = 0
            for enc_ht, enc_hg, enc_ag in recent:
                if enc_ht == ht:
                    gd = enc_hg - enc_ag
                    if gd > 0:
                        h_wins += 1
                else:
                    gd = enc_ag - enc_hg
                    if gd > 0:
                        h_wins += 1
                gd_sum += gd

            h2h_h_win_rate.append(h_wins / len(recent))
            h2h_goal_diff.append(gd_sum / len(recent))
            h2h_total_matches.append(len(recent))
        else:
            h2h_h_win_rate.append(0.33)  # default prior
            h2h_goal_diff.append(0.0)
            h2h_total_matches.append(0)

        # Record this encounter after calculating features. Skip unplayed
        # fixtures (NaN goals) so inference frames never crash here.
        if pair_key not in h2h_history:
            h2h_history[pair_key] = []
        try:
            hg_val = row["home_goals"]
            ag_val = row["away_goals"]
            if pd.notna(hg_val) and pd.notna(ag_val):
                h2h_history[pair_key].append((ht, int(hg_val), int(ag_val)))
        except (ValueError, TypeError, KeyError):
            pass

    matches_df["h2h_home_win_rate"] = h2h_h_win_rate
    matches_df["h2h_goal_diff"] = h2h_goal_diff
    matches_df["h2h_matches_count"] = h2h_total_matches
    return matches_df


# Pre-kickoff bookmaker market features. Odds are legal features only when
# struck before kickoff (closing aggregates qualify); results never enter.
ODDS_FEATURE_COLUMNS: List[str] = [
    "odds_implied_home",
    "odds_implied_draw",
    "odds_implied_away",
    "odds_overround",
    "odds_move_home",
    "odds_missing",
]

# Neutral market row: maximum-entropy distribution, used whenever a fixture
# has no pre-kickoff odds (and flagged via odds_missing).
ODDS_NEUTRAL: Dict[str, float] = {
    "odds_implied_home": 1.0 / 3.0,
    "odds_implied_draw": 1.0 / 3.0,
    "odds_implied_away": 1.0 / 3.0,
    "odds_overround": 0.0,
    "odds_move_home": 0.0,
    "odds_missing": 1.0,
}


def resolve_odds_features(
    home_team: str,
    away_team: str,
    match_date: datetime,
    odds_row: Optional[Dict[str, float]] = None,
    odds_df: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Resolves the six market features for one fixture.

    Precedence: explicit ``odds_row`` (live override) -> lookup in
    ``odds_df`` (historical frame) -> neutral row with ``odds_missing=1``.
    An explicit or looked-up row reports ``odds_missing=0``; NaN legs
    inside a found row degrade to neutral rather than crashing.
    """
    from src.odds_loader import lookup_odds

    candidate = odds_row
    if candidate is None and odds_df is not None:
        candidate = lookup_odds(odds_df, home_team, away_team, match_date)
    if not candidate:
        return dict(ODDS_NEUTRAL)
    resolved: Dict[str, float] = {}
    for col in ODDS_FEATURE_COLUMNS:
        if col == "odds_missing":
            resolved[col] = 0.0
            continue
        try:
            value = float(candidate[col])
        except (KeyError, TypeError, ValueError):
            return dict(ODDS_NEUTRAL)
        if value != value:  # NaN
            return dict(ODDS_NEUTRAL)
        resolved[col] = value
    return resolved


def build_engineered_dataset(
    raw_matches: pd.DataFrame,
    odds_df: Optional[pd.DataFrame] = None,
    odds_mask_rate: float = 0.0,
) -> pd.DataFrame:
    """End-to-end dataset builder merging rolling stats and H2H features for modeling.

    ``odds_df`` (historical pre-kickoff odds frame) is left-joined on
    (date, home, away); unmatched rows are neutral-filled and flagged via
    ``odds_missing``. When None, every row is neutral (tests/offline).
    ``odds_mask_rate`` randomly neutralizes that fraction of TRAINING rows
    (seeded) so the model also learns the no-market regime served for
    fixtures without cached market odds; 0.0 disables (tests).
    """
    raw_matches = raw_matches.copy()
    raw_matches["_source_order"] = np.arange(len(raw_matches), dtype=int)
    raw_matches = _chronological(raw_matches, "match_id")
    raw_matches["match_id"] = raw_matches.index

    # 1. Transform and compute team rolling stats
    team_df = transform_matches_to_team_perspective(raw_matches)
    team_features = compute_team_rolling_features(team_df)

    # Separate home and away features
    home_feats = team_features[team_features["is_home"] == 1].copy()
    away_feats = team_features[team_features["is_home"] == 0].copy()

    # Prefix columns
    feat_cols = [
        c for c in home_feats.columns
        if c.startswith("roll_") or c.startswith("venue_roll_") or c.startswith("ewm_")
        or c in ("rest_days", "congestion_14d")
    ]

    home_rename = {c: f"home_{c}" for c in feat_cols}
    away_rename = {c: f"away_{c}" for c in feat_cols}

    home_subset = home_feats[["match_id"] + feat_cols].rename(columns=home_rename)
    away_subset = away_feats[["match_id"] + feat_cols].rename(columns=away_rename)

    # 2. Compute dynamic Elo ratings and momentum features
    home_elos, away_elos, elo_diffs, elo_mom, _, _ = compute_dynamic_elo(raw_matches)
    raw_matches["home_elo"] = home_elos
    raw_matches["away_elo"] = away_elos
    raw_matches["elo_diff"] = elo_diffs
    for k, vals in elo_mom.items():
        raw_matches[k] = vals

    # 3. Compute H2H features
    matches_with_h2h = compute_head_to_head_features(raw_matches)

    # 4. Merge together
    merged = matches_with_h2h.merge(home_subset, on="match_id", how="left")
    merged = merged.merge(away_subset, on="match_id", how="left")

    # Differential features (home - away advantage indicators)
    for w in WINDOWS:
        merged[f"diff_roll_goals_for_{w}"] = merged[f"home_roll_goals_for_{w}"] - merged[f"away_roll_goals_for_{w}"]
        merged[f"diff_roll_goals_against_{w}"] = merged[f"home_roll_goals_against_{w}"] - merged[f"away_roll_goals_against_{w}"]
        merged[f"diff_roll_points_{w}"] = merged[f"home_roll_points_{w}"] - merged[f"away_roll_points_{w}"]
        merged[f"diff_roll_shots_target_{w}"] = merged[f"home_roll_shots_target_for_{w}"] - merged[f"away_roll_shots_target_for_{w}"]
        merged[f"diff_roll_possession_{w}"] = merged[f"home_roll_possession_{w}"] - merged[f"away_roll_possession_{w}"]

    merged["diff_rest_days"] = merged["home_rest_days"] - merged["away_rest_days"]
    merged["diff_congestion_14d"] = merged["home_congestion_14d"] - merged["away_congestion_14d"]

    # Away-team travel distance from static stadium geography (zero leakage).
    merged["away_travel_km"] = [
        haversine_km(ht, at) for ht, at in zip(merged["home_team"], merged["away_team"])
    ]

    # 5. Pre-kickoff market signals, joined on (date, home, away).
    # Unmatched rows stay NaN here and are neutral-filled + flagged below.
    merged["_odds_date"] = pd.to_datetime(merged["date"]).dt.normalize()
    if odds_df is not None and not odds_df.empty:
        odds_part = odds_df[
            ["date", "home_team", "away_team",
             "odds_implied_home", "odds_implied_draw", "odds_implied_away",
             "odds_overround", "odds_move_home"]
        ].copy()
        odds_part["_odds_date"] = pd.to_datetime(odds_part["date"]).dt.normalize()
        odds_part = odds_part.drop(columns=["date"]).drop_duplicates(
            subset=["_odds_date", "home_team", "away_team"]
        )
        merged = merged.merge(
            odds_part, on=["_odds_date", "home_team", "away_team"], how="left"
        )
    else:
        for _col in ("odds_implied_home", "odds_implied_draw", "odds_implied_away",
                     "odds_overround", "odds_move_home"):
            merged[_col] = np.nan
    merged = merged.drop(columns=["_odds_date"])
    merged["odds_missing"] = merged["odds_implied_home"].isna().astype(float)
    merged["odds_implied_home"] = merged["odds_implied_home"].fillna(1.0 / 3.0)
    merged["odds_implied_draw"] = merged["odds_implied_draw"].fillna(1.0 / 3.0)
    merged["odds_implied_away"] = merged["odds_implied_away"].fillna(1.0 / 3.0)
    merged["odds_overround"] = merged["odds_overround"].fillna(0.0)
    merged["odds_move_home"] = merged["odds_move_home"].fillna(0.0)

    # Market dropout for the no-live-odds serving regime: neutralize a
    # seeded fraction of rows so the model learns odds_missing=1 inputs.
    # Coverage gates run on the unmasked join, so data drift still fails
    # loudly upstream of here.
    if odds_mask_rate and 0.0 < odds_mask_rate < 1.0:
        rng = np.random.RandomState(42)
        mask = rng.rand(len(merged)) < float(odds_mask_rate)
        merged.loc[mask, "odds_implied_home"] = 1.0 / 3.0
        merged.loc[mask, "odds_implied_draw"] = 1.0 / 3.0
        merged.loc[mask, "odds_implied_away"] = 1.0 / 3.0
        merged.loc[mask, "odds_overround"] = 0.0
        merged.loc[mask, "odds_move_home"] = 0.0
        merged.loc[mask, "odds_missing"] = 1.0

    # Target encodings:
    # result: H -> 2, D -> 1, A -> 0
    res_map = {"H": 2, "D": 1, "A": 0}
    merged["target_outcome"] = merged["result"].map(res_map)
    merged["target_home_goals"] = merged["home_goals"]
    merged["target_away_goals"] = merged["away_goals"]

    # Fill any initial missing rolling averages with fixed league priors.
    # Never use dataset medians here: they are computed from future rows and
    # would leak validation/test information into training features.
    feature_columns = get_feature_column_names()
    for col in feature_columns:
        if col in merged.columns and merged[col].isna().any():
            # Differential and momentum features are centered on zero: with no
            # history there is no home-away advantage, so fill 0.0 (never a
            # league level, which would fake a signal).
            if col.startswith("diff_") or "momentum" in col:
                merged[col] = merged[col].fillna(0.0)
                continue
            fallback = 0.0
            for key, val in LEAGUE_DEFAULTS.items():
                if col.endswith(key) or key in col:
                    fallback = val
                    break
            # Elo columns fall back to neutral ratings rather than zero.
            if col in ("home_elo", "away_elo"):
                fallback = 1600.0
            elif col == "elo_diff":
                from src.config import get_config

                fallback = float(get_config()["model"].get("home_advantage", 65.0))
            elif "momentum" in col:
                fallback = 0.0
            merged[col] = merged[col].fillna(fallback)

    return merged


def get_feature_column_names() -> List[str]:
    """Returns the ordered list of predictive feature column names."""
    cols: List[str] = []

    # Elo rating and momentum features
    cols.extend([
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_elo_momentum_3",
        "away_elo_momentum_3",
        "diff_elo_momentum_3",
        "home_elo_momentum_5",
        "away_elo_momentum_5",
        "diff_elo_momentum_5",
    ])

    # Home & Away rolling metrics
    for side in ["home", "away"]:
        cols.append(f"{side}_rest_days")
        cols.append(f"{side}_congestion_14d")
        for w in WINDOWS:
            for m in ["goals_for", "goals_against", "goal_diff", "shots_for", "shots_target_for", "possession", "points"]:
                cols.append(f"{side}_roll_{m}_{w}")
        for span in EWM_SPANS:
            for m in EWM_METRICS:
                cols.append(f"{side}_ewm_{m}_{span}")
        for m in ["goals_for", "goals_against", "points"]:
            cols.append(f"{side}_venue_roll_{m}_5")

    # Differential features
    for w in WINDOWS:
        cols.extend([
            f"diff_roll_goals_for_{w}",
            f"diff_roll_goals_against_{w}",
            f"diff_roll_points_{w}",
            f"diff_roll_shots_target_{w}",
            f"diff_roll_possession_{w}",
        ])
    cols.append("diff_rest_days")
    cols.append("diff_congestion_14d")

    # H2H features
    cols.extend(["h2h_home_win_rate", "h2h_goal_diff", "h2h_matches_count"])

    # Away-team travel distance (static geography, leakage-free)
    cols.append("away_travel_km")

    # Pre-kickoff bookmaker market signals (neutral-filled when unavailable)
    cols.extend(ODDS_FEATURE_COLUMNS)
    return cols


def build_feature_context(
    history_matches_df: pd.DataFrame,
    as_of_date: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Precomputes dynamic Elo state, rating histories, and perspective table for fast batch inference."""
    if as_of_date is not None:
        history = history_matches_df[history_matches_df["date"] < as_of_date].copy()
    else:
        history = history_matches_df.copy()
    history = _chronological(history, "match_id")
    _, _, _, _, current_ratings, rating_histories = compute_dynamic_elo(history)
    team_df = transform_matches_to_team_perspective(history)
    return {
        "history": history,
        "current_ratings": current_ratings,
        "rating_histories": rating_histories,
        "team_df": team_df,
        "as_of_date": as_of_date,
    }


def build_fixture_features(
    home_team: str,
    away_team: str,
    match_date: datetime,
    history_matches_df: pd.DataFrame,
    precomputed_context: Optional[Dict[str, Any]] = None,
    odds_row: Optional[Dict[str, float]] = None,
    odds_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Builds a single-row feature DataFrame for an upcoming match using past history.

    Used for real-time inference and upcoming 2026/2027 fixtures.
    Supports optional precomputed_context for sub-millisecond batch forecasting.
    Market features resolve from an explicit ``odds_row`` (live override)
    first, then ``odds_df`` (historical frame), else neutral (flagged).
    """
    feature_cols = get_feature_column_names()

    use_cache = False
    if precomputed_context is not None:
        ctx_history = precomputed_context.get("history")
        ctx_as_of = precomputed_context.get("as_of_date")
        if ctx_history is not None:
            # Cache is valid only when it contains no matches at/after match_date
            # and its as_of bound (if any) does not exceed match_date.
            no_future = bool((ctx_history["date"] < match_date).all()) if len(ctx_history) else True
            if no_future and (ctx_as_of is None or ctx_as_of <= match_date):
                use_cache = True

    if use_cache:
        assert precomputed_context is not None
        history = precomputed_context["history"]
        current_ratings = precomputed_context["current_ratings"]
        rating_histories = precomputed_context["rating_histories"]
        team_df = precomputed_context["team_df"]
    else:
        history = history_matches_df[history_matches_df["date"] < match_date].copy()
        # An empty history still gets team-specific priors below.  Returning a
        # zero-filled row here made the differential features meaningless for
        # true cold starts and diverged from the serving path's shrinkage.
        if history.empty:
            current_ratings = dict(BASE_ELO)
            rating_histories = {team: [] for team in BASE_ELO}
            team_df = pd.DataFrame(columns=["team", "date", "is_home"])
        else:
            # Compute current dynamic Elo and rating history up to match date.
            history = _chronological(history, "match_id")
            _, _, _, _, current_ratings, rating_histories = compute_dynamic_elo(history)
            team_df = transform_matches_to_team_perspective(history)

    h_elo = current_ratings.get(home_team, BASE_ELO.get(home_team, 1420.0))
    a_elo = current_ratings.get(away_team, BASE_ELO.get(away_team, 1420.0))

    h_hist = rating_histories.get(home_team, [])
    a_hist = rating_histories.get(away_team, [])
    h_m3 = (h_elo - h_hist[-3]) if len(h_hist) >= 3 else (h_elo - BASE_ELO.get(home_team, 1420.0))
    a_m3 = (a_elo - a_hist[-3]) if len(a_hist) >= 3 else (a_elo - BASE_ELO.get(away_team, 1420.0))
    h_m5 = (h_elo - h_hist[-5]) if len(h_hist) >= 5 else (h_elo - BASE_ELO.get(home_team, 1420.0))
    a_m5 = (a_elo - a_hist[-5]) if len(a_hist) >= 5 else (a_elo - BASE_ELO.get(away_team, 1420.0))

    def extract_latest_team_stats(team_name: str, is_home: int) -> Dict[str, float]:
        sub = team_df[team_df["team"] == team_name]
        if not sub.empty and "date" in sub.columns:
            sub = sub[sub["date"] < match_date]
        stats: Dict[str, float] = {}

        p_info = CLUB_POWER_INDEX.get(team_name, {
            "gf_baseline": 1.10,
            "ga_baseline": 1.65,
            "points_baseline": 1.0,
            "shots_baseline": 11.0,
            "target_baseline": 3.5,
            "poss_baseline": 45.0,
        })

        if sub.empty:
            stats["rest_days"] = 7.0
            stats["congestion_14d"] = 0
            for w in WINDOWS:
                stats[f"roll_goals_for_{w}"] = p_info["gf_baseline"]
                stats[f"roll_goals_against_{w}"] = p_info["ga_baseline"]
                stats[f"roll_goal_diff_{w}"] = p_info["gf_baseline"] - p_info["ga_baseline"]
                stats[f"roll_shots_for_{w}"] = p_info["shots_baseline"]
                stats[f"roll_shots_target_for_{w}"] = p_info["target_baseline"]
                stats[f"roll_possession_{w}"] = p_info["poss_baseline"]
                stats[f"roll_points_{w}"] = p_info["points_baseline"]
            for span in EWM_SPANS:
                stats[f"ewm_goals_for_{span}"] = p_info["gf_baseline"]
                stats[f"ewm_goals_against_{span}"] = p_info["ga_baseline"]
                stats[f"ewm_goal_diff_{span}"] = p_info["gf_baseline"] - p_info["ga_baseline"]
                stats[f"ewm_shots_for_{span}"] = p_info["shots_baseline"]
                stats[f"ewm_shots_target_for_{span}"] = p_info["target_baseline"]
                stats[f"ewm_points_{span}"] = p_info["points_baseline"]
            stats["venue_roll_goals_for_5"] = p_info["gf_baseline"]
            stats["venue_roll_goals_against_5"] = p_info["ga_baseline"]
            stats["venue_roll_points_5"] = p_info["points_baseline"]
            return stats

        last_date = sub["date"].max()
        rest = (match_date - last_date).total_seconds() / (24 * 3600)
        stats["rest_days"] = float(np.clip(rest, 1.0, 30.0))
        # Congestion mirrors training: prior team matches in [T-14d, T).
        window_start = match_date - pd.Timedelta(days=14)
        stats["congestion_14d"] = int(
            ((sub["date"] >= window_start) & (sub["date"] < match_date)).sum()
        )

        # NOTE (train/serve consistency): training rolling features in
        # compute_team_rolling_features use raw observed means, so serving
        # must use raw observed means too. No Bayesian shrinkage here: a
        # shrunk serving feature the model never saw during training is a
        # train/serve skew. Cold starts (no history) still use priors above.
        # Rolling overall
        for w in WINDOWS:
            recent_w = sub.tail(w)
            stats[f"roll_goals_for_{w}"] = float(recent_w["goals_for"].mean())
            stats[f"roll_goals_against_{w}"] = float(recent_w["goals_against"].mean())
            stats[f"roll_goal_diff_{w}"] = float(recent_w["goals_for"].mean() - recent_w["goals_against"].mean())
            stats[f"roll_shots_for_{w}"] = float(recent_w["shots_for"].mean())
            stats[f"roll_shots_target_for_{w}"] = float(recent_w["shots_target_for"].mean())
            stats[f"roll_possession_{w}"] = float(recent_w["possession"].mean())
            stats[f"roll_points_{w}"] = float(recent_w["points"].mean())
        for span in EWM_SPANS:
            for metric in EWM_METRICS:
                values = sub[metric].astype(float)
                stats[f"ewm_{metric}_{span}"] = float(
                    values.ewm(span=span, adjust=False, min_periods=1).mean().iloc[-1]
                )

        # Venue specific: raw venue means; fall back to overall rolling when
        # the side has no history at this venue (matches training, where a
        # missing venue average falls back to the overall rolling average).
        venue_sub = sub[sub["is_home"] == is_home].tail(5)
        if not venue_sub.empty:
            stats["venue_roll_goals_for_5"] = float(venue_sub["goals_for"].mean())
            stats["venue_roll_goals_against_5"] = float(venue_sub["goals_against"].mean())
            stats["venue_roll_points_5"] = float(venue_sub["points"].mean())
        else:
            stats["venue_roll_goals_for_5"] = stats["roll_goals_for_5"]
            stats["venue_roll_goals_against_5"] = stats["roll_goals_against_5"]
            stats["venue_roll_points_5"] = stats["roll_points_5"]

        return stats

    h_stats = extract_latest_team_stats(home_team, is_home=1)
    a_stats = extract_latest_team_stats(away_team, is_home=0)

    feature_dict: Dict[str, float] = {}

    # Elo and momentum features
    feature_dict["home_elo"] = h_elo
    feature_dict["away_elo"] = a_elo
    from src.config import get_config as _get_cfg2

    feature_dict["elo_diff"] = (h_elo + float(_get_cfg2()["model"].get("home_advantage", 65.0))) - a_elo
    feature_dict["home_elo_momentum_3"] = round(h_m3, 2)
    feature_dict["away_elo_momentum_3"] = round(a_m3, 2)
    feature_dict["diff_elo_momentum_3"] = round(h_m3 - a_m3, 2)
    feature_dict["home_elo_momentum_5"] = round(h_m5, 2)
    feature_dict["away_elo_momentum_5"] = round(a_m5, 2)
    feature_dict["diff_elo_momentum_5"] = round(h_m5 - a_m5, 2)

    for k, v in h_stats.items():
        feature_dict[f"home_{k}"] = v
    for k, v in a_stats.items():
        feature_dict[f"away_{k}"] = v

    # Differentials
    for w in WINDOWS:
        feature_dict[f"diff_roll_goals_for_{w}"] = feature_dict[f"home_roll_goals_for_{w}"] - feature_dict[f"away_roll_goals_for_{w}"]
        feature_dict[f"diff_roll_goals_against_{w}"] = feature_dict[f"home_roll_goals_against_{w}"] - feature_dict[f"away_roll_goals_against_{w}"]
        feature_dict[f"diff_roll_points_{w}"] = feature_dict[f"home_roll_points_{w}"] - feature_dict[f"away_roll_points_{w}"]
        feature_dict[f"diff_roll_shots_target_{w}"] = feature_dict[f"home_roll_shots_target_for_{w}"] - feature_dict[f"away_roll_shots_target_for_{w}"]
        feature_dict[f"diff_roll_possession_{w}"] = feature_dict[f"home_roll_possession_{w}"] - feature_dict[f"away_roll_possession_{w}"]
    feature_dict["diff_rest_days"] = feature_dict["home_rest_days"] - feature_dict["away_rest_days"]
    feature_dict["diff_congestion_14d"] = (
        feature_dict["home_congestion_14d"] - feature_dict["away_congestion_14d"]
    )
    feature_dict["away_travel_km"] = haversine_km(home_team, away_team)

    # H2H
    h2h_sub = history[
        (history["date"] < match_date)
        & (
            ((history["home_team"] == home_team) & (history["away_team"] == away_team))
            | ((history["home_team"] == away_team) & (history["away_team"] == home_team))
        )
    ].tail(5)

    if not h2h_sub.empty:
        h_wins = 0
        gd_sum = 0
        for _, m in h2h_sub.iterrows():
            if m["home_team"] == home_team:
                gd = m["home_goals"] - m["away_goals"]
                if gd > 0:
                    h_wins += 1
            else:
                gd = m["away_goals"] - m["home_goals"]
                if gd > 0:
                    h_wins += 1
            gd_sum += gd
        feature_dict["h2h_home_win_rate"] = h_wins / len(h2h_sub)
        feature_dict["h2h_goal_diff"] = gd_sum / len(h2h_sub)
        feature_dict["h2h_matches_count"] = len(h2h_sub)
    else:
        feature_dict["h2h_home_win_rate"] = 0.33
        feature_dict["h2h_goal_diff"] = 0.0
        feature_dict["h2h_matches_count"] = 0

    # Pre-kickoff market signals (live override -> historical frame -> neutral)
    feature_dict.update(resolve_odds_features(home_team, away_team, match_date, odds_row, odds_df))

    return pd.DataFrame([feature_dict])[feature_cols]
