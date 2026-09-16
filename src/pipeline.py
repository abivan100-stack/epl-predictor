"""Main orchestration pipeline for data loading, training, benchmarking, and forecasting.

Provides end-to-end execution and exports predictions for the 2026/2027 Premier League season.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.data_loader import (
    load_2026_2027_fixtures,
    load_historical_stats,
)
from src.evaluate import (
    plot_confusion_matrices,
    plot_feature_importance,
    plot_goal_error_distribution,
    plot_metrics_comparison,
    plot_reliability_curves,
    plot_rps_comparison,
)
from src.feature_engineering import (
    build_engineered_dataset,
    build_feature_context,
    build_fixture_features,
    get_feature_column_names,
)
from src.models import (
    OUTCOME_CODES,
    OUTCOME_NAMES,
    MatchPredictorModel,
    favor_outcome_from_proba,
    split_calibration_evaluation,
    train_and_benchmark_models,
)
from src.logging_config import get_logger

logger = get_logger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "production_model.joblib")


def _round_probabilities(probabilities: List[float]) -> List[float]:
    """Round percentages to one decimal place while preserving a 100% total."""
    raw = np.asarray(probabilities, dtype=float) * 1000.0
    units = np.floor(raw + 1e-9).astype(int)
    remaining = int(1000 - units.sum())
    remainders = raw - units
    order = np.argsort(-remainders, kind="stable")
    for idx in order[:max(0, remaining)]:
        units[int(idx)] += 1
    if remaining < 0:
        for idx in np.argsort(remainders, kind="stable")[:abs(remaining)]:
            units[int(idx)] -= 1
    return [round(int(value) / 10.0, 1) for value in units]


class PremierLeaguePredictionPipeline:
    """End-to-end management pipeline for training, evaluation, and inference."""

    def __init__(self):
        self.raw_historical: Optional[pd.DataFrame] = None
        self.engineered_df: Optional[pd.DataFrame] = None
        self.fixtures_2026_2027: Optional[pd.DataFrame] = None
        self.odds_df: Optional[pd.DataFrame] = None
        self.feature_cols: List[str] = get_feature_column_names()
        self.models: Dict[str, MatchPredictorModel] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.best_model_name: str = "XGBoost"
        self.best_model: Optional[MatchPredictorModel] = None

    def load_data(
        self, force_download: bool = False, offline: bool | None = None
    ) -> PremierLeaguePredictionPipeline:
        """Loads historical match data and upcoming fixtures without running full dataset engineering."""
        if offline is None:
            offline = os.environ.get("EPL_OFFLINE", "").lower() in ("1", "true", "yes")
        if self.raw_historical is None:
            self.raw_historical = load_historical_stats(force_download=force_download, offline=offline)
        if self.fixtures_2026_2027 is None:
            self.fixtures_2026_2027 = load_2026_2027_fixtures(offline=offline)
        return self

    def prepare_data(
        self, force_download: bool = False, offline: bool | None = None
    ) -> PremierLeaguePredictionPipeline:
        """Loads historical stats and 2026/27 fixtures and performs feature engineering."""
        logger.info("[1/5] Loading historical match data and 2026/2027 fixtures...")
        self.load_data(force_download=force_download, offline=offline)

        logger.info(f"      - Loaded {len(self.raw_historical)} historical matches across 6 seasons.")
        logger.info(f"      - Loaded {len(self.fixtures_2026_2027)} matches for 2026/2027 Premier League.")

        logger.info("[2/5] Engineering rolling form, venue splits, and head-to-head metrics...")
        from src.odds_loader import load_odds_frame
        from src.validation import assert_odds_coverage, assert_odds_frame_clean

        logger.info("      - Loading pre-kickoff bookmaker odds...")
        self.odds_df = load_odds_frame(force_download=force_download, offline=offline)
        assert_odds_frame_clean(self.odds_df)
        coverage = assert_odds_coverage(self.raw_historical, self.odds_df)
        logger.info(f"      - Odds coverage on historical rows: {coverage:.3f}.")
        from src.config import get_config as _get_cfg

        mask_rate = float(_get_cfg()["model"].get("odds_mask_rate", 0.15))
        self.engineered_df = build_engineered_dataset(
            self.raw_historical, odds_df=self.odds_df, odds_mask_rate=mask_rate
        )
        logger.info(f"      - Engineered dataset shape: {self.engineered_df.shape} ({len(self.feature_cols)} features).")
        return self

    def train_and_evaluate(self, split_date_str: Optional[str] = None) -> Dict[str, Any]:
        """Performs time-series cross-validation split, fits RF & XGBoost, and generates plots."""
        from src.config import get_split_date

        if split_date_str is None:
            split_date_str = get_split_date()
        if self.engineered_df is None:
            self.prepare_data()

        logger.info(f"[3/5] Benchmarking Random Forest vs XGBoost with time-series split (cutoff: {split_date_str})...")
        split_date = pd.to_datetime(split_date_str)
        train_mask = self.engineered_df["date"] < split_date
        val_mask = self.engineered_df["date"] >= split_date

        train_df = self.engineered_df[train_mask].copy()
        val_df = self.engineered_df[val_mask].copy()

        logger.info(f"      - Training matches: {len(train_df)} | Validation matches: {len(val_df)}")

        self.models, self.metrics = train_and_benchmark_models(train_df, val_df, self.feature_cols)

        # Best model: primary key is Ranked Probability Score (lower),
        # tie-broken by log-loss (lower), accuracy (higher), goal MAE
        # (lower). RPS is the proper scoring rule for ordered Home/Draw/
        # Away outcomes: near-misses outrank far misses.
        def _rank(m: Dict[str, Any]) -> tuple:
            return (m["rps"], m["log_loss"], -m["accuracy"], m["avg_goal_mae"])

        self.best_model_name = min(self.metrics, key=lambda name: _rank(self.metrics[name]))
        self.best_model = self.models[self.best_model_name]

        for _name, _m in self.metrics.items():
            logger.info(
                f"      - {_name:13s} Accuracy: {_m['accuracy']:.3f} | "
                f"RPS: {_m['rps']:.4f} | LogLoss: {_m['log_loss']:.3f} | "
                f"Goal MAE: {_m['avg_goal_mae']:.3f}"
            )
        logger.info(f"      -> Best Performing Model Selected: {self.best_model_name} (lowest RPS)")

        # Generate Matplotlib visualizations
        logger.info("[4/5] Generating Matplotlib diagnostic visualization suite...")
        # The benchmark metrics use the held-out tail after the calibration
        # slice.  Reuse those exact labels for diagnostic plots.
        y_val_outcome = self.metrics[self.best_model_name]["eval_y_outcome"]
        y_val_hg = self.metrics[self.best_model_name]["eval_y_home_goals"]
        y_val_ag = self.metrics[self.best_model_name]["eval_y_away_goals"]

        plot_feature_importance(
            self.metrics["Random Forest"]["feature_importances"],
            self.metrics["XGBoost"]["feature_importances"],
            stacked_importances=self.metrics["Stacked"]["feature_importances"],
        )
        plot_confusion_matrices(
            y_val_outcome,
            self.metrics["Random Forest"]["val_preds"],
            self.metrics["XGBoost"]["val_preds"],
            stacked_preds=self.metrics["Stacked"]["val_preds"],
        )
        plot_metrics_comparison(self.metrics)
        plot_reliability_curves(
            y_val_outcome,
            {name: vals["val_proba"] for name, vals in self.metrics.items()},
        )
        plot_rps_comparison(self.metrics)
        plot_goal_error_distribution(
            y_val_hg,
            y_val_ag,
            self.metrics[self.best_model_name]["eval_pred_scores"],
        )
        logger.info("      - Diagnostic charts saved to 'visuals/' directory.")

        # Re-train best model on 100% of historical data for maximum forecasting accuracy
        logger.info("      - Refitting best model on full historical dataset for upcoming forecasts...")
        from src.config import get_config as _get_cfg_refit
        from src.feature_engineering import recency_weights as _recency_weights_refit

        _refit_weights = _recency_weights_refit(
            self.engineered_df["date"],
            half_life_days=float(_get_cfg_refit()["model"].get("recency_half_life_days", 730)),
        )
        self.best_model.fit(
            self.engineered_df[self.feature_cols],
            self.engineered_df["target_outcome"],
            self.engineered_df["target_home_goals"],
            self.engineered_df["target_away_goals"],
            sample_weight=_refit_weights,
        )
        # Refit invalidates any prefit sigmoid/isotonic mapping. Rebuild the
        # selected calibration strategy against the refit classifier while
        # keeping the evaluation tail untouched.
        calibration_slice, _ = split_calibration_evaluation(len(val_df))
        if calibration_slice.stop:
            self.best_model.calibrate_goals(
                val_df[self.feature_cols].iloc[calibration_slice],
                val_df["target_home_goals"].iloc[calibration_slice],
                val_df["target_away_goals"].iloc[calibration_slice],
            )
            self.best_model.calibrate_market_blend(
                val_df[self.feature_cols].iloc[calibration_slice],
                val_df["target_outcome"].iloc[calibration_slice],
            )
            self.best_model.calibrate_outcome(
                val_df[self.feature_cols].iloc[calibration_slice],
                val_df["target_outcome"].iloc[calibration_slice],
            )
        self.save_model(DEFAULT_MODEL_PATH)

        return self.metrics

    def save_model(self, filepath: Optional[str] = None) -> str:
        """Saves the current fitted production model checkpoint to disk."""
        if self.best_model is None:
            raise ValueError("No fitted production model available to save.")
        import json as _json

        path = filepath or DEFAULT_MODEL_PATH
        self.best_model.save(path)
        logger.info(f"      - Model checkpoint saved to: {path}")
        # Versioned metrics sidecar (committed): proves which benchmark the
        # checkpoint corresponds to without committing the large .joblib.
        try:
            sidecar = os.path.join(os.path.dirname(path), "metrics.json")
            payload = {
                "production_model": self.best_model_name,
                "calibration_temperature": getattr(self.best_model, "calibration_temperature", 1.0),
                "calibration_method": getattr(self.best_model, "calibration_method", "temperature"),
                "home_goal_correction": getattr(self.best_model, "home_goal_correction", 1.0),
                "away_goal_correction": getattr(self.best_model, "away_goal_correction", 1.0),
                "market_blend_weight": getattr(self.best_model, "market_blend_weight", 0.0),
                "no_market_blend_weight": getattr(self.best_model, "no_market_blend_weight", 0.0),
                "feature_count": len(self.feature_cols),
                "models": {
                    name: {
                        k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                        for k, v in vals.items()
                        if k in ("accuracy", "log_loss", "macro_f1", "rps",
                                 "mae_home_goals",
                                 "mae_away_goals", "avg_goal_mae", "exact_score_acc",
                                 "within_1_goal_acc", "calibration_temperature",
                                 "calibration_method",
                                 "home_goal_correction", "away_goal_correction")
                    }
                    for name, vals in (self.metrics or {}).items()
                },
            }
            os.makedirs(os.path.dirname(os.path.abspath(sidecar)), exist_ok=True)
            with open(sidecar, "w", encoding="utf-8") as f:
                _json.dump(payload, f, indent=2)
            logger.info(f"      - Metrics sidecar saved to: {sidecar}")
        except Exception as exc:
            logger.warning(f"Could not write metrics sidecar: {exc}")
        return path

    def load_model(self, filepath: Optional[str] = None) -> MatchPredictorModel:
        """Loads a production model checkpoint from disk."""
        path = filepath or DEFAULT_MODEL_PATH
        self.best_model = MatchPredictorModel.load(path)
        model_type = getattr(self.best_model, "model_type", "xgboost")
        self.best_model_name = {"rf": "Random Forest", "xgboost": "XGBoost"}.get(
            model_type, "Stacked"
        )
        from src.validation import assert_model_compatible
        assert_model_compatible(self.best_model, strict=True)
        self.feature_cols = self.best_model.feature_names
        return self.best_model

    def resolve_serving_odds(
        self, home_team: str, away_team: str, match_date: datetime
    ) -> Optional[Dict[str, float]]:
        """Resolves market features for one upcoming fixture.

        Uses the cached historical pre-kickoff row when available, otherwise
        returns None (neutral-filled downstream and flagged via odds_missing).
        Rows with non-finite legs are skipped as if missing.
        """
        from src.odds_loader import lookup_odds

        def _finite(row: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
            if not row:
                return None
            legs = (row.get("odds_implied_home"), row.get("odds_implied_draw"),
                    row.get("odds_implied_away"))
            try:
                if all(float(v) == float(v) for v in legs):
                    return row
            except (TypeError, ValueError):
                pass
            return None

        if self.odds_df is not None:
            return _finite(lookup_odds(self.odds_df, home_team, away_team, match_date))
        return None

    def forecast_2026_2027_season(self) -> pd.DataFrame:
        """Forecasts all 380 fixtures for the 2026/2027 season and exports results."""
        if self.best_model is None:
            self.train_and_evaluate()

        logger.info("[5/5] Generating match outcome probabilities and scoreline forecasts for 2026/2027...")
        if self.raw_historical is None or self.raw_historical.empty:
            raise ValueError("Historical data is empty; call prepare_data() before forecasting.")
        if self.fixtures_2026_2027 is None or self.fixtures_2026_2027.empty:
            raise ValueError("2026/2027 fixtures are empty; call prepare_data() before forecasting.")
        fixtures = self.fixtures_2026_2027.copy().sort_values(by=["date", "gameweek"]).reset_index(drop=True)
        rolling_history = self.raw_historical.copy().sort_values(by="date").reset_index(drop=True)
        # Historical means for unobserved in-play stats of played 2026/27
        # matches (shots/corners/possession are not in openfootball feeds).
        hist_means = {
            c: float(rolling_history[c].mean())
            for c in ["home_shots", "away_shots", "home_shots_target", "away_shots_target",
                      "home_corners", "away_corners", "home_possession", "away_possession"]
            if c in rolling_history.columns
        }
        feature_context = build_feature_context(rolling_history)
        context_history_len = len(rolling_history)
        if self.odds_df is None:
            try:
                from src.odds_loader import load_odds_frame

                self.odds_df = load_odds_frame(offline=True)
            except FileNotFoundError:
                logger.warning("Historical odds cache missing; serving rows will be odds-neutral.")
                self.odds_df = None
        predictions: List[Dict[str, Any]] = []

        for _, fix in fixtures.iterrows():
            m_date = fix["date"]
            ht = fix["home_team"]
            at = fix["away_team"]
            gw = fix["gameweek"]
            raw_time = fix.get("time", "")
            m_time = raw_time.strip() if isinstance(raw_time, str) and raw_time.strip() else "TBC"

            # Feature vector using precomputed context for sub-millisecond extraction
            X_match = build_fixture_features(
                ht, at, m_date, rolling_history,
                precomputed_context=feature_context,
                odds_row=self.resolve_serving_odds(ht, at, m_date),
                odds_df=self.odds_df,
            )

            probas = self.best_model.predict_outcome_proba(X_match)[0]  # [p_away, p_draw, p_home]
            p_away = float(probas[0])
            p_draw = float(probas[1])
            p_home = float(probas[2])

            pred_scores = self.best_model.predict_scoreline(X_match)[0]
            pred_hg, pred_ag = pred_scores

            # Favorite outcome uses the shared draw-aware rule so it always
            # agrees with home/draw/away probs (scoreline is already
            # constrained to that outcome in predict_scoreline).
            fav_idx = favor_outcome_from_proba(
                probas, odds_missing=float(X_match["odds_missing"].iloc[0])
            )
            fav_outcome = "Away Win" if fav_idx == 0 else ("Draw" if fav_idx == 1 else "Home Win")

            rounded_probs = _round_probabilities([p_home, p_draw, p_away])
            pred_item = {
                "gameweek": gw,
                "date": m_date.strftime("%Y-%m-%d") if isinstance(m_date, datetime) or hasattr(m_date, "strftime") else str(m_date),
                "time": m_time,
                "home_team": ht,
                "away_team": at,
                "home_win_prob": rounded_probs[0],
                "draw_prob": rounded_probs[1],
                "away_win_prob": rounded_probs[2],
                "predicted_outcome": fav_outcome,
                "predicted_score": f"{pred_hg} - {pred_ag}",
                "pred_home_goals": pred_hg,
                "pred_away_goals": pred_ag,
            }

            # Check if match was already played in 2026/27 (early weeks)
            if fix.get("status") == "played" and pd.notna(fix.get("home_goals")):
                act_hg = int(fix["home_goals"])
                act_ag = int(fix["away_goals"])
                act_res = "H" if act_hg > act_ag else ("A" if act_hg < act_ag else "D")
                pred_item["actual_score"] = f"{act_hg} - {act_ag}"
                pred_item["status"] = "Played"

                # Append to rolling history and refresh context so subsequent gameweeks reflect real results.
                # Shots/corners/possession are unobserved in openfootball feeds,
                # so use historical means (not hardcoded constants) to avoid
                # diluting form signals with league-average placeholders.
                new_row = {
                    "season": "2026-27",
                    "date": m_date,
                    "home_team": ht,
                    "away_team": at,
                    "home_goals": act_hg,
                    "away_goals": act_ag,
                    "result": act_res,
                    "home_shots": hist_means.get("home_shots", 12.0),
                    "away_shots": hist_means.get("away_shots", 10.0),
                    "home_shots_target": hist_means.get("home_shots_target", 4.0),
                    "away_shots_target": hist_means.get("away_shots_target", 3.0),
                    "home_corners": hist_means.get("home_corners", 5.0),
                    "away_corners": hist_means.get("away_corners", 4.0),
                    "home_possession": hist_means.get("home_possession", 50.0),
                    "away_possession": hist_means.get("away_possession", 50.0),
                }
                rolling_history = pd.concat([rolling_history, pd.DataFrame([new_row])], ignore_index=True)
                # Rebuild only when history actually grew; upcoming fixtures
                # reuse the cached Elo/team state via build_fixture_features.
                if len(rolling_history) != context_history_len:
                    feature_context = build_feature_context(rolling_history)
                    context_history_len = len(rolling_history)
            else:

                pred_item["actual_score"] = "-"
                pred_item["status"] = "Upcoming"

            predictions.append(pred_item)

        pred_df = pd.DataFrame(predictions)

        # Export CSV
        os.makedirs(DATA_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, "predictions_2026_2027.csv")
        pred_df.to_csv(csv_path, index=False)
        logger.info(f"      - Exported CSV to: {csv_path}")

        # Export Markdown
        md_path = os.path.join(DATA_DIR, "predictions_2026_2027.md")
        self._export_markdown_report(pred_df, md_path)
        logger.info(f"      - Exported Markdown summary to: {md_path}")

        return pred_df

    def _export_markdown_report(self, df: pd.DataFrame, output_path: str):
        """Creates a formatted markdown document summarizing the predictions."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Premier League 2026/2027 Season Match Predictions\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Primary Prediction Engine: **{self.best_model_name}**\n\n")

            f.write("## Model Performance Benchmark Summary\n\n")
            bench_data = []
            for m_name, m_vals in self.metrics.items():
                bench_data.append([
                    m_name,
                    f"{m_vals['accuracy']*100:.1f}%",
                    f"{m_vals['macro_f1']:.3f}",
                    f"{m_vals['log_loss']:.3f}",
                    f"{m_vals['avg_goal_mae']:.2f}",
                    f"{m_vals['within_1_goal_acc']*100:.1f}%",
                ])
            headers = ["Model", "Accuracy", "Macro F1", "Log Loss", "Goal MAE", "Within 1 Goal Acc"]
            f.write(tabulate(bench_data, headers=headers, tablefmt="github"))
            f.write("\n\n---\n\n")

            f.write("## Season Fixtures and Forecasts\n\n")
            for gw, gw_matches in df.groupby("gameweek"):
                f.write(f"### Gameweek {gw}\n\n")
                tbl_rows = []
                for _, r in gw_matches.iterrows():
                    tbl_rows.append([
                        r["date"],
                        r["home_team"],
                        r["predicted_score"],
                        r["away_team"],
                        f"{r['home_win_prob']}%",
                        f"{r['draw_prob']}%",
                        f"{r['away_win_prob']}%",
                        r["predicted_outcome"],
                        r["status"],
                    ])
                gw_headers = ["Date", "Home", "Pred Score", "Away", "Home %", "Draw %", "Away %", "Fav Outcome", "Status"]
                f.write(tabulate(tbl_rows, headers=gw_headers, tablefmt="github"))
                f.write("\n\n")

    def predict_custom_match(self, home_team: str, away_team: str, match_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Instant prediction for an arbitrary matchup between two clubs."""
        if self.best_model is None:
            self.train_and_evaluate()
        if self.raw_historical is None or self.raw_historical.empty:
            # Fast-load path (load_model only) may skip history; load it now.
            self.load_data()
        if self.odds_df is None:
            try:
                from src.odds_loader import load_odds_frame

                self.odds_df = load_odds_frame(offline=True)
            except FileNotFoundError:
                self.odds_df = None
        from src.validation import assert_model_compatible, canonical_team

        assert_model_compatible(self.best_model, strict=True)
        ht_std = canonical_team(home_team)
        at_std = canonical_team(away_team)
        if ht_std == at_std:
            raise ValueError("Home and away clubs must differ.")
        m_date = match_date or datetime.now()

        market = self.resolve_serving_odds(ht_std, at_std, m_date)
        X_match = build_fixture_features(
            ht_std, at_std, m_date, self.raw_historical,
            odds_row=market, odds_df=self.odds_df,
        )
        probas = self.best_model.predict_outcome_proba(X_match)[0]
        pred_scores = self.best_model.predict_scoreline(X_match)[0]
        exp_hg, exp_ag = self.best_model.predict_expected_goals(X_match)

        p_away, p_draw, p_home = probas[0], probas[1], probas[2]
        fav_idx = favor_outcome_from_proba(
            probas, odds_missing=float(X_match["odds_missing"].iloc[0])
        )
        fav_outcome = "Away Win" if fav_idx == 0 else ("Draw" if fav_idx == 1 else "Home Win")

        rounded_probs = _round_probabilities([float(p_home), float(p_draw), float(p_away)])
        return {
            "home_team": ht_std,
            "away_team": at_std,
            "match_date": m_date.strftime("%Y-%m-%d"),
            "predicted_score": f"{pred_scores[0]} - {pred_scores[1]}",
            "pred_home_goals": pred_scores[0],
            "pred_away_goals": pred_scores[1],
            "expected_home_goals": round(float(exp_hg[0]), 2),
            "expected_away_goals": round(float(exp_ag[0]), 2),
            "home_win_prob": rounded_probs[0],
            "draw_prob": rounded_probs[1],
            "away_win_prob": rounded_probs[2],
            "predicted_outcome": fav_outcome,
            "model_used": self.best_model_name,
            "market_home_prob": round(100 * market["odds_implied_home"], 1) if market else None,
            "market_draw_prob": round(100 * market["odds_implied_draw"], 1) if market else None,
            "market_away_prob": round(100 * market["odds_implied_away"], 1) if market else None,
            "market_missing": market is None,
        }
