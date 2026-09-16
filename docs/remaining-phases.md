# Delivery status and deferred research

This document is the current release-oriented status for the Premier League predictor. It replaces the earlier phase checklist, which contained historical season counts and no longer represented the dashboard or model contract.

## Current status

Phases 0 through 8 are implemented in the current release candidate. The production classifier is selected by Ranked Probability Score (RPS), not raw accuracy. The current held-out benchmark is sourced from `models/metrics.json`:

| Model | RPS | Accuracy | Goal MAE | Status |
| :--- | ---: | ---: | ---: | :--- |
| Random Forest | 0.2048 | 48.3% | 0.91 | Benchmark |
| XGBoost | 0.2051 | 48.7% | 0.92 | Benchmark |
| Stacked | 0.2044 | 48.9% | 0.89 | Production |

These are time-ordered validation results, not a guarantee of future match accuracy. The dashboard distinguishes recorded scores from projected fixtures; projected fixtures must never be presented as observed results.

## Completed phases

### Phase 0 — Data foundation

Historical results, the published fixture schedule, cached football-data.co.uk odds, team aliases, and kickoff-time fallbacks are represented in reproducible loaders. Missing kickoff times remain `TBC`; the pipeline does not invent a kickoff time.

### Phase 1 — Leakage-safe features

Rolling form, venue splits, head-to-head history, rest and congestion measures, travel distance, Elo ratings, and market signals are built chronologically. Fixture features filter history to dates strictly earlier than the fixture date, and odds validation rejects result columns and insufficient joins.

### Phase 2 — Model training

Random Forest, XGBoost, multinomial logistic regression, and Elo-Poisson members are trained as separate classification and goal-regression components. Tree tuning is deterministic and recorded in `models/tuning.json`.

### Phase 3 — Honest evaluation and selection

Time-series validation separates training, calibration, and evaluation data. The stacked meta-learner uses out-of-fold probabilities, calibration is selected on a disjoint slice, and production selection uses RPS with documented tie-breakers.

### Phase 4 — Forecast outputs

The pipeline writes the forecast CSV and Markdown report, persists compatible model metadata, and uses the shared draw-decision rule for probabilities, forecasts, and scorelines.

### Phase 5 — Serving and interfaces

The FastAPI surface provides health, readiness, model metadata, dataset, prediction, and gameweek endpoints. CLI commands validate gameweeks and club names, return non-zero status for invalid input, and support cached checkpoints or explicit retraining.

### Phase 6 — Dashboard

The React dashboard provides Fixtures, Simulator, Table, Clubs, and Analytics views from the serialized dataset. Official/projected labels, loading and retry states, accessible tables, keyboard-dismissible diagnostic previews, and responsive mobile layouts are part of the UI contract.

### Phase 7 — Diagnostics and publication

Diagnostic charts are generated with the dashboard's light theme and copied to `web/public/visuals/`. The dashboard data export and generated forecast artifacts are kept synchronized through the pipeline/export workflow.

### Phase 8 — Quality gates

The Python test suite covers leakage, odds contracts, model compatibility, calibration, scoreline consistency, tuning, interfaces, and API behavior. The web gate runs TypeScript checking and a production Vite build. CI runs the supported Python matrix and frontend build.

## Release checklist

Before publishing a refreshed forecast or model:

1. Run the offline pipeline when the cached source data is sufficient.
2. Regenerate the web dataset with `.venv\Scripts\python.exe export_web_data.py`.
3. Confirm generated CSV, Markdown, JSON, metrics, and PNG artifacts are synchronized. Validate forecast labels with the shared draw-decision rule rather than assuming the raw probability argmax is always the published outcome.
4. Run `.venv\Scripts\python.exe -m pytest tests/ -v` from the repository root.
5. Run `npm run build` from `web/`.
6. Inspect the dashboard at desktop and narrow phone widths, including Analytics diagnostics and tables.
7. Review the staged file list before committing; keep each commit scoped to one coherent change.

## Deferred research

The following are intentionally deferred research items, not release blockers:

- Player availability, injuries, suspensions, lineups, and verified squad news. These require timestamped sources and a leakage-safe availability snapshot.
- Additional historical seasons and richer competition context. Any extension must preserve chronological splits and re-run calibration and RPS selection.
- A live odds integration. This would require an explicit provider, licensing review, caching policy, failure behavior, and a new pre-kickoff leakage contract.

Any deferred feature must ship with its data provenance, time-of-availability definition, validation coverage, and dashboard/API documentation before it becomes part of the production model.
