import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_benchmark_matches_authoritative_model_metrics():
    metrics = json.loads((REPOSITORY_ROOT / "models/metrics.json").read_text(encoding="utf-8"))
    dashboard = json.loads(
        (REPOSITORY_ROOT / "web/src/data/eplData.json").read_text(encoding="utf-8")
    )

    benchmark = dashboard["benchmark"]
    assert benchmark["productionModel"] == metrics["production_model"]
    assert len(benchmark["models"]) == len(metrics["models"])

    for model in benchmark["models"]:
        source = metrics["models"][model["name"]]
        assert model["accuracy"] == round(source["accuracy"] * 100, 1)
        assert model["rps"] == round(source["rps"], 4)
        assert model["avgGoalMae"] == round(source["avg_goal_mae"], 2)
