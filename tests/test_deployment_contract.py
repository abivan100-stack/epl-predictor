import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_vercel_configuration_builds_the_web_dashboard_from_repository_root():
    config = json.loads((REPOSITORY_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert config["framework"] == "vite"
    assert config["installCommand"] == "npm --prefix web ci"
    assert config["buildCommand"] == "npm --prefix web run build"
    assert config["outputDirectory"] == "web/dist"


def test_vercel_configuration_serves_direct_dashboard_routes_through_the_spa_entrypoint():
    config = json.loads((REPOSITORY_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert {
        "source": "/((?!assets/|visuals/|sw\\.js$).*)",
        "destination": "/index.html",
    } in config["rewrites"]


def test_vercel_configuration_does_not_publish_backend_functions_accidentally():
    config = json.loads((REPOSITORY_ROOT / "vercel.json").read_text(encoding="utf-8"))

    assert "functions" not in config
    assert "api" not in config


def test_vercel_configuration_caches_hashed_assets_and_refreshes_the_service_worker():
    config = json.loads((REPOSITORY_ROOT / "vercel.json").read_text(encoding="utf-8"))
    headers = {
        entry["source"]: {
            header["key"]: header["value"] for header in entry["headers"]
        }
        for entry in config["headers"]
    }

    assert headers["/assets/(.*)"]["Cache-Control"].endswith("immutable")
    assert headers["/sw.js"]["Cache-Control"] == "no-cache, no-store, must-revalidate"
