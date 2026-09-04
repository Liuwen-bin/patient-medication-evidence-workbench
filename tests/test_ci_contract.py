from __future__ import annotations

import json
from pathlib import Path

import yaml


def test_ci_keeps_async_and_playwright_suites_in_separate_jobs() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    core_commands = json.dumps(jobs["core"], ensure_ascii=False)
    browser_commands = json.dumps(jobs["browser"], ensure_ascii=False)
    assert "test_web_render.py" not in core_commands
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in core_commands
    assert "test_web_render.py" in browser_commands
    assert "playwright install" in browser_commands


def test_live_integration_is_manual_only() -> None:
    workflow = yaml.safe_load(
        Path(".github/workflows/live-integration.yml").read_text(encoding="utf-8")
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
