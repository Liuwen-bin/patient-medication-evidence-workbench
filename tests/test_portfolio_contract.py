from __future__ import annotations

import json
from pathlib import Path


EXPECTED_SECTIONS = [
    "## 问题与用户",
    "## 三分钟看懂业务闭环",
    "## 架构与责任边界",
    "## 为什么是一个受控 Agent",
    "## Human-in-the-loop 与写回安全",
    "## 本地运行",
    "## 测试与评测",
    "## 结果：离线与在线",
    "## 演示路径",
    "## 关键取舍与失败案例",
    "## 医疗、数据和部署边界",
    "## 后续演进条件",
]


def test_readme_has_fixed_interview_story_order() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert text.startswith("# Patient Medication Evidence Workbench")
    positions = [text.index(section) for section in EXPECTED_SECTIONS]
    assert positions == sorted(positions)
    assert text.count("\n## ") == len(EXPECTED_SECTIONS)


def test_portfolio_contains_rendered_diagrams_and_business_screenshots() -> None:
    assets = Path("docs/portfolio/assets")
    for name in (
        "architecture.png",
        "state-machine.png",
        "01-ambiguous-product.png",
        "02-finding-evidence.png",
        "03-writeback-preview.png",
    ):
        path = assets / name
        assert path.is_file()
        assert path.stat().st_size > 1_000


def test_published_results_keep_offline_and_online_truth_separate() -> None:
    results = Path("docs/portfolio/results")
    offline = json.loads(
        (results / "offline-regression-summary.json").read_text(encoding="utf-8")
    )
    online = json.loads(
        (results / "online-integration-summary.json").read_text(encoding="utf-8")
    )

    assert offline["execution"]["mode"] == "offline_fixture"
    assert offline["execution"]["realModel"] is False
    assert offline["summary"]["caseCount"] == 15
    assert online["execution"]["mode"] == "online_integration"
    assert online["execution"]["realModel"] is True
    assert online["execution"]["realDatabases"] is True
    assert online["acceptancePassed"] is True
    assert online["acceptanceFailureCodes"] == []
    assert len(online["cases"]) == 5
    published = json.dumps(online, ensure_ascii=False).casefold()
    assert "http://" not in published
    assert "api_key" not in published
    assert "patientid" not in published
