from fastapi.testclient import TestClient


def test_root_serves_pharmacist_workbench(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "用药复核工作台" in response.text
    assert 'id="patient-panel"' in response.text
    assert 'id="review-queue"' in response.text
    assert 'id="evidence-panel"' in response.text


def test_static_assets_are_local(client: TestClient) -> None:
    response = client.get("/assets/styles.css")

    assert response.status_code == 200
    assert "workbench-grid" in response.text
