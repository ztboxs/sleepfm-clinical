from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import documentation


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for name in documentation.PUBLISHED_DOCS:
        (tmp_path / f"{name}.md").write_text(
            f"# {name}标题\n\n{name}正文\n", encoding="utf-8"
        )
    (tmp_path / "任务进度.md").write_text("# 内部进度\n", encoding="utf-8")
    monkeypatch.setattr(documentation, "DOCS_DIR", str(tmp_path))

    app = FastAPI()
    app.include_router(documentation.router)
    return TestClient(app)


def test_list_documents_uses_published_order(client: TestClient) -> None:
    response = client.get("/api/v1/docs")
    assert response.status_code == 200
    documents = response.json()["documents"]
    assert [item["name"] for item in documents] == list(documentation.PUBLISHED_DOCS)
    assert documents[0]["title"] == "使用文档标题"
    assert documents[0]["url"].startswith("/api/v1/docs/")


def test_get_document_returns_markdown(client: TestClient) -> None:
    response = client.get("/api/v1/docs/模型算法")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# 模型算法标题")


def test_unknown_or_internal_document_is_not_published(client: TestClient) -> None:
    assert client.get("/api/v1/docs/不存在").status_code == 404
    assert client.get("/api/v1/docs/任务进度").status_code == 404
