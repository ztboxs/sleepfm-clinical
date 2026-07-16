from pathlib import Path

from api.routers.documentation import PUBLISHED_DOCS


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def read(name: str) -> str:
    return (DOCS / f"{name}.md").read_text(encoding="utf-8")


def test_all_published_documents_exist_with_titles() -> None:
    for name in PUBLISHED_DOCS:
        content = read(name)
        assert content.startswith("# ")


def test_usage_document_covers_required_workflow() -> None:
    content = read("使用文档")
    for token in (
        "/api/v1/health",
        "/api/v1/preprocess",
        "/api/v1/embed",
        "/api/v1/sleep_staging",
        "/api/v1/disease_prediction",
        "/api/v1/predict",
        "/api/v1/download/{file_id}",
        "preprocessed_file_id",
        "embedding_file_id",
        "age / 100",
        "0 = 女性",
        "1 = 男性",
        "24 小时",
        "busy",
    ):
        assert token in content


def test_algorithm_and_tuning_documents_cover_client_topics() -> None:
    algorithm = read("模型算法")
    for token in ("128 Hz", "5 秒", "LOO-CL", "LSTM", "CoxPH"):
        assert token in algorithm

    tuning = read("训练与调优")
    for token in ("pretrain.py", "generate_embeddings.py", "验收指标", "调优请求"):
        assert token in tuning


def test_api_and_test_guides_use_current_async_contracts() -> None:
    api = read("接口文档")
    testing = read("测试指南")
    for content in (api, testing):
        assert "/api/v1/subtask_result/{task_id}" in content
        assert '"status": "accepted"' in content
        assert "/api/v1/task_result" in content


def test_deployment_document_covers_documentation_center() -> None:
    deployment = read("部署文档")
    for token in (
        "/help",
        "/api/v1/docs",
        "api/routers/documentation.py",
        "api/static/docs.html",
        "帮助",
        "使用文档",
    ):
        assert token in deployment
