import os
from urllib.parse import quote

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from api.config import DOCS_DIR


router = APIRouter(prefix="/api/v1", tags=["documentation"])

PUBLISHED_DOCS = (
    "使用文档",
    "模型算法",
    "训练与调优",
    "接口文档",
    "部署文档",
    "测试指南",
)


def _document_path(name: str) -> str:
    if name not in PUBLISHED_DOCS:
        raise HTTPException(status_code=404, detail="文档不存在")
    path = os.path.join(DOCS_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="文档不存在")
    return path


def _document_title(path: str, fallback: str) -> str:
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.startswith("# "):
                return line[2:].strip()
    return fallback


@router.get("/docs")
async def list_documents():
    documents = []
    for name in PUBLISHED_DOCS:
        path = _document_path(name)
        documents.append(
            {
                "name": name,
                "title": _document_title(path, name),
                "url": f"/api/v1/docs/{quote(name)}",
            }
        )
    return {"documents": documents}


@router.get("/docs/{name}", response_class=PlainTextResponse)
async def get_document(name: str):
    path = _document_path(name)
    with open(path, encoding="utf-8") as file:
        return PlainTextResponse(file.read(), media_type="text/markdown")
