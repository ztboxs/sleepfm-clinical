# Client Documentation Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver six Chinese client-facing Markdown documents through a read-only API and a dedicated `/help` documentation page linked from the console header.

**Architecture:** Markdown under `docs/` remains the only content source. A whitelisted FastAPI router reads those files on demand; the console header links to a separate static documentation shell that fetches and renders Markdown without embedding a second copy.

**Tech Stack:** Python 3.10, FastAPI, pytest, HTML/CSS/vanilla JavaScript, marked.js, Markdown

---

## File map

- Create `api/routers/documentation.py`: published-document whitelist and read-only API.
- Modify `api/config.py`: add the repository `DOCS_DIR`.
- Modify `api/main.py`: include the documentation router and serve `/help`.
- Modify `api/static/index.html`: add the header help menu only; keep all business tabs unchanged.
- Create `api/static/docs.html`: dedicated responsive documentation reader.
- Create `docs/使用文档.md`, `docs/模型算法.md`, `docs/训练与调优.md`: client-facing source documents.
- Modify `docs/接口文档.md`, `docs/部署文档.md`, `docs/测试指南.md`: correct async behavior and document the new center.
- Create `tests/test_documentation_api.py`, `tests/test_help_page.py`, `tests/test_documentation_content.py`: API, shell, and content-contract tests.

### Task 1: Add the read-only documentation API

**Files:**
- Create: `tests/test_documentation_api.py`
- Create: `api/routers/documentation.py`
- Modify: `api/config.py`
- Modify: `api/main.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_documentation_api.py`:

```python
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
```

- [ ] **Step 2: Run the tests and confirm the missing module failure**

Run: `pytest -q tests/test_documentation_api.py`

Expected: collection fails because `api.routers.documentation` does not exist.

- [ ] **Step 3: Implement the minimal API**

Append to `api/config.py`:

```python
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
```

Create `api/routers/documentation.py`:

```python
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
        documents.append({
            "name": name,
            "title": _document_title(path, name),
            "url": f"/api/v1/docs/{quote(name)}",
        })
    return {"documents": documents}


@router.get("/docs/{name}", response_class=PlainTextResponse)
async def get_document(name: str):
    path = _document_path(name)
    with open(path, encoding="utf-8") as file:
        return PlainTextResponse(file.read(), media_type="text/markdown")
```

Update `api/main.py` imports and router registration:

```python
from api.routers import documentation, health, predict

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(documentation.router)
```

- [ ] **Step 4: Run API tests**

Run: `pytest -q tests/test_documentation_api.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit the API slice**

```bash
git add api/config.py api/main.py api/routers/documentation.py tests/test_documentation_api.py
git commit -m "feat: add read-only documentation API"
```

### Task 2: Add the header help menu and dedicated reader

**Files:**
- Create: `tests/test_help_page.py`
- Create: `api/static/docs.html`
- Modify: `api/static/index.html`
- Modify: `api/main.py`

- [ ] **Step 1: Read `frontend-design` before changing UI**

Read `/Users/Apple/.codex/skills/frontend-design/SKILL.md` completely. Preserve the established Meta-style tokens and do not redesign unrelated controls.

- [ ] **Step 2: Write failing shell tests**

Create `tests/test_help_page.py`:

```python
from pathlib import Path

from api.main import app


ROOT = Path(__file__).resolve().parents[1]


def test_help_route_is_registered() -> None:
    assert "/help" in {route.path for route in app.routes}


def test_console_header_links_to_help() -> None:
    html = (ROOT / "api/static/index.html").read_text(encoding="utf-8")
    assert 'class="help-menu"' in html
    assert 'href="/help"' in html
    assert 'href="/docs"' in html


def test_documentation_shell_fetches_markdown_instead_of_copying_it() -> None:
    html = (ROOT / "api/static/docs.html").read_text(encoding="utf-8")
    assert 'id="docNavigation"' in html
    assert 'id="docContent"' in html
    assert "/api/v1/docs" in html
    assert "marked.parse" in html
    assert "LOO-CL 预训练" not in html
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `pytest -q tests/test_help_page.py`

Expected: failures for missing `/help`, help menu, and `api/static/docs.html`.

- [ ] **Step 4: Serve the documentation page**

Add to `api/main.py` after the root route:

```python
@app.get("/help")
async def help_page():
    return FileResponse(os.path.join(STATIC_DIR, "docs.html"))
```

- [ ] **Step 5: Add the console header menu**

In `api/static/index.html`, add styles for `.header-actions`, `.help-menu`, `.help-menu-popover`, and menu links using the existing `--card`, `--border`, `--primary`, `--shadow-md`, and `--radius` tokens. Place the menu before `#sysStats`:

```css
.header-actions {
  margin-left: auto;
  margin-right: 16px;
}

.help-menu { position: relative; }
.help-menu summary {
  list-style: none;
  padding: 8px 12px;
  border-radius: var(--radius);
  color: var(--text-secondary);
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: var(--transition);
}
.help-menu summary::-webkit-details-marker { display: none; }
.help-menu summary:hover,
.help-menu[open] summary { background: var(--bg); color: var(--primary); }
.help-menu-popover {
  position: absolute;
  top: calc(100% + 10px);
  right: 0;
  width: 260px;
  padding: 8px;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-lg);
  z-index: 220;
}
.help-menu-popover a {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 12px;
  border-radius: var(--radius);
  color: var(--text-primary);
  text-decoration: none;
}
.help-menu-popover a:hover { background: var(--primary-light); }
.help-menu-popover strong { font-size: 14px; }
.help-menu-popover span { font-size: 12px; color: var(--text-secondary); }
@media (max-width: 768px) {
  .header-actions { margin-right: 8px; }
  .help-menu-popover { right: -8px; width: 210px; }
  .help-menu-popover span { display: none; }
}
```

```html
<div class="header-actions">
  <details class="help-menu">
    <summary>帮助 <span aria-hidden="true">▾</span></summary>
    <div class="help-menu-popover">
      <a href="/help"><strong>使用文档</strong><span>算法、训练、接口与部署说明</span></a>
      <a href="/docs"><strong>API 调试文档</strong><span>打开 Swagger 接口页面</span></a>
    </div>
  </details>
</div>
```

Keep the existing seven business tabs unchanged. On screens below 768 px, hide the secondary menu descriptions and keep the help trigger visible.

- [ ] **Step 6: Create the dedicated documentation shell**

Create `api/static/docs.html` with these exact responsibilities:

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SleepFM Clinical - 使用文档</title>
  <script src="https://cdn.jsdelivr.net/npm/marked@15.0.12/marked.min.js"></script>
  <style>
    :root {
      --primary: #1877f2;
      --primary-hover: #166fe5;
      --primary-light: #e7f3ff;
      --bg: #f0f2f5;
      --card: #ffffff;
      --text-primary: #1c1e21;
      --text-secondary: #65676b;
      --text-tertiary: #8a8d91;
      --border: #dadde1;
      --border-light: #e4e6eb;
      --radius: 8px;
      --radius-lg: 12px;
      --shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
      --shadow-md: 0 2px 8px rgba(0, 0, 0, 0.1);
      --font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
        'Helvetica Neue', Arial, 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text-primary);
      font-family: var(--font);
      line-height: 1.7;
    }
    .docs-header {
      position: sticky;
      top: 0;
      z-index: 100;
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 28px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--border);
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .brand {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      color: var(--text-primary);
      font-weight: 700;
      text-decoration: none;
    }
    .logo {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border-radius: var(--radius);
      background: linear-gradient(135deg, var(--primary), #6c5ce7);
      color: white;
    }
    .docs-header nav { display: flex; gap: 8px; }
    .docs-header nav a {
      padding: 8px 12px;
      border-radius: var(--radius);
      color: var(--text-secondary);
      font-size: 14px;
      font-weight: 600;
      text-decoration: none;
    }
    .docs-header nav a:hover { background: var(--primary-light); color: var(--primary); }
    .docs-layout {
      width: min(1320px, calc(100% - 40px));
      margin: 28px auto 56px;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
      gap: 24px;
      align-items: start;
    }
    .docs-sidebar {
      position: sticky;
      top: 92px;
      padding: 20px;
      background: var(--card);
      border: 1px solid var(--border-light);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }
    .eyebrow {
      margin: 0 0 4px;
      color: var(--primary);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: 0.12em;
    }
    .docs-sidebar h1 { margin: 0 0 18px; font-size: 22px; }
    #docNavigation { display: grid; gap: 6px; }
    #docNavigation button {
      width: 100%;
      padding: 10px 12px;
      border: 0;
      border-radius: var(--radius);
      background: transparent;
      color: var(--text-secondary);
      font: 500 14px/1.35 var(--font);
      text-align: left;
      cursor: pointer;
    }
    #docNavigation button:hover { background: var(--bg); color: var(--text-primary); }
    #docNavigation button.active { background: var(--primary-light); color: var(--primary); font-weight: 700; }
    .docs-reader {
      min-height: calc(100vh - 148px);
      padding: 40px clamp(24px, 5vw, 64px);
      background: var(--card);
      border: 1px solid var(--border-light);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
    }
    .doc-status { padding: 72px 20px; color: var(--text-secondary); text-align: center; }
    .doc-status button {
      margin-top: 12px;
      padding: 8px 14px;
      border: 0;
      border-radius: var(--radius);
      background: var(--primary);
      color: white;
      font: 600 14px var(--font);
      cursor: pointer;
    }
    .markdown-body { max-width: 820px; margin: 0 auto; }
    .markdown-body h1 { margin: 0 0 28px; font-size: clamp(30px, 4vw, 42px); line-height: 1.2; }
    .markdown-body h2 { margin: 42px 0 16px; padding-bottom: 8px; border-bottom: 1px solid var(--border-light); font-size: 24px; }
    .markdown-body h3 { margin: 28px 0 12px; font-size: 18px; }
    .markdown-body p, .markdown-body ul, .markdown-body ol { margin: 0 0 16px; }
    .markdown-body ul, .markdown-body ol { padding-left: 24px; }
    .markdown-body a { color: var(--primary); text-underline-offset: 3px; }
    .markdown-body code {
      padding: 2px 6px;
      border-radius: 5px;
      background: #f5f7fa;
      color: #b42318;
      font-size: 0.9em;
    }
    .markdown-body pre {
      margin: 18px 0;
      padding: 18px;
      overflow: auto;
      border-radius: var(--radius-lg);
      background: #101828;
      color: #f8fafc;
    }
    .markdown-body pre code { padding: 0; background: transparent; color: inherit; }
    .markdown-body table { width: 100%; margin: 18px 0; border-collapse: collapse; font-size: 14px; }
    .markdown-body th, .markdown-body td { padding: 10px 12px; border: 1px solid var(--border); text-align: left; }
    .markdown-body th { background: var(--bg); }
    .markdown-body blockquote {
      margin: 18px 0;
      padding: 12px 16px;
      border-left: 4px solid var(--primary);
      background: var(--primary-light);
      color: var(--text-secondary);
    }
    @media (max-width: 768px) {
      .docs-header { height: auto; min-height: 60px; padding: 10px 16px; }
      .docs-header nav a { padding: 7px 8px; font-size: 13px; }
      .docs-layout { width: min(100% - 24px, 720px); margin-top: 16px; grid-template-columns: 1fr; }
      .docs-sidebar { position: static; }
      #docNavigation { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .docs-reader { min-height: auto; padding: 28px 18px; }
      .markdown-body table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <header class="docs-header">
    <a class="brand" href="/"><span class="logo">S</span><span>SleepFM Clinical</span></a>
    <nav><a href="/">返回控制台</a><a href="/docs">API 参考</a></nav>
  </header>
  <main class="docs-layout">
    <aside class="docs-sidebar">
      <p class="eyebrow">DOCUMENTATION</p>
      <h1>使用文档</h1>
      <nav id="docNavigation" aria-label="文档列表"></nav>
    </aside>
    <section class="docs-reader">
      <div id="docStatus" class="doc-status">正在加载文档…</div>
      <article id="docContent" class="markdown-body" hidden></article>
    </section>
  </main>
  <script>
    const API_BASE = '';
    let documents = [];
    let activeDocument = '';

    function showError(message) {
      const status = document.getElementById('docStatus');
      status.hidden = false;
      status.innerHTML = '';
      const text = document.createElement('p');
      text.textContent = message;
      const retry = document.createElement('button');
      retry.type = 'button';
      retry.textContent = '重新加载';
      retry.addEventListener('click', loadDocuments);
      status.append(text, retry);
      document.getElementById('docContent').hidden = true;
    }

    function renderNavigation() {
      const navigation = document.getElementById('docNavigation');
      navigation.innerHTML = '';
      documents.forEach(documentInfo => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = documentInfo.title;
        button.className = documentInfo.name === activeDocument ? 'active' : '';
        button.addEventListener('click', () => loadDocument(documentInfo.name));
        navigation.appendChild(button);
      });
    }

    async function loadDocument(name) {
      const status = document.getElementById('docStatus');
      const content = document.getElementById('docContent');
      status.hidden = false;
      status.textContent = '正在加载文档…';
      content.hidden = true;
      try {
        const response = await fetch(`${API_BASE}/api/v1/docs/${encodeURIComponent(name)}`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        if (!window.marked) throw new Error('Markdown 渲染器加载失败');
        const markdown = await response.text();
        content.innerHTML = marked.parse(markdown);
        content.querySelectorAll('a[href^="http"]').forEach(link => {
          link.target = '_blank';
          link.rel = 'noopener noreferrer';
        });
        activeDocument = name;
        window.location.hash = encodeURIComponent(name);
        renderNavigation();
        status.hidden = true;
        content.hidden = false;
        window.scrollTo({ top: 0, behavior: 'smooth' });
      } catch (error) {
        showError(`文档加载失败：${error.message}`);
      }
    }

    async function loadDocuments() {
      try {
        const response = await fetch(`${API_BASE}/api/v1/docs`);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        documents = (await response.json()).documents;
        if (!documents.length) throw new Error('没有可用文档');
        const requested = decodeURIComponent(window.location.hash.slice(1));
        const initial = documents.some(item => item.name === requested)
          ? requested
          : documents[0].name;
        renderNavigation();
        await loadDocument(initial);
      } catch (error) {
        showError(`文档列表加载失败：${error.message}`);
      }
    }

    loadDocuments();
  </script>
</body>
</html>
```

Do not add document prose to this file.

- [ ] **Step 7: Run shell tests**

Run: `pytest -q tests/test_help_page.py`

Expected: `3 passed`.

- [ ] **Step 8: Commit the UI shell**

```bash
git add api/main.py api/static/index.html api/static/docs.html tests/test_help_page.py
git commit -m "feat: add dedicated documentation center"
```

### Task 3: Write the client-facing source documents

**Files:**
- Create: `tests/test_documentation_content.py`
- Create: `docs/使用文档.md`
- Create: `docs/模型算法.md`
- Create: `docs/训练与调优.md`

- [ ] **Step 1: Write failing content-contract tests**

Create `tests/test_documentation_content.py`:

```python
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
        "/api/v1/health", "/api/v1/preprocess", "/api/v1/embed",
        "/api/v1/sleep_staging", "/api/v1/disease_prediction",
        "/api/v1/predict", "/api/v1/download/{file_id}",
        "preprocessed_file_id", "embedding_file_id", "age / 100",
        "0 = 女性", "1 = 男性", "24 小时", "busy",
    ):
        assert token in content


def test_algorithm_and_tuning_documents_cover_client_topics() -> None:
    algorithm = read("模型算法")
    for token in ("128 Hz", "5 秒", "LOO-CL", "LSTM", "CoxPH"):
        assert token in algorithm

    tuning = read("训练与调优")
    for token in ("pretrain.py", "generate_embeddings.py", "验收指标", "调优请求"):
        assert token in tuning
```

- [ ] **Step 2: Run the tests and confirm missing-file failures**

Run: `pytest -q tests/test_documentation_content.py`

Expected: failures for `使用文档.md`, `模型算法.md`, and `训练与调优.md`.

- [ ] **Step 3: Write `docs/使用文档.md`**

Use this exact heading structure and facts:

```markdown
# SleepFM-Clinical 使用文档
## 快速开始
### 打开控制台
### 健康检查
### 使用 Demo EDF 完成一次推理
## 控制台功能
### 完整推理
### 数据预处理
### 生成嵌入
### 睡眠分期
### 疾病预测
### 系统状态与疾病标签
## API 调用流程
### 异步子任务接口
### 完整推理接口
### 下载导出文件
## 参数说明
## 结果与文件复用
## 常见问题
```

Document that `/preprocess`, `/embed`, `/sleep_staging`, and `/disease_prediction` return `{"status":"accepted","task_id":"..."}` and require polling `/api/v1/subtask_result/{task_id}`. Document that `/predict` uses `/task_status` and `/task_result`. Include concise curl examples, real-age normalization, gender codes, `exports/`, 24-hour cleanup, required BAS/RESP/EKG/EMG modalities, busy handling, GPU serialization, and the existing 30–60 second demo reference with a hardware/data caveat.

- [ ] **Step 4: Write `docs/模型算法.md`**

Use this exact heading structure:

```markdown
# SleepFM-Clinical 模型算法说明
## 模型定位
## 输入与预处理
## 基座模型
### 5 秒 Tokenizer
### 通道无关与时序建模
### LOO-CL 预训练目标
## 下游模型
### 睡眠分期
### 疾病风险预测
## 当前服务推理链路
## 能力边界
## 参考资料
```

Explain 128 Hz resampling, 5-second/640-point tokens, six Conv1d blocks, 128-dimensional embeddings, modality pooling, transformer context, LOO-CL positives/negatives, LSTM heads, five sleep stages, age/sex concatenation, and CoxPH hazards. State explicitly that the published model is robust to heterogeneous montages, while this deployed pipeline currently requires at least one matched channel from all four modality groups. Link the Nature Medicine article and prototype repository, and state that hazard scores are relative model outputs rather than diagnoses or calibrated absolute probabilities.

- [ ] **Step 5: Write `docs/训练与调优.md`**

Use this exact heading structure:

```markdown
# SleepFM-Clinical 训练与调优方案
## 适用范围
## 数据准备
## 训练流程
### EDF 预处理
### 自监督预训练
### 嵌入生成
### 睡眠分期微调
### 疾病预测微调
## 配置与起始参数
## 评估与验收
## 向项目方提交调优请求
## 交付物
## 注意事项
```

Use repository-real commands from `sleepfm/pipeline/` and `sleepfm/preprocessing/preprocessing.py`. Separate the paper reference configuration (batch 32, learning rate 0.001, pretraining one epoch, fine-tuning ten epochs) from the sample YAML values. Warn that paths in YAML are prototype absolute paths and must be replaced. The tuning request checklist must require task definition, target population, channel inventory, sample/recording counts, label definitions, split policy, acceptance metrics, compute limits, privacy authorization, and expected deliverables.

- [ ] **Step 6: Run content tests**

Run: `pytest -q tests/test_documentation_content.py`

Expected: `3 passed`.

- [ ] **Step 7: Commit the new source documents**

```bash
git add docs/使用文档.md docs/模型算法.md docs/训练与调优.md tests/test_documentation_content.py
git commit -m "docs: add client model and usage guides"
```

### Task 4: Correct the API and testing documents

**Files:**
- Modify: `docs/接口文档.md`
- Modify: `docs/测试指南.md`
- Modify: `tests/test_documentation_content.py`

- [ ] **Step 1: Extend content tests for asynchronous contracts**

Append:

```python
def test_api_and_test_guides_use_current_async_contracts() -> None:
    api = read("接口文档")
    testing = read("测试指南")
    for content in (api, testing):
        assert "/api/v1/subtask_result/{task_id}" in content
        assert '"status": "accepted"' in content
        assert "/api/v1/task_result" in content
```

- [ ] **Step 2: Run the test and confirm the outdated docs fail**

Run: `pytest -q tests/test_documentation_content.py::test_api_and_test_guides_use_current_async_contracts`

Expected: failure because the current guides omit the subtask polling contract.

- [ ] **Step 3: Update `docs/接口文档.md`**

Keep health, task status/result, label mapping, input formats, result models, errors, channel requirements, and curl examples. Correct all independent inference endpoints to show the accepted response and polling result. Add `/api/v1/subtask_result/{task_id}`, `/api/v1/docs`, and `/api/v1/docs/{name}`. Keep `/predict` documented as the separate global tracker flow. Remove response examples that imply independent endpoints return full results synchronously.

- [ ] **Step 4: Update `docs/测试指南.md`**

Keep the Web walkthrough and demo facts. Change each independent curl/Python flow to submit, read `task_id`, poll `/subtask_result/{task_id}`, and then use returned file IDs. Add a documentation-center smoke test: open `/`, select “帮助” → “使用文档”, verify `/help`, switch all six documents, and check one Markdown code block/table renders.

- [ ] **Step 5: Run content tests**

Run: `pytest -q tests/test_documentation_content.py`

Expected: `4 passed`.

- [ ] **Step 6: Commit corrected operational docs**

```bash
git add docs/接口文档.md docs/测试指南.md tests/test_documentation_content.py
git commit -m "docs: align API guides with async tasks"
```

### Task 5: Update deployment documentation

**Files:**
- Modify: `docs/部署文档.md`
- Modify: `tests/test_documentation_content.py`

- [ ] **Step 1: Add a failing deployment contract test**

Append:

```python
def test_deployment_document_covers_documentation_center() -> None:
    deployment = read("部署文档")
    for token in (
        "/help", "/api/v1/docs", "api/routers/documentation.py",
        "api/static/docs.html", "帮助", "使用文档",
    ):
        assert token in deployment
```

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `pytest -q tests/test_documentation_content.py::test_deployment_document_covers_documentation_center`

Expected: failure because the deployment guide predates the documentation center.

- [ ] **Step 3: Update `docs/部署文档.md` minimally**

Add `documentation.py`, `docs.html`, and the three new Markdown sources to the project tree. Add `/help` to access addresses and both documentation API routes to the endpoint table. Add a “文档中心验证” subsection with the exact header-menu flow. Do not change Docker or systemd instructions and do not alter model deployment behavior.

- [ ] **Step 4: Run deployment and full content tests**

Run: `pytest -q tests/test_documentation_content.py`

Expected: `5 passed`.

- [ ] **Step 5: Commit deployment documentation**

```bash
git add docs/部署文档.md tests/test_documentation_content.py
git commit -m "docs: document the help center deployment"
```

### Task 6: Full verification and manual handoff

**Files:**
- Verify: all files changed by Tasks 1–5

- [ ] **Step 1: Run the full automated suite**

Run: `pytest -q`

Expected: all tests pass with zero failures.

- [ ] **Step 2: Verify Python syntax**

Run: `python -m compileall -q api tests`

Expected: exit code 0 with no syntax errors.

- [ ] **Step 3: Verify Markdown single-source behavior and repository hygiene**

Run:

```bash
rg -n "LOO-CL 预训练|age / 100|preprocessed_file_id" api/static/docs.html
git diff --check
git status --short
```

Expected: the first command prints nothing because document prose is not embedded in HTML; `git diff --check` prints nothing; status only shows intended changes if any remain.

- [ ] **Step 4: Inspect the final change set**

Run: `git log --oneline -6 && git status --short`

Expected: focused commits for API, UI shell, new documents, async corrections, and deployment documentation; clean worktree.

- [ ] **Step 5: Record the user smoke test**

Handoff instructions:

```text
1. Start the service with `python -m api.main`.
2. Open `http://<host>:6006/`.
3. Select “帮助” → “使用文档”.
4. Confirm the browser opens `/help` and displays the six Markdown documents.
5. Switch documents and verify headings, tables, links, and curl code blocks render correctly.
```
