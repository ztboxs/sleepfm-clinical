# Repository Guidelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a concise, English `AGENTS.md` that gives AI coding agents accurate contribution guidance for SleepFM-Clinical.

**Architecture:** Add one repository-root Markdown file organized around structure, commands, style, testing, collaboration, and sensitive-data handling. Derive every instruction from tracked files, documented workflows, or recent Git history.

**Tech Stack:** Markdown, Python 3.10, Conda, FastAPI/Uvicorn, Git

---

### Task 1: Create and verify the contributor guide

**Files:**
- Create: `AGENTS.md`
- Reference: `README.md`, `docs/测试指南.md`, `docs/部署文档.md`, `env.yml`, `.gitignore`

- [ ] **Step 1: Confirm the target is not already tracked**

Run: `git ls-files --error-unmatch AGENTS.md`

Expected: exits non-zero because `AGENTS.md` does not exist yet.

- [ ] **Step 2: Create the approved guide**

Create `AGENTS.md` with exactly this content:

```markdown
# Repository Guidelines

## Project Structure & Module Organization

`api/` contains the FastAPI service: routers live in `api/routers/`, request and response models in `api/schemas.py`, inference orchestration in `api/inference.py`, and the browser console in `api/static/index.html`. Core model, preprocessing, and training code belongs under `sleepfm/`; YAML experiment settings are in `sleepfm/configs/`, while model artifacts are under `sleepfm/checkpoints/`. Use `scripts/` for repository utilities, `notebooks/` for demonstrations, and `docs/` for deployment, API, and testing guidance.

## Build, Test, and Development Commands

- `conda env create -f env.yml` creates the Python 3.10 environment.
- `conda activate sleepfm_env` activates it for subsequent commands.
- `pip install fastapi "uvicorn[standard]" python-multipart` installs the API-only dependencies documented separately from the base environment.
- `python -m api.main` starts the service and Web console on port `6006`.
- `curl http://localhost:6006/api/v1/health` checks model and GPU readiness.

Training and evaluation scripts live in `sleepfm/pipeline/`; update the matching YAML configuration before running one.

## Coding Style & Naming Conventions

Use four-space indentation and follow PEP 8. Name modules, functions, and variables with `snake_case`, classes with `PascalCase`, and constants with `UPPER_SNAKE_CASE`. Group imports as standard library, third-party, then local modules. Add type annotations to schemas and public interfaces. No formatter or linter is configured, so match the surrounding style and keep changes focused.

## Testing Guidelines

The repository currently has no automated test suite or coverage threshold. Start the API, run the health check, then exercise affected endpoints with the workflow in `docs/测试指南.md` and demo assets under `notebooks/demo_data/`. For inference changes, run `python scripts/compare_notebook_vs_api.py` when the required models, GPU, and demo data are available. Put future automated tests in `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines

Recent history uses Conventional Commit-style prefixes such as `feat:` and `docs:`; use `fix:`, `refactor:`, or `test:` when appropriate. Keep each commit scoped to one logical change. Pull requests should explain motivation and behavior, list verification commands, link relevant issues, and include screenshots for `api/static/index.html` changes or request/response examples for API changes.

## Security & Data Handling

Never commit `.env` files, credentials, patient data, uploaded EDF files, generated HDF5/JSON exports, logs, or new large checkpoints. Use synthetic or repository-provided demo data in examples, and document any required local model or dataset paths without embedding private values.
```

- [ ] **Step 3: Validate length, structure, and whitespace**

Run: `wc -w AGENTS.md && rg -n '^#|^## ' AGENTS.md && git diff --check`

Expected: word count is between 200 and 400; the title and six second-level headings are present; `git diff --check` prints nothing.

- [ ] **Step 4: Review the final diff**

Run: `git diff -- AGENTS.md`

Expected: only the approved repository guide appears, with no unrelated file changes.

- [ ] **Step 5: Commit the guide**

```bash
git add AGENTS.md
git commit -m "docs: add repository contributor guidelines"
```

Expected: one new tracked file named `AGENTS.md`.
