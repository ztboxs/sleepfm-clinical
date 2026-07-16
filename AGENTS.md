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
