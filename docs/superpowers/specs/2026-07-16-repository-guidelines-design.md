# Repository Guidelines Design

## Goal

Create an English `AGENTS.md` for AI coding agents and human contributors. The guide must be concise, repository-specific, professional, and between 200 and 400 words.

## Content

The document will use the required title, `Repository Guidelines`, and cover:

- the roles of `api/`, `sleepfm/`, `scripts/`, `notebooks/`, and `docs/`;
- verified setup and local-run commands based on `env.yml`, `README.md`, and the existing deployment and testing guides;
- the Python conventions already visible in the repository, without inventing formatter or linter requirements;
- the current manual API and end-to-end testing workflow, explicitly noting that no automated test suite or coverage threshold is configured;
- commit prefixes observed in history (`feat:`, `fix:`, and `docs:`) and practical pull-request expectations;
- handling rules for PSG/clinical data, generated HDF5/JSON exports, checkpoints, secrets, and large artifacts.

## Constraints

Examples must use real paths and commands. The guide will not claim unsupported compatibility, tooling, or coverage requirements. It will tell agents to keep changes focused, avoid committing sensitive or generated data, and update relevant documentation when API behavior changes.

## Verification

After creation, verify the Markdown headings, word count, referenced paths, commands, and Git-derived conventions. Review the final diff to ensure only the approved guide is added during implementation.
