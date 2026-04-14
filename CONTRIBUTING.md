# Contributing to stereo2spatial

This repository contains the training, inference, and workflow tooling
for `stereo2spatial`. Contributions are most useful when they are scoped,
reproducible, and aligned with the current Python 3.10 + CI workflow.

## Before You Start

- Search existing issues and pull requests before opening a new one.
- For larger design changes, open an issue first so architecture and scope can
  be aligned before implementation work starts.
- Use the issue forms for bugs and feature requests so maintainers get the
  details needed to reproduce and review the change.
- If you are reporting a vulnerability, follow [SECURITY.md](SECURITY.md)
  instead of opening a public issue.
- If your interaction concerns community behavior rather than code, see
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Development Setup

This repo targets Python 3.10.

```bash
python -m venv .venv
. .venv/Scripts/activate
pip install -e .[dev]
```

Core local checks:

```bash
ruff check .
mypy stereo2spatial tests
pytest -q
```

These are the same checks enforced by CI in `.github/workflows/ci.yml`.

## Repository Map

- `stereo2spatial/`: package code for modeling, training, inference, configs,
  and codec integration
- `tests/`: regression and unit tests
- `configs/`: runnable training presets
- `scripts/`: dataset prep, QC, Atmos conversion, and export helpers
- `docs/`: architecture and reference docs
- `stereo2spatial/vendor/`: vendored upstream code; keep changes here minimal
  and well-justified

## Contribution Expectations

- Keep changes focused. Avoid mixing refactors, formatting churn, and behavior
  changes in the same pull request.
- Preserve Python 3.10 compatibility unless the project explicitly raises its
  minimum supported version.
- Add or update tests whenever behavior changes in `stereo2spatial/`.
- Update documentation when changing CLI flags, config behavior, exported
  bundle structure, or user-visible workflows.
- Prefer modifying project code over vendored code. If you must touch
  `stereo2spatial/vendor/`, explain why in the pull request.
- Keep public APIs typed and keep new code consistent with the repo's current
  style.

## Data and Artifact Hygiene

This project works with large local datasets, checkpoints, and generated audio.
Do not commit local working artifacts.

The current `.gitignore` already covers common local artifact paths, but review
your diff before pushing anyway.

## Working on Bugs

Bug reports are easiest to act on when they include:

- the exact command that failed
- the config file or config fragment used
- whether the failure happened in training, inference, export, or dataset
  tooling
- a traceback or log excerpt
- environment details such as OS, Python version, CUDA availability, and
  relevant hardware

If the bug depends on local data, provide the smallest reproducible slice you
can describe without sharing private or copyrighted media.

## Working on Features

For feature work, describe the user problem first, then the proposed change.
In this repo, maintainers will usually need to understand one or more of the
following up front:

- whether the change affects training, inference, dataset tooling, or export
- whether it changes config schema or CLI surface area
- whether it changes model checkpoint or bundle compatibility
- whether it adds new third-party dependencies

## Pull Request Guidelines

- Use a short, descriptive title.
- Link the issue being addressed when one exists.
- Summarize the behavior change, not just the files touched.
- Call out any breaking changes to configs, checkpoint loading, bundle export,
  or CLI flags.
- Include the exact validation you ran. If you could not run one of the normal
  checks, explain why.

Small, reviewable pull requests move faster than large batches of unrelated
changes.
