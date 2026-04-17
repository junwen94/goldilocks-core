# goldilocks-core

`goldilocks-core` is a materials machine-learning package for analyzing crystal structures and recommending suitable DFT calculation inputs.

## Current scope

The current development focus is a clean and research-grade package structure for Quantum ESPRESSO SCF single-point recommendations.

At the current stage, the project includes:
- structure loading helpers based on `pymatgen.Structure`
- basic structure analysis utilities
- an initial machine-learning feature extraction layer
- early package architecture for model loading and inference

## Package layout

The package currently develops around the following subpackages:

- `helpers/`
- `advisors/`
- `processing/`
- `cli/`
- `ml/`

## Development status

This project is currently in an early design and implementation stage.

The current codebase focuses on:
- building a maintainable package structure
- defining shared domain objects and types
- implementing and testing core helper utilities
- sketching the machine-learning feature/model/inference architecture

## Installation

Clone the repository and install the development dependencies with `uv`:

```bash
git clone git@github.com:stfc/goldilocks-core.git
cd goldilocks-core
uv sync
```

## Devlopment

Run the test suite with:
```bash
uv run pytest
```
Run linting with:
```bash
uv run ruff check .
```
Run pre-commit checks with:
```bash
uv run pre-commit run --all-files
```
