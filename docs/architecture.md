# Architecture

## Overview

`goldilocks-core` is a research-grade Python package for recommending DFT
calculation inputs from structure-aware logic and machine-learning models.

The package is organized around domain-focused modules rather than generic
utility buckets. The goal is to keep parsing, scientific processing, model
inference, recommendation policy, and user-facing interfaces clearly separated.

## Design Principles

- Keep core scientific logic independent from command-line and UI concerns.
- Keep data models explicit and reusable across modules.
- Treat `advisors/` as orchestration and policy layers rather than low-level computation.
- Prefer small focused modules over large mixed-responsibility files.
- Support incremental migration from older generic namespaces such as `helpers/`
  and `processing/` into clearer domain-oriented modules.

## Target Package Layout

```text
src/goldilocks_core/
├── advisors/
├── cli/
├── io/
├── kmesh.py
├── ml/
├── pseudos/
└── shared/
