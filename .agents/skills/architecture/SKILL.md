---
name: architecture
description: Reference to the Goldilocks system architecture document. Load before making structural decisions about the codebase, adding new pipeline stages, or when you need to understand how Core fits into the wider system.
---

# Architecture

The Goldilocks system architecture is documented in a separate repository: `goldilocks-architecture-doc`.

**Read the architecture document before making structural decisions.** It describes the target system design — the pipeline stages, subsystem boundaries, and design principles that this codebase is evolving towards.

## Where to find it

If `goldilocks-architecture-doc` is available locally (sibling of this repo):

```bash
cat ../goldilocks-architecture-doc/architecture.md
```

Otherwise, check with the user for access.

## Key architectural concepts

These are the high-level principles. Read the full document for details.

**Six-stage pipeline:**
Load → Analyse → Advise → Select → Generate → Bundle

- Each stage has defined input/output types
- Stages do not reach backwards — data flows forward only
- The AI subsystem is Analyse + Advise + Select (intelligent decisions)
- Input generation is Generate + Bundle (mechanical translation)

**Three subsystem boundaries:**
- **Core** owns physics recommendations and input generation
- **Runner** owns execution optimisation and AiiDA submission
- **Analysis** owns interpretation of calculation outputs

**Tools first, web app second:**
Every capability is available as Python API → CLI → HTTP API, in that order. The web app is a thin orchestration layer.

**Provenance matters:**
Every parameter recommendation records where it came from: `analysis`, `default`, or `user_hint`.

**Operators are operators:**
Humans and agents are the same kind of architectural actor. No privileged agent paths.

## When to read the full doc

- Before adding a new pipeline stage
- Before changing the data flow between stages
- Before adding a new advice category (smearing, magnetism, convergence, etc.)
- Before writing a new per-code generator
- When unsure whether something belongs in Core, Runner, or Analysis