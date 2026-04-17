# goldilocks-core architecture

## Project name
`goldilocks-core`

## Goal
`goldilocks-core` is a materials machine learning package for analyzing crystal structures and recommending suitable DFT calculation inputs.

The long-term goal is to help users go from a material structure to practical calculation recommendations, including input settings, submission strategy, and workflow suggestions.

## v0.1 scope
The first version will focus on a minimal but complete workflow for SCF single-point energy calculations.

v0.1 should support:
- structure-based recommendation
- one DFT code at a time
- a small number of recommended input components
- a clean Python package structure
- tests, documentation, and CI from the beginning

v0.1 will not support:
- complex multi-step workflows
- broad multi-code support from day one
- large-scale autonomous workflow planning
- advanced agentic behavior
- model training pipelines inside this package

## Core inputs
The core inputs for the package are:
- a material structure
- a target DFT code
- a calculation task

For the first implementation, the internal structure representation will be based on `pymatgen.Structure`.

## Core outputs
The package is intended to produce:
- recommended DFT input settings
- recommended submission-script settings, including parallelization suggestions
- a simple calculation workflow description

## Initial scientific focus
The initial scientific target is:
- DFT code: `Quantum ESPRESSO`
- task: SCF single-point energy calculation

This narrow scope is intentional. The goal is to build one clean, testable, and extensible end-to-end workflow before expanding to additional codes or more complex calculation tasks.

## Architecture direction
The package should follow a clear and research-grade structure inspired by well-organized scientific Python packages such as `janus-core`, while keeping a `src` layout.

The package should emphasize:
- explicit module boundaries
- small focused components
- clean Python APIs before CLI expansion
- testability from the beginning
- maintainable research-grade code

## Initial package layout
The initial package layout inside `src/goldilocks_core/` will include:

- `advisors/`
- `helpers/`
- `processing/`
- `cli/`
- `ml/`

## Responsibilities
### `advisors/`
This subpackage contains recommendation logic for specific DFT calculation settings and input components.

Examples may include:
- `kpoints.py`
- `smearing.py`
- `cutoffs.py`

These modules should be focused, composable, and easy to test.

### `helpers/`
This subpackage contains shared utilities and core support objects used across the package.

The current helper layer includes:
- structure loading utilities based on `pymatgen.Structure`
- lightweight structure analysis for DFT-relevant element categories
- shared dataclasses and type definitions for recommendations, features, and model metadata

It should not become a dumping ground for unrelated code.

### `processing/`
This subpackage contains higher-level orchestration logic that connects structure input, advisor logic, ML components, and output generation.

This layer is responsible for combining lower-level components into a usable recommendation workflow.

### `cli/`
This subpackage contains command-line interfaces built on top of the Python API.

The CLI should remain a thin layer over well-defined library functions.

### `ml/`
This subpackage contains machine-learning-related code such as model loading, inference wrappers, and ML support utilities.

Model training workflows are out of scope for this package and should live elsewhere.

The initial internal split of `ml/` is expected to include:
- `features.py` for feature extraction from structures
- `models.py` for trained model loading and model metadata
- `inference.py` for running predictions with loaded models

This separation is intended to keep `advisors/` focused on recommendation logic rather than model internals.

Feature extraction is expected to operate directly on `pymatgen.Structure` objects rather than on higher-level analysis summaries.

An initial target is to support CSLR-style feature extraction for downstream machine-learning models.

## Core workflow
The exact workflow will continue to evolve, but the initial direction is:

1. Read and validate structure input.
2. Extract relevant structural information.
3. Apply rule-based and/or machine-learning-based recommendation methods.
4. Optionally use LLM-based reasoning or external MCP-connected tools.
5. Assemble final calculation recommendations.

The role of LLMs is intentionally flexible:
- they may be used inside the package
- or they may be accessed through external MCP-based integrations

## Initial design principles
The package should follow these principles:
- simple and explicit APIs
- small focused modules
- separation of concerns
- testable components
- maintainable research-grade code
- documentation written alongside implementation

## Near-term implementation direction
The first implementation is currently focused on building the support layers needed before advisor logic becomes stable.

The current implementation path includes:
- supporting `pymatgen.Structure` as the main structure input
- providing structure loading and lightweight structure analysis helpers
- defining shared domain objects for recommendations, structure features, and model metadata
- establishing the initial `ml/` architecture for feature extraction, model loading, and inference
- adding tests alongside each new component

## Immediate next design task
The next design steps are to continue refining the machine-learning support layer and connect it to future advisor logic.

Current priorities include:
- extending CSLR-style feature extraction beyond the initial lattice-based feature block
- defining how trained models will be described and loaded
- defining how model inference outputs will be translated into recommendation logic


