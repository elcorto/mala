# AGENTS.md

This file is for agentic assistants answering questions about this repository.
It is intentionally optimized for codebase navigation and "where is X computed"
queries, not for implementation workflow.

## Rule discovery

- Checked for Cursor rules in `.cursor/rules/` and `.cursorrules`: none found.
- Checked for Copilot rules in `.github/copilot-instructions.md`: not present.
- Therefore, this file plus the docs in `docs/source/` are the main guidance.

## What this project does

MALA (Materials Learning Algorithms) is an ML-DFT framework.
It predicts LDOS (local density of states) on a real-space grid from atomic
structure descriptors, then derives physical observables (density, DOS,
energies, etc.) from the LDOS.

At a high level, MALA replaces expensive parts of DFT post-processing with
neural-network inference while preserving access to physically meaningful
quantities.

## High-level data flow

1. Electronic-structure raw data is generated (primarily Quantum ESPRESSO).
2. `DataConverter` converts raw outputs into MALA-ready volumetric data:
   descriptors + targets + compact simulation metadata.
3. `DataHandler` loads and scales snapshot data for training/validation/test.
4. `Network` maps descriptors -> LDOS.
5. `Trainer` optimizes model weights; `Tester` evaluates on held-out snapshots.
6. `Predictor` (or ASE calculator interface) performs inference for new atoms.
7. `LDOS` / `Density` / `DOS` calculators compute observables from predicted LDOS.

## Core domain terms used in code

- Snapshot: one atomic configuration + associated volumetric arrays.
- Descriptor: grid-based input representation of atomic structure
  (Bispectrum or ACE).
- Target: supervised output, usually LDOS.
- Additional calculation data: parsed simulation metadata used later for
  physically consistent inference/postprocessing.
- On-the-fly descriptors: descriptors computed at runtime from simulation
  metadata instead of precomputed arrays.

## Package map (where to look first)

- `mala/common/`
  - Global configuration and parameter objects.
  - Parallel-safe output and MPI/DDP plumbing.
- `mala/datahandling/`
  - Data conversion, snapshot bookkeeping, scaling, shuffling, lazy loading.
- `mala/descriptors/`
  - Descriptor computation backends and descriptor base class.
- `mala/network/`
  - Network definitions, training/testing/prediction runners, hyperopt.
- `mala/targets/`
  - LDOS/Density/DOS calculators and observable computations.
- `mala/interfaces/`
  - ASE-facing calculator integration.
- `mala/datageneration/`
  - Utilities for trajectory analysis and OF-DFT initialization.

## "Where is X computed?" quick index

### LDOS and derived observables

- Main LDOS implementation: `mala/targets/ldos.py` (`class LDOS`).
- LDOS -> total energy: `LDOS.get_total_energy`.
- LDOS -> band energy: `LDOS.get_band_energy`.
- LDOS -> entropy contribution: `LDOS.get_entropy_contribution`.
- LDOS -> number of electrons: `LDOS.get_number_of_electrons`.
- LDOS -> density: `LDOS.get_density`.
- LDOS -> DOS: `LDOS.get_density_of_states`.
- Self-consistent Fermi level from LDOS: `LDOS.get_self_consistent_fermi_energy`.

### Density and total-energy-related pieces

- Main density implementation: `mala/targets/density.py` (`class Density`).
- Density integral / electrons: `Density.get_number_of_electrons`.
- Density-derived energy terms: `Density.get_energy_contributions`.
- Density from LDOS helper: `Density.from_ldos_calculator`.

### DOS calculations

- Main DOS implementation: `mala/targets/dos.py` (`class DOS`).
- DOS-derived band energy: `DOS.get_band_energy`.
- DOS-derived electrons: `DOS.get_number_of_electrons`.
- DOS-derived entropy: `DOS.get_entropy_contribution`.
- Self-consistent Fermi level from DOS: `DOS.get_self_consistent_fermi_energy`.

### Descriptor generation

- Descriptor base logic: `mala/descriptors/descriptor.py` (`class Descriptor`).
- Descriptor from QE output: `Descriptor.calculate_from_qe_out`.
- Descriptor from JSON metadata: `Descriptor.calculate_from_json`.
- Descriptor from ASE atoms: `Descriptor.calculate_from_atoms`.
- Descriptor implementations:
  - Bispectrum: `mala/descriptors/bispectrum.py`
  - ACE: `mala/descriptors/ace.py`
  - Atomic density: `mala/descriptors/atomic_density.py`

### Data conversion and loading

- Raw->MALA conversion: `mala/datahandling/data_converter.py`.
  - Add input snapshots: `DataConverter.add_snapshot`
  - Execute conversion: `DataConverter.convert_snapshots`
- Dataset preparation/scaling: `mala/datahandling/data_handler.py`.
  - Main preparation entrypoint: `DataHandler.prepare_data`
- Data randomization for lazy loading: `mala/datahandling/data_shuffler.py`.

### Training, testing, prediction

- Training loop: `mala/network/trainer.py` (`Trainer.train_network`).
- Generic run save/load and forward helpers: `mala/network/runner.py`.
- Evaluation flow: `mala/network/tester.py` (`Tester.test_all_snapshots`).
- Inference for new structures: `mala/network/predictor.py`
  (`Predictor.predict_for_atoms`, `Predictor.predict_from_qeout`).

### ASE integration

- ASE calculator class: `mala/interfaces/ase_calculator.py` (`class MALA`).
- Main ASE compute hook: `MALA.calculate` and `MALA.calculate_properties`.

### Parameters and global runtime behavior

- Main parameter tree: `mala/common/parameters.py` (`class Parameters`).
- Parameter subsets include network, descriptors, targets, data, running,
  hyperparameter optimization, and data generation.
- Parallel/rank-aware behavior: `mala/common/parallelizer.py`.

## Documentation map (read these before deep code search)

- Project overview and conceptual workflow: `docs/source/index.md`.
- End-to-end basic workflow: `docs/source/basic_usage/trainingmodel.rst`.
- Data generation/conversion details: `docs/source/basic_usage/more_data.rst`.
- Prediction flow and observables: `docs/source/basic_usage/predictions.rst`.
- Advanced performance + lazy loading/shuffling: `docs/source/advanced_usage/trainingmodel.rst`.
- Descriptor tuning and ACSD/ACE context: `docs/source/advanced_usage/descriptors.rst`.
- Scaled inference, MPI/GPU, visualization: `docs/source/advanced_usage/predictions.rst`.
- OpenPMD data path: `docs/source/advanced_usage/openpmd.rst`.

## How to answer navigation questions in this repo

- Start from docs terminology (LDOS, snapshot, descriptor, target).
- Map concept -> package -> concrete class -> method.
- Confirm with direct method definitions (not only call sites).
- Include path references in answers; include method names whenever possible.
- When behavior differs by mode (numpy vs openPMD, precomputed vs on-the-fly,
  serial vs MPI/DDP), call that out explicitly.

## Known navigation pitfalls

- Many observables are exposed as properties that call internal calculators;
  report both property and underlying `get_*` method.
- Some functionality exists in both `LDOS` and `DOS`/`Density`; identify the
  authoritative implementation for the asked quantity.
- There are legacy and advanced pathways (e.g., optional modules, MPI branches);
  avoid assuming one path is always active.
- Tests/examples may show usage patterns that are simpler than production paths.

## Scope note

This AGENTS file intentionally omits build/lint/test command guidance and other
coding-task workflow details, because the primary use case is codebase Q&A and
location tracing.
