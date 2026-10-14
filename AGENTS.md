# AGENTS.md

This file is for agentic assistants answering questions about this repository.
It is intentionally optimized for codebase navigation and "where is X computed"
queries, not for implementation workflow.

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
4. `Network` maps descriptors to the configured target, usually LDOS.
5. `Trainer` optimizes model weights; `Tester` evaluates on held-out snapshots.
6. `Predictor` (or ASE calculator interface) performs inference for new atoms.
7. `LDOS` / `Density` / `DOS` calculators postprocess predicted target data
   into observables.

## Core domain terms used in code

- Snapshot: one atomic configuration + associated volumetric arrays.
- Descriptor: grid-based input representation of atomic structure
  (Bispectrum, AtomicDensity, ACE, or deprecated MinterpyDescriptors).
- Target: supervised output (LDOS, DOS, or Density; usually LDOS).
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
- LDOS -> total free energy and its terms: see the
  [detailed path below](#predicted-total-energy-detailed-path).
- LDOS -> number of electrons: `LDOS.get_number_of_electrons`.
- LDOS -> density: `LDOS.get_density`.
- LDOS -> DOS: `LDOS.get_density_of_states`.
- Self-consistent Fermi level from LDOS: `LDOS.get_self_consistent_fermi_energy`.

#### Predicted total energy (detailed path)

Total energy and all terms below are in **eV per simulation cell**,
not per atom. The six-term sum represents finite-temperature free energy at
fixed nuclei, including the electronic entropy contribution.

| Term | Meaning |
| --- | --- |
| `e_band` | Occupation-weighted Kohn-Sham eigenvalue sum, $\int \epsilon f(\epsilon)D(\epsilon)\,\mathrm{d}\epsilon$; not kinetic energy alone |
| `e_rho_times_v_hxc` | Double-counting correction, $-\int n(\mathbf{r})[v_H(\mathbf{r})+v_{xc}(\mathbf{r})]\,\mathrm{d}^3r$, including the minus sign |
| `e_hartree` | Hartree electrostatic energy |
| `e_xc` | Exchange-correlation energy returned by the QE total-energy module |
| `e_ewald` | Ion-ion Ewald interaction energy, independent of electronic density |
| `e_entropy_contribution` | Electronic entropy contribution, $-T S_{\mathrm{el}}$, already including temperature; an energy in eV, not entropy in eV/K |
| `v_xc` | Derived scalar $\int n(\mathbf{r})v_{xc}(\mathbf{r})\,\mathrm{d}^3r$, calculated as `-e_rho_times_v_hxc - 2*e_hartree` |

The integrated exchange-correlation potential term (`v_xc` in this table) is
used in MALA papers' energy decomposition, but is not exposed as a standalone
term in MALA's Python energy-contribution dictionary. It is an energy-valued
diagnostic, not a spatial potential field or an additional total-energy
summand. The code returns and sums only the six `e_*` terms:

```text
etot = e_band + e_rho_times_v_hxc + e_hartree
     + e_xc + e_ewald + e_entropy_contribution
```

- Main total-energy assembly (predicted/inferred path):
  `LDOS.get_total_energy` in `mala/targets/ldos.py`;
  `return_energy_contributions=True` also returns the contribution dictionary.
- Band + entropy terms come from DOS integration: via `DOS.get_band_energy` and
  `DOS.get_entropy_contribution` (called from `LDOS.get_total_energy`),
  implemented in `mala/targets/dos.py`; the core integrals are
  `DOS.__band_energy_from_dos` and `DOS.__entropy_contribution_from_dos`.
  `LDOS.get_band_energy` and `LDOS.get_entropy_contribution` also provide these
  terms from LDOS.
- `Density.get_energy_contributions` in `mala/targets/density.py` computes the
  "density-based" terms `e_rho_times_v_hxc`, `e_hartree`, `e_xc`, and
  `e_ewald`.
- Those density terms come from the QE-backed total-energy module:
  `Density.get_energy_contributions` calls `te.get_energies()` after
  `Density.__setup_total_energy_module`; the Fortran binding is subroutine
  `get_energies` in
  `external_modules/total_energy_module/total_energy.f90`.
- `e_ewald` is grouped with density-based terms because it is calculated by
  the TEM. `e_rho_times_v_hxc` maps to QE's `deband` in the Fortran module's
  `get_energies` and `set_rho_of_r` routines.
- The TEM returns density-based energy contributions in Ry;
  `Density.get_energy_contributions` multiplies them by ASE's `Rydberg` to
  return eV. DOS-derived band and entropy contributions are calculated in eV
  directly. `Target.read_additional_calculation_data` likewise converts QE's
  reported total free energy from Ry to eV; its Fermi level comes from ASE in
  eV.
- Explicit `temperature` in `LDOS.get_total_energy` is honored by the
  DOS-derived terms only on the explicit-data path (pass `ldos_data`, or both
  `dos_data` and `density_data`). The cached-data path uses cached observables.
  When the method constructs density internally, it omits forwarding
  temperature; calculate density at the intended temperature and supply it
  explicitly when overriding temperature.
- `Density.__setup_total_energy_module` converts density from `1/A^3` to
  `1/Bohr^3` for QE and may modify the supplied array in place. Save density
  arrays and calculate their native-unit statistics before the TEM call, or
  pass a copy. See `mala/targets/density.py`.

### Density and total-energy-related pieces

- Main density implementation: `mala/targets/density.py` (`class Density`).
- Density integral / electrons: `Density.get_number_of_electrons`.
- Density-derived energy terms: see the
  [total-energy calculation path](#predicted-total-energy-detailed-path).
- Density from LDOS helper: `Density.from_ldos_calculator`.

### DOS calculations

- Main DOS implementation: `mala/targets/dos.py` (`class DOS`).
- DOS-derived band and entropy terms: see the
  [total-energy calculation path](#predicted-total-energy-detailed-path).
- DOS-derived electrons: `DOS.get_number_of_electrons`.
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
  - Deprecated Minterpy descriptors:
    `mala/descriptors/minterpy_descriptors.py`

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
- ASE energy-calculation hook: `MALA.calculate`.
- Additional post-calculation properties: `MALA.calculate_properties`.

### Parameters and global runtime behavior

- Main parameter tree: `mala/common/parameters.py` (`class Parameters`).
- Parameter subsets include network, descriptors, targets, data, running,
  hyperparameter optimization, and data generation.
- Parallel/rank-aware behavior: `mala/common/parallelizer.py`.

## Units and observable conventions

These are native units in the Python target calculators, before any explicit
output conversion. `A` in MALA's unit strings means Å (angstrom).

| Quantity | Native unit / normalization |
| --- | --- |
| LDOS | states/(eV·Å³), unit string `1/(eV*A^3)` |
| DOS | states/eV, unit string `1/eV`; spatially integrated over the whole cell |
| Electronic number density | electrons/Å³, unit string `1/A^3` |
| Number of electrons | Dimensionless electron count for the whole cell |
| Fermi level and LDOS/DOS energy grid | eV |
| Total energy and each energy contribution | See [total-energy units and terms](#predicted-total-energy-detailed-path) |
| Electronic temperature | K |
| ASE-derived voxel vectors and volume | Å and Å³, respectively |

Unit conversion is implemented in `LDOS.convert_units` / `backconvert_units`,
`DOS.convert_units` / `backconvert_units`, and `Density.convert_units` /
`backconvert_units` in `mala/targets/{ldos,dos,density}.py`. Density represents
positive electron number density, not signed charge density. In a
pseudopotential calculation, its integral counts the represented valence
electrons rather than all electrons.

### LDOS file units and conversion

- Standard QE LDOS cube input is in `1/(Ry*Bohr^3)`, as documented in
  `docs/source/basic_usage/more_data.rst` and handled by `LDOS.read_from_cube`.
- `LDOS.from_numpy_file(..., units=...)` passes the declared input unit through
  `PhysicalData.read_from_numpy_file` (`mala/common/physical_data.py`) to
  `LDOS._process_loaded_array`. Conversion to native `1/(eV*A^3)` is performed
  during loading; Ry/Bohr input is multiplied by `1/(Rydberg*Bohr**3)` using
  ASE's unit constants.
- `.npy` files do not encode physical units. `LDOS.from_numpy_file` defaults
  to native `1/(eV*A^3)`, but externally produced arrays can use other units.
  Determine the units from the data's provenance and supply `units=` correctly.
- The standard `DataConverter` NumPy-output path reads and converts the target
  before writing it, so LDOS output is already in native `1/(eV*A^3)`. See
  `DataConverter.__convert_single_snapshot` in
  `mala/datahandling/data_converter.py`. Declaring such an array as Ry/Bohr
  input would convert it a second time.
- `Target.read_additional_calculation_data` builds `voxel` from the ASE cell
  divided by grid dimensions, so its vectors are in Å and its volume in Å³.
  DOS and electron-count spatial integration use this volume.

## API semantics and pitfalls

- For `"espresso-out"`, `Target.read_additional_calculation_data` reconstructs
  electronic `temperature` from the printed Fermi-Dirac smearing width in Ry:
  `temperature = width_Ry * Rydberg / kB`. The printed width is rounded, so the
  reconstructed temperature can differ slightly from the originally intended
  temperature. It is not a precise record of the nominal input temperature.
  See `mala/targets/target.py`.
- `temperature=None` in the LDOS/DOS observable methods uses the calculator's
  stored temperature. For total-energy temperature overrides and TEM density
  handling, see the
  [detailed calculation path](#predicted-total-energy-detailed-path).
- `LDOS.get_self_consistent_fermi_energy` forwards `temperature` only when
  explicit `ldos_data` is supplied; otherwise it uses the cached DOS calculator's
  Fermi level. `DOS.get_self_consistent_fermi_energy(dos_data=...,
  temperature=...)` provides a direct alternative. "Self-consistent" here means
  reproducing the reference electron count, not a self-consistent DFT solve.
- Some LDOS docstrings incorrectly omit the volume factor (claiming `1/eV`)
  or describe the voxel as being in Bohr. For the ASE-derived workflow, use
  `1/(eV*A^3)` and Å as established by the conversion and voxel-construction
  code, respectively; see `LDOS.convert_units` and
  `Target.read_additional_calculation_data`.
- `Runner.parameters` is only `params.running`; the complete configuration is
  `Runner.parameters_full` (`mala/network/runner.py`).
- `DataHandler.add_snapshot()` stores registrations in the supplied
  `Parameters.data.snapshot_directories_list`, and `clear_data()` clears that
  shared list. Use one `Parameters` instance across the standard workflow if
  `save_run()` should preserve registered snapshots.
- `Runner.load_run(..., prepare_data=False)` returns an unprepared
  `DataHandler` and clears snapshot registrations from the returned parameters.
  With `prepare_data=True`, it retains them and calls
  `prepare_data(reparametrize_scaler=False)` with the archived scalers.
- `Runner.load_run(..., load_runner=False)` returns parameters, network, and
  data handler; with `load_runner=True`, it additionally returns the runner.
- `save_run(..., additional_calculation_data=...)` stores calculation metadata
  as `<run_name>.info.json` in zipped runs. `load_run()` restores it into the
  target calculator for prediction and observable postprocessing.
- `Predictor.predict_for_atoms()` calculates descriptors, removes coordinate
  columns, reshapes and input-scales them, then passes the scaled tensor to
  `_forward_snap_descriptors()`. Overrides that return predictions must handle
  output inverse-scaling and physical restriction; MPI calls may also pass
  `local_data_size`.
- Runtime descriptors are calculated from `Parameters.descriptors`; these
  settings must match those used to create the training data.
- `FastTensorDataset.__len__()` uses floor division, so iteration omits samples
  beyond the final complete batch when the sample count is not divisible by the
  batch size (`mala/datahandling/fast_tensor_dataset.py`).
- With snapshot-based splitting, `DataHandler.prepare_data()` accepts training
  plus validation snapshots, or test-only snapshots. Training data without a
  validation snapshot raises an exception.

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
