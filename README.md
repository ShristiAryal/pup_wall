# Simplified Micro-Model of the PUP Unreinforced Masonry Walls in OpenSeesPy

This repository contains the modelling, calibration, blind-prediction, and
verification codebase used to build and test a two-dimensional finite-element
(quad + zero-length interface spring) model of the six unreinforced masonry
(URM) wall specimens PUP1–PUP6, and to compare that model's displacement
field against the walls' photogrammetric (LED) measurements.

The model is calibrated on a single wall (PUP2) and driven, without further
parameter adjustment, through three further walls (PUP3, PUP4, PUP5) using
each wall's own recorded boundary conditions and loading history ("blind
prediction"). PUP1 (double-fixed) and PUP6 (cantilever) are deferred: the
current lever construction cannot represent their boundary conditions, and
they are gated off in code rather than run through an invalid representation
(see §3.3 and §8). All numerical results in this repository are
reproducible from the scripts and data described below.

---

## 1. Data source

All experimental data used here originate from:

> Petry, S. and Beyer, K. (2015). *Cyclic Test Data of Six Unreinforced
> Masonry Walls with Different Boundary Conditions [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.8443

The dataset comprises quasi-static cyclic in-plane tests on six identical
URM wall specimens (PUP1–PUP6), constructed from hollow clay brick units and
cement-based mortar, tested under differing axial load and top rotational
restraint. Wall deformation was recorded with a 312-point digital
photogrammetric (LED) system. The dataset is distributed under CC BY-SA 4.0.

This repository does not redistribute the raw dataset. To reproduce any
result, download the dataset from the DOI above and place it locally; the
loader scripts (`wall.py`, `optical_data.py`, `recorded_protocol.py`) expect
the following inputs, matched to the dataset's own folder structure:

| Script | Expects |
|---|---|
| `wall.py` | `{wall}/processed_data/{wall}_conventional_at_LS.asc` (via `optical_data.load_conventional_at_ls`) |
| `optical_data.py` | `{wall}/processed_data/{wall}_LED_C{cc}_R{rr}_at_LS.asc` (312 files per wall) and the continuous per-marker and conventional recordings |
| `recorded_protocol.py` | `PUP_force_displ_hystereses_incl_vert_displ.xls` (one sheet per wall) |

`optical_data.BASE_DIR` and `FIG_DIR`, and the equivalent paths in the other
scripts, are hardcoded to a local filesystem location in the version of the
code as written. Before running anything, update these path constants (or
refactor them to read from an environment variable / config file) to point
at your own copy of the dataset.** This is the one step of reproduction
that requires manual configuration on a new machine.

---

## 2. Repository layout

All scripts live in the repository and are run from there.

PUP_WALL/
  # core model + analysis
  wall.py
  optical_data.py
  recorded_protocol.py
  interface_laws.py
  model_builder.py
  blind_predict.py
  benchmarks.py
  model_optical_compare.py
  stage2_recorded_harness.py
  mesh_sensitivity.py
  # diagnostics (probes)
  probe_AB_divergence.py
  probe_yfloor_sweep.py
  probe_reload_gap.py
  # figure scripts
  fig6_pup3_monotonic_vs_cyclic.py
  fig7_theta_base.py
  fig8_sh_frac.py
  fig9_pup5_inversion.py
  #reference inputs
  material_properties.csv     # brick/mortar/masonry material inputs (E, G,
                              #   f_t, c, mu_peak, mu_res, G_fI, G_fII, damage
                              #   parameters, ...), read by interface_laws.py
  wall_test_matrix.csv        # published per-wall summary (peak force, peak
                              #   drift, ...), read by benchmarks.py as an
                              #   external cross-check
  # outputs
  results/
    model_optical_compare_PUP2.csv   # per-wall comparison tables 
    model_optical_compare_PUP3.csv
    model_optical_compare_PUP5.csv
  # --- meta ---
  README.md
  DATA.md
  requirements.txt
  LICENSE
```

All scripts are run from the repository root (e.g. `python blind_predict.py
PUP4`). The reference CSVs (`material_properties.csv`, `wall_test_matrix.csv`)
sit alongside the scripts; the comparison CSVs are
written to and read from `results/`. The raw Zenodo dataset not
included and must be downloaded separately and pointed at via the path
constants noted above.

---

## 3. What each script does

### 3.1 Data loaders

| Script | Purpose |
|---|---|
| `wall.py` | Loads one wall's `_at_LS` conventional-channel data (`Drift_TP` → `drift_percent`, `Disp_TP` → `top_plate_disp_mm`, `F1` → `force_horizontal_kN`). Row i of the loaded table corresponds to paper load step LS(i+1); the pre-axial-load state (LS0) is not present in these files. |
| `optical_data.py` | Loads the 312-marker photogrammetric field per wall (13 columns × 24 rows) into a tidy `(LS, C, R, x, y, z)` table, and constructs the LS0 reference configuration per wall (a pre-axial-ramp average for PUP1/3/4/5/6; the first `_at_LS` row for PUP2, whose optical recording starts mid-ramp). Documents two known data-quality issues used downstream: a hydraulic pressure-loss artifact at LS37 on PUP5, and a missing 0.15%/0.25% drift level on PUP1's protocol. |
| `recorded_protocol.py` | Reads the EPFL hysteresis workbook (`PUP_force_displ_hystereses_incl_vert_displ.xls`), one sheet per wall, auto-detecting each sheet's header length (it differs between PUP2 and the others). Also extracts the reversal (peak) indices of a drift history. |

### 3.2 Material law and mesh

| Script | Purpose |
|---|---|
| `interface_laws.py` | Builds the trilinear tension (Mode-I) and shear (Mode-II, cohesive–frictional with softening to a residual friction plateau) backbones used for every mortar-joint spring, from the parameters in `material_properties.csv`, and constructs the corresponding OpenSees `Hysteretic` materials. Includes a single-joint (2-node, zeroLength) test rig used to verify each backbone in isolation (peak stress, fracture energy, residual plateau) before it is used in the wall model, and an alternative "Option B" hybrid shear law (a cohesive spring in parallel with a friction-floor spring). Runnable standalone (`python interface_laws.py`) to reproduce these single-joint checks and plots. |
| `model_builder.py` | Builds the wall mesh (elastic quad elements for brick units and the loading beam, running-bond brick layout with half-unit bricks at alternating course ends) and connects courses/bricks with zero-length interface springs (families: `base`, `bed`, `head`, `middle`), whose stiffnesses are derived from measured brick/masonry moduli via a homogenisation relation. Runs a two-stage build per wall: (1) build with provisional joint strengths and apply gravity + axial load; (2) harvest the resulting compressive stress at every joint and rebuild with area-consistent joint strengths. Applies the lateral load through a rigid "lever" arm positioned at the wall's experimental load-application height, and drives it cyclically through a target displacement history with an adaptive step/retry scheme. |
| `mesh_sensitivity.py` | Two-axis mesh-convergence study on PUP2 (vertical refinement of each course into stacked sub-rows; horizontal refinement of each half-module column into sub-columns), gated to reproduce the baseline mesh exactly at unit subdivision. Confirms that refinement changes the peak base shear by at most ~0.5% and the initial stiffness by at most ~2.5%, i.e. that the reported response is not an artifact of interface density (manuscript §3.1). |

### 3.3 Calibration, blind prediction, and drift decomposition

| Script | Purpose |
|---|---|
| `blind_predict.py` | Drives the PUP2-calibrated, **frozen** model (frozen configuration described in the accompanying manuscript, §4–§5) through each of the other walls' own recorded load history and own boundary conditions, injecting only the experimentally-known axial stress and load-height ratio per wall — never a material or damage parameter. PUP1 (double-fixed top) and PUP6 (cantilever, top moment = 0) are explicitly gated off with a raised `NotImplementedError`, because the current lever construction cannot represent either boundary condition; running them would silently produce a physically invalid result. |
| `benchmarks.py` | (1) Extracts the cyclic force–drift backbone (envelope) per direction from each wall's conventional data, and cross-checks the resulting peaks against `wall_test_matrix.csv`. (2) Decomposes the optically-measured (or model-predicted) top displacement into a flexural and a shear component at a given load step, by pairing LED rows into virtual brick rows, determining the wall-wide compressed toe, fitting curvature and shear strain band by band, and integrating over height. |
| `model_optical_compare.py` | For a given wall: drives the frozen model through that wall's own **optical** load-step drift sequence (rather than the conventional-channel one used by `blind_predict.py`), and at every step, compares the model's displacement field to the optical field on two derived quantities — the whole-wall rigid base rotation `theta_base` (orthogonal Procrustes fit) and the shear fraction `sh_frac` of the deformational remainder (via `benchmarks.decompose_drift`, rocking excluded). The model field is constructed two ways (`model-A`: interpolated onto the optical LED grid; `model-B`: the model's own lattice, mapped into the optical coordinate frame), so that A can be reported as the headline comparison and B as a resolution-sensitivity check. A geometry gate verifies the model↔optical coordinate transform's orientation before any of these numbers are trusted. Writes `model_optical_compare_<wall>.csv` (see §6). |

### 3.4 Diagnostics

| Script | Purpose |
|---|---|
| `stage2_recorded_harness.py` | Reproduces, as a fixed test case, a driver non-convergence ("stall") that occurs when the calibrated PUP2 model is driven through its true recorded (34-peak) cyclic protocol rather than the simplified 4-peak calibration schedule. Used to test candidate driver fixes (finer displacement step, gentler post-reversal ramp) against both the full recorded protocol and a 4-peak regression-guard schedule that must remain convergent after any change. |
| `probe_AB_divergence.py` | Investigates why `sh_frac` from `model-A` and `model-B` agree at low/mid drift but diverge at high drift, by reporting the full per-band curvature/shear-strain table for both constructions at selected drift levels. |
| `probe_yfloor_sweep.py` | Confirms the `probe_AB_divergence.py` diagnosis by re-integrating `sh_frac` for both constructions after excluding all bands below a rising height floor (0, 300, 500, 700 mm), to test whether the A/B disagreement is confined to the lowest 1–2 bands. |
| `probe_reload_gap.py` | Follow-up specific to PUP5: checks whether a residual A/B gap on the reload (negative-drift) half-cycles, which does not close even at a 700 mm floor, is confined to a few steps (including the LS37 hydraulic-artifact region) or is systematic across all reload steps. |

The diagnostics in `stage2_recorded_harness.py` and the `probe_*.py` files
do not alter `model_builder.py`, `interface_laws.py`, or any calibrated
parameter. All masking or flooring is applied locally inside the diagnostic
script, on a copy of the output.

### 3.5 Figures

| Script | Purpose |
|---|---|
| `fig6_pup3_monotonic_vs_cyclic.py` | PUP3 monotonic pushover vs. the recorded cyclic response, isolating the loaded-side under-prediction as damage-independent (the loaded side is reproduced at zero damage; the reload side collapses only under the cyclic history). |
| `fig7_theta_base.py` | Three-panel figure: optical vs. model-A `theta_base` (absolute drift on the x-axis) for PUP2, PUP3, PUP5. Reads the three `model_optical_compare_<wall>.csv` files directly; does not re-run OpenSees. |
| `fig8_sh_frac.py` | Two-panel figure: full-field `sh_frac`, optical vs. model-A, against load-step index, for PUP2 and PUP3. |
| `fig9_pup5_inversion.py` | PUP5-specific figure: `sh_frac` vs. signed drift, showing the inverted (relative to optical) directional response on the reload half-cycle, with the LS37 artifact and the high-drift loaded steps shown greyed out and excluded from the directional claim. |

---

## 4. Modelling pipeline

1. **`interface_laws.py`** — verify the joint material laws in isolation
   (single-joint rig) before they are used in any wall model.
2. **`model_builder.py`** — build the mesh and material stiffnesses; not
   normally run standalone, but imported by every script below.
3. **`blind_predict.py`** (or the calibration build inside
   `model_optical_compare.py`) — run the PUP2 calibration build, and the
   blind (no-parameter-change) builds for PUP3, PUP4, PUP5. PUP1 and PUP6
   are gated off (see §3.3).
4. **`benchmarks.py`** — extract backbones from the conventional data, and
   provide the drift-decomposition routine used by both the optical
   post-processing and the model comparison.
5. **`model_optical_compare.py`** — drive the frozen model through each
   wall's optical load-step sequence and produce the comparison CSVs.
6. **`fig6_pup3_monotonic_vs_cyclic.py`, `fig7_theta_base.py`,
   `fig8_sh_frac.py`, `fig9_pup5_inversion.py`** — produce the manuscript
   figures. (`fig7`–`fig9` read the comparison CSVs from step 5;
   `fig6` re-runs a PUP3 monotonic pushover against the recorded cyclic
   response.)

The `stage2_recorded_harness.py`, `mesh_sensitivity.py`, and `probe_*.py`
scripts are diagnostics/convergence checks that were used during development
to establish driver robustness, mesh independence, and the resolution limits
of the drift decomposition; they are not required to reproduce steps 1–6
above, but are included for transparency, since they are referenced in the
accompanying manuscript's method and discussion sections.

---

## 5. Environment

Required packages (see `requirements.txt`): `openseespy`, `numpy`, `pandas`,
`scipy`, `matplotlib`, `xlrd`. All results reported in the accompanying
manuscript were produced and confirmed on Windows under **Python 3.12.10**
and **OpenSeesPy 3.8.0.0**; `requirements.txt` records the pinned versions of
the remaining packages. The cyclic driver is deterministic — identical runs
reproduce to the last printed digit.

Example commands (run from the repository root):

```bash
# 1. Verify the joint material laws in isolation
python interface_laws.py

# 2. Blind-predict one wall (no parameters moved)
python blind_predict.py PUP4

# 3. List boundary conditions / gating status for all walls
python blind_predict.py --list

# 4. Backbone check + drift-decomposition smoke test
python benchmarks.py

# 5. Compare the frozen model to the optical field for one wall
python model_optical_compare.py PUP3

# 6. Regenerate the manuscript figures (run after step 5 for each wall)
python fig7_theta_base.py
python fig8_sh_frac.py
python fig9_pup5_inversion.py
```

The exact flag names (`--list`, `--gate-only`, `--max-ls`, `--verbose-solver`,
etc.) are documented in each script's own `argparse` definitions and module
docstring.

---

## 6. Output files: `model_optical_compare_<wall>.csv`

Produced by `model_optical_compare.py`, one row per driven load step (`ls`).
Columns, as written by the script:

| Column | Meaning |
|---|---|
| `wall` | Wall identifier (e.g. `PUP2`) |
| `ls` | 0-based load-step index (paper load step is `ls + 1`) |
| `paper_ls` | 1-based paper load-step number |
| `drift_target` | Commanded drift for this step (%), from the wall's own optical load-step sequence |
| `drift_reached` | Drift actually reached by the driver (%) |
| `reached_ok` | `True` if the driver reached `drift_target` within tolerance without stalling |
| `theta_base_opt_mrad` | Rigid base rotation from the optical field (mrad) |
| `theta_base_modelA_mrad` | Rigid base rotation, model field interpolated onto the optical LED grid (mrad) |
| `theta_base_modelB_mrad` | Rigid base rotation, model's own lattice mapped into the optical frame (mrad) |
| `sh_frac_opt` | Shear fraction of the deformational remainder, optical field |
| `sh_frac_modelA` | Shear fraction, model-A |
| `sh_frac_modelB` | Shear fraction, model-B |
| `u_top_opt_mm` | Top-row horizontal displacement, optical field (mm) |
| `disp_tp_mm` | Top-plate displacement, conventional instrument channel (mm) |

Row counts in the three files provided alongside this repository:
`model_optical_compare_PUP2.csv` — 34 rows; `model_optical_compare_PUP3.csv`
— 43 rows; `model_optical_compare_PUP5.csv` — 38 rows. These match the
number of `_at_LS` rows documented for each wall in `wall.py`.

---

## 7. Known data caveats carried through the codebase

These are documented in `optical_data.py` and used by `benchmarks.py`,
`model_optical_compare.py`, and `fig9_pup5_inversion.py`:

- **PUP5, LS37**: a hydraulic pressure-loss event during the test produces
  an artifact in the last recorded hysteresis loop; it is excluded from
  quantitative comparisons rather than interpreted as strength degradation.
- **PUP1**: a cycle at nominal 0.2% drift accidentally overshot to 0.3%; the
  protocol has no recorded 0.15% or 0.25% drift level.
- **PUP2**: the optical recording starts mid-way through the axial-load
  ramp; the reference configuration for PUP2 is therefore its first
  `_at_LS` row (paper LS1), not a genuine pre-load state as for the other
  five walls.

---

## 8. Anti-refitting discipline

No material, damage, or stiffness parameter changes once PUP2 calibration
is frozen. The only per-wall inputs for PUP3, PUP4, and PUP5 are the recorded
axial stress and load-height ratio (`sigma0`, `H0_over_H`). PUP1 and PUP6 are
refused rather than run through a boundary-condition representation that is
invalid for them. Both rules are enforced in `blind_predict.py` via the
`BLIND_BC` table and a `NotImplementedError` guard.

---

## 9. License and citation

- **Code in this repository**: MIT License (see `LICENSE`).
- **Experimental data**: not redistributed here; cite Petry & Beyer (2015),
  DOI [10.5281/zenodo.8443](https://doi.org/10.5281/zenodo.8443),
  CC BY-SA 4.0.

Author: Shristi Aryal.
