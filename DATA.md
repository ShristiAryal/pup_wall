# Data availability and setup

This repository contains **code only**. The experimental dataset it analyses
is not redistributed here and must be obtained separately from its own
permanent archive.

## Dataset

> Petry, S. and Beyer, K. (2015). *Cyclic Test Data of Six Unreinforced
> Masonry Walls with Different Boundary Conditions* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.8443

- **DOI:** 10.5281/zenodo.8443
- **License:** Creative Commons Attribution-ShareAlike 4.0 International
  (CC BY-SA 4.0)
- **Size:** ~13.3 GB, distributed as ten split archives
  (`Data.part01.rar` … `Data.part10.rar`)
- **Contents:** quasi-static cyclic in-plane tests on six unreinforced
  masonry walls (PUP1–PUP6), including the 312-point photogrammetric (LED)
  field per test unit and the recorded force–displacement histories.

## How to obtain and set up the data

1. Download all ten `Data.partNN.rar` files from the Zenodo record above.
2. Extract them (they form a single multi-part archive; extracting
   `Data.part01.rar` with an archive tool that supports multi-part RAR will
   unpack the whole set).
3. Note the local path to the extracted dataset.
4. Update the hardcoded path constants in the loader scripts to point at
   that path — in particular `optical_data.BASE_DIR` (and the equivalent
   constants in `wall.py`, `recorded_protocol.py`). See the README, §1, for
   the exact files each loader expects.

The code does not download the data automatically; this manual step is the
one part of reproduction that requires local configuration on a new system.

## Licensing note

The **code** in this repository is released under the MIT License (see
`LICENSE`). The **experimental data** is the property of its authors and is
licensed CC BY-SA 4.0 by them; it is neither included in nor relicensed by
this repository. Any derived data files distributed here (the
`results/model_optical_compare_*.csv` comparison tables) are computed from
that dataset and should be understood in that context; cite Petry & Beyer
(2015) for the underlying measurements.
