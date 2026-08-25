
import argparse

import numpy as np

import model_builder as mb
import benchmarks as bench
from optical_data import load_optical, load_reference, WALL_H
from wall import load_wall

import model_optical_compare as moc
from probe_AB_divergence_simple import _band_table_A, _band_table_B

FLOOR = 700.0
LS37_ARTIFACT = 37   # PUP5 hydraulic pressure loss (optical_data docstring)


def _fmt(value, spec):

    return ("{:" + spec + "}").format(value)


def _sh_floored(bands, y_floor=FLOOR):

    y = bands["y_mm"].to_numpy(float)
    chi = bands["chi"].to_numpy(float)
    gam = bands["gamma"].to_numpy(float)

    keep = y >= y_floor
    y_f, chi_p = bench._pad_profile(y[keep], chi[keep])
    y_s, gam_p = bench._pad_profile(y[keep], gam[keep])

    if y_f is not None:
        u_fl = bench._trapz(chi_p * (WALL_H - y_f), y_f)
    else:
        u_fl = np.nan

    if y_s is not None:
        u_sh = bench._trapz(gam_p, y_s)
    else:
        u_sh = np.nan

    d = abs(u_sh) + abs(u_fl)
    if np.isfinite(d) and d > 1e-9:
        return abs(u_sh) / d
    else:
        return np.nan


def run(wall):
    mb.QUIET_SOLVER = True
    opt_all = load_optical(wall)
    ref_opt, mode = load_reference(wall)
    drift = load_wall(wall)["drift_percent"].to_numpy(float)
    n_ls = min(opt_all["LS"].nunique(), len(drift))
    ls_list = list(range(n_ls))

    targets = []
    for i in ls_list:
        targets.append(drift[i])

    mesh, handles = moc.build_calibrated(wall)
    snaps = moc.drive_and_snapshot(mesh, handles, targets, verbose=False)

    print("\n" + "=" * 78)
    print("RELOAD-GAP PROBE — " + wall + "   sh_frac at FLOOR " +
          str(int(FLOOR)) + " mm (mid-height field only)")
    print("=" * 78)
    print("  " + _fmt("ls", ">3") + " " + _fmt("drift%", ">8") + " " +
          _fmt("dir", ">4") + "  " + _fmt("shA@700", ">8") + " " +
          _fmt("shB@700", ">8") + " " + _fmt("|A-B|", ">6") + "   note")
    print("  " + "-" * 60)

    reload_gaps = []
    load_gaps = []

    for idx in range(len(ls_list)):
        ls = ls_list[idx]
        snap = snaps[idx]

        result_A = _band_table_A(ref_opt, snap["x0"], snap["y0"],
                                 snap["ux"], snap["uy"])
        bA = result_A[0]
        result_B = _band_table_B(snap["x0"], snap["y0"],
                                 snap["ux"], snap["uy"])
        bB = result_B[0]

        shA = _sh_floored(bA)
        shB = _sh_floored(bB)

        if np.isfinite(shA) and np.isfinite(shB):
            gap = abs(shA - shB)
        else:
            gap = np.nan

        d = drift[ls]
        if d < -1e-4:
            direction = "neg"
        elif d > 1e-4:
            direction = "pos"
        else:
            direction = "~0"

        note = ""
        if ls >= LS37_ARTIFACT:
            note = "LS37+ hydraulic artifact"
        elif abs(d) > 0.30:
            note = "high drift"

        if direction == "neg" and np.isfinite(gap) and ls < LS37_ARTIFACT:
            reload_gaps.append(gap)
        if direction == "pos" and np.isfinite(gap) and ls < LS37_ARTIFACT:
            load_gaps.append(gap)

        print("  " + _fmt(ls, ">3") + " " + _fmt(d, ">+8.3f") + " " +
              _fmt(direction, ">4") + "  " + _fmt(shA, ">8.3f") + " " +
              _fmt(shB, ">8.3f") + " " + _fmt(gap, ">6.3f") + "   " + note)

    print("\n  Summary (excluding LS37+ artifact):")
    if reload_gaps:
        rg = np.array(reload_gaps)
        n_over = int((rg > 0.10).sum())
        print("    reload (neg) |A-B|@700: mean " + _fmt(rg.mean(), ".3f") +
              "  median " + _fmt(np.median(rg), ".3f") + "  max " +
              _fmt(rg.max(), ".3f") + "  n=" + str(len(rg)) + "  (" +
              str(n_over) + " steps > 0.10)")
    if load_gaps:
        lg = np.array(load_gaps)
        print("    load   (pos) |A-B|@700: mean " + _fmt(lg.mean(), ".3f") +
              "  median " + _fmt(np.median(lg), ".3f") + "  max " +
              _fmt(lg.max(), ".3f") + "  n=" + str(len(lg)))

    print("\n  Verdict guide: if reload mean@700 << 0.10 and only the top "
          "high-drift")
    print("  steps exceed it -> step-specific (finding survives with top "
          "region excluded).")
    print("  If reload mean@700 ~ 0.12 across many steps -> systematic "
          "method limit.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wall")
    args = parser.parse_args()
    run(args.wall)


if __name__ == "__main__":
    main()
