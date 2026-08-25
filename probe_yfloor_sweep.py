
import argparse

import numpy as np

import model_builder as mb
import benchmarks as bench
from optical_data import load_optical, load_reference, WALL_H
from wall import load_wall

import model_optical_compare as moc
from probe_AB_divergence_simple import _band_table_A, _band_table_B

FLOORS = [0.0, 300.0, 500.0, 700.0]


def _fmt(value, spec):

    return ("{:" + spec + "}").format(value)


def _sh_frac_floored(bands, y_floor):

    y = bands["y_mm"].to_numpy(float)
    chi = bands["chi"].to_numpy(float)
    gam = bands["gamma"].to_numpy(float)

    keep = y >= y_floor
    yk = y[keep]
    chik = chi[keep]
    gamk = gam[keep]

    y_f, chi_p = bench._pad_profile(yk, chik)
    y_s, gam_p = bench._pad_profile(yk, gamk)

    if y_f is not None:
        u_fl_s = bench._trapz(chi_p * (WALL_H - y_f), y_f)
    else:
        u_fl_s = np.nan

    if y_s is not None:
        u_sh = bench._trapz(gam_p, y_s)
    else:
        u_sh = np.nan

    denom = abs(u_sh) + abs(u_fl_s)
    if np.isfinite(denom) and denom > 1e-9:
        sh = abs(u_sh) / denom
    else:
        sh = np.nan

    return u_fl_s, u_sh, sh


def run(wall):
    mb.QUIET_SOLVER = True
    opt_all = load_optical(wall)
    ref_opt, mode = load_reference(wall)
    drift = load_wall(wall)["drift_percent"].to_numpy(float)
    n_ls = min(opt_all["LS"].nunique(), len(drift))

    pos = []
    neg = []
    for i in range(n_ls):
        if drift[i] > 0:
            pos.append(i)
        elif drift[i] < 0:
            neg.append(i)

    pos_pairs = []
    for i in pos:
        pos_pairs.append((drift[i], i))
    pos_pairs.sort()
    pos_sorted = []
    for value, i in pos_pairs:
        pos_sorted.append(i)

    neg_pairs = []
    for i in neg:
        neg_pairs.append((drift[i], i))
    neg_pairs.sort()
    neg_sorted = []
    for value, i in neg_pairs:
        neg_sorted.append(i)

    picks = [
        ("low+", pos_sorted[len(pos_sorted) // 6]),
        ("mid+", pos_sorted[len(pos_sorted) // 2]),
        ("high+", pos_sorted[-1]),
    ]
    if neg_sorted:
        picks.append(("high-", neg_sorted[0]))

    mesh, handles = moc.build_calibrated(wall)

    targets = []
    for label, i in picks:
        targets.append(drift[i])

    snaps = moc.drive_and_snapshot(mesh, handles, targets, verbose=True)

    print("\n" + "=" * 78)
    print("Y-FLOOR SWEEP — " + wall +
          "   sh_frac(A) / sh_frac(B) / |A-B|  per floor")
    print("(prediction: |A-B| shrinks as floor rises if divergence is basal)")
    print("=" * 78)

    header_floor_cols = []
    for f in FLOORS:
        header_floor_cols.append("floor" + _fmt(int(f), ">4"))
    hdr = ("  " + _fmt("step", "6") + " " + _fmt("drift%", ">8") + "  " +
          "  ".join(header_floor_cols))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for idx in range(len(picks)):
        label, ls = picks[idx]
        snap = snaps[idx]

        result_A = _band_table_A(ref_opt, snap["x0"], snap["y0"],
                                 snap["ux"], snap["uy"])
        bA = result_A[0]
        result_B = _band_table_B(snap["x0"], snap["y0"],
                                 snap["ux"], snap["uy"])
        bB = result_B[0]

        cells = []
        for f in FLOORS:
            _, _, shA = _sh_frac_floored(bA, f)
            _, _, shB = _sh_frac_floored(bB, f)
            if np.isfinite(shA) and np.isfinite(shB):
                gap = abs(shA - shB)
            else:
                gap = np.nan
            cells.append(_fmt(shA, ".2f") + "/" + _fmt(shB, ".2f") + "/" +
                        _fmt(gap, ".2f"))

        row_cells = []
        for c in cells:
            row_cells.append(_fmt(c, ">13"))
        print("  " + _fmt(label, "6") + " " + _fmt(drift[ls], "+8.3f") +
              "  " + "  ".join(row_cells))

    print("\n  cell = sh_frac_A / sh_frac_B / |A-B|.  Watch the |A-B| (3rd) "
          "value")
    print("  across floors: collapse toward ~0 at floor 500-700 confirms "
          "the")
    print("  divergence is confined to the base bands (diagnosis H2).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wall")
    args = parser.parse_args()
    run(args.wall)


if __name__ == "__main__":
    main()
