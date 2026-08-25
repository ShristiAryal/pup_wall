
import argparse

import numpy as np
from scipy.interpolate import griddata

import model_builder as mb
import benchmarks as bench
from optical_data import load_optical, load_reference, N_COLS, WALL_L, WALL_H
from wall import load_wall

import model_optical_compare as moc   # reuse the locked harness pieces


def _fmt(value, spec):

    return ("{:" + spec + "}").format(value)


def _band_table_A(ref_opt, x0m, y0m, uxm, uym):

    lx_opt = ref_opt["x"].to_numpy(float)
    ly_opt = ref_opt["y"].to_numpy(float)
    lx_mod, ly_mod = moc.optical_to_model(lx_opt, ly_opt)

    pts = np.column_stack([x0m, y0m])
    ux_led = griddata(pts, uxm, (lx_mod, ly_mod), method="linear")
    uy_led = griddata(pts, uym, (lx_mod, ly_mod), method="linear")


    ux_led_opt = -ux_led
    uy_led_opt = uy_led

    opt = ref_opt[["C", "R"]].copy()
    opt["x"] = ref_opt["x"].to_numpy(float) + ux_led_opt
    opt["y"] = ref_opt["y"].to_numpy(float) + uy_led_opt
    opt["z"] = 0.0
    opt["LS"] = 0

    ref_df = ref_opt[["C", "R", "x", "y"]].copy()
    ref_df["z"] = 0.0

    X0, Y0 = bench._grid(ref_df)
    XL, YL = bench._grid(opt[opt["LS"] == 0])
    pairs = bench.detect_brick_pairs(ref_df)
    X0v, Y0v = bench.virtual_grid(X0, Y0, pairs)
    Xv, Yv = bench.virtual_grid(XL, YL, pairs)
    toe = bench._global_compressed_toe(X0v, Y0v, Xv, Yv)
    bands = bench.band_quantities(X0v, Y0v, Xv, Yv, global_toe=toe)

    y_f, chi = bench._pad_profile(bands["y_mm"].to_numpy(),
                                  bands["chi"].to_numpy())
    y_s, gam = bench._pad_profile(bands["y_mm"].to_numpy(),
                                  bands["gamma"].to_numpy())

    if y_f is not None:
        u_fl_s = bench._trapz(chi * (WALL_H - y_f), y_f)
    else:
        u_fl_s = np.nan

    if y_s is not None:
        u_sh = bench._trapz(gam, y_s)
    else:
        u_sh = np.nan

    denom = abs(u_sh) + abs(u_fl_s)
    if np.isfinite(denom) and denom > 1e-9:
        sh = abs(u_sh) / denom
    else:
        sh = np.nan

    top_ref_dx = float(np.nanmean((Xv - X0v)[-1]))
    return bands, u_fl_s, u_sh, sh, top_ref_dx


def _accumulate_cell(grid, row, col, new_value):

    if not np.isfinite(grid[row, col]):
        grid[row, col] = new_value
    else:
        grid[row, col] = 0.5 * (grid[row, col] + new_value)


def _band_table_B(x0m, y0m, uxm, uym):

    x0_opt, y0_opt = moc.model_to_optical(x0m, y0m)
    xd_opt, yd_opt = moc.model_to_optical(x0m + uxm, y0m + uym)

    y_levels = np.unique(np.round(y0_opt, 3))
    y_levels = y_levels[np.argsort(y_levels)]

    col_centres = np.linspace(-WALL_L / 2, WALL_L / 2, N_COLS)
    col_edges = np.concatenate([[-np.inf],
                                (col_centres[:-1] + col_centres[1:]) / 2,
                                [np.inf]])

    nR = len(y_levels)
    nC = N_COLS
    X0 = np.full((nR, nC), np.nan)
    Y0 = np.full((nR, nC), np.nan)
    XL = np.full((nR, nC), np.nan)
    YL = np.full((nR, nC), np.nan)

    y_index = {}
    for i in range(len(y_levels)):
        y_index[round(y_levels[i], 3)] = i

    for i in range(len(x0_opt)):
        ri = y_index[round(y0_opt[i], 3)]
        ci = int(np.searchsorted(col_edges, x0_opt[i]) - 1)
        if ci < 0:
            ci = 0
        if ci > nC - 1:
            ci = nC - 1

        _accumulate_cell(X0, ri, ci, x0_opt[i])
        _accumulate_cell(Y0, ri, ci, y0_opt[i])
        _accumulate_cell(XL, ri, ci, xd_opt[i])
        _accumulate_cell(YL, ri, ci, yd_opt[i])

    pairs = []
    for r in range(1, nR + 1):
        pairs.append((r, r))

    X0v, Y0v = bench.virtual_grid(X0, Y0, pairs)
    Xv, Yv = bench.virtual_grid(XL, YL, pairs)
    toe = bench._global_compressed_toe(X0v, Y0v, Xv, Yv)
    bands = bench.band_quantities(X0v, Y0v, Xv, Yv, global_toe=toe)

    y_f, chi = bench._pad_profile(bands["y_mm"].to_numpy(),
                                  bands["chi"].to_numpy())
    y_s, gam = bench._pad_profile(bands["y_mm"].to_numpy(),
                                  bands["gamma"].to_numpy())

    if y_f is not None:
        u_fl_s = bench._trapz(chi * (WALL_H - y_f), y_f)
    else:
        u_fl_s = np.nan

    if y_s is not None:
        u_sh = bench._trapz(gam, y_s)
    else:
        u_sh = np.nan

    denom = abs(u_sh) + abs(u_fl_s)
    if np.isfinite(denom) and denom > 1e-9:
        sh = abs(u_sh) / denom
    else:
        sh = np.nan

    top_ref_dx = float(np.nanmean((Xv - X0v)[-1]))
    return bands, u_fl_s, u_sh, sh, top_ref_dx


def _format_band_cell(row_subset, column, format_spec, nan_width):

    if len(row_subset) == 0:
        return "nan".rjust(nan_width)
    value = row_subset[column].iloc[0]
    if not np.isfinite(value):
        return "nan".rjust(nan_width)
    return ("{:" + format_spec + "}").format(value)


def run(wall):
    mb.QUIET_SOLVER = True
    opt_all = load_optical(wall)
    ref_opt, mode = load_reference(wall)
    wdf = load_wall(wall)
    drift = wdf["drift_percent"].to_numpy(float)
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

    lo = pos_sorted[len(pos_sorted) // 6]
    mid = pos_sorted[len(pos_sorted) // 2]
    hi = pos_sorted[-1]

    if neg_pairs:
        hineg = neg_pairs[0][1]
    else:
        hineg = None

    picks = [("low+", lo), ("mid+", mid), ("high+", hi)]
    if hineg is not None:
        picks.append(("high-", hineg))

    mesh, handles = moc.build_calibrated(wall)

    targets = []
    for label, i in picks:
        targets.append(drift[i])

    snaps = moc.drive_and_snapshot(mesh, handles, targets, verbose=True)

    for idx in range(len(picks)):
        label, ls = picks[idx]
        snap = snaps[idx]

        print("\n" + "=" * 74)
        print(wall + " " + label + "  LS " + str(ls) + " (paper LS" +
              str(ls + 1) + ")  target drift " + _fmt(drift[ls], "+.4f") +
              "%  reached " + _fmt(snap["reached"], "+.4f") + "%")
        print("=" * 74)

        result_A = _band_table_A(ref_opt, snap["x0"], snap["y0"],
                                 snap["ux"], snap["uy"])
        bA, uflA, ushA, shA, topA = result_A

        result_B = _band_table_B(snap["x0"], snap["y0"],
                                 snap["ux"], snap["uy"])
        bB, uflB, ushB, shB, topB = result_B

        print("  A: u_fl_smooth " + _fmt(uflA, "+8.3f") + "  u_sh " +
              _fmt(ushA, "+8.3f") + "  sh_frac " + _fmt(shA, ".3f") +
              "   top-row ref dx " + _fmt(topA, "+.3f") + " mm")
        print("  B: u_fl_smooth " + _fmt(uflB, "+8.3f") + "  u_sh " +
              _fmt(ushB, "+8.3f") + "  sh_frac " + _fmt(shB, ".3f") +
              "   top-row ref dx " + _fmt(topB, "+.3f") + " mm")
        print("  |A-B| sh_frac = " + _fmt(abs(shA - shB), ".3f") +
              "   top-row dx gap = " + _fmt(abs(topA - topB), ".3f") +
              " mm (H3 flag if this ~ drift magnitude)")

        # align band tables by y for side-by-side reading
        print("\n  per-band chi (1/mm) and gamma, A vs B (by height):")
        print("  " + _fmt("y_mm", ">7") + " | " + _fmt("chiA", ">10") +
              " " + _fmt("gamA", ">9") + " " + _fmt("ncA", ">4") + " | " +
              _fmt("chiB", ">10") + " " + _fmt("gamB", ">9") + " " +
              _fmt("ncB", ">4"))

        bA2 = bA.assign(yk=bA["y_mm"].round(-1))
        bB2 = bB.assign(yk=bB["y_mm"].round(-1))
        ys = sorted(set(bA2["yk"]) | set(bB2["yk"]))

        for yk in ys:
            ra = bA2[bA2["yk"] == yk]
            rb = bB2[bB2["yk"] == yk]

            chiA = _format_band_cell(ra, "chi", "10.3e", 10)
            gamA = _format_band_cell(ra, "gamma", "9.3e", 9)
            chiB = _format_band_cell(rb, "chi", "10.3e", 10)
            gamB = _format_band_cell(rb, "gamma", "9.3e", 9)

            if len(ra):
                ncA = int(ra["n_closed"].iloc[0])
            else:
                ncA = 0
            if len(rb):
                ncB = int(rb["n_closed"].iloc[0])
            else:
                ncB = 0

            print("  " + _fmt(yk, "7.0f") + " | " + chiA + " " + gamA +
                  " " + _fmt(ncA, ">4") + " | " + chiB + " " + gamB +
                  " " + _fmt(ncB, ">4"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wall")
    args = parser.parse_args()
    run(args.wall)


if __name__ == "__main__":
    main()
