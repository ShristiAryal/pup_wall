
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import model_builder as mb
from blind_predict import build_calibrated, BLIND_BC, CALIB_WALL
import benchmarks as bench
from benchmarks import decompose_drift, MASONRY_ROWS
from optical_data import (load_optical, load_reference, N_COLS, N_ROWS,
                          WALL_L, WALL_H)
from wall import load_wall


try:
    import openseespy.opensees as ops
    _HAVE_OPS = True
except Exception:
    ops = None
    _HAVE_OPS = False


try:
    from scipy.interpolate import griddata
    _HAVE_SCIPY = True
except Exception:
    griddata = None
    _HAVE_SCIPY = False


def _fmt(value, spec):

    return ("{:" + spec + "}").format(value)

def optical_to_model(x_opt, y_opt):
    x_model = WALL_L / 2.0 - np.asarray(x_opt, float)
    y_model = np.asarray(y_opt, float)
    return x_model, y_model

def model_to_optical(x_mod, y_mod):
    x_opt = WALL_L / 2.0 - np.asarray(x_mod, float)
    y_opt = np.asarray(y_mod, float)
    return x_opt, y_opt

def masonry_node_tags(mesh):

    seen = {}
    tags = []
    for t in mesh.nodes:
        x, y = mesh.nodes[t]
        course = t // 1_000_000
        if course < 1 or course > mb.N_COURSES:
            continue    # skip foundation & beam
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen[key] = t
        tags.append(t)
    return tags


def model_field(mesh, tags):

    x0_list = []
    y0_list = []
    ux_list = []
    uy_list = []
    for t in tags:
        x0_list.append(mesh.nodes[t][0])
        y0_list.append(mesh.nodes[t][1])
        ux_list.append(ops.nodeDisp(t, 1))
        uy_list.append(ops.nodeDisp(t, 2))

    x0 = np.array(x0_list, float)
    y0 = np.array(y0_list, float)
    ux = np.array(ux_list, float)
    uy = np.array(uy_list, float)
    return x0, y0, ux, uy

def geometry_gate(mesh, wall, tol_mm=None, verbose=True):

    ref, mode = load_reference(wall)
    ref = ref.dropna(subset=["x", "y"])
    led = ref[ref["R"].isin(MASONRY_ROWS)]

    tags = masonry_node_tags(mesh)
    mx_list = []
    my_list = []
    for t in tags:
        mx_list.append(mesh.nodes[t][0])
        my_list.append(mesh.nodes[t][1])
    mx = np.array(mx_list)
    my = np.array(my_list)

    mx_opt, my_opt = model_to_optical(mx, my)   # model lattice in optical frame
    lx = led["x"].to_numpy(float)
    ly = led["y"].to_numpy(float)

    parity_corr = float(np.corrcoef(mx, mx_opt)[0, 1])
    parity_ok = parity_corr < -0.99

    mapped_x_centre = 0.5 * (mx_opt.min() + mx_opt.max())
    opt_x_centre = 0.5 * (lx.min() + lx.max())
    centre_off = abs(mapped_x_centre - opt_x_centre)
    centre_ok = centre_off < 40.0

    ydir_corr = float(np.corrcoef(my, my_opt)[0, 1])
    ydir_ok = ydir_corr > 0.99

    base_below_top = my_opt[np.argmin(my)] < my_opt[np.argmax(my)]
    base_off = abs(my_opt.min() - ly.min())
    base_ok = base_below_top and base_off < 120.0

    ok = parity_ok and centre_ok and ydir_ok and base_ok

    n_model_cols = len(np.unique(np.round(mx)))
    n_opt_cols = len(np.unique(np.round(lx / 80.0)))
    span_dx = (mx_opt.max() - mx_opt.min()) - (lx.max() - lx.min())
    span_dy = (my_opt.max() - my_opt.min()) - (ly.max() - ly.min())

    if verbose:
        print("[" + wall + "] GEOMETRY GATE (transform x_model = L/2 - "
              "x_opt) — orientation tests:")
        print("    optical ref mode = " + str(mode) + ", masonry LEDs = " +
              str(len(led)) + ", model nodes = " + str(len(tags)))

        if parity_ok:
            parity_word = "OK"
        else:
            parity_word = "FAIL: not reversed"
        print("    (1) x-parity     corr(mx, mx_opt) = " +
              _fmt(parity_corr, "+.4f") + "  [" + parity_word + "]")

        if centre_ok:
            centre_word = "OK"
        else:
            centre_word = "FAIL"
        print("    (2) x-centering  mapped centre " +
              _fmt(mapped_x_centre, "+.1f") + " vs optical " +
              _fmt(opt_x_centre, "+.1f") + " mm (off " +
              _fmt(centre_off, ".1f") + ")  [" + centre_word + "]")

        if ydir_ok:
            ydir_word = "OK"
        else:
            ydir_word = "FAIL: inverted"
        print("    (3) y-direction  corr(my, my_opt) = " +
              _fmt(ydir_corr, "+.4f") + "  [" + ydir_word + "]")

        if base_ok:
            base_word = "OK"
        else:
            base_word = "FAIL"
        print("    (4) base-stays-base  base below top = " +
              str(base_below_top) + ", base off " + _fmt(base_off, ".1f") +
              " mm  [" + base_word + "]")

        print("    context (expected, not judged): model cols ~" +
              str(n_model_cols) + " vs LED cols ~" + str(n_opt_cols) +
              "; span dx " + _fmt(span_dx, "+.0f") + " dy " +
              _fmt(span_dy, "+.0f") + " mm (different grid pitch + base datum)")

        if ok:
            gate_word = "PASS"
            gate_extra = ""
        else:
            gate_word = "FAIL"
            gate_extra = "  <-- TRANSFORM ORIENTATION WRONG; numbers void"
        print("    gate " + gate_word + gate_extra)

    return {
        "ok": ok,
        "parity_corr": parity_corr,
        "centre_off": centre_off,
        "ydir_corr": ydir_corr,
        "base_off": base_off,
        "mode": mode,
        "n_leds": len(led),
        "n_nodes": len(tags),
    }

def procrustes_rotation(x0, y0, x1, y1):

    x0 = np.asarray(x0, float)
    y0 = np.asarray(y0, float)
    x1 = np.asarray(x1, float)
    y1 = np.asarray(y1, float)

    valid = (np.isfinite(x0) & np.isfinite(y0) &
            np.isfinite(x1) & np.isfinite(y1))
    if valid.sum() < 3:
        return np.nan

    x0 = x0[valid]
    y0 = y0[valid]
    x1 = x1[valid]
    y1 = y1[valid]

    # centre both clouds on their own centroid, so only rotation remains
    ax = x0 - x0.mean()
    ay = y0 - y0.mean()
    bx = x1 - x1.mean()
    by = y1 - y1.mean()

    # 2x2 cross-covariance H = A^T B; optimal angle = atan2(Hxy-Hyx, Hxx+Hyy)
    Hxx = np.sum(ax * bx)
    Hxy = np.sum(ax * by)
    Hyx = np.sum(ay * bx)
    Hyy = np.sum(ay * by)

    return float(np.arctan2(Hxy - Hyx, Hxx + Hyy))



def _pivot_ref_positions(ref):
    """optical reference DataFrame -> dict (C,R) -> (x,y) for masonry rows."""
    positions = {}
    for r in ref.itertuples():
        if np.isfinite(r.x) and np.isfinite(r.y):
            positions[(int(r.C), int(r.R))] = (r.x, r.y)
    return positions


def model_field_A(ref, x0m, y0m, uxm, uym):

    lx_opt = ref["x"].to_numpy(float)
    ly_opt = ref["y"].to_numpy(float)
    lx_mod, ly_mod = optical_to_model(lx_opt, ly_opt)

    pts = np.column_stack([x0m, y0m])
    ux_led = griddata(pts, uxm, (lx_mod, ly_mod), method="linear")
    uy_led = griddata(pts, uym, (lx_mod, ly_mod), method="linear")

    ux_led_opt = -ux_led
    uy_led_opt = uy_led

    opt = ref[["C", "R"]].copy()
    opt["x"] = ref["x"].to_numpy(float) + ux_led_opt
    opt["y"] = ref["y"].to_numpy(float) + uy_led_opt
    opt["z"] = 0.0
    opt["LS"] = 0

    ref_df = ref[["C", "R", "x", "y"]].copy()
    ref_df["z"] = 0.0

    return opt[["LS", "C", "R", "x", "y", "z"]], ref_df


def _accumulate_cell(grid, row, col, new_value):

    if not np.isfinite(grid[row, col]):
        grid[row, col] = new_value
    else:
        grid[row, col] = 0.5 * (grid[row, col] + new_value)


def model_field_B(x0m, y0m, uxm, uym):

    x0_opt, y0_opt = model_to_optical(x0m, y0m)
    xd_opt, yd_opt = model_to_optical(x0m + uxm, y0m + uym)

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

    # degenerate self-pairs: each model course row IS a virtual row already
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
        sh_frac = abs(u_sh) / denom
    else:
        sh_frac = np.nan

    th = procrustes_rotation(X0v.ravel(), Y0v.ravel(), Xv.ravel(), Yv.ravel())
    if np.isfinite(th):
        th_mrad = 1e3 * th
    else:
        th_mrad = np.nan

    return sh_frac, th_mrad


def sh_frac_from_decomp(res):

    u_sh = res.get("u_sh_mm", np.nan)
    u_fl_s = res.get("u_fl_smooth_mm", np.nan)
    denom = abs(u_sh) + abs(u_fl_s)
    if not np.isfinite(denom) or denom < 1e-9:
        return np.nan
    return abs(u_sh) / denom

def drive_and_snapshot(mesh, handles, targets_pct, dstep=0.0125,
                       reach_frac=1.0, verbose=True):

    lever = handles["lever"]

    beam_row = []
    for c in range(N_COLS + 1):
        beam_row.append(mb.tag(12, c, 0, 1))

    tags = masonry_node_tags(mesh)

    ops.timeSeries("Linear", 7)
    ops.pattern("Plain", 7, 7)
    ops.load(lever, 1.0, 0.0)

    def u_drift():

        beam_disps = []
        for t in beam_row:
            beam_disps.append(ops.nodeDisp(t, 1))
        return 100.0 * float(np.mean(beam_disps)) / mb.H_EXP

    dstep_pct = dstep / (mb.H_EXP / 100.0)
    reach_tol = reach_frac * dstep_pct
    ramp = mb._REVERSAL_RAMP

    snaps = []
    for i in range(len(targets_pct)):
        target = targets_pct[i]
        since_rev = 0
        stalled = False

        while True:
            cur = u_drift()
            remaining = target - cur
            if abs(remaining) <= reach_tol:
                break

            if remaining > 0:
                sgn = 1.0
            else:
                sgn = -1.0

            if since_rev < len(ramp):
                frac = ramp[since_rev]
            else:
                frac = 1.0

            base_step = dstep * frac
            remaining_mm = abs(remaining) * (mb.H_EXP / 100.0)
            step_mm = sgn * min(base_step, remaining_mm)

            ok = mb._try_step(step_mm, lever)
            if not ok:
                stalled = True
                break
            since_rev = since_rev + 1

        reached = u_drift()
        reached_ok = (abs(reached - target) <= max(reach_tol, 1e-6)
                     and not stalled)
        x0, y0, ux, uy = model_field(mesh, tags)

        snaps.append({
            "ls": i,
            "target": target,
            "reached": reached,
            "reached_ok": bool(reached_ok),
            "x0": x0,
            "y0": y0,
            "ux": ux,
            "uy": uy,
        })

        if verbose:
            if reached_ok:
                flag = ""
            else:
                flag = "  <-- NOT REACHED (flagged)"
            print("    LS " + _fmt(i, "2d") + ": target " +
                  _fmt(target, "+.4f") + "%  reached " +
                  _fmt(reached, "+.4f") + "%" + flag)

    return snaps

def _format_theta_for_display(value):

    if np.isfinite(value):
        return _fmt(value, "+.3f")
    return "  nan"


def _format_sh_frac_for_display(value):

    if np.isfinite(value):
        return _fmt(value, ".3f")
    return " nan"


def compare_wall(wall, dstep=0.0125, max_ls=None, gate_tol=40.0,
                 out_csv=None, verbose=True):

    if not _HAVE_OPS:
        raise RuntimeError("openseespy not importable — run on Windows.")
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy not importable — needed for field "
                           "interpolation (pip install scipy).")

    
    opt_all = load_optical(wall)
    ref_opt, mode = load_reference(wall)

    
    wdf = load_wall(wall)
    drift_tp = wdf["drift_percent"].to_numpy(float)
    n_ls = min(opt_all["LS"].nunique(), len(drift_tp))
    ls_list = list(range(n_ls))
    if max_ls is not None:
        ls_list = ls_list[:max_ls]

    targets = []
    for i in ls_list:
        targets.append(drift_tp[i])

    
    mesh, handles = build_calibrated(wall)
    gate = geometry_gate(mesh, wall, tol_mm=gate_tol, verbose=verbose)

    if verbose:
        print("[" + wall + "] driving " + str(len(targets)) +
              " optical LS holds (dstep=" + str(dstep) + " mm)")
    snaps = drive_and_snapshot(mesh, handles, targets, dstep=dstep,
                               verbose=verbose)

    rows = []
    for snap in snaps:
        ls = snap["ls"]

        opt_res = decompose_drift(wall, ls, opt=opt_all, ref=ref_opt,
                                  verbose=False)
        Xo, Yo = bench._grid(opt_all[opt_all["LS"] == ls])
        X0o, Y0o = bench._grid(ref_opt)
        th_opt = procrustes_rotation(X0o.ravel(), Y0o.ravel(),
                                     Xo.ravel(), Yo.ravel())
        sh_opt = sh_frac_from_decomp(opt_res)


        optA, refA = model_field_A(ref_opt, snap["x0"], snap["y0"],
                                   snap["ux"], snap["uy"])
        resA = decompose_drift(wall, 0, opt=optA, ref=refA, verbose=False)
        XA, YA = bench._grid(optA[optA["LS"] == 0])
        X0A, Y0A = bench._grid(refA)
        th_A = procrustes_rotation(X0A.ravel(), Y0A.ravel(),
                                   XA.ravel(), YA.ravel())
        sh_A = sh_frac_from_decomp(resA)

        try:
            sh_B, th_B = model_field_B(snap["x0"], snap["y0"],
                                       snap["ux"], snap["uy"])
        except Exception as e:
            sh_B = np.nan
            th_B = np.nan
            if verbose:
                print("    LS " + _fmt(ls, "2d") +
                      ": model-B decomposition skipped (" + str(e) + ")")

        if np.isfinite(th_opt):
            theta_base_opt_mrad = 1e3 * th_opt
        else:
            theta_base_opt_mrad = np.nan

        if np.isfinite(th_A):
            theta_base_modelA_mrad = 1e3 * th_A
        else:
            theta_base_modelA_mrad = np.nan

        rows.append({
            "wall": wall,
            "ls": ls,
            "paper_ls": ls + 1,
            "drift_target": snap["target"],
            "drift_reached": snap["reached"],
            "reached_ok": snap["reached_ok"],
            "theta_base_opt_mrad": theta_base_opt_mrad,
            "theta_base_modelA_mrad": theta_base_modelA_mrad,
            "theta_base_modelB_mrad": th_B,   # already mrad from model_field_B
            "sh_frac_opt": sh_opt,
            "sh_frac_modelA": sh_A,
            "sh_frac_modelB": sh_B,
            "u_top_opt_mm": opt_res.get("u_top_optical_mm", np.nan),
            "disp_tp_mm": opt_res.get("disp_tp_mm", np.nan),
        })

    tbl = pd.DataFrame(rows)

    if verbose:
        print("\n" + "=" * 78)
        if gate["ok"]:
            gate_word = "PASS"
        else:
            gate_word = "FAIL"
        print("MODEL vs OPTICAL — " + wall + "  (gate " + gate_word +
              ", ref " + str(mode) + ")")
        print("theta_base in mrad; sh_frac = u_sh/(u_sh+u_fl_smooth), "
              "remainder only (rock excluded)")
        print("=" * 78)

        show = tbl.copy()
        for c in ("theta_base_opt_mrad", "theta_base_modelA_mrad",
                  "theta_base_modelB_mrad"):
            show[c] = show[c].map(_format_theta_for_display)
        for c in ("sh_frac_opt", "sh_frac_modelA", "sh_frac_modelB"):
            show[c] = show[c].map(_format_sh_frac_for_display)

        cols = ["ls", "drift_target", "drift_reached", "reached_ok",
                "theta_base_opt_mrad", "theta_base_modelA_mrad",
                "theta_base_modelB_mrad",
                "sh_frac_opt", "sh_frac_modelA", "sh_frac_modelB"]
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(show[cols].to_string(index=False))

        n_bad = int((~tbl["reached_ok"]).sum())
        if n_bad:
            print("\nNOTE: " + str(n_bad) + " LS not reached within tol — "
                  "rows flagged reached_ok=False (documented, not dropped).")

    if out_csv is None:
        out_csv = mb.FIG_DIR.parent / ("model_optical_compare_" + wall + ".csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(out_csv, index=False)
    if verbose:
        print("\n[" + wall + "] table saved: " + str(out_csv))

    return tbl, gate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wall", help="PUP2 or PUP3 (first harness set)")
    parser.add_argument("--dstep", type=float, default=0.0125)
    parser.add_argument("--max-ls", type=int, default=None,
                        help="limit to first N optical LS holds (smoke test)")
    parser.add_argument("--gate-tol", type=float, default=40.0,
                        help="(vestigial: gate now uses orientation tests "
                             "with internal thresholds, not a single "
                             "tolerance)")
    parser.add_argument("--gate-only", action="store_true",
                        help="run the transform geometry gate and exit")
    parser.add_argument("--verbose-solver", action="store_true",
                        help="show OpenSees convergence firehose (default "
                             "quiet)")
    args = parser.parse_args()

    valid_walls = list(BLIND_BC) + [CALIB_WALL]
    if args.wall not in valid_walls:
        raise SystemExit(
            args.wall + ": choose PUP2 (calibration) or a blind wall " +
            str(list(BLIND_BC)) + ".")

    mb.QUIET_SOLVER = not args.verbose_solver
    if args.verbose_solver:
        solver_log_word = "VERBOSE"
    else:
        solver_log_word = "QUIET"
    print("[solver log] " + solver_log_word)

    if args.gate_only:
        mesh, handles = build_calibrated(args.wall)
        geometry_gate(mesh, args.wall, tol_mm=args.gate_tol, verbose=True)
        return

    compare_wall(args.wall, dstep=args.dstep, max_ls=args.max_ls,
                gate_tol=args.gate_tol)


if __name__ == "__main__":
    main()
