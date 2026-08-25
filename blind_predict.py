
import argparse
from pathlib import Path

import numpy as np

import model_builder as mb
from interface_laws import load_params
from recorded_protocol import load_recorded_protocol
from stage2_recorded_harness import significant_peaks

BLIND_BC = {
    "PUP1": {
        "sigma0": 1.055,
        "H0_over_H": 0.50,
        "bc": "double-fixed",
        "lever_ok": False,
        "note": "H0/H=0.50: lever node below beam; needs double-fixed BC",
    },
    "PUP3": {
        "sigma0": 1.055,
        "H0_over_H": 1.50,
        "bc": "lever @ 1.5H",
        "lever_ok": True,
        "note": "flexural rocking; drop-in",
    },
    "PUP4": {
        "sigma0": 1.524,
        "H0_over_H": 1.50,
        "bc": "lever @ 1.5H",
        "lever_ok": True,
        "note": "hybrid; drop-in",
    },
    "PUP5": {
        "sigma0": 0.527,
        "H0_over_H": 0.75,
        "bc": "lever @ 0.75H",
        "lever_ok": True,
        "note": "low axial (0.09); expect worst tracking (Decision 015)",
    },
    "PUP6": {
        "sigma0": 0.996,
        "H0_over_H": float("inf"),
        "bc": "cantilever (M_top=0)",
        "lever_ok": False,
        "note": "H0/H=inf: cantilever; needs top free-rotation BC",
    },
}


CALIB_WALL = "PUP2"
DEFAULT_XLS = (Path(__file__).resolve().parents[1] / "data" /
               "PUP_force_displ_hystereses_incl_vert_displ.xls")


RECORDED_DSTEP = 0.0125       
RECORDED_REACH_FRAC = 1.0     

DEFAULT_THRESH = 0.02


def build_calibrated(wall, shear_law="B"):
  
    if wall != CALIB_WALL:
        bc = BLIND_BC[wall]
        if not bc["lever_ok"]:
            raise NotImplementedError(
                wall + ": " + bc["bc"] + " — " + bc["note"] + ". The "
                "current lever construction cannot represent this BC. "
                "This wall needs a dedicated boundary-condition "
                "implementation (tracked as separate work). Refusing to "
                "run it through the lever harness so a broken geometry "
                "cannot produce a fake curve.")
        
        mb.WALL_CONFIG[wall] = {
            "sigma0": bc["sigma0"],
            "H0_over_H": bc["H0_over_H"],
        }


    params = load_params()
    K = mb.derive_stiffnesses(params, verbose=False)
    mesh = mb.build_geometry()


    handles_stage1 = mb.build_opensees_model(
        mesh, params, K, wall=wall, sigma_map=None, shear_law=shear_law)
    mb.apply_gravity(mesh, handles_stage1["cfg"])
    sigma_map, _ = mb.harvest_sigma(mesh, handles_stage1)


    handles_stage2 = mb.build_opensees_model(
        mesh, params, K, wall=wall, sigma_map=sigma_map, shear_law=shear_law)
    mb.apply_gravity(mesh, handles_stage2["cfg"])

    return mesh, handles_stage2


def blind_run(wall, thresh=DEFAULT_THRESH, xls_path=None, shear_law="B",
              dstep=RECORDED_DSTEP, max_steps=None):
    
    if xls_path:
        xls_path = Path(xls_path)
    else:
        xls_path = DEFAULT_XLS

    if max_steps is not None:
        mb.MAX_STEPS = int(max_steps)

    print("[" + wall + "] driver: dstep=" + str(dstep) + " mm  reach_frac=" +
          str(RECORDED_REACH_FRAC) + "  MAX_STEPS=" + str(mb.MAX_STEPS))


    rec = load_recorded_protocol(wall, xls_path)
    exp_drift = rec["drift_pct"]
    exp_force = rec["force_kN"]

    
    peaks, _ = significant_peaks(exp_drift, thresh=thresh)
    if len(peaks) and abs(peaks[0]) < 0.01:
        peaks = peaks[1:]     

    print("[" + wall + "] protocol: " + str(len(peaks)) +
          " significant peaks (" + str(round(float(peaks.min()), 3)) +
          ".." + str(round(float(peaks.max()), 3)) + "%), thresh=" +
          str(thresh) + "  | experimental peak|F| = " +
          str(round(rec["peak_abs_force"], 1)) + " kN")


    mesh, handles = build_calibrated(wall, shear_law=shear_law)
    drift, shear, status, log = mb.cyclic(
        mesh, handles, peaks.tolist(),
        dstep=dstep, reach_frac=RECORDED_REACH_FRAC)

    if len(drift):
        reached = float(np.max(np.abs(drift)))
    else:
        reached = float("nan")

    if len(shear):
        Vmax = float(np.max(shear))
        Vmin = float(np.min(shear))
    else:
        Vmax = float("nan")
        Vmin = float("nan")

    complete = (status == "sequence complete")


    exp_Vmax = float(np.nanmax(exp_force))
    exp_Vmin = float(np.nanmin(exp_force))

    if exp_Vmax:
        peak_err_pos = 100.0 * (Vmax - exp_Vmax) / exp_Vmax
    else:
        peak_err_pos = float("nan")

    if exp_Vmin:
        peak_err_neg = 100.0 * (abs(Vmin) - abs(exp_Vmin)) / abs(exp_Vmin)
    else:
        peak_err_neg = float("nan")

    print("[" + wall + "] drive: " + status + "  |  reached |drift| " +
          str(round(reached, 3)) + "%")
    print("[" + wall + "] model  V+ " + str(round(Vmax, 1)) +
          "  V- " + str(round(Vmin, 1)) + " kN")
    print("[" + wall + "] exp    V+ " + str(round(exp_Vmax, 1)) +
          "  V- " + str(round(exp_Vmin, 1)) + " kN")

    if complete:
        completion_word = "COMPLETE"
    else:
        completion_word = "STALLED"

    print("[" + wall + "] peak error (recorded protocol): +side " +
          str(round(peak_err_pos, 1)) + "%   -side " +
          str(round(peak_err_neg, 1)) + "%   " + completion_word)

    return {
        "wall": wall,
        "model_drift": drift,
        "model_shear": shear,
        "exp_drift": exp_drift,
        "exp_force": exp_force,
        "status": status,
        "complete": complete,
        "reached": reached,
        "Vmax": Vmax,
        "Vmin": Vmin,
        "exp_Vmax": exp_Vmax,
        "exp_Vmin": exp_Vmin,
        "peak_err_pos": peak_err_pos,
        "peak_err_neg": peak_err_neg,
        "peaks": peaks,
        "log": log,
    }


def plot_overlay(res, fname=None):
   
    import matplotlib.pyplot as plt

    wall = res["wall"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(res["exp_drift"], res["exp_force"], "-", color="0.6", lw=1.0,
            label=wall + " experiment (recorded)")
    ax.plot(res["model_drift"], res["model_shear"], "-", color="crimson",
            lw=1.4, label="blind model (frozen config)")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("drift [%]")
    ax.set_ylabel("base shear V [kN]")

    if res["complete"]:
        tag = "COMPLETE"
    else:
        tag = "STALLED @ " + str(round(res["reached"], 3)) + "%"

    title = (wall + " blind prediction — " + tag + "\n" +
             "peak err (recorded): +" + str(round(res["peak_err_pos"], 1)) +
             "% / " + str(round(res["peak_err_neg"], 1)) +
             "%   [NO parameters tuned]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    if fname is None:
        fname = mb.FIG_DIR / ("blind_" + wall + "_overlay.png")
    Path(fname).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("[" + wall + "] figure saved: " + str(fname))
    return fname


def print_table():
    
    print("Blind-prediction BC table")
    print("  calibration wall: " + CALIB_WALL + " (not a blind target)\n")

    header = ("  wall  sig0[MPa]   H0/H  runnable  bc / note")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for wall_name in BLIND_BC:
        bc = BLIND_BC[wall_name]

        if bc["H0_over_H"] == float("inf"):
            h0_over_h_text = "inf"
        else:
            h0_over_h_text = str(round(bc["H0_over_H"], 2))

        if bc["lever_ok"]:
            runnable_text = "YES"
        else:
            runnable_text = "BC-work"

        row = ("  " + wall_name.ljust(5) + " " +
               str(round(bc["sigma0"], 3)).rjust(9) + " " +
               h0_over_h_text.rjust(6) + "  " +
               runnable_text.rjust(8) + "  " +
               bc["bc"] + " — " + bc["note"])
        print(row)

    print("\n  Drop-in (smoke-test one of these first): PUP3, PUP4, PUP5")
    print("  BC re-derivation needed:  PUP1, PUP6")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wall", nargs="?", help="PUP1/PUP3/PUP4/PUP5/PUP6")
    parser.add_argument("--thresh", type=float, default=DEFAULT_THRESH,
                        help="significant-reversal threshold %% (per-wall)")
    parser.add_argument("--xls", default=None,
                        help="path to the hysteresis .xls")
    parser.add_argument("--dstep", type=float, default=RECORDED_DSTEP,
                        help="driver step size mm (default frozen 0.0125; "
                             "coarsen e.g. 0.025 for ~2x faster peak-only "
                             "runs)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="driver step ceiling (default harness value; "
                             "raise e.g. 80000 for long protocols like PUP3)")
    parser.add_argument("--no-fig", action="store_true",
                        help="skip overlay figure")
    parser.add_argument("--list", action="store_true",
                        help="print BC table and exit")
    parser.add_argument("--verbose", action="store_true",
                        help="show OpenSees solver convergence warnings "
                             "(default: quiet)")
    args = parser.parse_args()

    if args.list or not args.wall:
        print_table()
        return

    if args.wall not in BLIND_BC:
        raise SystemExit(
            args.wall + " is not a blind wall. Choose from " +
            str(list(BLIND_BC.keys())) + " (PUP2 is calibration).")

 
    mb.QUIET_SOLVER = not args.verbose

    if args.verbose:
        solver_log_word = "VERBOSE"
    else:
        solver_log_word = "QUIET"
    print("[solver log] " + solver_log_word + " (mb.QUIET_SOLVER=" +
          str(mb.QUIET_SOLVER) + ")")

    res = blind_run(args.wall, thresh=args.thresh, xls_path=args.xls,
                    dstep=args.dstep, max_steps=args.max_steps)
    if not args.no_fig:
        plot_overlay(res)


if __name__ == "__main__":
    main()
