
import argparse
from pathlib import Path

import numpy as np

import model_builder as mb
from interface_laws import load_params
from recorded_protocol import load_recorded_protocol


mb.MAX_STEPS = 40000

REPO = Path(__file__).resolve().parents[1]
XLS = REPO / "data" / "PUP_force_displ_hystereses_incl_vert_displ.xls"

# 4-peak calibration control (regression guard)
CONTROL_PEAKS = [+0.36, -0.36, +0.36, -0.36]


def _fmt(value, spec):

    return ("{:" + spec + "}").format(value)


def significant_peaks(d, thresh=0.02):

    peaks = [d[0]]
    idxs = [0]

    direction = 0
    last_ext = d[0]
    last_i = 0

    for i in range(1, len(d)):
        if direction == 0:
            if abs(d[i] - last_ext) > thresh:
                direction = np.sign(d[i] - last_ext)
                last_ext = d[i]
                last_i = i
        else:
            if np.sign(d[i] - d[last_i]) == direction or d[i] == d[last_i]:
                last_ext = d[i]
                last_i = i
            if (d[i] - last_ext) * direction < -thresh:
                peaks.append(last_ext)
                idxs.append(last_i)
                direction = -direction
                last_ext = d[i]
                last_i = i

    peaks.append(d[last_i])
    idxs.append(last_i)
    return np.asarray(peaks), np.asarray(idxs)


def recorded_peak_schedule(wall="PUP2", thresh=0.02):
    """Load the recorded protocol and reduce it to its significant peaks."""
    r = load_recorded_protocol(wall, XLS)
    pk, _ = significant_peaks(r["drift_pct"], thresh)
    if abs(pk[0]) < 0.01:      # drop the tiny leading seed peak
        pk = pk[1:]
    return pk, r


def build_pup2_calibrated(shear_law="B"):

    params = load_params()
    K = mb.derive_stiffnesses(params, verbose=False)
    mesh = mb.build_geometry()

    
    h1 = mb.build_opensees_model(mesh, params, K, wall="PUP2",
                                 sigma_map=None, shear_law=shear_law)
    mb.apply_gravity(mesh, h1["cfg"])
    sigma_map, _ = mb.harvest_sigma(mesh, h1)

    
    h2 = mb.build_opensees_model(mesh, params, K, wall="PUP2",
                                 sigma_map=sigma_map, shear_law=shear_law)
    mb.apply_gravity(mesh, h2["cfg"])
    return mesh, h2


def drive(peaks, dstep=mb.DSTEP, reversal_ramp=mb._REVERSAL_RAMP,
          reach_frac=1.0, shear_law="B", label="", cyclic_fn=None):

    mesh, handles = build_pup2_calibrated(shear_law=shear_law)

    if cyclic_fn is not None:
        fn = cyclic_fn
    else:
        fn = mb.cyclic

    drift, shear, status, log = fn(
        mesh, handles, peaks, dstep=dstep,
        reversal_ramp=reversal_ramp, reach_frac=reach_frac)

    reached = float(np.max(np.abs(drift)))
    Vmax = float(np.max(shear))
    Vmin = float(np.min(shear))

    if Vmin != 0:
        asym = 100.0 * (Vmax - abs(Vmin)) / abs(Vmin)
    else:
        asym = float("nan")

    complete = (status == "sequence complete")

    print("\n[" + label + "] status: " + status)

    if complete:
        complete_word = "COMPLETE"
    else:
        complete_word = "STALLED"

    print("    reached |drift| = " + _fmt(reached, ".3f") + "%   V+ " +
          _fmt(Vmax, ".1f") + "  V- " + _fmt(Vmin, ".1f") + "  asym " +
          _fmt(asym, "+.2f") + "%   " + complete_word)

    return {
        "drift": drift,
        "shear": shear,
        "status": status,
        "log": log,
        "reached": reached,
        "Vmax": Vmax,
        "Vmin": Vmin,
        "asym": asym,
        "complete": complete,
    }


def drive_verbose(peaks, dstep, reversal_ramp, label="", shear_law="B",
                  reach_frac=1.0, progress_path=None):

    import openseespy.opensees as ops

    mesh, handles = build_pup2_calibrated(shear_law=shear_law)
    lever = handles["lever"]

    beam_row = []
    for c in range(mb.N_COLS + 1):
        beam_row.append(mb.tag(12, c, 0, 1))

    ops.timeSeries("Linear", 3)
    ops.pattern("Plain", 3, 3)
    ops.load(lever, 1.0, 0.0)

    def u_drift():

        beam_disps = []
        for t in beam_row:
            beam_disps.append(ops.nodeDisp(t, 1))
        return 100.0 * float(np.mean(beam_disps)) / mb.H_EXP

    def shear_now():
       
        ops.reactions()
        total_reaction_x = 0.0
        for t in mesh.foundation_nodes.values():
            total_reaction_x = total_reaction_x + ops.nodeReaction(t)[0]
        return -total_reaction_x / 1000.0

    dstep_pct = dstep / (mb.H_EXP / 100.0)
    reach_tol = reach_frac * dstep_pct

    drift = [u_drift()]
    shear = [shear_now()]
    status = "sequence complete"
    total = 0

    if progress_path:
        pf = open(progress_path, "w")
    else:
        pf = None

    for hc in range(len(peaks)):
        target = peaks[hc]
        since = 0

        exit_reason = None

        while total < mb.MAX_STEPS:
            cur = drift[-1]
            rem = target - cur
            if abs(rem) <= reach_tol:
                exit_reason = "reached"
                break

            if rem > 0:
                sgn = 1.0
            else:
                sgn = -1.0

            if since < len(reversal_ramp):
                frac = reversal_ramp[since]
            else:
                frac = 1.0

            step_mm = sgn * min(dstep * frac, abs(rem) * (mb.H_EXP / 100.0))

            if not mb._try_step(step_mm, lever):
                status = ("non-convergence at drift " +
                          _fmt(drift[-1], "+.3f") + "% heading to " +
                          _fmt(target, "+.3f") + "% (half-cycle " +
                          str(hc) + ")")
                if pf:
                    pf.write("STALL hc=" + str(hc) + " at " +
                             _fmt(drift[-1], "+.4f") + "%->" +
                             _fmt(target, "+.3f") + "%\n")
                    pf.flush()
                drift.append(u_drift())
                shear.append(shear_now())
                exit_reason = "stalled"
                break

            drift.append(u_drift())
            shear.append(shear_now())
            total = total + 1
            since = since + 1

        if exit_reason is None:
            status = "MAX_STEPS reached"

        if pf:
            pf.write("hc " + _fmt(hc, "2d") + " -> " +
                     _fmt(target, "+.3f") + "% : reached " +
                     _fmt(drift[-1], "+.4f") + "%  V " +
                     _fmt(shear[-1], "+.1f") + "  steps " + str(total) + "\n")
            pf.flush()

        if status != "sequence complete":
            break

    if pf:
        pf.close()

    drift = np.asarray(drift)
    shear = np.asarray(shear)
    Vmax = float(np.max(shear))
    Vmin = float(np.min(shear))
    asym = 100.0 * (Vmax - abs(Vmin)) / abs(Vmin)
    complete = (status == "sequence complete")

    if complete:
        complete_word = "COMPLETE"
    else:
        complete_word = "STALLED"

    print("[" + label + "] " + status + " | reached " +
          _fmt(float(np.max(np.abs(drift))), ".3f") + "% | V+ " +
          _fmt(Vmax, ".1f") + " V- " + _fmt(Vmin, ".1f") + " asym " +
          _fmt(asym, "+.2f") + "% | " + complete_word)

    return {
        "drift": drift,
        "shear": shear,
        "status": status,
        "reached": float(np.max(np.abs(drift))),
        "Vmax": Vmax,
        "Vmin": Vmin,
        "asym": asym,
        "complete": complete,
    }


def run_control(**kw):
 
    print("=" * 70)
    print("REGRESSION CONTROL — 4-peak +/-0.36% (must stay COMPLETE)")
    return drive(CONTROL_PEAKS, label="control", **kw)


def run_protocol(**kw):
    
    pk, _ = recorded_peak_schedule()
    print("=" * 70)
    print("RECORDED PROTOCOL — " + str(len(pk)) + " peaks, " +
          _fmt(float(pk.min()), "+.3f") + ".." +
          _fmt(float(pk.max()), "+.3f") + "%")
    return drive(pk.tolist(), label="protocol", **kw)



FIXES = {
    "baseline": (mb.DSTEP, mb._REVERSAL_RAMP, "Day-1 driver, unchanged"),
    "a1": (0.025, mb._REVERSAL_RAMP, "2a: dstep halved 0.05->0.025"),
    "a2": (0.0125, mb._REVERSAL_RAMP, "2a: dstep quartered 0.05->0.0125"),
    "b1": (mb.DSTEP, (0.05, 0.1, 0.2, 0.35, 0.5, 0.7),
           "2b: deeper+gentler ramp (6 steps)"),
    "b2": (mb.DSTEP, (0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85),
           "2b: 8-step very gentle ramp"),
    "ab": (0.025, (0.05, 0.1, 0.2, 0.35, 0.5, 0.7),
           "2a+2b: half dstep + deeper ramp"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix", default="baseline", choices=list(FIXES))
    parser.add_argument("--control-only", action="store_true")
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()

    dstep, ramp, note = FIXES[args.fix]
    print("\n### FIX '" + args.fix + "': " + note)
    print("    dstep=" + str(dstep) + "  reversal_ramp=" + str(ramp) + "\n")

    if not args.protocol_only:
        run_control(dstep=dstep, reversal_ramp=ramp)
    if not args.control_only:
        run_protocol(dstep=dstep, reversal_ramp=ramp)


if __name__ == "__main__":
    main()
