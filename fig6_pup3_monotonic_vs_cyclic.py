
import argparse
from pathlib import Path

import numpy as np

import model_builder as mb
from blind_predict import build_calibrated, blind_run


def push(mesh, handles, target_pct, dstep, label):

    import openseespy.opensees as ops
    lever = handles["lever"]
    beam_row = [mb.tag(12, c, 0, 1) for c in range(mb.N_COLS + 1)]

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(lever, 1.0, 0.0)
    ops.test("NormDispIncr", 1.0e-7, 100, 0)
    ops.algorithm("Newton")

    sgn = 1.0 if target_pct >= 0 else -1.0
    step = sgn * abs(dstep)
    drift, shear = [0.0], [0.0]
    status = "target reached"
    for _ in range(mb.MAX_STEPS):
        if not mb._try_step(step, lever):
            status = f"non-convergence at drift {drift[-1]:+.3f}%"
            break
        u_top = float(np.mean([ops.nodeDisp(t, 1) for t in beam_row]))
        ops.reactions()
        V = -sum(ops.nodeReaction(t)[0]
                 for t in mesh.foundation_nodes.values())
        drift.append(100.0 * u_top / mb.H_EXP)
        shear.append(V / 1000.0)
        if (sgn > 0 and drift[-1] >= target_pct) or \
           (sgn < 0 and drift[-1] <= target_pct):
            break
    else:
        status = "MAX_STEPS reached"
    drift, shear = np.asarray(drift), np.asarray(shear)
    Vext = float(shear[np.argmax(np.abs(shear))])   # signed extreme shear
    print(f"  [{label}] {status} | reached {drift[-1]:+.3f}% | "
          f"peak V {Vext:+.1f} kN")
    return drift, shear, status, Vext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.6,
                    help="monotonic drift magnitude %% each way (default 0.6, "
                         "the value that produced the locked +104.6/-105.2)")
    ap.add_argument("--dstep", type=float, default=0.02,
                    help="pushover driver step (default 0.02, as locked)")
    ap.add_argument("--push-max-steps", type=int, default=20000,
                    help="ceiling for each monotonic push (default 20000)")
    ap.add_argument("--cyclic-max-steps", type=int, default=80000,
                    help="ceiling for the recorded cyclic drive (PUP3 needs "
                         "~80000 for its 46-peak protocol)")
    ap.add_argument("--fname", default=None,
                    help="output PNG path (default results/figures/"
                         "fig6_pup3_monotonic_vs_cyclic.png)")
    args = ap.parse_args()

    t = abs(args.target)

    print("=" * 70)
    print("FIGURE 6 — PUP3 monotonic pushover vs recorded cyclic")
    print(f"  push target +/-{t}%  dstep={args.dstep}  "
          f"push MAX_STEPS={args.push_max_steps}")
    print("  (frozen config; fresh build each curve; nothing tuned)\n")

    print("+x monotonic push:")
    mb.MAX_STEPS = args.push_max_steps
    mesh_p, h_p = build_calibrated("PUP3", shear_law="B")
    d_pos, v_pos, s_pos, Vpos = push(mesh_p, h_p, +t, args.dstep, "+x")

    print("\n-x monotonic push:")
    mesh_n, h_n = build_calibrated("PUP3", shear_law="B")
    d_neg, v_neg, s_neg, Vneg = push(mesh_n, h_n, -t, args.dstep, "-x")

    print("\nrecorded cyclic drive (this reruns the PUP3 blind drive; slow):")
    res = blind_run("PUP3", max_steps=args.cyclic_max_steps)
    d_cyc = np.asarray(res["model_drift"])
    v_cyc = np.asarray(res["model_shear"])
    Vcyc_p, Vcyc_n = res["Vmax"], res["Vmin"]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    ax.plot(d_cyc, v_cyc, "-", color="crimson", lw=1.3,
            label=f"recorded cyclic (V+ {Vcyc_p:+.1f} / V- {Vcyc_n:+.1f})")

    d_mono = np.concatenate([d_neg[::-1], d_pos])
    v_mono = np.concatenate([v_neg[::-1], v_pos])
    ax.plot(d_mono, v_mono, "-", color="0.25", lw=1.8,
            label=f"monotonic pushover (V+ {Vpos:+.1f} / V- {Vneg:+.1f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("top drift [%]")
    ax.set_ylabel("base shear V [kN]")
    ax.set_title("PUP3 — monotonic pushover vs recorded cyclic "
                 "[NO parameters tuned]")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    fname = Path(args.fname) if args.fname else (
        mb.FIG_DIR / "fig6_pup3_monotonic_vs_cyclic.png")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname, dpi=200)
    plt.close(fig)

    print("\n" + "-" * 70)
    print("FIGURE 6 NUMBERS (verify against locked table before locking):")
    print(f"  monotonic  +x : {Vpos:+.1f} kN   (locked +104.6)")
    print(f"  monotonic  -x : {Vneg:+.1f} kN   (locked -105.2)")
    print(f"  cyclic     V+ : {Vcyc_p:+.1f} kN   (locked +100.8)")
    print(f"  cyclic     V- : {Vcyc_n:+.1f} kN   (locked  -0.9)")
    gap = 100.0 * (Vpos - Vcyc_p) / Vpos if Vpos else float("nan")
    print(f"  +side pushover-vs-cyclic gap : {gap:+.1f}%  (locked ~3.8%)")
    print(f"\nfigure saved: {fname}")


if __name__ == "__main__":
    main()
