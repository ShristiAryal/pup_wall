

import os
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WALL = "PUP5"
LS37_DROP = 37
LS_LOADED_EXCLUDED = {32, 34, 36}


def _fmt(value, spec):

    return ("{:" + spec + "}").format(value)


def _classify_side(drift_value):

    if drift_value >= 0:
        return "loaded"
    else:
        return "reload"


def find_csv(wall):
  
    name = "model_optical_compare_" + wall + ".csv"
    here = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join(here, "results", name),
        os.path.join(here, name),
        os.path.join(os.getcwd(), "results", name),
        os.path.join(os.getcwd(), name),
        os.path.join(here, "..", "results", name),
    ]

    for c in candidates:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return None


def main():
    path = find_csv(WALL)
    if path is None:
        sys.exit("ERROR: could not find model_optical_compare_" + WALL +
                 ".csv")
    print("Reading: " + path)
    df = pd.read_csv(path)

    n0 = len(df)
    df = df[df["ls"] != LS37_DROP].copy()
    print("  dropped LS37 (hydraulic artifact): " + str(n0) + " -> " +
          str(len(df)) + " rows")

    df["side"] = df["drift_reached"].apply(_classify_side)
    df["excluded"] = df["ls"].isin(LS_LOADED_EXCLUDED)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    C_OPT = "#000000"
    C_MOD = "#c0392b"
    C_EX = "#b0b0b0"
    C_GAP = "#c0392b"

    
    reload_robust = df[(df["side"] == "reload") & (~df["excluded"])]
    first_gap = True
    for _, r in reload_robust.iterrows():
        if first_gap:
            gap_label = "reload gap (model\u2192optical)"
        else:
            gap_label = None
        ax.plot([r["drift_reached"], r["drift_reached"]],
                [r["sh_frac_modelA"], r["sh_frac_opt"]],
                color=C_GAP, lw=1.0, alpha=0.35, zorder=1, label=gap_label)
        first_gap = False

    for side, mk in (("loaded", "o"), ("reload", "s")):
        sub = df[(df["side"] == side) & (~df["excluded"])]
        ax.scatter(sub["drift_reached"], sub["sh_frac_opt"],
                   marker=mk, s=46, facecolors="none", edgecolors=C_OPT,
                   linewidths=1.3, label="optical (" + side + ")", zorder=3)
        ax.scatter(sub["drift_reached"], sub["sh_frac_modelA"],
                   marker=mk, s=28, color=C_MOD,
                   label="model (" + side + ")", zorder=2)

    ex = df[df["excluded"]]
    if not ex.empty:
        ax.scatter(ex["drift_reached"], ex["sh_frac_opt"],
                   marker="o", s=46, facecolors="none", edgecolors=C_EX,
                   linewidths=1.3, zorder=1)
        ax.scatter(ex["drift_reached"], ex["sh_frac_modelA"],
                   marker="o", s=28, color=C_EX, zorder=1,
                   label="loaded, resolution-limited (excluded)")

    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.axvline(0.0, color="0.4", lw=0.7)
    ax.set_xlabel(r"signed drift  $\delta$  [%]   (reload $<0$  |  loaded $>0$)")
    ax.set_ylabel(r"shear fraction  $\mathrm{sh\_frac}$  (full field)")
    ax.set_title("PUP5 (low-axial shear): inverted directional response",
                 fontsize=11, loc="left")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=7.5, loc="lower center", framealpha=0.9, ncol=2)

    fig.tight_layout()
    out = os.path.join(os.getcwd(), "fig9_pup5_inversion.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Wrote " + out)

    print("\nVerification (D021 reload table, optical vs model-A):")
    for ls in (27, 29, 31, 33, 35):
        r = df[df["ls"] == ls]
        if r.empty:
            print("  ls" + str(ls) + ": MISSING")
            continue
        r = r.iloc[0]
        print("  ls" + str(ls) + " drift " +
              _fmt(r["drift_reached"], "+.3f") + "%  opt " +
              _fmt(r["sh_frac_opt"], ".3f") + "  model " +
              _fmt(r["sh_frac_modelA"], ".3f") + "  gap " +
              _fmt(r["sh_frac_opt"] - r["sh_frac_modelA"], ".3f"))

    print("\nExcluded loaded steps (D021 pt3, greyed, no gap-line):")
    for ls in sorted(LS_LOADED_EXCLUDED):
        r = df[df["ls"] == ls]
        if r.empty:
            print("  ls" + str(ls) + ": MISSING")
            continue
        r = r.iloc[0]
        print("  ls" + str(ls) + " drift " +
              _fmt(r["drift_reached"], "+.3f") + "%  opt " +
              _fmt(r["sh_frac_opt"], ".3f") + "  model " +
              _fmt(r["sh_frac_modelA"], ".3f"))


if __name__ == "__main__":
    main()
