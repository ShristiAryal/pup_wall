
import os
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WALLS = ["PUP2", "PUP3"]
PANEL_LABEL = {
    "PUP2": "PUP2 (shear wall)",
    "PUP3": "PUP3 (flexural wall)",
}


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


def load(wall):

    path = find_csv(wall)
    if path is None:
        sys.exit("ERROR: could not find model_optical_compare_" + wall +
                 ".csv in results/, script dir, or cwd.")
    print("  " + wall + ": " + path)
    df = pd.read_csv(path).sort_values("ls").reset_index(drop=True)
    df["side"] = df["drift_reached"].apply(_classify_side)
    return df


def main():
    print("Reading CSVs:")
    data = {}
    for w in WALLS:
        data[w] = load(w)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)

    C_OPT = "#000000"     # optical: black, open markers
    C_MOD = "#c0392b"     # model-A: red, filled

    for idx in range(len(WALLS)):
        wall = WALLS[idx]
        ax = axes[idx]
        df = data[wall]

        for side, mk in (("loaded", "o"), ("reload", "s")):
            sub = df[df["side"] == side]
            ax.scatter(sub["ls"], sub["sh_frac_opt"],
                       marker=mk, s=40, facecolors="none",
                       edgecolors=C_OPT, linewidths=1.2,
                       label="optical (" + side + ")", zorder=3)
            ax.scatter(sub["ls"], sub["sh_frac_modelA"],
                       marker=mk, s=24, color=C_MOD,
                       label="model (" + side + ")", zorder=2)

        ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel("load step  (ls)")
        ax.set_title(PANEL_LABEL[wall], fontsize=10, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[0].set_ylabel(r"shear fraction  $\mathrm{sh\_frac}$  (full field)")


    handles, labels = axes[0].get_legend_handles_labels()
    seen_labels = set()
    dedup_handles = []
    dedup_labels = []
    for i in range(len(handles)):
        if labels[i] not in seen_labels:
            seen_labels.add(labels[i])
            dedup_handles.append(handles[i])
            dedup_labels.append(labels[i])
    axes[0].legend(dedup_handles, dedup_labels, fontsize=7.5,
                  loc="upper left", framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(os.getcwd(), "fig8_sh_frac.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("\nWrote " + out)

    print("\nVerification (full-field sh_frac, optical vs model-A):")
    for wall in WALLS:
        df = data[wall]
        o = df["sh_frac_opt"]
        a = df["sh_frac_modelA"]
        print("  " + wall + ":")
        print("    optical range [" + _fmt(o.min(), ".3f") + ", " +
              _fmt(o.max(), ".3f") + "]  spread " +
              _fmt(o.max() - o.min(), ".3f"))
        print("    model-A range [" + _fmt(a.min(), ".3f") + ", " +
              _fmt(a.max(), ".3f") + "]  spread " +
              _fmt(a.max() - a.min(), ".3f") + "  mean " +
              _fmt(a.mean(), ".3f"))


if __name__ == "__main__":
    main()
