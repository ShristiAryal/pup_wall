

import os
import sys
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WALLS = ["PUP2", "PUP3", "PUP5"]
PANEL_LABEL = {
    "PUP2": "PUP2 (calibration, shear)",
    "PUP3": "PUP3 (flexural reference)",
    "PUP5": "PUP5 (low-axial shear)",
}
LS37_EXCLUDE_WALL = "PUP5"   # LS37 = hydraulic pressure-loss artifact


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
        sys.exit(
            "ERROR: could not find model_optical_compare_" + wall +
            ".csv in results/, script dir, or cwd. Run 'dir /s "
            "model_optical_compare_" + wall + ".csv' to locate it and "
            "place the script alongside results/, or copy the CSVs next "
            "to it.")
    print("  " + wall + ": " + path)
    df = pd.read_csv(path)

    if wall == LS37_EXCLUDE_WALL:
        n0 = len(df)
        df = df[df["ls"] != 37].copy()
        if len(df) < n0:
            print("         excluded LS37 (hydraulic artifact) -> " +
                  str(len(df)) + " rows")

    
    df["drift_abs"] = df["drift_reached"].abs()
    df["side"] = df["drift_reached"].apply(_classify_side)
    return df


def main():
    print("Reading CSVs:")
    data = {}
    for w in WALLS:
        data[w] = load(w)

    fig, axes = plt.subplots(3, 1, figsize=(6.5, 9.0), sharex=True)

    C_OPT = "#000000"
    C_MOD = "#c0392b"

    for idx in range(len(WALLS)):
        wall = WALLS[idx]
        ax = axes[idx]
        df = data[wall]

        for side, mk in (("loaded", "o"), ("reload", "s")):
            sub = df[df["side"] == side]
            ax.scatter(sub["drift_abs"], sub["theta_base_opt_mrad"].abs(),
                       marker=mk, s=34, facecolors="none",
                       edgecolors=C_OPT, linewidths=1.1,
                       label="optical (" + side + ")")
            ax.scatter(sub["drift_abs"], sub["theta_base_modelA_mrad"].abs(),
                       marker=mk, s=20, color=C_MOD,
                       label="model (" + side + ")")

        ax.set_ylabel(r"$|\theta_{\mathrm{base}}|$  [mrad]")
        ax.set_title(PANEL_LABEL[wall], fontsize=10, loc="left")
        ax.grid(True, alpha=0.25, linewidth=0.5)

    axes[-1].set_xlabel(r"absolute drift  $|\delta|$  [%]")


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
    out = os.path.join(os.getcwd(), "fig7_theta_base.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("\nWrote " + out)


    print("\nVerification (max |theta_base| mrad, model-A vs optical):")
    for wall in WALLS:
        df = data[wall]
        for side in ("loaded", "reload"):
            sub = df[df["side"] == side]
            if sub.empty:
                continue
            i = sub["drift_abs"].idxmax()
            r = sub.loc[i]
            print("  " + _fmt(wall, "5s") + " " + _fmt(side, "6s") +
                  " @|drift|=" + _fmt(r["drift_abs"], ".3f") + "%  ls=" +
                  _fmt(int(r["ls"]), "2d") + "  opt=" +
                  _fmt(r["theta_base_opt_mrad"], "+.3f") + "  modelA=" +
                  _fmt(r["theta_base_modelA_mrad"], "+.3f"))


    print("\nSpot-check (theta_base model-A vs model-B agreement, mrad):")
    for wall in WALLS:
        df = data[wall]
        ab = (df["theta_base_modelA_mrad"] -
             df["theta_base_modelB_mrad"]).abs()
        i = ab.idxmax()
        r = df.loc[i]
        print("  " + _fmt(wall, "5s") + " |A-B| max " +
              _fmt(ab.max(), ".4f") + "  median " + _fmt(ab.median(), ".4f") +
              "  (worst @ls=" + _fmt(int(r["ls"]), "2d") + ", |drift|=" +
              _fmt(r["drift_abs"], ".3f") + "%: A=" +
              _fmt(r["theta_base_modelA_mrad"], "+.3f") + " B=" +
              _fmt(r["theta_base_modelB_mrad"], "+.3f") + ")")


if __name__ == "__main__":
    main()
