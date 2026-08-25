

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(r"C:\Users\ADMIN\Downloads\URM_Data\Data_Part_01")
FIG_DIR = Path(r"C:\Users\ADMIN\Downloads\URM_Data\pup-opensees-skeleton"
               r"\results\figures")

WALL_L = 2010.0   # mm
WALL_H = 2250.0   # mm

N_COLS = 13
N_ROWS = 24
N_LEDS = N_COLS * N_ROWS  # 312

RAMP_THRESHOLD_KN = 50.0   
WINDOW_TRIM = 10           
MIN_VALID_SAMPLES = 20     


LS0_CONFIG = {
    "PUP1": {"mode": "preramp"},
    "PUP2": {"mode": "ls1_reference"},
    "PUP3": {"mode": "preramp"},
    "PUP4": {"mode": "preramp"},
    "PUP5": {"mode": "preramp", "window": (3, 32)},
    "PUP6": {"mode": "preramp", "window": (3, 52)},
}

def load_optical(wall, base_dir=BASE_DIR, variant="at_LS"):
   
    pd_dir = base_dir / wall / "processed_data"
    files = sorted(pd_dir.glob(wall + "_LED_*_" + variant + ".asc"))
    if len(files) != N_LEDS:
        raise ValueError(
            wall + ": expected " + str(N_LEDS) + " LED files, found " +
            str(len(files)))

    frames = []
    for f in files:
        match = re.search(r"C(\d+)_R(\d+)", f.name)
        c = int(match.group(1))
        r = int(match.group(2))

        df = pd.read_csv(f, sep="\t", skiprows=[1])   
        df.columns = ["x", "y", "z"]
        df["C"] = c
        df["R"] = r
        df["LS"] = range(len(df))
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)[["LS", "C", "R", "x", "y", "z"]]

    counts = out.groupby("LS")["C"].count()
    if not (counts == N_LEDS).all():
        bad_rows = counts[counts != N_LEDS]
        raise ValueError(
            wall + ": point count per LS not " + str(N_LEDS) + ":\n" +
            str(bad_rows))
    return out


def load_conventional_at_ls(wall, base_dir=BASE_DIR):
    """Conventional _at_LS file (forces, displacements at the same holds)."""
    f = base_dir / wall / "processed_data" / (wall + "_conventional_at_LS.asc")
    df = pd.read_csv(f, sep="\t", skiprows=[1])
    return df.loc[:, ~df.columns.str.startswith("empty")]


def load_conventional_continuous(wall, base_dir=BASE_DIR):
    """Conventional continuous recording (full test incl. pre-load phase)."""
    f = base_dir / wall / "processed_data" / (wall + "_conventional.asc")
    df = pd.read_csv(f, sep="\t", skiprows=[1])
    return df.loc[:, ~df.columns.str.startswith("empty")]


def load_led_continuous(wall, c, r, base_dir=BASE_DIR):
    """Continuous recording for ONE LED marker (column c, row r)."""
    filename = (wall + "_LED_C" + str(c).zfill(2) +
               "_R" + str(r).zfill(2) + ".asc")
    f = base_dir / wall / "processed_data" / filename
    df = pd.read_csv(f, sep="\t", skiprows=[1])
    df.columns = ["x", "y", "z"]
    return df


def find_preramp_window(wall, base_dir=BASE_DIR,
                        threshold=RAMP_THRESHOLD_KN, trim=WINDOW_TRIM):
   
    conv_full = load_conventional_continuous(wall, base_dir)
    ramp_start = int((conv_full["N"].abs() > threshold).idxmax())

    if ramp_start <= 2 * trim + MIN_VALID_SAMPLES:
        raise ValueError(
            wall + ": pre-ramp window too short (ramp starts at row " +
            str(ramp_start) +
            ") — supply an explicit window in LS0_CONFIG.")

    return trim, ramp_start - trim


def load_ls0(wall, base_dir=BASE_DIR, window=None):

    if window is None:
        start, stop = find_preramp_window(wall, base_dir)
    else:
        start, stop = window

    pd_dir = base_dir / wall / "processed_data"

    records = []
    for c in range(1, N_COLS + 1):
        for r in range(1, N_ROWS + 1):
            filename = (wall + "_LED_C" + str(c).zfill(2) +
                       "_R" + str(r).zfill(2) + ".asc")
            f = pd_dir / filename

            df = pd.read_csv(f, sep="\t", skiprows=[1])
            df.columns = ["x", "y", "z"]
            win = df.iloc[start:stop].dropna()

            records.append({
                "C": c,
                "R": r,
                "x": win["x"].mean(),
                "y": win["y"].mean(),
                "z": win["z"].mean(),
                "n_valid": len(win),
            })

    ls0 = pd.DataFrame(records)

    n_bad = (ls0["n_valid"] < MIN_VALID_SAMPLES).sum()
    if n_bad:
        print("NOTE " + wall + ": " + str(n_bad) + " markers with < " +
              str(MIN_VALID_SAMPLES) + " valid pre-ramp samples (expected "
              "for steel-plate placeholder rows R01/R24 on PUP1/PUP2; "
              "verify otherwise).")

    return ls0


def load_reference(wall, base_dir=BASE_DIR):
 
    cfg = LS0_CONFIG[wall]

    if cfg["mode"] == "preramp":
        ref = load_ls0(wall, base_dir, window=cfg.get("window"))
        return ref, "preramp"

    opt = load_optical(wall, base_dir)
    ref = (opt[opt["LS"] == 0][["C", "R", "x", "y", "z"]]
           .reset_index(drop=True))
    return ref, "ls1_reference"


def plot_led_grid(opt, wall, ls=0, ax=None, ls0=None):
    """Sanity plot: LED positions at one load step over the wall outline."""
    pts = opt[opt["LS"] == ls]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 7))

    half_l = WALL_L / 2
    ax.plot([-half_l, half_l, half_l, -half_l, -half_l],
            [0, 0, WALL_H, WALL_H, 0], "k-", lw=1.5)

    if ls0 is not None:
        ax.scatter(ls0["x"], ls0["y"], s=26, facecolors="none",
                   edgecolors="tab:blue", label="Reference config")

    n_tracked = pts[["x", "y"]].notna().all(axis=1).sum()
    ax.scatter(pts["x"], pts["y"], s=12, c="tab:red",
               label="LS index " + str(ls) + " (" + str(n_tracked) +
               " tracked)")

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(wall)
    ax.legend(loc="upper right", fontsize=7)
    return ax


if __name__ == "__main__":
    walls = ["PUP1", "PUP2", "PUP3", "PUP4", "PUP5", "PUP6"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))

    axes_flat = axes.flat
    for i in range(len(walls)):
        wall = walls[i]
        ax = axes_flat[i]

        print("=" * 60)
        opt = load_optical(wall)
        conv = load_conventional_at_ls(wall)
        ref, mode = load_reference(wall)

        n_ls = opt["LS"].nunique()
        nan_frac = opt[["x", "y"]].isna().any(axis=1).mean()
        n_dead = ref[["x", "y"]].isna().any(axis=1).sum()
        rows_match = (len(conv) == n_ls)

        print(wall + ": " + str(n_ls) + " LS (conv rows match: " +
              str(rows_match) + "), NaN " +
              str(round(nan_frac * 100, 1)) + "%, reference: " + mode +
              ", NaN ref markers: " + str(n_dead))

        plot_led_grid(opt, wall, ls=0, ax=ax, ls0=ref)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outpath = FIG_DIR / "led_grids_all_walls.png"
    plt.savefig(outpath, dpi=150)
    print("=" * 60)
    print("Figure saved: " + str(outpath))
    plt.show()
