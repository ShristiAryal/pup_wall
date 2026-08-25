
from pathlib import Path
import numpy as np
import pandas as pd

DEFAULT_XLS = Path(__file__).resolve().parents[1] / "data" / \
    "PUP_force_displ_hystereses_incl_vert_displ.xls"

COLS = ["drift_pct", "force_kN", "axial_kN", "Mtop_kNm", "Mbot_kNm",
        "vert_LED_mm", "vert_LVDT_mm"]


def load_recorded_protocol(wall="PUP2", xls_path=None):

    if xls_path:
        xls_path = Path(xls_path)
    else:
        xls_path = DEFAULT_XLS

    raw = pd.read_excel(xls_path, sheet_name=wall, header=None)


    column0 = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    first = column0.first_valid_index()
    if first is None:
        raise ValueError(wall + ": no numeric data found in column 0")

    block = raw.iloc[first:, :7].apply(pd.to_numeric, errors="coerce")


    block = block[block.iloc[:, 0].notna() & block.iloc[:, 1].notna()]

    out = {}
    n_cols = min(7, block.shape[1])
    for i in range(n_cols):
        out[COLS[i]] = block.iloc[:, i].to_numpy(dtype=float)

    out["n"] = len(out["drift_pct"])

    force = out["force_kN"]
    biggest_negative = abs(np.nanmin(force))
    biggest_positive = abs(np.nanmax(force))
    out["peak_abs_force"] = float(max(biggest_negative, biggest_positive))

    out["first_row"] = int(first)
    out["wall"] = wall
    return out


def reversal_peaks(drift, tol=1e-6):

    d = np.asarray(drift)
    step_direction = np.diff(d)
    sign = np.sign(step_direction)

    idx = []
    last_sign = 0
    for i in range(1, len(sign)):
        if sign[i] != 0 and sign[i] != last_sign:
            idx.append(i)
            last_sign = sign[i]
        elif sign[i] != 0:
            last_sign = sign[i]

    return np.asarray(idx, dtype=int)


def summarise(wall="PUP2", xls_path=None):
    """Load one wall's recorded protocol, print a one-line summary, and
    return the loaded data."""
    r = load_recorded_protocol(wall, xls_path)
    d = r["drift_pct"]
    peaks = reversal_peaks(d)

    print(wall + ": first_row " + str(r["first_row"]) + "  n " +
          str(r["n"]) + "  drift " + str(round(float(d.min()), 3)) +
          ".." + str(round(float(d.max()), 3)) + "%  force " +
          str(round(float(r["force_kN"].min()), 1)) + ".." +
          str(round(float(r["force_kN"].max()), 1)) + " kN  peak|F| " +
          str(round(r["peak_abs_force"], 1)) + "  reversals " +
          str(len(peaks)))

    return r


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        xls = sys.argv[1]
    else:
        xls = "/mnt/user-data/uploads/PUP_force_displ_hystereses_incl_vert_displ.xls"

    for w in ["PUP1", "PUP2", "PUP3", "PUP4", "PUP5", "PUP6"]:
        summarise(w, xls)
