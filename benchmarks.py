
import numpy as np
import pandas as pd
from pathlib import Path

from wall import load_wall, WALLS
from optical_data import (load_optical, load_reference,
                          N_COLS, WALL_H, LS0_CONFIG)

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
elif hasattr(np, "trapz"):
    _trapz = np.trapz
else:
    _trapz = None

DOCS_DIR = Path(r"C:\Users\ADMIN\Downloads\URM_Data\pup-opensees-skeleton\docs")
MATRIX_CSV = DOCS_DIR / "wall_test_matrix.csv"

MASONRY_ROWS = list(range(2, 24))
JOINT_OPEN_TOL_MM = 0.02            
MIN_FIT_POINTS = 3                  


def extract_backbone(wall):
  
    df = load_wall(wall, "_at_LS")
    d = df["drift_percent"].to_numpy(float)
    F = df["force_horizontal_kN"].to_numpy(float)

    out = {}
    peaks = {}


    directions = [("pos", 1.0), ("neg", -1.0)]
    for name, sign in directions:
        rows = []
        best_so_far = 0.0
        for i in range(len(d)):
            signed_drift = sign * d[i]
            if signed_drift > best_so_far + 1e-9:
                best_so_far = signed_drift
                rows.append((i, d[i], F[i]))
        out[name] = pd.DataFrame(
            rows, columns=["ls", "drift_percent", "force_kN"])

        mask = (sign * d) > 0
        if mask.any():
            signed_force = sign * np.where(mask, F, np.nan)
            signed_drift_values = np.where(mask, sign * d, np.nan)
            iF = np.nanargmax(signed_force)
            iD = np.nanargmax(signed_drift_values)
            peaks[name] = {
                "F_peak_kN": F[iF],
                "ls_F_peak": int(iF),
                "drift_at_F_peak": d[iF],
                "drift_max": d[iD],
                "ls_drift_max": int(iD),
            }
        else:
            peaks[name] = None

    out["peaks"] = peaks
    return out


def check_backbones(matrix_csv=MATRIX_CSV):
    """Print peaks for all six walls next to the wall_test_matrix.csv row."""
    if Path(matrix_csv).exists():
        matrix = pd.read_csv(matrix_csv)
    else:
        matrix = None
        print("NOTE: " + str(matrix_csv) +
              " not found — printing computed peaks only.")

    def row_mentions_wall(row, wall_name):
        """True if any cell in this table row contains the wall's name."""
        return row.astype(str).str.contains(wall_name).any()

    for w in WALLS:
        bb = extract_backbone(w)
        print("-" * 70)
        print(w)

        for direc in ("pos", "neg"):
            p = bb["peaks"][direc]
            if p is None:
                print("  " + direc + ": no data")
                continue
            print("  " + direc + ": F_peak = " +
                  str(round(p["F_peak_kN"], 1)) + " kN at LS " +
                  str(p["ls_F_peak"]) + " (drift " +
                  str(round(p["drift_at_F_peak"], 3)) + "%), drift_max = " +
                  str(round(p["drift_max"], 3)) + "% at LS " +
                  str(p["ls_drift_max"]))

        if w == "PUP5":
            print("  NOTE: hydraulic pressure loss at LS37 — ignore any "
                  "apparent strength drop in the last loop.")

        if matrix is not None:
            row = matrix[matrix.apply(
                lambda r: row_mentions_wall(r, w), axis=1)]
            if len(row):
                print("  wall_test_matrix.csv row:")
                print("  " + row.to_string(index=False).replace("\n", "\n  "))


def _grid(df):
    
    x_table = df.pivot(index="R", columns="C", values="x")
    x_table = x_table.reindex(index=range(1, 25), columns=range(1, N_COLS + 1))

    y_table = df.pivot(index="R", columns="C", values="y")
    y_table = y_table.reindex(index=range(1, 25), columns=range(1, N_COLS + 1))

    return x_table.to_numpy(float), y_table.to_numpy(float)


def detect_brick_pairs(ref):
  
    y_med = ref.groupby("R")["y"].median()

    rows = []
    for r in MASONRY_ROWS:
        if r in y_med.index and np.isfinite(y_med[r]):
            rows.append(r)

    gaps = {}
    for i in range(len(rows) - 1):
        r = rows[i]
        gaps[r] = y_med[rows[i + 1]] - y_med[r]

    r0 = rows[0]
    if gaps[r0] > gaps[rows[1]]:
        start = r0
    else:
        start = rows[1]

    pairs = []
    for r in range(start, rows[-1], 2):
        pairs.append((r, r + 1))

    print("  brick-pair detection: pairs start at R" + str(start).zfill(2) +
          " -> " + str(len(pairs)) + " virtual rows (gap R" +
          str(r0).zfill(2) + "->R" + str(r0 + 1).zfill(2) + " = " +
          str(round(gaps[r0])) + " mm, R" + str(rows[1]).zfill(2) + "->R" +
          str(rows[2]).zfill(2) + " = " + str(round(gaps[rows[1]])) + " mm)")
    return pairs


def virtual_grid(X, Y, pairs):

    Xv_rows = []
    Yv_rows = []
    for a, b in pairs:
        Xv_rows.append((X[a - 1] + X[b - 1]) / 2)
        Yv_rows.append((Y[a - 1] + Y[b - 1]) / 2)
    Xv = np.array(Xv_rows)
    Yv = np.array(Yv_rows)
    return Xv, Yv


def _global_compressed_toe(X0v, Y0v, Xv, Yv):
    
    V = Yv - Y0v
    n_v = X0v.shape[0]

    open_by_col = None
    for j in range(n_v - 1):
        dv = V[j + 1] - V[j]
        pos_open = np.where(np.isfinite(dv), np.maximum(dv, 0.0), 0.0)
        if open_by_col is None:
            open_by_col = pos_open
        else:
            open_by_col = open_by_col + pos_open

    finite_any = np.isfinite(np.nanmean(X0v, axis=0))
    cols = np.where(finite_any)[0]
    left = cols[0]
    right = cols[-1]

    if open_by_col[left] <= open_by_col[right]:
        return "left"
    else:
        return "right"


def _point(arr_x, arr_y, row, col):
   
    return np.array([arr_x[row, col], arr_y[row, col]])


def band_quantities(X0v, Y0v, Xv, Yv, global_toe=None):

    n_v = X0v.shape[0]
    U = Xv - X0v
    V = Yv - Y0v

    if global_toe is None:
        global_toe = _global_compressed_toe(X0v, Y0v, Xv, Yv)

    recs = []
    for j in range(n_v - 1):
        x0 = np.nanmean([X0v[j], X0v[j + 1]], axis=0)   
        ly = Y0v[j + 1] - Y0v[j]                          
        dv = V[j + 1] - V[j]                              
        eps = dv / ly                                     
        y_band = np.nanmean([Y0v[j], Y0v[j + 1]])

        valid = np.isfinite(dv) & np.isfinite(x0)
        if valid.sum() < MIN_FIT_POINTS:
            recs.append((y_band, np.nan, np.nan, np.nan, 0))
            continue

        cols = np.where(valid)[0]
        left = cols[0]
        right = cols[-1]

        if global_toe == "left":
            edge = left
        else:
            edge = right

        if edge == left:
            step = 1
            stop = right + 1
        else:
            step = -1
            stop = left - 1

        closed = []
        for c in range(edge, stop, step):
            if not valid[c]:
                continue
            if dv[c] > JOINT_OPEN_TOL_MM:
                break
            closed.append(c)

        if len(closed) < MIN_FIT_POINTS:
            if closed:
                lc_partial = abs(x0[closed[-1]] - x0[edge])
            else:
                lc_partial = 0.0
            recs.append((y_band, np.nan, np.nan, lc_partial, len(closed)))
            continue

        closed = np.array(closed)
        Lc = abs(x0[closed[-1]] - x0[closed[0]])

    
        chi = -np.polyfit(x0[closed], eps[closed], 1)[0]


        closed_set = set(closed.tolist())
        gammas = []
        for c in closed[:-1]:
            if (c + 1) not in closed_set:
                continue

        
            d1_0 = np.linalg.norm(_point(X0v, Y0v, j, c) -
                                  _point(X0v, Y0v, j + 1, c + 1))
            d2_0 = np.linalg.norm(_point(X0v, Y0v, j, c + 1) -
                                  _point(X0v, Y0v, j + 1, c))
            d1 = np.linalg.norm(_point(Xv, Yv, j, c) -
                                _point(Xv, Yv, j + 1, c + 1))
            d2 = np.linalg.norm(_point(Xv, Yv, j, c + 1) -
                                _point(Xv, Yv, j + 1, c))
            lx = abs(x0[c + 1] - x0[c])
            lyq = np.nanmean([ly[c], ly[c + 1]])
            if lx > 0 and np.isfinite(lyq):
                gammas.append(((d1 ** 2 - d1_0 ** 2) - (d2 ** 2 - d2_0 ** 2))
                              / (4 * lx * lyq))

        if gammas:
            gamma = np.nanmean(gammas)
        else:
            gamma = np.nan

        recs.append((y_band, chi, gamma, Lc, len(closed)))

    return pd.DataFrame(recs, columns=["y_mm", "chi", "gamma",
                                       "Lc_mm", "n_closed"])


def _pad_profile(y, q):

    finite_mask = np.isfinite(q)
    y_finite = y[finite_mask]
    q_finite = q[finite_mask]

    if len(y_finite) == 0:
        return None, None

    order = np.argsort(y_finite)
    y_sorted = y_finite[order]
    q_sorted = q_finite[order]

    y_padded = np.concatenate([[0.0], y_sorted, [WALL_H]])
    q_padded = np.concatenate([[q_sorted[0]], q_sorted, [q_sorted[-1]]])
    return y_padded, q_padded


def _rocking_band_drift(X0v, Y0v, Xv, Yv, bands):

    U = Xv - X0v
    n_v = X0v.shape[0]
    total = 0.0

    for j in range(n_v - 1):
        if np.isfinite(bands.iloc[j]["chi"]):
            continue

        ubot = U[j]
        utop = U[j + 1]
        ly = Y0v[j + 1] - Y0v[j]
        good = (np.isfinite(ubot) & np.isfinite(utop) &
                np.isfinite(ly) & (ly > 0))

        if good.sum() < 2:
            continue    

        theta = np.nanmean((utop[good] - ubot[good]) / ly[good])
        y_band = np.nanmean([np.nanmean(Y0v[j]), np.nanmean(Y0v[j + 1])])
        total = total + theta * (WALL_H - y_band)

    return total



def decompose_drift(wall, ls, opt=None, ref=None, verbose=True):

    if opt is None:
        opt = load_optical(wall)

    if ref is None:
        ref, mode = load_reference(wall)
    else:
        mode = LS0_CONFIG[wall]["mode"]

    X0, Y0 = _grid(ref)
    XL, YL = _grid(opt[opt["LS"] == ls])

    pairs = detect_brick_pairs(ref)
    X0v, Y0v = virtual_grid(X0, Y0, pairs)
    Xv, Yv = virtual_grid(XL, YL, pairs)

    toe = _global_compressed_toe(X0v, Y0v, Xv, Yv)
    bands = band_quantities(X0v, Y0v, Xv, Yv, global_toe=toe)

    y_f, chi = _pad_profile(bands["y_mm"].to_numpy(), bands["chi"].to_numpy())
    y_s, gam = _pad_profile(bands["y_mm"].to_numpy(), bands["gamma"].to_numpy())

    if y_f is not None:
        u_fl_smooth = _trapz(chi * (WALL_H - y_f), y_f)
    else:
        u_fl_smooth = np.nan

    if y_s is not None:
        u_sh = _trapz(gam, y_s)
    else:
        u_sh = np.nan


    u_rock = _rocking_band_drift(X0v, Y0v, Xv, Yv, bands)
    u_fl = u_fl_smooth + u_rock

    disp_tp = float(load_wall(wall, "_at_LS")["top_plate_disp_mm"].iloc[ls])


    u_top = float(np.nanmean((Xv - X0v)[-1]))
    flipped = (np.isfinite(u_top) and np.isfinite(disp_tp) and
              u_top * disp_tp < 0)

    res = {
        "u_fl_mm": u_fl,
        "u_sh_mm": u_sh,
        "u_total_mm": u_fl + u_sh,
        "u_fl_smooth_mm": u_fl_smooth,
        "u_rock_mm": u_rock,
        "toe": toe,
        "u_top_optical_mm": u_top,
        "disp_tp_mm": disp_tp,
        "residual_internal_mm": (u_fl + u_sh) - u_top,
        "residual_mm": (u_fl + u_sh) - disp_tp,
        "optical_x_flipped_vs_conventional": flipped,
        "bands": bands,
        "reference_mode": mode,
    }

    if verbose:
        n_rock = int((bands["n_closed"] < MIN_FIT_POINTS).sum())

        print("  " + wall + " LS index " + str(ls) + " (paper LS" +
              str(ls + 1) + "), ref = " + str(mode) + ", toe = " + str(toe))

        print("  u_fl = " + str(round(u_fl, 2)) + " mm (smooth " +
              str(round(u_fl_smooth, 2)) + " + rock " +
              str(round(u_rock, 2)) + ")   u_sh = " + str(round(u_sh, 2)) +
              " mm   sum = " + str(round(u_fl + u_sh, 2)) + " mm")

        if u_top:
            residual_pct = 100 * res["residual_internal_mm"] / u_top
        else:
            residual_pct = np.nan
        print("  u_top (optical, same coords)   = " + str(round(u_top, 2)) +
              " mm   internal residual = " +
              str(round(res["residual_internal_mm"], 2)) + " mm (" +
              str(round(residual_pct)) + "%)")

        if flipped:
            flip_note = "   [optical x-axis FLIPPED vs conventional +]"
        else:
            flip_note = ""
        print("  Disp_TP (conventional channel) = " + str(round(disp_tp, 2)) +
              " mm" + flip_note)

        if n_rock:
            print("  NOTE: " + str(n_rock) +
                  " band(s) skipped (joint fully open / markers lost).")

    return res


if __name__ == "__main__":
    print("=" * 70)
    print("BACKBONE CHECK — all six walls vs wall_test_matrix.csv")
    print("=" * 70)
    check_backbones()

    print()
    print("=" * 70)
    print("DECOMPOSITION SMOKE TEST — PUP1, one mid-amplitude load step")
    print("=" * 70)
    bb = extract_backbone("PUP1")
    df = load_wall("PUP1", "_at_LS")
    d = df["drift_percent"].to_numpy(float)


    target = bb["peaks"]["pos"]["drift_max"] / 3.0
    ls_test = int(np.argmax(d >= target))
    print("chosen LS index " + str(ls_test) + ": drift = " +
          str(round(d[ls_test], 3)) + "% (1/3 of +peak " +
          str(round(bb["peaks"]["pos"]["drift_max"], 3)) + "%)")

    result = decompose_drift("PUP1", ls_test)

    print()
    print("band profiles (chi in 1/mm, gamma dimensionless, Lc in mm):")

    def _format_scientific(value):
        return "{: .3e}".format(value)

    print(result["bands"].to_string(index=False,
                                    float_format=_format_scientific))
