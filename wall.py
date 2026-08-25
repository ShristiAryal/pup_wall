
import pandas as pd
from optical_data import load_conventional_at_ls, BASE_DIR

WALLS = ["PUP1", "PUP2", "PUP3", "PUP4", "PUP5", "PUP6"]

FORCE_COLUMN = "F1"

def load_wall(wall, variant="_at_LS", base_dir=BASE_DIR):
  
    if variant != "_at_LS":
        raise NotImplementedError(
            "load_wall only knows how to load variant '_at_LS', "
            "but you asked for: " + str(variant)
        )

    df = load_conventional_at_ls(wall, base_dir=base_dir)
    df = df.copy()

    needed_columns = ["Drift_TP", "Disp_TP"]
    missing_columns = []
    for col in needed_columns:
        if col not in df.columns:
            missing_columns.append(col)

    if len(missing_columns) > 0:
        raise KeyError(
            wall + ": missing column(s) " + str(missing_columns) +
            ". Columns found in file: " + str(list(df.columns))
        )

    df["drift_percent"] = pd.to_numeric(df["Drift_TP"], errors="coerce")
    df["top_plate_disp_mm"] = pd.to_numeric(df["Disp_TP"], errors="coerce")

    if FORCE_COLUMN in df.columns:
        df["force_horizontal_kN"] = pd.to_numeric(df[FORCE_COLUMN], errors="coerce")
    else:
    
        df["force_horizontal_kN"] = float("nan")
        print(
            "NOTE " + wall + ": column '" + FORCE_COLUMN + "' not found, "
            "so force_horizontal_kN is set to NaN (missing)."
        )

    return df


if __name__ == "__main__":
    walls_to_check = ["PUP2", "PUP3"]

    for wall_name in walls_to_check:
        data = load_wall(wall_name)

        first_drift = data["drift_percent"].iloc[0]
        first_disp = data["top_plate_disp_mm"].iloc[0]
        second_force = data["force_horizontal_kN"].iloc[1]

        print(
            wall_name + ": " + str(len(data)) + " rows  "
            "drift_percent[0]=" + "{:+.5f}".format(first_drift) + "  "
            "Disp_TP[0]=" + "{:+.5f}".format(first_disp) + "  "
            "F1[1]=" + "{:+.3f}".format(second_force) + " kN"
        )


        assert abs(first_drift) < 1e-3, wall_name + ": row 0 should be LS1 (drift ~ 0)"

    print("wall.py self-check OK (row 0 = paper LS1, drift ~ 0).")
