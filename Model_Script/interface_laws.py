
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openseespy.opensees as ops
import pandas as pd

if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid
elif hasattr(np, "trapz"):
    _trapz = np.trapz
else:
    _trapz = None


REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "docs" / "material_properties.csv"
FIG_DIR = REPO_ROOT / "results" / "figures"


ENERGY_TOL = 0.03


K_N_BED = 25.4   # N/mm^3
K_S_BED = 5.54   # N/mm^3


def load_params(csv_path=CSV_PATH):
    
    df = pd.read_csv(csv_path)
    params = {}

    for _, row in df.iterrows():
        symbol = str(row["symbol"]).strip()
        value = row["value"]
        prior = row["prior"]

        if pd.notna(value):
            params[symbol] = float(value)
        elif pd.notna(prior):
            params[symbol] = float(prior)
        

    return params


def get_param(params, symbol):

    if symbol not in params:
        raise KeyError(
            "'" + symbol + "' has neither value nor prior in " +
            CSV_PATH.name + " — fill the prior column before running."
        )
    return params[symbol]

def _softening_points(delta1, peak, base, area_soft):
    
    if area_soft <= 0:
        raise ValueError("softening area budget must be positive")

    a = area_soft / peak
    delta2 = delta1 + a
    delta3 = delta1 + 3.0 * a
    stress2 = base + peak / 3.0
    stress3 = base + max(1e-3 * peak, 1e-6)
    return delta2, stress2, delta3, stress3


def mode_i_backbone(f_t, G_fI, k_n):

    delta1 = f_t / k_n
    elastic_area = 0.5 * f_t * delta1
    if elastic_area >= G_fI:
        raise ValueError("elastic energy exceeds G_fI — check k_n / f_t units")

    delta2, stress2, delta3, stress3 = _softening_points(
        delta1, f_t, 0.0, G_fI - elastic_area)
    return [(f_t, delta1), (stress2, delta2), (stress3, delta3)]

def mode_i_compression_provisional(f_u, k_n):

    delta1 = f_u / k_n
    return [(-f_u, -delta1),
            (-0.95 * f_u, -3.0 * delta1),
            (-0.90 * f_u, -6.0 * delta1)]

def mode_ii_backbone(c, mu_peak, mu_res, sigma_n, G_fII, k_s):

    tau_p = c + mu_peak * sigma_n
    tau_r = mu_res * sigma_n
    excess = tau_p - tau_r
    if excess <= 0:
        raise ValueError("tau_p <= tau_r: no softening branch to construct")

    delta1 = tau_p / k_s
    delta2, stress2, delta3, stress3 = _softening_points(
        delta1, excess, tau_r, G_fII)
    backbone_points = [(tau_p, delta1), (stress2, delta2), (stress3, delta3)]
    return backbone_points, tau_r

def make_normal_material(tag, f_t, G_fI, f_u, k_n, area=1.0, beta=1.0,
                         damage1=0.0, damage2=0.0):

    tension_stress_points = mode_i_backbone(f_t, G_fI, k_n)
    tension_force_points = []
    for stress, disp in tension_stress_points:
        tension_force_points.append((stress * area, disp))

    compression_stress_points = mode_i_compression_provisional(f_u, k_n)
    compression_force_points = []
    for stress, disp in compression_stress_points:
        compression_force_points.append((stress * area, disp))

    ops.uniaxialMaterial(
        "Hysteretic", tag,
        *tension_force_points[0], *tension_force_points[1],
        *tension_force_points[2],
        *compression_force_points[0], *compression_force_points[1],
        *compression_force_points[2],
        1.0, 1.0, damage1, damage2, beta,
    )
    return tension_force_points, compression_force_points


def make_shear_material(tag, c, mu_peak, mu_res, sigma_n, G_fII, k_s,
                        area=1.0, beta=1.0, damage1=0.0, damage2=0.0):

    backbone_stress_points, tau_r = mode_ii_backbone(
        c, mu_peak, mu_res, sigma_n, G_fII, k_s)

    positive_side = []
    for stress, disp in backbone_stress_points:
        positive_side.append((stress * area, disp))

    negative_side = []
    for force, disp in positive_side:
        negative_side.append((-force, -disp))

    ops.uniaxialMaterial(
        "Hysteretic", tag,
        *positive_side[0], *positive_side[1], *positive_side[2],
        *negative_side[0], *negative_side[1], *negative_side[2],
        1.0, 1.0, damage1, damage2, beta,
    )
    return positive_side, tau_r * area

def _fresh_joint():

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    ops.fix(1, 1, 1)

def _attach(mats_dirs):

    tags = []
    dirs = []
    for material_tag, direction in mats_dirs:
        tags.append(material_tag)
        dirs.append(direction)
    ops.element("zeroLength", 1, 1, 2, "-mat", *tags, "-dir", *dirs)


def _analysis_setup():
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1.0e-8, 200, 0)
    ops.algorithm("Newton")


def _run_disp_history(dof, targets, d_incr, record_dofs=(1, 2)):
    
    disp = {}
    force = {}
    for d in record_dofs:
        disp[d] = [ops.nodeDisp(2, d)]
        force[d] = [-ops.eleForce(1)[d - 1]]

    current = ops.nodeDisp(2, dof)
    for target in targets:
        n_steps = max(1, int(round(abs(target - current) / d_incr)))
        step = (target - current) / n_steps
        ops.integrator("DisplacementControl", 2, dof, step)
        ops.analysis("Static")

        for _ in range(n_steps):
            ok = ops.analyze(1)
            if ok != 0:
                
                ops.algorithm("ModifiedNewton", "-initial")
                ok = ops.analyze(1)
                ops.algorithm("Newton")
            if ok != 0:
                raise RuntimeError(
                    "single-joint rig failed to converge at u" + str(dof) +
                    " = " + str(round(ops.nodeDisp(2, dof), 4)) + " mm")
            for d in record_dofs:
                disp[d].append(ops.nodeDisp(2, d))

                force[d].append(-ops.eleForce(1)[d - 1])
        current = target

    disp_arrays = {}
    force_arrays = {}
    for d in record_dofs:
        disp_arrays[d] = np.asarray(disp[d])
        force_arrays[d] = np.asarray(force[d])
    return disp_arrays, force_arrays

def test_tension(params, cyclic=False):

    f_t = get_param(params, "f_t_joint")
    G_fI = get_param(params, "G_fI")
    f_u = get_param(params, "f_u")

    _fresh_joint()
    ops.fix(2, 1, 0)
    tension_points, _ = make_normal_material(101, f_t, G_fI, f_u, K_N_BED)
    _attach([(101, 2)])
    _analysis_setup()

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, 1.0)

    delta3 = tension_points[2][1]
    d_end = 1.15 * delta3
    if cyclic:
        d_mid = 0.5 * (tension_points[0][1] + tension_points[1][1])
        targets = [d_mid, 1e-4, d_end]
    else:
        targets = [d_end]

    disp, force = _run_disp_history(2, targets, d_incr=delta3 / 400.0)
    return disp[2], force[2], tension_points


def test_shear(params, sigma_n, cyclic=False):
   
    c = get_param(params, "c")
    mu_peak = get_param(params, "mu_peak")
    mu_res = get_param(params, "mu_res")
    G_fII = get_param(params, "G_fII")
    f_t = get_param(params, "f_t_joint")
    G_fI = get_param(params, "G_fI")
    f_u = get_param(params, "f_u")

    _fresh_joint()
    make_normal_material(101, f_t, G_fI, f_u, K_N_BED)
    shear_points, tau_r = make_shear_material(
        102, c, mu_peak, mu_res, sigma_n, G_fII, K_S_BED)
    _attach([(102, 1), (101, 2)])
    _analysis_setup()

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, -sigma_n)          
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("normal pre-compression stage failed")
    ops.loadConst("-time", 0.0)

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 1.0, 0.0)

    delta3 = shear_points[2][1]
    d_end = 1.2 * delta3
    if cyclic:
        d_mid = 0.5 * (shear_points[0][1] + shear_points[1][1])
        targets = [d_mid, 1e-4, d_end]
    else:
        targets = [d_end]

    disp, force = _run_disp_history(1, targets, d_incr=delta3 / 400.0)
    return disp[1], force[1], shear_points, tau_r


def _monotonic_mask(u):
    
    return np.ones_like(u, dtype=bool)

def check_mode_i(u, f, G_fI, f_t):
    
    results = {}

    measured_area = float(_trapz(f, u))
    results["G_fI measured [N/mm]"] = measured_area
    results["G_fI target   [N/mm]"] = G_fI
    results["G_fI rel. err"] = abs(measured_area - G_fI) / G_fI

    peak_stress = float(f.max())
    results["peak stress [MPa]"] = peak_stress
    results["peak target [MPa]"] = f_t
    results["peak rel. err"] = abs(peak_stress - f_t) / f_t

    ok = (results["G_fI rel. err"] < ENERGY_TOL
          and results["peak rel. err"] < 0.01)
    return ok, results

def check_mode_ii(u, f, G_fII, tau_p_target, tau_r, d3):

    results = {}

    i_peak = int(np.argmax(f))
    excess = f[i_peak:] - tau_r
    measured_area = float(_trapz(excess, u[i_peak:]))
    results["G_fII measured [N/mm]"] = measured_area
    results["G_fII target   [N/mm]"] = G_fII
    results["G_fII rel. err"] = abs(measured_area - G_fII) / G_fII

    peak_stress = float(f.max())
    results["tau_p measured [MPa]"] = peak_stress
    results["tau_p target   [MPa]"] = tau_p_target
    results["tau_p rel. err"] = abs(peak_stress - tau_p_target) / tau_p_target

    tail = f[u > d3]
    if tail.size:
        plateau = float(tail.mean())
    else:
        plateau = float("nan")
    results["plateau measured [MPa]"] = plateau
    results["plateau target   [MPa]"] = tau_r
    if tail.size:
        results["plateau rel. err"] = abs(plateau - tau_r) / tau_r
    else:
        results["plateau rel. err"] = float("inf")

    ok = (results["G_fII rel. err"] < ENERGY_TOL
          and results["tau_p rel. err"] < 0.01
          and results["plateau rel. err"] < 0.02)
    return ok, results


def report_cyclic(u, f, label):

    du = np.diff(u)
    turning_points = np.where(du < 0)[0]
    if not turning_points.size:
        print("    " + label + ": no unloading detected")
        return float("nan")

    i_peak = int(turning_points[0])

    i_end = i_peak + int(np.argmin(u[i_peak:]))
    seg_u = u[i_peak:i_end + 1]
    seg_f = f[i_peak:i_end + 1]

    crossing = np.where(seg_f <= 0.0)[0]
    if crossing.size:
        j = crossing[0]
        if j == 0:
            u_zero = float(seg_u[0])
        else:

            f1, f2 = seg_f[j - 1], seg_f[j]
            u1, u2 = seg_u[j - 1], seg_u[j]
            u_zero = float(u1 + (0.0 - f1) * (u2 - u1) / (f2 - f1))
        print("    " + label + ": unloaded from u = " +
              str(round(float(u[i_peak]), 4)) + " mm; force crosses 0 at u = " +
              str(round(u_zero, 4)) + " mm (secant-to-origin -> 0.0000)")
        return u_zero

    print("    " + label + ": unloaded from u = " +
          str(round(float(u[i_peak]), 4)) + " mm to u = " +
          str(round(float(seg_u[-1]), 4)) +
          " mm without force crossing 0 (residual force there = " +
          str(round(float(seg_f[-1]), 4)) + " MPa — secant-like)")
    return 0.0

def _plot(u, f, backbone_pts, title, fname, tau_r=None):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(u, f, lw=1.4, label="Hysteretic response")

    # build the backbone line, starting from the origin
    backbone_disp = [0.0]
    backbone_stress = [0.0]
    for stress, disp in backbone_pts:
        backbone_disp.append(disp)
        backbone_stress.append(stress)
    ax.plot(backbone_disp, backbone_stress, "o--", ms=4, lw=0.9,
            label="design backbone")

    if tau_r is not None:
        ax.axhline(tau_r, ls=":", lw=0.9, color="k",
                   label=r"$\tau_r=\mu_{res}\,\sigma_n$")

    ax.set_xlabel("relative displacement [mm]")
    ax.set_ylabel("stress [MPa]  (unit tributary area)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=200)
    plt.close(fig)

def main():
    params = load_params()
    G_fI = get_param(params, "G_fI")
    G_fII = get_param(params, "G_fII")
    f_t = get_param(params, "f_t_joint")
    c = get_param(params, "c")
    mu_peak = get_param(params, "mu_peak")
    mu_res = get_param(params, "mu_res")

    sigma_n = 1.05  # MPa

    verdicts = {}

    print("=" * 68)
    print("TEST 1 — direct tension, monotonic (Mode-I / G_fI)")
    u, f, tension_points = test_tension(params, cyclic=False)
    ok, results = check_mode_i(u, f, G_fI, f_t)
    verdicts["T1 tension monotonic"] = ok
    for key in results:
        print("    " + key.ljust(26) + " " + str(round(results[key], 5)))
    _plot(u, f, tension_points, "Mode-I tension — monotonic",
          "day3_tension_mono.png")

    print("TEST 2 — direct tension, one unload-reload cycle")
    u, f, tension_points = test_tension(params, cyclic=True)
    report_cyclic(u, f, "tension")
    verdicts["T2 tension cyclic (observational)"] = True
    _plot(u, f, tension_points, "Mode-I tension — unload/reload (beta=1.0)",
          "day3_tension_cyclic.png")

    print("TEST 3 — shear under sigma_n = " + str(round(sigma_n, 2)) +
          " MPa, monotonic (Mode-II)")
    u, f, shear_points, tau_r = test_shear(params, sigma_n, cyclic=False)
    tau_p_target = c + mu_peak * sigma_n
    ok, results = check_mode_ii(u, f, G_fII, tau_p_target, tau_r,
                                shear_points[2][1])
    verdicts["T3 shear monotonic"] = ok
    for key in results:
        print("    " + key.ljust(26) + " " + str(round(results[key], 5)))
    _plot(u, f, shear_points, "Mode-II shear — monotonic",
          "day3_shear_mono.png", tau_r=tau_r)

    print("TEST 4 — shear, one unload-reload cycle")
    u, f, shear_points, tau_r = test_shear(params, sigma_n, cyclic=True)
    report_cyclic(u, f, "shear")
    verdicts["T4 shear cyclic (observational)"] = True
    _plot(u, f, shear_points, "Mode-II shear — unload/reload (beta=1.0)",
          "day3_shear_cyclic.png", tau_r=tau_r)

    print("=" * 68)
    all_ok = True
    for name in verdicts:
        if not verdicts[name]:
            all_ok = False
        if verdicts[name]:
            status = "PASS"
        else:
            status = "FAIL"
        print("  [" + status + "] " + name)
    print("=" * 68)
    print("Concrete02 fallback decision: inspect day3_tension_cyclic.png —"
          "\n  if unloading/reloading shows spurious force overshoot or"
          "\n  non-physical crack re-closure, invoke the pre-authorised"
          "\n  fallback and log it under Decision 004.")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

def cohesive_backbone(c, G_fII, k_s):

    if c <= 0:
        raise ValueError("cohesion c must be > 0 for the cohesive spring")

    delta1 = c / k_s
    delta2, stress2, delta3, stress3 = _softening_points(delta1, c, 0.0, G_fII)
    return [(c, delta1), (stress2, delta2), (stress3, delta3)]


def make_shear_material_hybrid(tag_coh, tag_fric, c, mu, sigma_n, G_fII, k_s,
                               area=1.0, beta=1.0, damage1=0.0, damage2=0.0,
                               fric_frac=0.05):

    friction_stress = mu * sigma_n
    if friction_stress <= 0.0:
        friction_stress = 1e-9

    d_y = (c + friction_stress) / k_s
    k_s_cohesion = c / d_y

    cohesion_stress_points = cohesive_backbone(c, G_fII, k_s_cohesion)
    cohesion_force_points = []
    for stress, disp in cohesion_stress_points:
        cohesion_force_points.append((stress * area, disp))

    cohesion_force_points_negative = []
    for force, disp in cohesion_force_points:
        cohesion_force_points_negative.append((-force, -disp))

    ops.uniaxialMaterial(
        "Hysteretic", tag_coh,
        *cohesion_force_points[0], *cohesion_force_points[1],
        *cohesion_force_points[2],
        *cohesion_force_points_negative[0],
        *cohesion_force_points_negative[1],
        *cohesion_force_points_negative[2],
        1.0, 1.0, damage1, damage2, beta,
    )

    F_fric = friction_stress * area
    floor_point_1 = (F_fric, d_y)
    floor_point_2 = (F_fric, 100.0 * d_y)
    floor_point_3 = (F_fric, 1.0e4 * d_y)
    floor_point_1_neg = (-floor_point_1[0], -floor_point_1[1])
    floor_point_2_neg = (-floor_point_2[0], -floor_point_2[1])
    floor_point_3_neg = (-floor_point_3[0], -floor_point_3[1])

    ops.uniaxialMaterial(
        "Hysteretic", tag_fric,
        *floor_point_1, *floor_point_2, *floor_point_3,
        *floor_point_1_neg, *floor_point_2_neg, *floor_point_3_neg,
        1.0, 1.0, 0.0, 0.0, beta,
    )
    return cohesion_force_points, F_fric


def coulomb_available():
   
    return True

def test_shear_hybrid(params, sigma_n, cyclic=False):

    c = get_param(params, "c")
    mu_res = get_param(params, "mu_res")
    G_fII = get_param(params, "G_fII")
    f_t = get_param(params, "f_t_joint")
    G_fI = get_param(params, "G_fI")
    f_u = get_param(params, "f_u")

    _fresh_joint()
    make_normal_material(101, f_t, G_fI, f_u, K_N_BED)
    cohesion_points, F_fric = make_shear_material_hybrid(
        102, 103, c, mu_res, sigma_n, G_fII, K_S_BED)

    ops.element("zeroLength", 1, 1, 2, "-mat", 102, 103, 101,
                "-dir", 1, 1, 2)
    _analysis_setup()

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, -sigma_n)
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    if ops.analyze(10) != 0:
        raise RuntimeError("hybrid rig: normal pre-compression stage failed")
    ops.loadConst("-time", 0.0)

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 1.0, 0.0)

    delta3 = cohesion_points[2][1]
    d_end = 2.0 * delta3   
    if cyclic:
        d_mid = 0.5 * (cohesion_points[0][1] + cohesion_points[1][1])
        targets = [d_mid, 1e-4, d_end]
    else:
        targets = [d_end]

    disp, force = _run_disp_history(1, targets, d_incr=delta3 / 400.0)
    return disp[1], force[1], cohesion_points, F_fric, mu_res * sigma_n


def check_hybrid(u, f, c, tau_r, k_s_area):

    results = {}

    peak = float(f.max())
    tau_p_target = c + tau_r
    results["tau_p measured"] = peak
    results["tau_p target (c+mu*sig)"] = tau_p_target
    results["tau_p rel. err"] = abs(peak - tau_p_target) / tau_p_target

    n = len(u)
    tail = f[int(0.8 * n):]
    if tail.size:
        plateau = float(tail.mean())
    else:
        plateau = float("nan")
    results["plateau measured"] = plateau
    results["plateau target (mu*sig)"] = tau_r
    if tau_r:
        results["plateau rel. err"] = abs(plateau - tau_r) / tau_r
    else:
        results["plateau rel. err"] = float("inf")

    tau_p_for_dy = c + tau_r
    if k_s_area:
        d_y = tau_p_for_dy / k_s_area
    else:
        d_y = 0.0

    if d_y > 0:
        window = (u > 1e-9) & (u < 0.5 * d_y)
        if window.sum() >= 2:
            K0 = float(np.polyfit(u[window], f[window], 1)[0])
        else:
            rising = np.where(u > 1e-9)[0]
            if rising.size >= 2:
                K0 = float((f[rising[1]] - f[rising[0]]) /
                          (u[rising[1]] - u[rising[0]]))
            else:
                K0 = float("nan")
    else:
        K0 = float("nan")

    results["K0 measured"] = K0
    results["K0 target (k_s*area)"] = k_s_area
    if k_s_area:
        results["K0 rel. err"] = abs(K0 - k_s_area) / k_s_area
    else:
        results["K0 rel. err"] = float("inf")

    ok = (results["tau_p rel. err"] < 0.05
          and results["plateau rel. err"] < 0.05
          and results["K0 rel. err"] < 0.10)
    return ok, results
