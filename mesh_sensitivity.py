
import sys
import matplotlib
matplotlib.use("Agg")
import numpy as np

import model_builder as mb
from model_builder import (Mesh, tag, course_layout, _brick_edge_nodes,
                           _trib_cols, WALL_T, DY, N_COLS, N_COURSES, H_HALF,
                           H_MASONRY, BEAM_H)

def build_geometry_refined(nsub=2):
    
    m = Mesh()
    dy_sub = DY / nsub

    def cn(course, brick, subcol, r, which):

        side = 'L' if which in ('bl', 'tl') else 'R'
        level = r if which in ('bl', 'br') else r + 1   
        base = 1 if side == 'L' else 2                    
        return tag(course, brick, subcol, 100 + 10 * level + base)

    for course in range(1, N_COURSES + 1):
        y_bot = (course - 1) * DY
        for brick, c0, ncols in course_layout(course):
            x_l, x_r = c0 * H_HALF, (c0 + ncols) * H_HALF
            subcols = ((0, x_l, (c0 + 1) * H_HALF),
                       (1, (c0 + 1) * H_HALF, x_r)) if ncols == 2 \
                else ((0, x_l, x_r),)
            for sub, xa, xb in subcols:
                for r in range(nsub):
                    ya, yb = y_bot + r * dy_sub, y_bot + (r + 1) * dy_sub
                    bl = cn(course, brick, sub, r, 'bl')
                    br = cn(course, brick, sub, r, 'br')
                    tr = cn(course, brick, sub, r, 'tr')
                    tl = cn(course, brick, sub, r, 'tl')
                    for t_, xx, yy in ((bl, xa, ya), (br, xb, ya),
                                       (tr, xb, yb), (tl, xa, yb)):
                        if t_ not in m.nodes:
                            m.add_node(t_, xx, yy)
                    m.brick_quads.append(([bl, br, tr, tl], course, brick, sub))


            info = dict(col_start=c0, ncols=ncols, nsub=nsub)
            m.bricks[(course, brick)] = info

    
    for c in range(N_COLS + 1):
        t = tag(0, c, 0, 1)
        m.add_node(t, c * H_HALF, 0.0)
        m.foundation_nodes[c] = t


    for c in range(N_COLS + 1):
        m.add_node(tag(12, c, 0, 1), c * H_HALF, H_MASONRY)
        m.add_node(tag(12, c, 0, 4), c * H_HALF, H_MASONRY + BEAM_H)
    for c in range(N_COLS):
        m.beam_quads.append(([tag(12, c, 0, 1), tag(12, c + 1, 0, 1),
                              tag(12, c + 1, 0, 4), tag(12, c, 0, 4)], 12, c, 0))


    def add_iface(family, nI, nJ, area, normal_dir, meta):
        m.interfaces.append(dict(family=family, nI=nI, nJ=nJ, area=area,
                                 normal_dir=normal_dir,
                                 shear_dir=1 if normal_dir == 2 else 2,
                                 meta=meta))

    def cn_(course, brick, subcol, r, which):
        return cn(course, brick, subcol, r, which)


    def brick_bottom(course, brick, c0, ncols):
        bottom = {}
        if ncols == 2:
            bottom[c0] = [cn_(course, brick, 0, 0, 'bl')]
            bottom[c0 + 1] = [cn_(course, brick, 0, 0, 'br'),
                              cn_(course, brick, 1, 0, 'bl')]
            bottom[c0 + 2] = [cn_(course, brick, 1, 0, 'br')]
        else:
            bottom[c0] = [cn_(course, brick, 0, 0, 'bl')]
            bottom[c0 + 1] = [cn_(course, brick, 0, 0, 'br')]
        return bottom

    def brick_top(course, brick, c0, ncols):
        top = {}
        rt = nsub - 1
        if ncols == 2:
            top[c0] = [cn_(course, brick, 0, rt, 'tl')]
            top[c0 + 1] = [cn_(course, brick, 0, rt, 'tr'),
                           cn_(course, brick, 1, rt, 'tl')]
            top[c0 + 2] = [cn_(course, brick, 1, rt, 'tr')]
        else:
            top[c0] = [cn_(course, brick, 0, rt, 'tl')]
            top[c0 + 1] = [cn_(course, brick, 0, rt, 'tr')]
        return top


    for (course, brick), info in m.bricks.items():
        if info["ncols"] == 2:
            c0 = info["col_start"]
            levels = [('bl_br', r, 'bot') for r in range(nsub)] + \
                     [('tl_tr', nsub - 1, 'top')]
           
            wsum = nsub  
            for k, (_, r, tb) in enumerate(levels):
                w = 0.5 if k in (0, len(levels) - 1) else 1.0
                a = (w / wsum) * DY * WALL_T
                if tb == 'bot':
                    nI = cn_(course, brick, 0, r, 'br')
                    nJ = cn_(course, brick, 1, r, 'bl')
                else:
                    nI = cn_(course, brick, 0, r, 'tr')
                    nJ = cn_(course, brick, 1, r, 'tl')
                add_iface("middle", nI, nJ, a, 1, (course, brick, tb, k))


    for course in range(1, N_COURSES + 1):
        bricks = course_layout(course)
        for (b1, c1, n1), (b2, c2, n2) in zip(bricks[:-1], bricks[1:]):
            sub1 = 1 if n1 == 2 else 0        
            levels = list(range(nsub))         
            n_levels = nsub + 1
            wsum = nsub
            for k in range(n_levels):
                w = 0.5 if k in (0, n_levels - 1) else 1.0
                a = (w / wsum) * DY * WALL_T
                if k < nsub:                   
                    nI = cn_(course, b1, sub1, k, 'br')
                    nJ = cn_(course, b2, 0, k, 'bl')
                else:                          
                    nI = cn_(course, b1, sub1, nsub - 1, 'tr')
                    nJ = cn_(course, b2, 0, nsub - 1, 'tl')
                add_iface("head", nI, nJ, a, 1, (course, b1, b2, k))


    def bed_between(course_up, lowers, joint_name):
        for brick, c0, ncols in course_layout(course_up):
            up = brick_bottom(course_up, brick, c0, ncols)
            for lb_meta, ltop, lc0, lnc in lowers:
                a0, b0 = max(c0, lc0), min(c0 + ncols, lc0 + lnc)
                for col, w in _trib_cols(a0, b0).items():
                    ups, lows = up[col], ltop[col]
                    area = w * H_HALF * WALL_T / (len(ups) * len(lows))
                    for u in ups:
                        for lo in lows:
                            add_iface(joint_name, lo, u, area, 2,
                                      (course_up, brick, lb_meta, col))

    found_top = {c: [t] for c, t in m.foundation_nodes.items()}
    bed_between(1, [("foundation", found_top, 0, N_COLS)], "base")
    for course in range(2, N_COURSES + 1):
        lowers = [((course - 1, b), brick_top(course - 1, b, c0, nc), c0, nc)
                  for b, c0, nc in course_layout(course - 1)]
        bed_between(course, lowers, "bed")


    beam_bot = {c: [tag(12, c, 0, 1)] for c in range(N_COLS + 1)}
    for brick, c0, ncols in course_layout(N_COURSES):
        topn = brick_top(N_COURSES, brick, c0, ncols)
        for col, w in _trib_cols(c0, c0 + ncols).items():
            ups = beam_bot[col]
            lows = topn[col]
            area = w * H_HALF * WALL_T / (len(ups) * len(lows))
            for u in ups:
                for lo in lows:
                    add_iface("top", lo, u, area, 2,
                              (N_COURSES, brick, "beam", col))

    return m


def verify_refined(m, nsub):

    fam_area = {}
    for f in m.interfaces:
        fam_area[f["family"]] = fam_area.get(f["family"], 0.0) + f["area"]
    joint_area = mb.WALL_L * WALL_T
    for fam, n_levels in (("base", 1), ("top", 1), ("bed", N_COURSES - 1)):
        expect = n_levels * joint_area
        assert abs(fam_area[fam] - expect) < 1e-6 * expect, \
            (fam, fam_area[fam], expect)
    assert abs(fam_area["head"] - 6 * N_COURSES * DY * WALL_T) < 1e-6
    assert abs(fam_area["middle"] - 6 * N_COURSES * DY * WALL_T) < 1e-6
    for f in m.interfaces:
        xi, yi = m.nodes[f["nI"]]
        xj, yj = m.nodes[f["nJ"]]
        assert abs(xi - xj) < 1e-9 and abs(yi - yj) < 1e-9, ("noncoincident", f)
    counts = {fam: sum(1 for f in m.interfaces if f["family"] == fam)
              for fam in fam_area}
    print(f"  nsub={nsub}: invariants OK | interfaces={len(m.interfaces)} "
          f"quads={len(m.brick_quads)} nodes={len(m.nodes)}")
    print(f"           counts={counts}")
    print(f"           areas[m2]={ {k: round(v/1e6,3) for k,v in fam_area.items()} }")
    return fam_area


def build_geometry_hrefined(hsub=2):
    m = Mesh()
    hf = H_HALF / hsub                    
    def hn(course, brick, sub, j, lohi):
        return tag(course, brick, sub, 200 + 10 * j + (1 if lohi == 0 else 4))

    for course in range(1, N_COURSES + 1):
        y0, y1 = (course - 1) * DY, course * DY
        for brick, c0, ncols in course_layout(course):
            halves = ((0, c0), (1, c0 + 1)) if ncols == 2 else ((0, c0),)
            for sub, base_col in halves:
                x_left = base_col * H_HALF
                for j in range(hsub):
                    xa, xb = x_left + j * hf, x_left + (j + 1) * hf
                    bl, br = hn(course, brick, sub, j, 0), hn(course, brick, sub, j + 1, 0)
                    tl, tr = hn(course, brick, sub, j, 1), hn(course, brick, sub, j + 1, 1)
                    for t_, xx, yy in ((bl, xa, y0), (br, xb, y0),
                                       (tr, xb, y1), (tl, xa, y1)):
                        if t_ not in m.nodes:
                            m.add_node(t_, xx, yy)
                    m.brick_quads.append(([bl, br, tr, tl], course, brick, sub))
            m.bricks[(course, brick)] = dict(col_start=c0, ncols=ncols)

    
    NF = N_COLS * hsub
    for fc in range(NF + 1):
        t = tag(0, fc, 0, 1)
        m.add_node(t, fc * hf, 0.0)
        m.foundation_nodes[fc] = t

    for fc in range(NF + 1):
        m.add_node(tag(12, fc, 0, 1), fc * hf, H_MASONRY)          
    for c in range(N_COLS + 1):
        m.add_node(tag(12, c, 0, 4), c * H_HALF, H_MASONRY + BEAM_H)  
    for c in range(N_COLS):
        bl, br = tag(12, c * hsub, 0, 1), tag(12, (c + 1) * hsub, 0, 1)
        tl, tr = tag(12, c, 0, 4), tag(12, c + 1, 0, 4)
        m.beam_quads.append(([bl, br, tr, tl], 12, c, 0))

    def add_iface(family, nI, nJ, area, normal_dir, meta):
        m.interfaces.append(dict(family=family, nI=nI, nJ=nJ, area=area,
                                 normal_dir=normal_dir,
                                 shear_dir=1 if normal_dir == 2 else 2, meta=meta))


    for (course, brick), info in m.bricks.items():
        if info["ncols"] == 2:
            a = 0.5 * DY * WALL_T
    
            add_iface("middle", hn(course, brick, 0, hsub, 0),
                      hn(course, brick, 1, 0, 0), a, 1, (course, brick, "bot"))
            add_iface("middle", hn(course, brick, 0, hsub, 1),
                      hn(course, brick, 1, 0, 1), a, 1, (course, brick, "top"))

    for course in range(1, N_COURSES + 1):
        bricks = course_layout(course)
        for (b1, c1, n1), (b2, c2, n2) in zip(bricks[:-1], bricks[1:]):
            sub1 = 1 if n1 == 2 else 0
            a = 0.5 * DY * WALL_T
            add_iface("head", hn(course, b1, sub1, hsub, 0),
                      hn(course, b2, 0, 0, 0), a, 1, (course, b1, b2, "bot"))
            add_iface("head", hn(course, b1, sub1, hsub, 1),
                      hn(course, b2, 0, 0, 1), a, 1, (course, b1, b2, "top"))

   
    def brick_bottom_fine(course, brick, c0, ncols):
        out = {}
        halves = ((0, c0), (1, c0 + 1)) if ncols == 2 else ((0, c0),)
        for sub, base_col in halves:
            for j in range(hsub + 1):
                fcol = base_col * hsub + j
                out.setdefault(fcol, []).append(hn(course, brick, sub, j, 0))
        return out

    def brick_top_fine(course, brick, c0, ncols):
        out = {}
        halves = ((0, c0), (1, c0 + 1)) if ncols == 2 else ((0, c0),)
        for sub, base_col in halves:
            for j in range(hsub + 1):
                fcol = base_col * hsub + j
                out.setdefault(fcol, []).append(hn(course, brick, sub, j, 1))
        return out

    def _trib_fine(cols_sorted):
        if len(cols_sorted) < 2:
            return {cols_sorted[0]: 1.0} if cols_sorted else {}
        w = {c: 1.0 for c in cols_sorted}
        w[cols_sorted[0]] = 0.5
        w[cols_sorted[-1]] = 0.5
        return w

    def bed_between(course_up, lowers, joint_name):
        for brick, c0, ncols in course_layout(course_up):
            up = brick_bottom_fine(course_up, brick, c0, ncols)
            for lb_meta, ltop, lf_lo, lf_hi in lowers:
                f_lo = max(c0 * hsub, lf_lo)
                f_hi = min((c0 + ncols) * hsub, lf_hi)
                cols = [fc for fc in range(f_lo, f_hi + 1)
                        if fc in up and fc in ltop]
                if len(cols) < 2:
                    continue
                for fc, w in _trib_fine(cols).items():
                    ups, lows = up[fc], ltop[fc]
                    area = w * hf * WALL_T / (len(ups) * len(lows))
                    for u in ups:
                        for lo in lows:
                            add_iface(joint_name, lo, u, area, 2,
                                      (course_up, brick, lb_meta, fc))

    found_top = {fc: [t] for fc, t in m.foundation_nodes.items()}
    bed_between(1, [("foundation", found_top, 0, NF)], "base")
    for course in range(2, N_COURSES + 1):
        lowers = []
        for b, c0, nc in course_layout(course - 1):
            top = brick_top_fine(course - 1, b, c0, nc)
            lowers.append(((course - 1, b), top, c0 * hsub, (c0 + nc) * hsub))
        bed_between(course, lowers, "bed")

    beam_bot = {fc: [tag(12, fc, 0, 1)] for fc in range(NF + 1)}
    for brick, c0, ncols in course_layout(N_COURSES):
        topn = brick_top_fine(N_COURSES, brick, c0, ncols)
        cols = sorted(topn)
        for fc, w in _trib_fine(cols).items():
            ups, lows = beam_bot[fc], topn[fc]
            area = w * hf * WALL_T / (len(ups) * len(lows))
            for u in ups:
                for lo in lows:
                    add_iface("top", lo, u, area, 2,
                              (N_COURSES, brick, "beam", fc))

    return m


def verify_hrefined(m, hsub):
    fam_area = {}
    for f in m.interfaces:
        fam_area[f["family"]] = fam_area.get(f["family"], 0.0) + f["area"]
    joint_area = mb.WALL_L * WALL_T
    for fam, n_levels in (("base", 1), ("top", 1), ("bed", N_COURSES - 1)):
        expect = n_levels * joint_area
        assert abs(fam_area[fam] - expect) < 1e-6 * expect, \
            (fam, fam_area[fam], expect)
    assert abs(fam_area["head"] - 6 * N_COURSES * DY * WALL_T) < 1e-6, fam_area["head"]
    assert abs(fam_area["middle"] - 6 * N_COURSES * DY * WALL_T) < 1e-6, fam_area["middle"]
    for f in m.interfaces:
        xi, yi = m.nodes[f["nI"]]
        xj, yj = m.nodes[f["nJ"]]
        assert abs(xi - xj) < 1e-9 and abs(yi - yj) < 1e-9, ("noncoincident", f)
    counts = {fam: sum(1 for f in m.interfaces if f["family"] == fam)
              for fam in fam_area}
    print(f"  hsub={hsub}: invariants OK | interfaces={len(m.interfaces)} "
          f"quads={len(m.brick_quads)} nodes={len(m.nodes)}")
    print(f"           counts={counts}")
    print(f"           areas[m2]={ {k: round(v/1e6,3) for k,v in fam_area.items()} }")
    return fam_area


import openseespy.opensees as ops


def _wrapped_step(step_mm, lever, depth=0, max_depth=6, log=None):

    ladders = [("Newton", {}, 1.0e-7),
               ("ModifiedNewton", {"-initial": True}, 1.0e-7),
               ("KrylovNewton", {}, 1.0e-7),
               ("Newton", {}, 1.0e-6),
               ("KrylovNewton", {}, 1.0e-5)]
    ops.integrator("DisplacementControl", lever, 1, step_mm)
    for name, kw, tol in ladders:
        ops.test("NormDispIncr", tol, 200, 0)
        if kw:
            ops.algorithm(name, "-initial")
        else:
            ops.algorithm(name)
        if ops.analyze(1) == 0:
            if log is not None and (name != "Newton" or tol != 1.0e-7):
                log.append(f"  step ok via {name} tol={tol:g} "
                           f"(depth {depth}, step {step_mm:.4f} mm)")
            ops.algorithm("Newton")
            ops.test("NormDispIncr", 1.0e-7, 200, 0)
            return True

    if depth >= max_depth:
        if log is not None:
            log.append(f"  BISECTION EXHAUSTED at depth {depth}, "
                       f"step {step_mm:.5f} mm")
        ops.algorithm("Newton")
        ops.test("NormDispIncr", 1.0e-7, 200, 0)
        return False
    if log is not None:
        log.append(f"  bisecting step {step_mm:.4f} -> {step_mm/2:.4f} mm "
                   f"(depth {depth}->{depth+1})")
    ok = (_wrapped_step(step_mm / 2.0, lever, depth + 1, max_depth, log) and
          _wrapped_step(step_mm / 2.0, lever, depth + 1, max_depth, log))
    ops.algorithm("Newton")
    ops.test("NormDispIncr", 1.0e-7, 200, 0)
    return ok


def pushover_wrapped(mesh, handles, target_drift=None, dstep=None):

    lever = handles["lever"]
    beam_row = [mb.tag(12, c, 0, 1) for c in range(mb.N_COLS + 1)]
    if target_drift is None:
        target_drift = mb.TARGET_DRIFT
    if dstep is None:
        dstep = mb.DSTEP

    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(lever, 1.0, 0.0)
    ops.test("NormDispIncr", 1.0e-7, 200, 0)
    ops.algorithm("Newton")

    drift, shear = [0.0], [0.0]
    log = []
    status = "target reached"
    for step in range(mb.MAX_STEPS):
        if not _wrapped_step(dstep, lever, log=log):
            status = f"non-convergence at drift {drift[-1]:.3f}% (wrapper exhausted)"
            break
        u_top = float(np.mean([ops.nodeDisp(t, 1) for t in beam_row]))
        ops.reactions()
        V = -sum(ops.nodeReaction(t)[0] for t in mesh.foundation_nodes.values())
        drift.append(100.0 * u_top / mb.H_EXP)
        shear.append(V / 1000.0)
        if drift[-1] >= target_drift:
            break
    else:
        status = "MAX_STEPS reached"
    if log:
        print(f"    [wrapper log: {len(log)} events]")
        for e in log[:12]:
            print(e)
        if len(log) > 12:
            print(f"    ... (+{len(log)-12} more)")
    return np.asarray(drift), np.asarray(shear), status


def run_level(nsub, axis="v", wrapped=False):

    params = mb.load_params()
    K = mb.derive_stiffnesses(params, verbose=False)
    if nsub == 1:
        mesh = mb.build_geometry()
        mb.verify(mesh)
    elif axis == "v":
        mesh = build_geometry_refined(nsub)
        verify_refined(mesh, nsub)
    else:
        mesh = build_geometry_hrefined(nsub)
        verify_hrefined(mesh, nsub)

    handles = mb.build_opensees_model(mesh, params, K, wall="PUP2",
                                      sigma_map=None)
    N_total = mb.apply_gravity(mesh, handles["cfg"])
    sigma_map, comp_base = mb.harvest_sigma(mesh, handles)
    W_self = mb.GAMMA * mb.WALL_L * H_MASONRY * WALL_T
    expect = N_total + W_self

    import openseespy.opensees as _ops
    _ops.reactions()
    Ry = sum(_ops.nodeReaction(t, 2) for t in mesh.foundation_nodes.values())
    err = abs(abs(Ry) - expect) / expect
    assert err < 0.005, f"nsub={nsub} base equilibrium failed ({100*err:.3f}%)"

    handles = mb.build_opensees_model(mesh, params, K, wall="PUP2",
                                      sigma_map=sigma_map)
    mb.apply_gravity(mesh, handles["cfg"])

    if wrapped:
        drift, shear, status = pushover_wrapped(mesh, handles)
    else:
        drift, shear, status = mb.pushover(mesh, handles)
    i_pk = int(np.argmax(shear))
    n0 = min(5, len(drift) - 1)
    K0 = (shear[n0] - shear[0]) / ((drift[n0] - drift[0]) / 100.0 * mb.H_EXP) \
        if n0 > 0 else float("nan")
    dmg = mb.damage_state(handles)
    return dict(nsub=nsub, peakV=shear[i_pk], peak_drift=drift[i_pk],
                endV=shear[-1], end_drift=drift[-1], K0=K0, status=status,
                n_iface=len(mesh.interfaces), n_quad=len(mesh.brick_quads),
                n_shear=sum(d["sheared"] for d in dmg),
                n_open=sum(d["opened"] for d in dmg),
                comp_base=abs(Ry) / 1e3)


def main(nsub_fine=2, axis="v", wrapped=False):
    levels = [1, nsub_fine]
    ax = "vertical" if axis == "v" else "horizontal (sub-column)"
    label = "NSUB" if axis == "v" else "HSUB"
    print("=" * 72)
    print(f"MESH SENSITIVITY — PUP2 pushover, {ax} subdivision {label} in {levels}"
          + ("  [wrapped solver]" if wrapped else ""))
    print("=" * 72)
    res = []
    for ns in levels:
        print(f"\n--- {label} = {ns} " + "-" * 50)
        res.append(run_level(ns, axis=axis, wrapped=wrapped))

    b, f = res[0], res[1]
    def pct(a, c): return 100.0 * (c - a) / a if a else float("nan")
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'quantity':<22}{'nsub=1 (base)':>16}{'nsub='+str(nsub_fine):>16}{'delta':>12}")
    print(f"{'peak V [kN]':<22}{b['peakV']:>16.1f}{f['peakV']:>16.1f}{pct(b['peakV'],f['peakV']):>11.1f}%")
    print(f"{'peak drift [%]':<22}{b['peak_drift']:>16.3f}{f['peak_drift']:>16.3f}{pct(b['peak_drift'],f['peak_drift']):>11.1f}%")
    print(f"{'end V [kN]':<22}{b['endV']:>16.1f}{f['endV']:>16.1f}{pct(b['endV'],f['endV']):>11.1f}%")
    print(f"{'K0 [kN/mm]':<22}{b['K0']:>16.1f}{f['K0']:>16.1f}{pct(b['K0'],f['K0']):>11.1f}%")
    print(f"{'interfaces':<22}{b['n_iface']:>16d}{f['n_iface']:>16d}")
    print(f"{'brick quads':<22}{b['n_quad']:>16d}{f['n_quad']:>16d}")
    print(f"{'shear-yielded':<22}{b['n_shear']:>16d}{f['n_shear']:>16d}")
    print(f"{'tension-opened':<22}{b['n_open']:>16d}{f['n_open']:>16d}")
    print(f"{'base comp [kN]':<22}{b['comp_base']:>16.1f}{f['comp_base']:>16.1f}")
    print("-" * 72)
    dV = abs(pct(b['peakV'], f['peakV']))
    dK = abs(pct(b['K0'], f['K0']))
    verdict = ("CONVERGED" if dV < 5 and dK < 10 else
               "SENSITIVE — investigate")
    print(f"peak V shift {dV:.1f}%, K0 shift {dK:.1f}%  ->  {verdict}")
    print(f"status base: {b['status']}")
    print(f"status fine: {f['status']}")
    return res


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    axis = sys.argv[2] if len(sys.argv) > 2 else "v"
    wrapped = len(sys.argv) > 3 and sys.argv[3].lower().startswith("w")
    main(ns, axis=axis, wrapped=wrapped)

