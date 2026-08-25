

import openseespy.opensees as ops

from interface_laws import (get_param, load_params, make_normal_material,
                            make_shear_material, make_shear_material_hybrid)


WALL_L = 2010.0          
WALL_T = 200.0           
N_COURSES = 11           
N_COLS = 13              
H_HALF = WALL_L / N_COLS 
DY = 200.0               
H_MASONRY = N_COURSES * DY   
H_EXP = 2250.0           
BEAM_H = 200.0           

H_M = 10.0               
HB_MODULE = 190.0        
GAMMA = 9.0e-6           

WALL_CONFIG = {
    "PUP2": {"sigma0": 1.05, "H0_over_H": 0.75},
}

E_BEAM_FACTOR = 50.0     
TRUSS_EA = 2.0e10        


def tag(course, brick, subcol, corner):
    
    return 1_000_000 * course + 10_000 * brick + 100 * subcol + corner

def course_layout(course):
  
    bricks_in_this_course = []
    current_col = 0

    if course % 2 == 1:
        
        bricks_in_this_course.append((0, 0, 1))
        current_col = 1
        brick_index = 1
        while current_col < N_COLS:
            bricks_in_this_course.append((brick_index, current_col, 2))
            current_col = current_col + 2
            brick_index = brick_index + 1
    else:
    
        brick_index = 0
        while current_col < N_COLS - 1:
            bricks_in_this_course.append((brick_index, current_col, 2))
            current_col = current_col + 2
            brick_index = brick_index + 1
        bricks_in_this_course.append((brick_index, N_COLS - 1, 1))

    return bricks_in_this_course

class Mesh:

    def __init__(self):
        self.nodes = {}
        self.brick_quads = []      
        self.beam_quads = []       
        self.foundation_nodes = {} 
        self.interfaces = []       
        self.bricks = {}           

    def add_node(self, node_tag, x, y):
        if node_tag in self.nodes:
            raise ValueError("duplicate node tag: " + str(node_tag))
        self.nodes[node_tag] = (float(x), float(y))


def _brick_edge_nodes(course, brick, col_start, ncols):
    
    bottom = {}
    top = {}

    if ncols == 2:
        bottom[col_start] = [tag(course, brick, 0, 1)]
        bottom[col_start + 1] = [tag(course, brick, 0, 2),
                                  tag(course, brick, 1, 1)]
        bottom[col_start + 2] = [tag(course, brick, 1, 2)]

        top[col_start] = [tag(course, brick, 0, 4)]
        top[col_start + 1] = [tag(course, brick, 0, 3),
                               tag(course, brick, 1, 4)]
        top[col_start + 2] = [tag(course, brick, 1, 3)]
    else:
        bottom[col_start] = [tag(course, brick, 0, 1)]
        bottom[col_start + 1] = [tag(course, brick, 0, 2)]

        top[col_start] = [tag(course, brick, 0, 4)]
        top[col_start + 1] = [tag(course, brick, 0, 3)]

    return bottom, top


def _trib_cols(a, b):
  
    weights = {}
    if b <= a:
        return weights

    for c in range(a, b + 1):
        weights[c] = 1.0
    weights[a] = 0.5
    weights[b] = 0.5
    return weights


def build_geometry():
    m = Mesh()

    for course in range(1, N_COURSES + 1):
        y0 = (course - 1) * DY
        y1 = course * DY
        bricks_in_course = course_layout(course)

        for brick, c0, ncols in bricks_in_course:
            x_left = c0 * H_HALF
            x_right = (c0 + ncols) * H_HALF

            if ncols == 2:
                
                x_mid = (c0 + 1) * H_HALF
                halves = [(0, x_left, x_mid), (1, x_mid, x_right)]
                for sub, xa, xb in halves:
                    m.add_node(tag(course, brick, sub, 1), xa, y0)
                    m.add_node(tag(course, brick, sub, 2), xb, y0)
                    m.add_node(tag(course, brick, sub, 3), xb, y1)
                    m.add_node(tag(course, brick, sub, 4), xa, y1)
                    quad_nodes = [tag(course, brick, sub, 1),
                                  tag(course, brick, sub, 2),
                                  tag(course, brick, sub, 3),
                                  tag(course, brick, sub, 4)]
                    m.brick_quads.append((quad_nodes, course, brick, sub))
            else:
                
                m.add_node(tag(course, brick, 0, 1), x_left, y0)
                m.add_node(tag(course, brick, 0, 2), x_right, y0)
                m.add_node(tag(course, brick, 0, 3), x_right, y1)
                m.add_node(tag(course, brick, 0, 4), x_left, y1)
                quad_nodes = [tag(course, brick, 0, 1),
                              tag(course, brick, 0, 2),
                              tag(course, brick, 0, 3),
                              tag(course, brick, 0, 4)]
                m.brick_quads.append((quad_nodes, course, brick, 0))

            bottom_nodes, top_nodes = _brick_edge_nodes(course, brick, c0, ncols)
            m.bricks[(course, brick)] = {
                "col_start": c0,
                "ncols": ncols,
                "bottom": bottom_nodes,
                "top": top_nodes,
            }


    for c in range(N_COLS + 1):
        foundation_tag = tag(0, c, 0, 1)
        m.add_node(foundation_tag, c * H_HALF, 0.0)
        m.foundation_nodes[c] = foundation_tag


    for c in range(N_COLS + 1):
        m.add_node(tag(12, c, 0, 1), c * H_HALF, H_MASONRY)
        m.add_node(tag(12, c, 0, 4), c * H_HALF, H_MASONRY + BEAM_H)
    for c in range(N_COLS):
        quad_nodes = [tag(12, c, 0, 1), tag(12, c + 1, 0, 1),
                      tag(12, c + 1, 0, 4), tag(12, c, 0, 4)]
        m.beam_quads.append((quad_nodes, 12, c, 0))


    def add_iface(family, nI, nJ, area, normal_dir, meta):
        
        if normal_dir == 2:
            shear_dir = 1
        else:
            shear_dir = 2
        new_interface = {
            "family": family,
            "nI": nI,
            "nJ": nJ,
            "area": area,
            "normal_dir": normal_dir,
            "shear_dir": shear_dir,
            "meta": meta,
        }
        m.interfaces.append(new_interface)


    for key in m.bricks:
        course, brick = key
        info = m.bricks[key]
        if info["ncols"] == 2:
            a = 0.5 * DY * WALL_T
            add_iface("middle", tag(course, brick, 0, 2),
                      tag(course, brick, 1, 1), a, 1, (course, brick, "bot"))
            add_iface("middle", tag(course, brick, 0, 3),
                      tag(course, brick, 1, 4), a, 1, (course, brick, "top"))


    for course in range(1, N_COURSES + 1):
        bricks_in_course = course_layout(course)
        for i in range(len(bricks_in_course) - 1):
            b1, c1, n1 = bricks_in_course[i]
            b2, c2, n2 = bricks_in_course[i + 1]
            a = 0.5 * DY * WALL_T
            if n1 == 2:
                right_corner_sub = 1
            else:
                right_corner_sub = 0
            add_iface("head", tag(course, b1, right_corner_sub, 2),
                      tag(course, b2, 0, 1), a, 1, (course, b1, b2, "bot"))
            add_iface("head", tag(course, b1, right_corner_sub, 3),
                      tag(course, b2, 0, 4), a, 1, (course, b1, b2, "top"))

    def bed_between(course_up, lower_bricks_edges, joint_name):
        
        bricks_in_course = course_layout(course_up)
        for brick, c0, ncols in bricks_in_course:
            upper_nodes = m.bricks[(course_up, brick)]["bottom"]
            for lb_meta, lower_top_nodes, lc0, lnc in lower_bricks_edges:
                overlap_start = max(c0, lc0)
                overlap_end = min(c0 + ncols, lc0 + lnc)
                weights = _trib_cols(overlap_start, overlap_end)
                for col in weights:
                    w = weights[col]
                    upper_node_list = upper_nodes[col]
                    lower_node_list = lower_top_nodes[col]
                    area = (w * H_HALF * WALL_T /
                            (len(upper_node_list) * len(lower_node_list)))
                    for u in upper_node_list:
                        for lo in lower_node_list:
                            add_iface(joint_name, lo, u, area, 2,
                                      (course_up, brick, lb_meta, col))


    found_top = {}
    for c in m.foundation_nodes:
        found_top[c] = [m.foundation_nodes[c]]
    bed_between(1, [("foundation", found_top, 0, N_COLS)], "base")


    for course in range(2, N_COURSES + 1):
        lowers = []
        for b, c0, nc in course_layout(course - 1):
            lower_info = ((course - 1, b), m.bricks[(course - 1, b)]["top"],
                          c0, nc)
            lowers.append(lower_info)
        bed_between(course, lowers, "bed")


    beam_bottom_nodes = {}
    for c in range(N_COLS + 1):
        beam_bottom_nodes[c] = [tag(12, c, 0, 1)]

    for brick, c0, ncols in course_layout(N_COURSES):
        top_nodes = m.bricks[(N_COURSES, brick)]["top"]
        weights = _trib_cols(c0, c0 + ncols)
        for col in weights:
            w = weights[col]
            upper_node_list = beam_bottom_nodes[col]
            lower_node_list = top_nodes[col]
            area = (w * H_HALF * WALL_T /
                    (len(upper_node_list) * len(lower_node_list)))
            for u in upper_node_list:
                for lo in lower_node_list:
                    new_interface = {
                        "family": "top",
                        "nI": lo,
                        "nJ": u,
                        "area": area,
                        "normal_dir": 2,
                        "shear_dir": 1,
                        "meta": (N_COURSES, brick, "beam", col),
                    }
                    m.interfaces.append(new_interface)

    return m



def verify(m):
    
    fam_area = {}
    for f in m.interfaces:
        family = f["family"]
        if family not in fam_area:
            fam_area[family] = 0.0
        fam_area[family] = fam_area[family] + f["area"]

    joint_area = WALL_L * WALL_T
    horizontal_families = [("base", 1), ("top", 1), ("bed", N_COURSES - 1)]
    for fam, n_levels in horizontal_families:
        expect = n_levels * joint_area
        got = fam_area[fam]
        assert abs(got - expect) < 1e-6 * expect, (fam, got, expect)

    head_count = 0
    for f in m.interfaces:
        if f["family"] == "head":
            head_count = head_count + 1
    n_head = head_count // 2
    assert n_head == 6 * N_COURSES, n_head
    assert abs(fam_area["head"] - 6 * N_COURSES * DY * WALL_T) < 1e-6

    assert abs(fam_area["middle"] - 6 * N_COURSES * DY * WALL_T) < 1e-6

    for f in m.interfaces:
        xi, yi = m.nodes[f["nI"]]
        xj, yj = m.nodes[f["nJ"]]
        assert abs(xi - xj) < 1e-9 and abs(yi - yj) < 1e-9, f


    counts = {}
    for fam in fam_area:
        count = 0
        for f in m.interfaces:
            if f["family"] == fam:
                count = count + 1
        counts[fam] = count

    areas_in_m2 = {}
    for fam in fam_area:
        areas_in_m2[fam] = round(fam_area[fam] / 1e6, 3)

    print("invariants OK  | interface counts:", counts)
    print("               | areas per family [m2]:", areas_in_m2)
    print("               | nodes =", len(m.nodes),
          ", brick quads =", len(m.brick_quads),
          ", beam quads =", len(m.beam_quads),
          ", interfaces =", len(m.interfaces))



def derive_stiffnesses(params, verbose=True):
   
    E_b = get_param(params, "E_b")
    nu_b = get_param(params, "nu_brick")
    E_mas = get_param(params, "E_masonry")
    G_mas = get_param(params, "G_masonry")

    G_b = E_b / (2.0 * (1.0 + nu_b))

    denom_E = (HB_MODULE + H_M) / E_mas - HB_MODULE / E_b
    denom_G = (HB_MODULE + H_M) / G_mas - HB_MODULE / G_b
    assert denom_E > 0 and denom_G > 0, "homogenisation denominators <= 0"
    E_m = H_M / denom_E
    G_m = H_M / denom_G

    def kn_of(Em):
        return Em * E_b / (H_M * (E_b - Em))

    def ks_of(Gm):
        return Gm * G_b / (H_M * (G_b - Gm))

    K = {
        "kn_bed": kn_of(E_m),
        "ks_bed": ks_of(G_m),
        "kn_head": kn_of(E_m / 3.0),
        "ks_head": ks_of(G_m / 3.0),
        "kn_mid": 400.0,
        "ks_mid": 400.0,
        "E_m": E_m,
        "G_m": G_m,
        "G_b": G_b,
    }

    if verbose:
        print("derive_stiffnesses (Wilding Eqs. 2-6, our inputs):")
        print("  E_b =", round(E_b), " nu_b =", nu_b,
              " E_mas =", round(E_mas), " G_mas =", round(G_mas),
              " h_m =", H_M)
        print("  E_m =", round(E_m, 2), " G_m =", round(G_m, 2),
              " G_b =", round(G_b, 1), " MPa")
        print("  bed : k_n =", round(K["kn_bed"], 2),
              " k_s =", round(K["ks_bed"], 2), "N/mm3")
        print("  head: k_n =", round(K["kn_head"], 2),
              " k_s =", round(K["ks_head"], 2), "N/mm3")
        print("  mid : k_n =", round(K["kn_mid"], 1),
              " k_s =", round(K["ks_mid"], 1), "N/mm3")

    assert 0.6 < K["kn_bed"] / 25.4 < 1.4
    assert 0.6 < K["ks_bed"] / 5.54 < 1.4
    return K

def family_material_params(params, K):

    bed = {
        "f_t": get_param(params, "f_t_joint"),
        "G_fI": get_param(params, "G_fI"),
        "c": get_param(params, "c"),
        "mu_p": get_param(params, "mu_peak"),
        "mu_r": get_param(params, "mu_res"),
        "G_fII": get_param(params, "G_fII"),
        "k_n": K["kn_bed"],
        "k_s": K["ks_bed"],
    }
    head = {
        "f_t": get_param(params, "f_t_head"),
        "G_fI": get_param(params, "G_fI_head"),
        "c": get_param(params, "c_head"),
        "mu_p": get_param(params, "mu_peak"),
        "mu_r": get_param(params, "mu_res"),
        "G_fII": get_param(params, "G_fII_head"),
        "k_n": K["kn_head"],
        "k_s": K["ks_head"],
    }
    mid = {
        "f_t": get_param(params, "f_t_brick"),
        "G_fI": get_param(params, "G_fI_brick"),
        "c": get_param(params, "c_brick"),
        "mu_p": get_param(params, "mu_brick"),
        "mu_r": get_param(params, "mu_brick"),
        "G_fII": get_param(params, "G_fII_brick"),
        "k_n": K["kn_mid"],
        "k_s": K["ks_mid"],
    }

    return {"base": bed, "bed": bed, "top": bed, "head": head, "middle": mid}

def build_opensees_model(mesh, params, K, wall="PUP2", sigma_map=None,
                         shear_law="A"):

    cfg = WALL_CONFIG[wall]
    fam = family_material_params(params, K)
    f_u = get_param(params, "f_u")
    E_b = get_param(params, "E_b")
    nu_b = get_param(params, "nu_brick")

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)


    for node_tag in mesh.nodes:
        x, y = mesh.nodes[node_tag]
        ops.node(node_tag, x, y)
    for foundation_tag in mesh.foundation_nodes.values():
        ops.fix(foundation_tag, 1, 1)


    ops.nDMaterial("ElasticIsotropic", 1, E_b, nu_b)
    ops.nDMaterial("ElasticIsotropic", 2, E_BEAM_FACTOR * E_b, nu_b)

    element_id = 1
    for brick_quad in mesh.brick_quads:
        quad_nodes = brick_quad[0]
        ops.element("quad", element_id, *quad_nodes, WALL_T, "PlaneStress", 1)
        element_id = element_id + 1

    element_id = 500
    for beam_quad in mesh.beam_quads:
        quad_nodes = beam_quad[0]
        ops.element("quad", element_id, *quad_nodes, WALL_T, "PlaneStress", 2)
        element_id = element_id + 1

    H0 = cfg["H0_over_H"] * H_EXP
    lever = tag(13, 0, 0, 1)
    ops.node(lever, WALL_L / 2.0, H0)
    ops.uniaxialMaterial("Elastic", 5, TRUSS_EA)
    beam_left = tag(12, 0, 0, 4)
    beam_right = tag(12, N_COLS, 0, 4)
    ops.element("Truss", 90001, lever, beam_left, 1.0, 5)
    ops.element("Truss", 90002, lever, beam_right, 1.0, 5)

    sigma0 = cfg["sigma0"]
    damage1 = get_param(params, "damage1")
    damage2 = get_param(params, "damage2")

    iface_handles = []
    for i in range(len(mesh.interfaces)):
        f = mesh.interfaces[i]
        p = fam[f["family"]]

        if sigma_map is not None:
            sigma_n = sigma_map[i]
        else:
            if f["normal_dir"] == 2:
                sigma_n = sigma0
            else:
                sigma_n = 0.0

        normal_mat_tag = 10000 + 2 * i
        shear_mat_tag = 10001 + 2 * i
        element_tag = 100000 + i

        normal_backbone, _ = make_normal_material(
            normal_mat_tag, p["f_t"], p["G_fI"], f_u, p["k_n"],
            area=f["area"], damage1=damage1, damage2=damage2)

        if shear_law == "B":

            friction_mat_tag = 300000 + i
            shear_backbone, _ = make_shear_material_hybrid(
                shear_mat_tag, friction_mat_tag, p["c"], p["mu_r"],
                sigma_n, p["G_fII"], p["k_s"], area=f["area"],
                damage1=damage1, damage2=damage2)
            shear_dof = f["shear_dir"]
            normal_dof = f["normal_dir"]
            ops.element("zeroLength", element_tag, f["nI"], f["nJ"],
                        "-mat", shear_mat_tag, friction_mat_tag,
                        normal_mat_tag,
                        "-dir", shear_dof, shear_dof, normal_dof)
        else:
            
            shear_backbone, _ = make_shear_material(
                shear_mat_tag, p["c"], p["mu_p"], p["mu_r"], sigma_n,
                p["G_fII"], p["k_s"], area=f["area"],
                damage1=damage1, damage2=damage2)
            dof_to_mat = {}
            dof_to_mat[f["normal_dir"]] = normal_mat_tag
            dof_to_mat[f["shear_dir"]] = shear_mat_tag
            ops.element("zeroLength", element_tag, f["nI"], f["nJ"],
                        "-mat", dof_to_mat[1], dof_to_mat[2],
                        "-dir", 1, 2)

        handle = {
            "ele": element_tag,
            "idx": i,
            "d1_n": normal_backbone[0][1],
            "d1_s": shear_backbone[0][1],
            "family": f["family"],
            "nI": f["nI"],
            "nJ": f["nJ"],
            "area": f["area"],
            "normal_dir": f["normal_dir"],
            "shear_dir": f["shear_dir"],
        }
        iface_handles.append(handle)

    return {"iface": iface_handles, "lever": lever, "cfg": cfg}


def apply_gravity(mesh, cfg, n_steps=10):
    
    N_total = cfg["sigma0"] * WALL_L * WALL_T

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)

    for c in range(N_COLS + 1):
        if c == 0 or c == N_COLS:
            weight = 0.5
        else:
            weight = 1.0
        node_load = -N_total * weight / N_COLS
        ops.load(tag(12, c, 0, 4), 0.0, node_load)

    for brick_quad in mesh.brick_quads:
        quad_nodes = brick_quad[0]
        corner_coords = []
        for node_tag in quad_nodes:
            corner_coords.append(mesh.nodes[node_tag])
        x1, y0 = corner_coords[0]
        x2, _ = corner_coords[1]
        _, y3 = corner_coords[2]
        brick_weight = GAMMA * abs(x2 - x1) * abs(y3 - y0) * WALL_T
        for node_tag in quad_nodes:
            ops.load(node_tag, 0.0, -brick_weight / 4.0)

    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 100, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0 / n_steps)
    ops.analysis("Static")

    result = ops.analyze(n_steps)
    if result != 0:
        raise RuntimeError("gravity + axial stage failed to converge")

    ops.loadConst("-time", 0.0)
    return N_total


def harvest_sigma(mesh, handles):
    
    sigma_map = {}
    total_base_compression = 0.0

    for interface_handle in handles["iface"]:
        forces = ops.eleForce(interface_handle["ele"])

        index_in_forces = 2 + (interface_handle["normal_dir"] - 1)
        N_tension = forces[index_in_forces]

        compression = -N_tension
        if compression < 0.0:
            compression = 0.0

        sigma = compression / interface_handle["area"]
        sigma_map[interface_handle["idx"]] = sigma

        if interface_handle["family"] == "base":
            total_base_compression = total_base_compression + compression

    return sigma_map, total_base_compression
