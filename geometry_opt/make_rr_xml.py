"""
Generate the RR variant of the MuJoCo model, with the two RR bar mechanisms coupled to each leg:

    python -m geometry_opt.make_rr_xml [--policy v140] [--material PLA] [--no-linkage] [--out PATH]

Input : seneca_loco/simulation/assets/xml/senecabot_loco.xml (RRR, thigh present) and, for the linkage,
        results/linkage_<policy>.csv (design of FR and BR written by `run_pipeline --linkage`; FL / BL copy them,
        the left legs of the model are identical to the right ones).
Output: seneca_loco/simulation/assets/xml/senecabot_loco_rr.xml (config.XML_RR_PATH).

RR model (always): the first link of every leg (thigh `*_upper`, hip joint, hip actuator) is removed; the tibia
`*_lower` (knee joint) hangs from the torso at the place where the knee was with the hip at 0, and tibia and
metatarsus keep length, geometry, masses, joint axes, ranges and damping.

With the linkage (default), every leg also gets the two RR bar mechanisms of linkage.py, drawn in the sagittal plane
(mechanism plane x = torso x, plane y = torso z):
    RR 1: bars l1 (hinge `<leg>_rr1_joint` on the torso at B) and l2 (elbow `<leg>_rr1_elbow`), closed on the tibia
          at a point f1 along it by a `connect` equality; motor `<leg>_rr1` on `<leg>_rr1_joint`.
    RR 2: bars l3 (hinge `<leg>_rr2_joint` on the torso at O = the knee pivot) and l4 (elbow `<leg>_rr2_elbow`),
          closed on the metatarsus at a point f2 along it; motor `<leg>_rr2` on `<leg>_rr2_joint`.
The knee and ankle joints are then passive (driven through the loops) and the hip, knee and ankle motors are gone:
the 8 actuators are the two crank motors of each leg. The model's reference pose (qpos0) is the closed pose at the
centre of the range of motion (joint `ref` attributes), so the loops are closed from the first step.

Looks: the bars are capsules of the same radius as the leg link they drive (RR 1 as the tibia, RR 2 as the
metatarsus), RR 1 and RR 2 in the leg plane (PLANE_OFFSET > 0 would move them inside / outside it); the knee pivot is raised so
every leg is mounted inside the torso box: the front knee pivot O sits PIVOT_INSET above the torso underside, the
hind knee pivots the RRR model's offset (35 mm) plus HIND_EXTRA_RAISE (45.6 mm, level trunk in the gait) higher, and
the hind legs move forward until their crank pivot B is inside the box; thin mounting plates join the
torso to the pivots below it. Bars and plates are visual only (no contacts).
Mass: each bar is a hollow PLA tube of outer radius BAR_TUBE_RO (wall = config.MIN_WALL_THICKNESS_MM), not a solid
capsule; the plates have no mass.
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

from . import config as C
from . import linkage as LK

PLANE_OFFSET = 0.0       # [m] lateral distance of the planes of RR 1 (inside) and RR 2 (outside) from the leg plane (0 = all in the leg plane)
BAR_TUBE_RO = 0.009      # [m] outer radius of the hollow tube whose mass each bar gets (pipeline sizes ~ 18 mm tubes)
PLATE_HALF_X = 0.02      # [m] half width (along x) of the mounting plates
PLATE_HALF_Y = 0.003     # [m] half thickness of the mounting plates
CTRL_DEFAULT = 100.0     # [N·m] crank motor range when no motor selection is available
TORSO_RGBA = "0.8 0.6 0.4 1"
PIVOT_INSET = 0.002       # [m] the pivots O and B of every leg lie inside the torso box by at least this margin
HIND_EXTRA_RAISE = 0.0456  # [m] extra raise of the hind knee pivots beyond the RRR offset (35 mm): with it the stance
                           # feet of the RR reference touch the floor with the trunk level (pitch 0); without it 4.2°
DEG = np.degrees


def _fmt(*v):
    return " ".join(f"{x:.6g}" for x in v)


def _leg_rows(key):
    df = pd.read_csv(C.RESULTS_DIR / f"linkage_{key}.csv")
    rows = {r["leg"].lower(): r.to_dict() for _, r in df.iterrows()}
    if not {"fr", "br"} <= set(rows):
        raise SystemExit(f"linkage_{key}.csv needs the legs FR and BR (has {sorted(rows)}): run run_pipeline --linkage")
    return {"fr": rows["fr"], "fl": rows["fr"], "br": rows["br"], "bl": rows["br"]}


def _ctrl_ranges(key, material=None):
    """Crank motor torque range [N·m] from the selected motors (linkage_actuators_<key>_<material>.csv)."""
    files = sorted(C.RESULTS_DIR.glob(f"linkage_actuators_{key}_{material or '*'}.csv"))
    if not files:
        return {}
    df = pd.read_csv(files[0])
    return {r["actuator"]: float(r["motor_peak"]) for _, r in df.iterrows()}      # keys: fr_rr1, fr_rr2, br_rr1, br_rr2


def _bar_mass(length):
    t = C.MIN_WALL_THICKNESS_MM / 1000.0
    area = np.pi * (BAR_TUBE_RO ** 2 - (BAR_TUBE_RO - t) ** 2)
    return C.MATERIALS["PLA"]["density_kg_m3"] * area * length


def _bars(leg, pos, y_in, cfg, row, r_tibia, r_foot, z_torso):
    """XML of the two RR mechanisms of a leg and of its mounting plates (children of the torso body).
    `pos` = knee pivot O in the torso frame, `y_in` = +1/-1 direction towards the torso centre."""
    O = np.array([pos[0], pos[2]])                      # knee pivot in the (x, z) plane of the torso
    B = O - cfg["O"]                                    # B = O - (dx, -dy)  (plane vector O - B = (dx, -dy))
    y1, y2 = pos[1] + y_in * PLANE_OFFSET, pos[1] - y_in * PLANE_OFFSET      # RR 1 inside, RR 2 outside
    col1, col2 = "0.95 0.55 0.1 1", "0.85 0.2 0.2 1"
    bar = lambda L, r, rgba: (f'<geom type="capsule" fromto="0 0 0 {L:.6g} 0 0" size="{r}" rgba="{rgba}" '
                              f'mass="{_bar_mass(L):.6g}" contype="0" conaffinity="0"/>')
    a = {k: cfg[k] for k in ("l1", "l2", "l3", "l4_angle")}
    L1, L2, L3, L4 = row["l1"], row["l2"], row["l3"], cfg["l4"]
    # a body written with euler y = -alpha has its +x axis at plane angle alpha; the hinge axis (0 -1 0) makes the
    # joint value equal to the angle, and `ref` makes qpos0 the closed pose.
    plate = lambda name, y, x, zlo: (f'<geom name="{leg}_mount_{name}" type="box" pos="{_fmt(x, y, 0.5 * (zlo + z_torso))}" '
                                      f'size="{_fmt(PLATE_HALF_X, PLATE_HALF_Y, 0.5 * (z_torso - zlo))}" rgba="{TORSO_RGBA}" '
                                      f'mass="0" contype="0" conaffinity="0"/>')
    return f"""
            <!-- ===== {leg.upper()}: RR 1 (l1, l2) -> tibia at f1 = {row['f1']:.2f}; RR 2 (l3, l4) -> metatarsus at f2 = {row['f2']:.2f} ===== -->
            {plate('knee', pos[1], O[0], O[1] - PLATE_HALF_X) if O[1] < z_torso - 1e-6 else ''}
            {plate('rr1', y1, B[0], B[1] - PLATE_HALF_X) if abs(y1 - pos[1]) > 1e-9 or B[1] < z_torso - 1e-6 else ''}
            {plate('rr2', y2, O[0], O[1] - PLATE_HALF_X) if abs(y2 - pos[1]) > 1e-9 else ''}
            <site name="{leg}_rr_B" pos="{_fmt(B[0], y1, B[1])}" size="0.008" rgba="0 0 0 1"/>
            <body name="{leg}_rr1_l1" pos="{_fmt(B[0], y1, B[1])}" euler="0 {-DEG(a['l1']):.6g} 0">
                <joint name="{leg}_rr1_joint" type="hinge" axis="0 -1 0" ref="{DEG(a['l1']):.6g}"/>
                {bar(L1, r_tibia, col1)}
                <body name="{leg}_rr1_l2" pos="{L1:.6g} 0 0" euler="0 {-DEG(a['l2'] - a['l1']):.6g} 0">
                    <joint name="{leg}_rr1_elbow" type="hinge" axis="0 -1 0" ref="{DEG(a['l2'] - a['l1']):.6g}" damping="0.01" armature="0.001" frictionloss="0"/>
                    {bar(L2, r_tibia, col1)}
                </body>
            </body>
            <body name="{leg}_rr2_l3" pos="{_fmt(O[0], y2, O[1])}" euler="0 {-DEG(a['l3']):.6g} 0">
                <joint name="{leg}_rr2_joint" type="hinge" axis="0 -1 0" ref="{DEG(a['l3']):.6g}"/>
                {bar(L3, r_foot, col2)}
                <body name="{leg}_rr2_l4" pos="{L3:.6g} 0 0" euler="0 {-DEG(a['l4_angle'] - a['l3']):.6g} 0">
                    <joint name="{leg}_rr2_elbow" type="hinge" axis="0 -1 0" ref="{DEG(a['l4_angle'] - a['l3']):.6g}" damping="0.01" armature="0.001" frictionloss="0"/>
                    {bar(L4, r_foot, col2)}
                </body>
            </body>"""


def generate(src_text, key=None, with_linkage=True, material=None):
    rows = _leg_rows(key) if with_linkage else {}
    cfgs = {leg: LK.reference_configuration(r) for leg, r in rows.items()}
    ctrl = _ctrl_ranges(key, material) if with_linkage else {}
    box = re.search(r'<geom type="box" size="([\d.]+) [\d.]+ ([\d.]+)" mass="3"', src_text)
    hx, hz = float(box.group(1)), float(box.group(2))                   # torso half length / half height
    z_torso = -hz                                                       # torso underside
    equality, motors = [], []
    cap = r'<geom type="capsule" fromto="0 0 0 0 0 -([\d.]+)" size="([\d.]+)"'

    def leg_block(m):
        leg, upos, body = m.group(1), [float(v) for v in m.group(2).split()], m.group(3)
        ulen = float(re.search(cap, body).group(1))
        kpos = [upos[0], upos[1], round(upos[2] - ulen, 6)]
        lower = body[body.index(f'<body name="{leg}_lower"'):].rstrip()
        assert lower.endswith("</body>")
        lower = lower[: -len("</body>")].rstrip()                    # drop the closing tag of the removed thigh
        r_tibia, r_foot = (float(c[1]) for c in re.findall(cap, lower)[:2])
        extra, euler_l = "", ""
        if with_linkage:
            cfg, row = cfgs[leg], rows[leg]
            kpos[0], kpos[2] = kx[leg], kz[leg]                         # see korig / kx / kz below
            euler_l = f' euler="0 {DEG(-cfg["knee"]):.6g} 0"'           # tibia pitch at the reference pose = -knee
            lower = re.sub(rf'(<joint name="{leg}_knee_joint" class="knee" range="[^"]*")', rf'\1 ref="{DEG(cfg["knee"]):.6g}"', lower)
            lower = re.sub(rf'(<joint name="{leg}_ankle_joint" class="ankle" range="[^"]*")', rf'\1 ref="{DEG(cfg["ankle"]):.6g}"', lower)
            lower = re.sub(rf'(<body name="{leg}_foot" pos="[^"]*")', rf'\1 euler="0 {DEG(cfg["ankle"]):.6g} 0"', lower)
            y_in = 1.0 if kpos[1] < 0 else -1.0                         # towards the torso centre
            extra = _bars(leg, kpos, y_in, cfg, row, r_tibia, r_foot, z_torso)
            equality.append(
                f'<connect name="{leg}_rr1_loop" body1="{leg}_rr1_l2" body2="{leg}_lower" anchor="{row["l2"]:.6g} 0 0"/>\n'
                f'        <connect name="{leg}_rr2_loop" body1="{leg}_rr2_l4" body2="{leg}_foot" anchor="{cfg["l4"]:.6g} 0 0"/>')
            for nm in ("rr1", "rr2"):
                rng = ctrl.get(f"{'fr' if leg[0] == 'f' else 'br'}_{nm}", CTRL_DEFAULT)   # left legs copy the right ones
                motors.append(f'<motor name="{leg}_{nm}" joint="{leg}_{nm}_joint" ctrlrange="-{rng:.6g} {rng:.6g}"/>')
        lower = re.sub(rf'<body name="{leg}_lower" pos="[^"]*">',
                       f'<body name="{leg}_lower" pos="{_fmt(*kpos)}"{euler_l}>', lower, count=1)
        site = f'<site name="{leg}_thigh_mimic" class="mimic" pos="{_fmt(*kpos)}"/>'
        return f"{site}\n\n            {lower}{extra}\n            "

    pat = re.compile(r'<body name="(\w\w)_upper" pos="([^"]+)">(.*?)(?=\n\s*<!-- =+ Pata|\n\s*</body>\s*\n\s*</worldbody>)', re.S)
    # knee pivots: every leg is mounted inside the torso box (no part of the mounting sticks out of it).
    # Height: the legs keep the vertical offsets of the RRR model's knee pivots (hip at 0; there the front knee is
    # 35 mm lower than the hind one) plus HIND_EXTRA_RAISE for the hind legs (level trunk); the lowest pivot O (front)
    # sits PIVOT_INSET above the torso underside. Length: each leg moves the least along x that puts its pivots O and
    # B inside the box (the hind crank B lay 25 mm behind the torso).
    korig = {m.group(1): float(m.group(2).split()[2]) - float(re.search(cap, m.group(3)).group(1))
             for m in pat.finditer(src_text)}
    xorig = {m.group(1): float(m.group(2).split()[0]) for m in pat.finditer(src_text)}
    if with_linkage:
        rel = {leg: korig[leg] + (HIND_EXTRA_RAISE if leg[0] == "b" else 0.0) for leg in korig}
        low = min(rel, key=rel.get)
        kz = {leg: round(rel[leg] - rel[low] + z_torso + PIVOT_INSET, 6) for leg in rel}
        kx = {}
        for leg in korig:
            xs = np.array([xorig[leg], xorig[leg] - cfgs[leg]["O"][0]])          # O_x, B_x
            lo, hi = -hx + PIVOT_INSET, hx - PIVOT_INSET
            kx[leg] = round(xorig[leg] + max(0.0, lo - xs.min()) - max(0.0, xs.max() - hi), 6)
            zs = np.array([kz[leg], kz[leg] - cfgs[leg]["O"][1]])                 # O_z, B_z
            assert zs.min() >= z_torso + PIVOT_INSET - 1e-9 and zs.max() <= hz - PIVOT_INSET + 1e-9, \
                f"{leg}: pivots at z {zs} do not fit in the torso box (half height {hz})"
    out = pat.sub(leg_block, src_text)
    out = re.sub(r'\n\s*<exclude body1="base" body2="\w\w_upper"/>', "", out)
    out = re.sub(r'\n\s*<motor name="\w\w_(hip|knee|ankle)"[^\n]*', "", out) if with_linkage else \
        re.sub(r'\n\s*<motor name="\w\w_hip"[^\n]*', "", out)
    out = out.replace('<mujoco model="senecabot">', '<mujoco model="senecabot_rr">', 1)
    if with_linkage:
        out = out.replace("</actuator>", "        " + "\n        ".join(motors) + "\n    </actuator>", 1)
        out = out.replace("</mujoco>", "    <!-- Closed loops of the two RR mechanisms: the tip of l2 / l4 is pinned to the tibia / metatarsus -->\n"
                          "    <equality>\n        " + "\n        ".join(equality) + "\n    </equality>\n</mujoco>", 1)
    return out


NOTE = """    <!-- RR VARIANT of senecabot_loco.xml, generated by geometry_opt/make_rr_xml.py (do not edit by hand).
         The first link of every leg (thigh `*_upper`, hip joint and hip actuator) is removed, leaving a 2R leg: tibia
         `*_lower` (knee joint) + metatarsus `*_foot` (ankle joint), unchanged in length, geometry, masses, joint axes,
         ranges and damping. The knee pivot is fixed to the torso where the knee was with the hip at 0
         ((upper pos) + (0, 0, -thigh length)); every body/joint/site name is kept (`*_thigh_mimic` now sits on the torso at
         the knee pivot). The tibia pitch is now the knee angle alone (hip - knee with hip = 0): policies trained on
         the RRR model do NOT transfer, the gait has to be retrained.
         __LINKAGE__ -->
"""
NOTE_LINK = ("With the linkage: each leg carries the RR 1 (`*_rr1_l1`, `*_rr1_l2`) and RR 2 (`*_rr2_l3`, `*_rr2_l4`) bar mechanisms "
             "of geometry_opt/linkage.py, closed on the tibia and the metatarsus by `connect` equalities; the two crank hinges "
             "`*_rr1_joint`, `*_rr2_joint` are the only actuated joints (motors `*_rr1`, `*_rr2`), the knee and ankle are "
             "passive. qpos0 is the closed reference pose (centre of the range of motion, joint `ref`). Bars are capsules "
             "as wide as the link they drive; mounting plates `*_mount_*` are visual only.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="v140", choices=list(C.MODELS), help="linkage design to use (results/linkage_<policy>.csv)")
    ap.add_argument("--material", default=None, help="take the crank motor torque range from linkage_actuators_<policy>_<material>.csv")
    ap.add_argument("--no-linkage", action="store_true", help="plain RR model without the bar mechanisms")
    ap.add_argument("--out", default=str(C.XML_RR_PATH))
    args = ap.parse_args(argv)
    src = C.XML_PATH.read_text(encoding="utf8")
    text = generate(src, args.policy, not args.no_linkage, args.material)
    note = NOTE.replace("__LINKAGE__", "Plain RR model, no bar mechanisms." if args.no_linkage else NOTE_LINK + f" Design: policy {args.policy}.")
    text = text.replace("    <!-- LEG SEGMENT NAMING.", note + "\n    <!-- LEG SEGMENT NAMING.", 1)
    open(args.out, "w", encoding="utf8").write(text)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
