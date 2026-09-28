"""Render the pipeline results as markdown (results_summary.md) and echo it to the console."""
from datetime import datetime

import numpy as np

from . import config as C
from .model_builder import JOINT_NAMES, LINK_NAMES

SHORT = {"hip": "h", "knee": "k", "ankle": "a"}
LINK_DESC = {"upper": "hip→knee", "lower": "knee→ankle", "foot": "ankle→foot"}


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _f(x, n=1):
    return "∞" if not np.isfinite(x) else f"{x:.{n}f}"


def header(thesis, thesis_check):
    L = thesis["segment_length_m"]
    lines = [
        "# Geometry optimization — results summary",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} by `geometry_opt/run_pipeline.py`._",
        "",
        "## Configuration",
        f"- FS_torque = {C.FS_TORQUE}, FS_material = {C.FS_MATERIAL}, FS_speed = {C.FS_SPEED}; "
        f"wall thickness t = {C.MIN_WALL_THICKNESS_MM} mm",
        f"- Left/right motors: {'same motor on both sides (pair sized with the larger requirement)' if C.SYMMETRIC_MOTORS else 'selected independently per joint'}",
        f"- Motor criteria: peak torque ≥ FS·|τ|max"
        + (", nominal torque ≥ FS·τ_RMS" if C.CHECK_RMS_TORQUE else "")
        + (", rated speed ≥ FS_speed·|ω|max" if C.CHECK_SPEED else ""),
        f"- Convergence: motors unchanged {'AND' if C.CONVERGENCE_MODE == 'all' else 'OR'} "
        f"total mass change ≤ {100*C.MASS_TOL:.0f}% (max {C.MAX_ITER} iterations)",
        f"- Link structure mass in the loop: {C.STRUCTURE_MASS_IN_LOOP} (hollow tubes, combined criterion); "
        f"foot sphere infill {C.FOOT_SPHERE_INFILL}",
        f"- Joint-limit stops carry their load: {C.LIMITS_CARRY_LOAD}; joint damping: "
        + ("thesis value (1.5 N·m·s/rad hip/knee, 0.05 ankle)" if C.JOINT_DAMPING_OVERRIDE is None
           else f"{C.JOINT_DAMPING_OVERRIDE} N·m·s/rad"),
        f"- Anisotropy factor on σ_yield: {C.ANISOTROPY_FACTOR}",
        "",
        _table(["Material", "σ_yield [MPa]", "σ_adm [MPa]", "E [GPa]", "ρ [kg/m³]", "Source"],
               [[k, v["sigma_yield_MPa"], _f(v["sigma_yield_MPa"] * C.ANISOTROPY_FACTOR / C.FS_MATERIAL),
                 v["E_GPa"], v["density_kg_m3"], v["source"]] for k, v in C.MATERIALS.items()]),
        "",
        "## Initial parameters (thesis, stored in `thesis_params.py`)",
        _table(["Joint / segment", "Length front [cm]", "Length back [cm]", "Torque limit [N·m]"],
               [[j, _f(L[("front", j)] * 100), _f(L[("back", j)] * 100), f"±{thesis['torque_limit_Nm'][j]:.0f}"]
                for j in ("hip", "knee", "ankle")]),
        "",
        f"- Torso mass: {thesis['torso_mass_kg']} kg (only explicit mass in the thesis model; leg masses "
        "come from the capsule geometry at MuJoCo's default density, 1000 kg/m³).",
        f"- {thesis_check[0]}",
        f"- Source: {thesis['source']}",
        "",
    ]
    return "\n".join(lines)


def model_section(key, gait, validation, results):
    spec = C.MODELS[key]
    v = validation
    lines = [
        f"## {spec['label']}",
        f"- Agent: `{spec['agent'].relative_to(C.WORKSPACE)}`; reference: `{spec['reference'].name}`",
        f"- Recorded gait: {gait['n_envs']} robots × {gait['n_per_env']} control steps "
        f"({gait['qpos'].shape[0]} states), forward speed {float(gait['speed']):.3f} m/s, "
        f"trained model mass {float(gait['mass_trained']):.2f} kg",
        f"- Inverse-dynamics validation on the trained model (iteration 0 vs forward simulation): max torque "
        f"RMS error {v['tau_rms_err'].max():.3f} N·m, peak ratio {v['tau_peak_ratio'].min():.3f}–"
        f"{v['tau_peak_ratio'].max():.3f}, base residual max {v['root_force_residual_max_pct']:.2f}% of weight, "
        f"hinge-moment check {v['hinge_check_Nm']:.3f} N·m",
        "",
    ]
    for res in results:
        lines += material_section(res)
    lines += structural_section(results)
    return "\n".join(lines)


def material_section(res):
    mat = res["material"]
    status = (f"converged in {res['iterations']} iterations" if res["converged"]
              else f"NOT converged after {res['iterations']} iterations")
    jh = [f"{j[:2].upper()}-{SHORT[j[3:]]}" for j in JOINT_NAMES]
    rows = []
    for h in res["history"]:
        m = h["mass"]
        rows.append([h["iter"], _f(m["total"], 2), _f(m["motors"], 2), _f(m["structure"], 2)]
                    + [_f(x, 1) for x in h["tau_max"]]
                    + ["—" if h["iter"] == 0 else ("no" if h["motors_same"] else "yes"), f"{100*h['dm']:.1f}%"])
    fm = res["final_mass"]
    lines = [
        f"### {mat} — {status}",
        "",
        "**1. Convergence history** (mass of the model simulated at each iteration; |τ|max per joint in N·m; "
        "Δ = relative change of the next design's mass)",
        "",
        _table(["Iter", "Total [kg]", "Motors [kg]", "Structure [kg]"] + jh + ["Motors changed", "Δ mass"], rows),
        "",
        f"Final design: **{fm['total']:.2f} kg** (torso {fm['torso']:.2f}, motors {fm['motors']:.2f}, "
        f"{mat} structure {fm['structure']:.2f}).",
        "",
        f"Consistency of the last iteration: unbalanced base force max "
        f"{100*res['history'][-1]['base_residual_max']:.2f}% (mean {100*res['history'][-1]['base_residual_mean']:.3f}%) "
        f"of the robot weight; hinge-moment check {res['history'][-1]['hinge_check']:.1e} N·m.",
        "",
        "**2. Selected actuators**",
        "",
    ]
    s = res["summary"]
    rows = []
    for i, x in enumerate(res["selection"]):
        rows.append([x["joint"], _f(s["tau_max"][i], 2), _f(s["tau_rms"][i], 2), _f(s["omega_max_rpm"][i], 0),
                     x["model"] + ("" if x["feasible"] else " ⚠ infeasible"), _f(x["peak"], 1), _f(x["nominal"], 1),
                     _f(x["speed"], 0), _f(x["mass"], 3), x["binding"],
                     _f(s["tau_damp_at_max"][i], 1),
                     _f(100 * res["base_residual"][np.abs(res["loads"].tau[:, i]).argmax()], 1)])
    lines += [
        _table(["Joint", "Torque max sim [N·m]", "Torque RMS [N·m]", "ω max [rpm]", "Selected motor",
                "Motor peak [N·m]", "Motor nominal [N·m]", "Motor speed [rpm]", "Motor mass [kg]",
                "Binding criterion", "Damping part of τmax [N·m]", "Base residual at τmax [% weight]"], rows),
        "",
        "_Base residual_: unbalanced base force of the inverse dynamics at the instant of the joint's peak "
        "(fixed-kinematics limitation, mainly in flight / single-support phases); > 5% marks a less reliable peak.",
        "",
        f"Total actuator mass: {sum(x['mass'] for x in res['selection']):.2f} kg.",
        "",
    ]
    return lines


def structural_section(results):
    by = {r["material"]: r for r in results}
    mats = [m for m in ("PLA", "PETG") if m in by]
    lines = ["### 3. Structural sizing (hollow tubes, t = %.1f mm)" % C.MIN_WALL_THICKNESS_MM, "",
             "**Axial criterion** σ = F/A ≤ σ_adm (loads of each material's converged design). "
             "`*` = the required area is below the smallest t = 4 mm tube: the minimum is a solid rod "
             f"Ø{2*C.MIN_OUTER_RADIUS_MM:.0f} mm.", ""]
    rows = []
    for i, link in enumerate(LINK_NAMES):
        f_ax = " / ".join(_f(by[m]["structure"][link]["N_max"], 0) for m in mats)
        f_res = " / ".join(_f(by[m]["loads"].F[:, i].max(), 0) for m in mats)
        row = [f"{link} ({LINK_DESC[link[3:]]})", _f(by[mats[0]]["structure"][link]["length"] * 1000, 0), f_ax, f_res]
        for m in mats:
            st = by[m]["structure"][link]
            row += [_f(st["A_min_axial"] * 1e6, 2), _f(2 * st["R_axial"] * 1000, 1) + ("*" if st["wall_limited_axial"] else "")]
        rows.append(row)
    hdr = ["Link", "L [mm]", "F axial max [N] (" + "/".join(mats) + ")", "F resultant max [N]"]
    for m in mats:
        hdr += [f"A_min {m} [mm²]", f"D_ext,min {m} [mm]"]
    lines += [_table(hdr, rows), ""]

    lines += ["**Combined criterion** (axial + bending + shear + torsion, von Mises, simultaneous components): "
              "this is the sizing used for the link masses in the loop.", ""]
    rows = []
    for i, link in enumerate(LINK_NAMES):
        L = by[mats[0]]["loads"]
        row = [link, _f(L.Mb[:, i].max(), 1), _f(np.abs(L.T[:, i]).max(), 2), _f(L.V[:, i].max(), 0)]
        for m in mats:
            st = by[m]["structure"][link]
            row += [_f(2 * st["R_combined"] * 1000, 1), _f(st["mass"] * 1000, 0)]
        row.append(by[mats[0]]["structure"][link]["governing"])
        rows.append(row)
    hdr = ["Link", f"Bending max [N·m] ({mats[0]})", "Torsion max [N·m]", "Shear max [N]"]
    for m in mats:
        hdr += [f"D_ext,min {m} [mm]", f"Tube mass {m} [g]"]
    hdr.append("Governing")
    lines += [_table(hdr, rows), ""]
    return lines


def comparison(all_results):
    rows = []
    for key, results in all_results.items():
        for r in results:
            fm = r["final_mass"]
            rows.append([C.MODELS[key]["label"], r["material"], "yes" if r["converged"] else "NO", r["iterations"],
                         _f(fm["total"], 2), _f(fm["motors"], 2), _f(fm["structure"], 2),
                         ", ".join(sorted(set(x["model"] for x in r["selection"])))])
    return "\n".join(["## Overview", "",
                      _table(["Policy", "Material", "Converged", "Iterations", "Final mass [kg]", "Motors [kg]",
                              "Structure [kg]", "Motor models used"], rows), ""])


def notes():
    return "\n".join([
        "## Method and limitations",
        "- The trained policies do not transfer to heavier robots (tested: at ~15 kg the joint error grows from "
        "7–8° to 20–25° and the 1.40 m/s gait slows to 0.5–0.65 m/s). Loads are therefore obtained by inverse "
        "dynamics of the gait each policy achieves on the thesis model: every design must perform that same "
        "gait; ground forces are re-solved per sample for the new masses (friction pyramid, NNLS) and "
        "joint-limit / dry-friction / self-contact forces are kept as simulated.",
        "- With fixed kinematics, phases with 0–1 feet on the ground cannot be balanced exactly for a "
        "different mass distribution (momentum is not conserved the same way); the residual is reported per "
        "iteration and at every joint's torque peak. The link bending peaks all occur at residuals of a few %.",
        "- Validation: on the trained model the inverse dynamics reproduces the forward simulation exactly "
        "(torques, ground forces, base balance), and the link wrenches satisfy the hinge-moment identity to "
        "machine precision for every design.",
        "- Joint torques include the thesis model's joint damping (a simulation regularizer); its share at "
        "each joint's peak is listed. Set `JOINT_DAMPING_OVERRIDE` in config.py to size for a real value.",
        "- Motors are point masses at the joints (the catalog has no dimensions); rotor inertia/gear "
        "reflected inertia is the model's armature (0.01 kg·m²), not motor-specific.",
        "- Torque–speed coupling is not checked beyond peak torque and rated speed separately.",
        "- FDM strength: horizontal-print TDS values; parts loaded across layers can retain only ~50–70% "
        "(`ANISOTROPY_FACTOR`). Stiffness, buckling, fatigue and joint/connection design are not covered.",
        "",
    ])
