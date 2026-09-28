"""
Iterative dynamics <-> actuator-selection <-> structure loop.

    design_0 = thesis model (solid capsules at water density, massless ideal motors, 3 kg torso)
    repeat (max MAX_ITER):
        model_k  = build(design_k)
        loads_k  = inverse dynamics of the recorded gait on model_k   (torques, link wrenches)
        motors_k = lightest catalog motor per joint meeting the torque / RMS / speed criteria
        tubes_k  = outer radius per link (combined criterion, chosen material)
        design_{k+1} = motors_k (point masses at the joints) + tubes_k (link masses)
        converged if motors_k == motors_{k-1} and |m(design_{k+1}) - m(design_k)| / m <= MASS_TOL
                    (CONVERGENCE_MODE "any": either condition)
Run once per trained policy and per material (the tube mass depends on the material).
"""
import numpy as np

from . import config as C
from . import inverse_dynamics as ID
from . import motor_selection as MS
from . import structural as ST
from .model_builder import Design, build_model, mass_breakdown, segment_lengths, JOINT_NAMES, LINK_NAMES


def optimize(gait, material, catalog, lengths=None, log=print):
    lengths = lengths or segment_lengths()
    design = Design()
    history, prev_sel = [], None
    converged = False
    for it in range(C.MAX_ITER):
        model = build_model(design)
        mb = mass_breakdown(model, design)
        loads = ID.run(model, gait)
        s = loads.summary()
        sel = MS.select(catalog, s)
        struct = ST.size_all(loads, material, lengths)
        next_design = Design(material=material,
                             motor_mass_kg={x["joint"]: x["mass"] for x in sel},
                             tube_ro_m={k: v["R_combined"] for k, v in struct.items()})
        next_mb = mass_breakdown(build_model(next_design), next_design)
        sel_names = tuple(x["model"] for x in sel)
        motors_same = prev_sel is not None and sel_names == prev_sel
        dm = abs(next_mb["total"] - mb["total"]) / mb["total"]
        if C.CONVERGENCE_MODE == "all":
            converged = it >= 1 and motors_same and dm <= C.MASS_TOL
        else:
            converged = it >= 1 and (motors_same or dm <= C.MASS_TOL)
        w = mb["total"] * 9.81
        res_f = np.linalg.norm(loads.root_residual[:, :3], axis=1) / w
        history.append(dict(iter=it, mass=mb, tau_max=s["tau_max"].copy(), tau_rms=s["tau_rms"].copy(),
                            base_residual_max=float(res_f.max()), base_residual_mean=float(res_f.mean()),
                            hinge_check=float(loads.hinge_check),
                            selection=sel_names, next_mass=next_mb, dm=dm, motors_same=motors_same,
                            infeasible=[x["joint"] for x in sel if not x["feasible"]]))
        log(f"    iter {it}: mass {mb['total']:.2f} kg (motors {mb['motors']:.2f}, structure {mb['structure']:.2f})"
            f" | tau_max {np.round(s['tau_max'], 1).tolist()} | next mass {next_mb['total']:.2f} kg"
            f" (d={100*dm:.1f}%) | motors {'same' if motors_same else 'changed'}")
        if converged:
            break
        prev_sel = sel_names
        design = next_design
    return dict(material=material, converged=converged, iterations=len(history), history=history,
                design=design, final_design=next_design, final_mass=next_mb, simulated_mass=mb,
                base_residual=res_f,
                selection=sel, structure=struct, summary=s, loads=loads)


def save_timeseries(res, key, dt, n_envs, n_per_env):
    """Final-iteration time series, shaped (step, robot, ...) - the gait samples are stored
    step-major (all robots at step 0, then step 1, ...)."""
    L = res["loads"]
    shp = lambda x: x.reshape(n_per_env, n_envs, *x.shape[1:])
    path = C.output_dir() / f"timeseries_{key}_{res['material']}.npz"
    np.savez_compressed(
        path, dt=dt, n_envs=n_envs, n_per_env=n_per_env, joint_names=np.array(JOINT_NAMES),
        link_names=np.array(LINK_NAMES),
        tau=shp(L.tau), tau_damp=shp(L.tau_damp), tau_limit=shp(L.tau_limit), omega=shp(L.omega),
        axial_N=shp(L.N), shear_V=shp(L.V), bending_Mb=shp(L.Mb), torsion_T=shp(L.T), resultant_F=shp(L.F),
        ground_forces=shp(L.f_contact),
    )
    return path
