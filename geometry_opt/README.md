# geometry_opt — actuator selection and link sizing for SenecaBot

Iterative dynamics → actuator selection → structural sizing loop, driven by the gaits of the two
100M-step policies (0.218 m/s no-slip reference and 1.40 m/s mocap-speed reference).

```bash
source ".venv/bin/activate"                      # from "Optimizacion Geometria/"
python -m geometry_opt.run_pipeline              # both policies, PLA and PETG (~1.5 min with cached gaits)
python -m geometry_opt.run_pipeline --models v140 --materials PLA
python -m geometry_opt.run_pipeline --no-cache   # re-record the gaits (needs the GPU, ~1 min each)
python -m geometry_opt.run_pipeline --symmetric  # same motor on the left and right leg of each joint
```

`--symmetric` (or `SYMMETRIC_MOTORS` in config.py, or `MISMOS_MOTORES_IZQ_DER` at the top of the notebook) sizes
each left/right pair (FR/FL, BR/BL) with the larger requirement of the two sides. Its outputs go to
`results/symmetric_motors/` and `results_summary_symmetric.md`, so both variants coexist.

Outputs: `results_summary.md` (report), `visualize_results.ipynb` (plots; reads `results/`, no GPU needed),
`results/timeseries_<policy>_<material>.npz` (torques, joint speeds, axial/shear/bending/torsion per link,
shaped (step, robot, ...)), `results/history_*.csv` (per-iteration masses and torques), `results/motors_*.csv`,
`results/structure_*.csv`, `results/gait_*.npz` (cached gaits, flattened step-major).

| Module | Role |
|---|---|
| `config.py` | all tunables: safety factors, wall thickness, convergence, motor criteria, PLA/PETG properties (Prusament TDS), damping/limit options |
| `thesis_params.py` | stores the thesis values (Table V: segment lengths, ROM, torque limits; Sec. 3.2.1: 3 kg torso; Table II) with their page references, and checks them against the XML. The PDF is not needed |
| `catalog.py` | reads `my actuator/motors_MyActuator_Series_L_H_X.csv` (same data as the .xlsx) |
| `model_builder.py` | MuJoCo model of a design: motor point masses at the joints, links as hollow tubes of the material |
| `simulation.py` | runs a trained policy (MJX) and replays each state on CPU MuJoCo: accelerations, ground contacts, joint-limit / friction / self-contact forces |
| `inverse_dynamics.py` | torques and link wrenches of the recorded gait on any design (ground forces re-solved for the new masses) |
| `motor_selection.py` | lightest motor with peak ≥ FS·τmax, nominal ≥ FS·τ_RMS, speed ≥ ω_max |
| `structural.py` | tube sizing: axial only (A_min, D_ext) and combined axial + bending + shear + torsion (von Mises) |
| `pipeline.py` | the iterative loop and convergence test |
| `report.py`, `run_pipeline.py` | markdown/console report, entry point |

Why inverse dynamics instead of re-simulating the policy: the policies were trained on the 7.8 kg thesis
model with massless ideal motors, and they degrade on heavier robots (at ~15 kg the joint error goes from
7–8° to 20–25°). Loads measured on a degraded gait would size the robot for a different, slower motion.
Instead, every design is required to perform the gait the policy achieves, and its loads are computed
exactly for that motion. On the trained model this reproduces the forward simulation to machine precision.
