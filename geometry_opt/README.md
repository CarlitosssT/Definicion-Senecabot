# geometry_opt — actuator selection and link sizing for SenecaBot

Iterative dynamics → actuator selection → structural sizing loop, driven by the gaits of the two
100M-step policies (0.218 m/s no-slip reference and 1.40 m/s mocap-speed reference).

```bash
source ".venv/bin/activate"                      # from "Optimizacion Geometria/"
python -m geometry_opt.run_pipeline              # both policies, PLA and PETG (~1.5 min with cached gaits)
python -m geometry_opt.run_pipeline --models v140 --materials PLA
python -m geometry_opt.run_pipeline --no-cache   # re-record the gaits (needs the GPU, ~1 min each)
python -m geometry_opt.run_pipeline --symmetric  # same motor on the left and right leg of each joint
python -m geometry_opt.run_pipeline --linkage    # also size the RR transmission linkage (+~1.5 min with 20 cores)
```

## RR transmission linkage (`linkage.py`)

The leg is being redesigned from RRR to **RR**: the first link (thigh, hip joint and hip motor) is removed and the
tibia and metatarsus keep their position and length (`senecabot_loco_rr.xml`, see below). Both remaining joints are
driven from the torso by two RR bar mechanisms; `linkage.py` sizes them (port of `MecanismoV2.ipynb`).

- Planar model, absolute angles. `B` = actuator of RR 1 at the origin, `O = (dx, -dy)` = tibia pivot. RR 1 (`l1`, `l2`)
  goes from `B` to the point `f1` along the tibia `r1`; RR 2 (`l3`, `l4`) goes from `O` to the point `f2` along the
  metatarsus `r2`. `l4` is derived: distance from the tip of `l3` (angle `a3`) to the point `f2·r2` in the reference
  pose (centre of the range). Unknowns: `l1, l2, l3, dx, dy, a3`.
- For every `(f1, f2)` in `LINKAGE_F_VALUES` a differential evolution minimises `l4` (first), the bar lengths and the
  growth of `dx`, `dy` over their minimum (`48·cos45° = 33.9 mm`, the smallest allowed), and maximises the part of the
  extended range of motion that both RR reach. The range of motion of the gait must be covered 100 % and **no two bars
  that do not share a joint may come closer than `LINKAGE_CLEARANCE_M`** (checked on every pose with the inverse
  kinematics of both RR; both elbow branches of RR 1 are tried).
- The range of motion is the orientation of the tibia and the metatarsus with respect to the torso in the recorded gait
  (MuJoCo forward kinematics, so it does not depend on the thigh). Lengths come from the `*_lower` / `*_foot` capsules.
- The sweep runs in parallel (`LINKAGE_WORKERS`, spawn processes) and is cached in `results/linkage_<policy>_<leg>.npz`
  (invalidated when the inputs or settings change). Outputs: `results/linkage_<policy>.csv`, `_maps.png`, `_design.png`,
  `_animation.gif`, and a section of the summary. FL / BL are not sized (mirror images of FR / BR; `--linkage-legs`).
- API: `design_legs(gait, key, out_dir)`; per leg `make_problem` → `design_leg` → `summary_row`, `check_trajectory`,
  `plot_maps`, `plot_design`, `animate`. All settings are `LINKAGE_*` in `config.py`.
- Actuators through the linkage: with the joint torque and speed of the gait, `actuator_requirements` gives what each
  actuator of the linkage must deliver (`tau_act = G^-T A^-T tau_joint`, `omega_act = G A omega_joint`, G = Jacobian of
  the two driven bars with respect to tibia and metatarsus) and `run_pipeline --linkage` selects motors with it
  (`motor_selection.select_requirements`, `results/linkage_actuators_<policy>_<material>.csv`). The design itself
  includes the **torque amplification** (actuator / joint peak torque, 1 = direct drive) in its cost
  (`LINKAGE_WEIGHTS['torque']`, evaluated on `LINKAGE_TORQUE_SAMPLES` poses of the gait; 0 turns it off).
  With weight 0 the compact designs need 2.5x (FR) and 8.6x (BR) the joint torque; with 1 (default) 1.6x and 4.2x;
  with 3 about 1.1x and 1.6x at the price of range of motion and longer bars.
- Not in the dynamics loop: the motors are not moved to the torso and the bars' mass is not added, so the joint
  torques of the gait (RRR model, motors at the joints) are only an estimate of the real loads.
- `a3` could not be fixed at ±45°: at +45° `l4` crosses `r1`, at −45° `l3` and `f2·r2` are almost aligned with `O` and no
  design exists, so `a3` is optimised (`LINKAGE_L3_ANGLE_RAD` fixes it).

### RR model with the linkage (`seneca_loco/simulation/assets/xml/senecabot_loco_rr.xml`)

Generated, not edited by hand: `python -m geometry_opt.make_rr_xml [--policy v140] [--material PLA] [--no-linkage]` reads
`senecabot_loco.xml` and `results/linkage_<policy>.csv` (run `run_pipeline --linkage` first; FL / BL copy the FR / BR
design, the left legs of the model are identical to the right ones). Picture: `results/rr_model_linkage.png`.

- **RR leg:** the four `*_upper` bodies, hip joints and hip actuators are removed. `*_lower` (knee) hangs from the torso
  where the knee was with the hip at 0 (`upper pos + (0, 0, -thigh length)`); tibia and metatarsus keep geometry, mass,
  axes, ranges and damping, and every remaining name (`*_thigh_mimic` sits on the torso at the knee pivot).
- **Linkage:** each leg carries RR 1 (`*_rr1_l1`, `*_rr1_l2`) and RR 2 (`*_rr2_l3`, `*_rr2_l4`) as bodies in the sagittal
  plane (mechanism plane x = torso x, plane y = torso z). `*_rr1_joint` sits on the torso at B, `*_rr2_joint` at the knee
  pivot O; the elbows `*_rr1_elbow`, `*_rr2_elbow` are free; two `connect` equalities per leg pin the tip of `l2` to the
  tibia at `f1·r1` and the tip of `l4` to the metatarsus at `f2·r2`. All joint values are absolute angles (axis -y,
  `ref` = reference value).
- **Looks:** the bars are capsules as wide as the link they drive (RR 1 like the tibia, 25 mm radius; RR 2 like the
  metatarsus, 20 mm), RR 1 in a plane 60 mm inside the leg plane and RR 2 60 mm outside it, so no capsules overlap. Thin
  mounting plates (`*_mount_*`) join the torso underside to B and to O in each of the three planes. Bars and plates are
  visual only (no contacts); each bar weighs as a hollow PLA tube of 9 mm outer radius (4 mm wall), the plates weigh
  nothing (total mass 6.06 kg).
- **Actuation:** the only motors are the two cranks of each leg (`*_rr1`, `*_rr2`, torque range = peak of the motor
  selected in `linkage_actuators_<policy>_<material>.csv`, 100 N·m if there is none). Knee and ankle are passive,
  driven through the loops. nq/nv/nu/neq = 31/30/8/8.
- **Reference pose (qpos0):** the closed pose at the centre of the range of motion of the gait (knee and ankle `ref`,
  body `euler`), so the loops are closed at reset. Checked: equality violation 3e-17 m at qpos0 and <= 1.5 µm over 442
  gait poses of FR and BR set from the IK of `linkage.py`; 1000 steps with PD on the cranks stay stable.
- The policies trained on the RRR model do not transfer: the gait has to be retrained on this XML (control = the 8 crank
  torques). `LINKAGE_XML` stays on the RRR model (the recorded gaits); `config.XML_RR_PATH` points to this file.
