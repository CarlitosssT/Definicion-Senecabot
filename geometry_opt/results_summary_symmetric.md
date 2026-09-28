# Geometry optimization — results summary
_Generated 2026-09-27 22:00 by `geometry_opt/run_pipeline.py`._

## Configuration
- FS_torque = 1.2, FS_material = 1.2, FS_speed = 1.0; wall thickness t = 4.0 mm
- Left/right motors: same motor on both sides (pair sized with the larger requirement)
- Motor criteria: peak torque ≥ FS·|τ|max, nominal torque ≥ FS·τ_RMS, rated speed ≥ FS_speed·|ω|max
- Convergence: motors unchanged AND total mass change ≤ 1% (max 10 iterations)
- Link structure mass in the loop: True (hollow tubes, combined criterion); foot sphere infill 1.0
- Joint-limit stops carry their load: True; joint damping: thesis value (1.5 N·m·s/rad hip/knee, 0.05 ankle)
- Anisotropy factor on σ_yield: 1.0

| Material | σ_yield [MPa] | σ_adm [MPa] | E [GPa] | ρ [kg/m³] | Source |
|---|---|---|---|---|---|
| PLA | 51.0 | 42.5 | 2.3 | 1240.0 | Prusament PLA TDS 2021-10: printed horizontal, ISO 527-1 |
| PETG | 47.0 | 39.2 | 1.5 | 1270.0 | Prusament PETG TDS 2021-10: printed horizontal, ISO 527-1 |

## Initial parameters (extracted from the thesis PDF)
| Joint / segment | Length front [cm] | Length back [cm] | Torque limit [N·m] |
|---|---|---|---|
| hip | 12.1 | 16.6 | ±80 |
| knee | 11.1 | 23.1 | ±80 |
| ankle | 15.3 | 14.8 | ±40 |

- Torso mass: 3.0 kg (only explicit mass in the thesis model; leg masses come from the capsule geometry at MuJoCo's default density, 1000 kg/m³).
- thesis <-> XML: all segment lengths, ROMs, torso mass and torque limits match

## Overview

| Policy | Material | Converged | Iterations | Final mass [kg] | Motors [kg] | Structure [kg] | Motor models used |
|---|---|---|---|---|---|---|---|
| 100M @ 0.218 m/s (no-slip reference) | PLA | yes | 3 | 9.71 | 6.00 | 0.71 | X4-36 36:1, X6-60 19,612:1, X8-32 9:1 |
| 100M @ 0.218 m/s (no-slip reference) | PETG | yes | 3 | 9.75 | 6.00 | 0.75 | X4-36 36:1, X6-60 19,612:1, X8-32 9:1 |
| 100M @ 1.40 m/s (mocap-speed reference) | PLA | yes | 2 | 9.72 | 6.00 | 0.72 | X4-36 36:1, X6-60 19,612:1, X8-32 9:1 |
| 100M @ 1.40 m/s (mocap-speed reference) | PETG | yes | 2 | 9.75 | 6.00 | 0.75 | X4-36 36:1, X6-60 19,612:1, X8-32 9:1 |

## 100M @ 0.218 m/s (no-slip reference)
- Agent: `seneca_loco/artifacts/trained_agents/2026-09-27/07-12-14/PPOJax_saved.pkl`; reference: `trajectory_adapted.noslip_0218.npz`
- Recorded gait: 13 robots × 850 control steps (11050 states), forward speed 0.211 m/s, trained model mass 7.83 kg
- Inverse-dynamics validation on the trained model (iteration 0 vs forward simulation): max torque RMS error 0.000 N·m, peak ratio 1.000–1.000, base residual max 0.00% of weight, hinge-moment check 0.000 N·m

### PLA — converged in 3 iterations

**1. Convergence history** (mass of the model simulated at each iteration; |τ|max per joint in N·m; Δ = relative change of the next design's mass)

| Iter | Total [kg] | Motors [kg] | Structure [kg] | FR-h | FR-k | FR-a | FL-h | FL-k | FL-a | BR-h | BR-k | BR-a | BL-h | BL-k | BL-a | Motors changed | Δ mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.83 | 0.00 | 4.83 | 9.1 | 27.7 | 12.6 | 10.6 | 22.8 | 12.9 | 7.1 | 15.0 | 8.4 | 6.3 | 18.7 | 8.5 | — | 22.6% |
| 1 | 9.61 | 5.94 | 0.67 | 8.0 | 31.0 | 11.4 | 11.4 | 26.7 | 11.9 | 11.1 | 18.0 | 8.4 | 8.5 | 21.3 | 9.0 | yes | 1.1% |
| 2 | 9.71 | 6.00 | 0.71 | 8.0 | 31.1 | 11.4 | 11.4 | 26.8 | 11.9 | 11.1 | 18.2 | 8.5 | 8.6 | 21.5 | 9.1 | no | 0.0% |

Final design: **9.71 kg** (torso 3.00, motors 6.00, PLA structure 0.71).

Consistency of the last iteration: unbalanced base force max 19.72% (mean 2.681%) of the robot weight; hinge-moment check 4.7e-14 N·m.

**2. Selected actuators**

| Joint | Torque max sim [N·m] | Torque RMS [N·m] | ω max [rpm] | Selected motor | Motor peak [N·m] | Motor nominal [N·m] | Motor speed [rpm] | Motor mass [kg] | Binding criterion | Damping part of τmax [N·m] | Base residual at τmax [% weight] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fr_hip | 8.04 | 3.56 | 59 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | speed | 6.0 | 3.1 |
| fr_knee | 31.06 | 8.04 | 125 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 18.0 | 0.9 |
| fr_ankle | 11.41 | 2.97 | 176 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.4 | 18.7 |
| fl_hip | 11.38 | 3.89 | 57 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | speed | 8.6 | 15.7 |
| fl_knee | 26.78 | 8.04 | 105 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 15.4 | 0.5 |
| fl_ankle | 11.91 | 2.91 | 227 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.7 | 11.1 |
| br_hip | 11.15 | 3.52 | 35 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 4.0 | 0.0 |
| br_knee | 18.22 | 7.58 | 67 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 5.6 | 0.5 |
| br_ankle | 8.52 | 2.26 | 84 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.2 | 0.5 |
| bl_hip | 8.59 | 2.72 | 34 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 4.6 | 0.5 |
| bl_knee | 21.46 | 7.82 | 68 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 7.9 | 0.9 |
| bl_ankle | 9.12 | 2.33 | 114 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.1 | 0.9 |

_Base residual_: unbalanced base force of the inverse dynamics at the instant of the joint's peak (fixed-kinematics limitation, mainly in flight / single-support phases); > 5% marks a less reliable peak.

Total actuator mass: 6.00 kg.

### PETG — converged in 3 iterations

**1. Convergence history** (mass of the model simulated at each iteration; |τ|max per joint in N·m; Δ = relative change of the next design's mass)

| Iter | Total [kg] | Motors [kg] | Structure [kg] | FR-h | FR-k | FR-a | FL-h | FL-k | FL-a | BR-h | BR-k | BR-a | BL-h | BL-k | BL-a | Motors changed | Δ mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.83 | 0.00 | 4.83 | 9.1 | 27.7 | 12.6 | 10.6 | 22.8 | 12.9 | 7.1 | 15.0 | 8.4 | 6.3 | 18.7 | 8.5 | — | 23.0% |
| 1 | 9.64 | 5.94 | 0.70 | 8.0 | 31.0 | 11.4 | 11.4 | 26.7 | 11.9 | 11.1 | 18.0 | 8.5 | 8.6 | 21.3 | 9.0 | yes | 1.1% |
| 2 | 9.75 | 6.00 | 0.75 | 8.0 | 31.1 | 11.5 | 11.4 | 26.8 | 12.0 | 11.2 | 18.3 | 8.5 | 8.6 | 21.5 | 9.2 | no | 0.0% |

Final design: **9.75 kg** (torso 3.00, motors 6.00, PETG structure 0.75).

Consistency of the last iteration: unbalanced base force max 19.40% (mean 2.630%) of the robot weight; hinge-moment check 5.2e-14 N·m.

**2. Selected actuators**

| Joint | Torque max sim [N·m] | Torque RMS [N·m] | ω max [rpm] | Selected motor | Motor peak [N·m] | Motor nominal [N·m] | Motor speed [rpm] | Motor mass [kg] | Binding criterion | Damping part of τmax [N·m] | Base residual at τmax [% weight] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fr_hip | 8.04 | 3.56 | 59 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | speed | 6.0 | 3.1 |
| fr_knee | 31.08 | 8.05 | 125 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 18.0 | 0.9 |
| fr_ankle | 11.46 | 2.98 | 176 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.4 | 18.5 |
| fl_hip | 11.39 | 3.89 | 57 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | speed | 8.6 | 15.4 |
| fl_knee | 26.80 | 8.05 | 105 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 15.4 | 0.5 |
| fl_ankle | 11.95 | 2.92 | 227 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.7 | 10.8 |
| br_hip | 11.17 | 3.52 | 35 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 4.0 | 0.0 |
| br_knee | 18.25 | 7.60 | 67 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 5.6 | 0.5 |
| br_ankle | 8.55 | 2.27 | 84 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.2 | 0.5 |
| bl_hip | 8.64 | 2.72 | 34 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 4.6 | 0.5 |
| bl_knee | 21.48 | 7.84 | 68 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 7.9 | 0.9 |
| bl_ankle | 9.16 | 2.34 | 114 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.1 | 0.9 |

_Base residual_: unbalanced base force of the inverse dynamics at the instant of the joint's peak (fixed-kinematics limitation, mainly in flight / single-support phases); > 5% marks a less reliable peak.

Total actuator mass: 6.00 kg.

### 3. Structural sizing (hollow tubes, t = 4.0 mm)

**Axial criterion** σ = F/A ≤ σ_adm (loads of each material's converged design). `*` = the required area is below the smallest t = 4 mm tube: the minimum is a solid rod Ø8 mm.

| Link | L [mm] | F axial max [N] (PLA/PETG) | F resultant max [N] | A_min PLA [mm²] | D_ext,min PLA [mm] | A_min PETG [mm²] | D_ext,min PETG [mm] |
|---|---|---|---|---|---|---|---|
| fr_upper (hip→knee) | 121 | 121 / 121 | 156 / 157 | 2.84 | 8.0* | 3.09 | 8.0* |
| fr_lower (knee→ankle) | 111 | 122 / 123 | 181 / 182 | 2.88 | 8.0* | 3.13 | 8.0* |
| fr_foot (ankle→foot) | 153 | 252 / 253 | 258 / 258 | 5.93 | 8.0* | 6.45 | 8.0* |
| fl_upper (hip→knee) | 121 | 149 / 150 | 188 / 188 | 3.52 | 8.0* | 3.83 | 8.0* |
| fl_lower (knee→ankle) | 111 | 168 / 168 | 213 / 213 | 3.95 | 8.0* | 4.29 | 8.0* |
| fl_foot (ankle→foot) | 153 | 314 / 315 | 317 / 317 | 7.39 | 8.0* | 8.04 | 8.0* |
| br_upper (hip→knee) | 166 | 106 / 107 | 134 / 134 | 2.51 | 8.0* | 2.73 | 8.0* |
| br_lower (knee→ankle) | 231 | 104 / 105 | 151 / 151 | 2.45 | 8.0* | 2.67 | 8.0* |
| br_foot (ankle→foot) | 148 | 162 / 163 | 181 / 182 | 3.82 | 8.0* | 4.16 | 8.0* |
| bl_upper (hip→knee) | 166 | 80 / 81 | 114 / 114 | 1.89 | 8.0* | 2.06 | 8.0* |
| bl_lower (knee→ankle) | 231 | 101 / 101 | 129 / 129 | 2.37 | 8.0* | 2.58 | 8.0* |
| bl_foot (ankle→foot) | 148 | 126 / 127 | 158 / 159 | 2.97 | 8.0* | 3.23 | 8.0* |

**Combined criterion** (axial + bending + shear + torsion, von Mises, simultaneous components): this is the sizing used for the link masses in the loop.

| Link | Bending max [N·m] (PLA) | Torsion max [N·m] | Shear max [N] | D_ext,min PLA [mm] | Tube mass PLA [g] | D_ext,min PETG [mm] | Tube mass PETG [g] | Governing |
|---|---|---|---|---|---|---|---|---|
| fr_upper | 17.9 | 3.57 | 135 | 16.8 | 24 | 17.4 | 26 | bending |
| fr_lower | 17.9 | 2.87 | 236 | 16.8 | 22 | 17.3 | 24 | bending |
| fr_foot | 13.1 | 0.06 | 62 | 15.1 | 27 | 15.6 | 28 | bending |
| fl_upper | 23.0 | 9.37 | 180 | 18.5 | 27 | 19.1 | 29 | bending |
| fl_lower | 23.0 | 6.96 | 278 | 18.4 | 25 | 19.1 | 27 | bending |
| fl_foot | 14.8 | 0.02 | 76 | 15.8 | 28 | 16.3 | 30 | bending |
| br_upper | 22.1 | 9.64 | 112 | 18.1 | 37 | 18.7 | 39 | bending |
| br_lower | 22.1 | 6.06 | 158 | 18.1 | 51 | 18.7 | 54 | bending |
| br_foot | 22.0 | 0.37 | 102 | 18.1 | 33 | 18.7 | 35 | bending |
| bl_upper | 14.8 | 10.51 | 104 | 16.7 | 33 | 17.2 | 35 | bending |
| bl_lower | 21.0 | 7.16 | 124 | 18.2 | 51 | 18.8 | 54 | bending |
| bl_foot | 22.2 | 0.45 | 108 | 18.2 | 33 | 18.8 | 35 | bending |

## 100M @ 1.40 m/s (mocap-speed reference)
- Agent: `seneca_loco/artifacts/trained_agents/2026-09-27/18-22-14/PPOJax_saved.pkl`; reference: `trajectory_adapted.preRootSpeed.bak.npz`
- Recorded gait: 13 robots × 850 control steps (11050 states), forward speed 1.394 m/s, trained model mass 7.83 kg
- Inverse-dynamics validation on the trained model (iteration 0 vs forward simulation): max torque RMS error 0.000 N·m, peak ratio 1.000–1.000, base residual max 0.00% of weight, hinge-moment check 0.000 N·m

### PLA — converged in 2 iterations

**1. Convergence history** (mass of the model simulated at each iteration; |τ|max per joint in N·m; Δ = relative change of the next design's mass)

| Iter | Total [kg] | Motors [kg] | Structure [kg] | FR-h | FR-k | FR-a | FL-h | FL-k | FL-a | BR-h | BR-k | BR-a | BL-h | BL-k | BL-a | Motors changed | Δ mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.83 | 0.00 | 4.83 | 18.9 | 32.8 | 8.9 | 15.3 | 35.3 | 13.0 | 10.3 | 17.4 | 16.2 | 10.9 | 14.2 | 12.8 | — | 23.6% |
| 1 | 9.68 | 6.00 | 0.68 | 22.2 | 35.9 | 13.5 | 19.0 | 45.7 | 17.0 | 12.9 | 22.3 | 15.6 | 14.1 | 19.2 | 12.1 | no | 0.4% |

Final design: **9.72 kg** (torso 3.00, motors 6.00, PLA structure 0.72).

Consistency of the last iteration: unbalanced base force max 37.66% (mean 5.573%) of the robot weight; hinge-moment check 5.9e-14 N·m.

**2. Selected actuators**

| Joint | Torque max sim [N·m] | Torque RMS [N·m] | ω max [rpm] | Selected motor | Motor peak [N·m] | Motor nominal [N·m] | Motor speed [rpm] | Motor mass [kg] | Binding criterion | Damping part of τmax [N·m] | Base residual at τmax [% weight] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fr_hip | 22.18 | 5.64 | 70 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 7.5 | 0.9 |
| fr_knee | 35.92 | 8.74 | 160 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 15.8 | 1.3 |
| fr_ankle | 13.53 | 3.45 | 178 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.3 | 1.2 |
| fl_hip | 18.96 | 5.64 | 42 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 2.6 | 1.1 |
| fl_knee | 45.67 | 9.78 | 163 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 20.0 | 1.4 |
| fl_ankle | 17.03 | 3.55 | 182 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.5 | 1.0 |
| br_hip | 12.89 | 4.15 | 32 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 2.1 | 1.6 |
| br_knee | 22.34 | 7.87 | 61 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 2.7 | 1.3 |
| br_ankle | 15.60 | 5.03 | 193 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | RMS torque | 0.1 | 26.9 |
| bl_hip | 14.12 | 4.30 | 29 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 2.9 | 2.2 |
| bl_knee | 19.25 | 6.94 | 68 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 2.0 | 1.6 |
| bl_ankle | 12.14 | 4.63 | 185 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | RMS torque | 0.4 | 2.5 |

_Base residual_: unbalanced base force of the inverse dynamics at the instant of the joint's peak (fixed-kinematics limitation, mainly in flight / single-support phases); > 5% marks a less reliable peak.

Total actuator mass: 6.00 kg.

### PETG — converged in 2 iterations

**1. Convergence history** (mass of the model simulated at each iteration; |τ|max per joint in N·m; Δ = relative change of the next design's mass)

| Iter | Total [kg] | Motors [kg] | Structure [kg] | FR-h | FR-k | FR-a | FL-h | FL-k | FL-a | BR-h | BR-k | BR-a | BL-h | BL-k | BL-a | Motors changed | Δ mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 7.83 | 0.00 | 4.83 | 18.9 | 32.8 | 8.9 | 15.3 | 35.3 | 13.0 | 10.3 | 17.4 | 16.2 | 10.9 | 14.2 | 12.8 | — | 24.0% |
| 1 | 9.71 | 6.00 | 0.71 | 22.1 | 36.0 | 13.5 | 18.9 | 45.7 | 17.0 | 12.9 | 22.4 | 15.6 | 14.2 | 19.3 | 12.2 | no | 0.5% |

Final design: **9.75 kg** (torso 3.00, motors 6.00, PETG structure 0.75).

Consistency of the last iteration: unbalanced base force max 36.93% (mean 5.461%) of the robot weight; hinge-moment check 5.6e-14 N·m.

**2. Selected actuators**

| Joint | Torque max sim [N·m] | Torque RMS [N·m] | ω max [rpm] | Selected motor | Motor peak [N·m] | Motor nominal [N·m] | Motor speed [rpm] | Motor mass [kg] | Binding criterion | Damping part of τmax [N·m] | Base residual at τmax [% weight] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fr_hip | 22.09 | 5.63 | 70 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 7.5 | 0.9 |
| fr_knee | 35.98 | 8.76 | 160 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 15.8 | 1.3 |
| fr_ankle | 13.51 | 3.45 | 178 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.3 | 1.2 |
| fl_hip | 18.92 | 5.65 | 42 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 2.6 | 1.0 |
| fl_knee | 45.73 | 9.79 | 163 | X6-60 19,612:1 | 60.0 | 20.0 | 176 | 0.820 | speed | 20.0 | 1.3 |
| fl_ankle | 17.02 | 3.55 | 182 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | speed | 0.5 | 1.0 |
| br_hip | 12.94 | 4.15 | 32 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 2.1 | 1.6 |
| br_knee | 22.38 | 7.89 | 61 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 2.7 | 1.3 |
| br_ankle | 15.63 | 5.04 | 193 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | RMS torque | 0.1 | 26.4 |
| bl_hip | 14.17 | 4.32 | 29 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | peak torque | 2.9 | 2.6 |
| bl_knee | 19.31 | 6.96 | 68 | X4-36 36:1 | 34.0 | 10.5 | 111 | 0.360 | RMS torque | 2.0 | 1.5 |
| bl_ankle | 12.17 | 4.65 | 185 | X8-32 9:1 | 32.0 | 8.0 | 277 | 0.550 | RMS torque | 0.4 | 2.5 |

_Base residual_: unbalanced base force of the inverse dynamics at the instant of the joint's peak (fixed-kinematics limitation, mainly in flight / single-support phases); > 5% marks a less reliable peak.

Total actuator mass: 6.00 kg.

### 3. Structural sizing (hollow tubes, t = 4.0 mm)

**Axial criterion** σ = F/A ≤ σ_adm (loads of each material's converged design). `*` = the required area is below the smallest t = 4 mm tube: the minimum is a solid rod Ø8 mm.

| Link | L [mm] | F axial max [N] (PLA/PETG) | F resultant max [N] | A_min PLA [mm²] | D_ext,min PLA [mm] | A_min PETG [mm²] | D_ext,min PETG [mm] |
|---|---|---|---|---|---|---|---|
| fr_upper (hip→knee) | 121 | 165 / 165 | 219 / 219 | 3.88 | 8.0* | 4.22 | 8.0* |
| fr_lower (knee→ankle) | 111 | 168 / 168 | 246 / 246 | 3.95 | 8.0* | 4.30 | 8.0* |
| fr_foot (ankle→foot) | 153 | 357 / 358 | 365 / 366 | 8.41 | 8.0* | 9.14 | 8.0* |
| fl_upper (hip→knee) | 121 | 144 / 144 | 180 / 180 | 3.38 | 8.0* | 3.68 | 8.0* |
| fl_lower (knee→ankle) | 111 | 166 / 166 | 209 / 209 | 3.90 | 8.0* | 4.23 | 8.0* |
| fl_foot (ankle→foot) | 153 | 321 / 322 | 328 / 328 | 7.56 | 8.0* | 8.21 | 8.0* |
| br_upper (hip→knee) | 166 | 102 / 103 | 127 / 127 | 2.41 | 8.0* | 2.62 | 8.0* |
| br_lower (knee→ankle) | 231 | 87 / 87 | 148 / 149 | 2.05 | 8.0* | 2.23 | 8.0* |
| br_foot (ankle→foot) | 148 | 149 / 149 | 170 / 171 | 3.50 | 8.0* | 3.81 | 8.0* |
| bl_upper (hip→knee) | 166 | 109 / 110 | 134 / 135 | 2.57 | 8.0* | 2.80 | 8.0* |
| bl_lower (knee→ankle) | 231 | 85 / 86 | 152 / 153 | 2.01 | 8.0* | 2.20 | 8.0* |
| bl_foot (ankle→foot) | 148 | 149 / 149 | 176 / 177 | 3.49 | 8.0* | 3.81 | 8.0* |

**Combined criterion** (axial + bending + shear + torsion, von Mises, simultaneous components): this is the sizing used for the link masses in the loop.

| Link | Bending max [N·m] (PLA) | Torsion max [N·m] | Shear max [N] | D_ext,min PLA [mm] | Tube mass PLA [g] | D_ext,min PETG [mm] | Tube mass PETG [g] | Governing |
|---|---|---|---|---|---|---|---|---|
| fr_upper | 20.4 | 7.43 | 200 | 17.9 | 26 | 18.5 | 28 | bending |
| fr_lower | 20.6 | 5.79 | 330 | 17.9 | 24 | 18.4 | 26 | bending |
| fr_foot | 20.3 | 0.30 | 79 | 17.8 | 33 | 18.4 | 35 | bending |
| fl_upper | 21.4 | 10.78 | 177 | 18.0 | 26 | 18.6 | 28 | bending |
| fl_lower | 21.4 | 7.98 | 283 | 17.9 | 24 | 18.5 | 26 | bending |
| fl_foot | 15.8 | 0.28 | 131 | 16.2 | 29 | 16.7 | 31 | bending |
| br_upper | 22.8 | 9.76 | 107 | 18.3 | 37 | 18.9 | 40 | bending |
| br_lower | 22.8 | 5.95 | 153 | 18.3 | 51 | 18.9 | 55 | bending |
| br_foot | 14.4 | 0.38 | 82 | 15.5 | 27 | 16.0 | 28 | bending |
| bl_upper | 20.1 | 12.89 | 108 | 18.0 | 36 | 18.6 | 39 | bending |
| bl_lower | 20.9 | 9.20 | 154 | 18.1 | 51 | 18.7 | 54 | bending |
| bl_foot | 19.5 | 0.70 | 103 | 17.3 | 31 | 17.9 | 33 | bending |

## Method and limitations
- The trained policies do not transfer to heavier robots (tested: at ~15 kg the joint error grows from 7–8° to 20–25° and the 1.40 m/s gait slows to 0.5–0.65 m/s). Loads are therefore obtained by inverse dynamics of the gait each policy achieves on the thesis model: every design must perform that same gait; ground forces are re-solved per sample for the new masses (friction pyramid, NNLS) and joint-limit / dry-friction / self-contact forces are kept as simulated.
- With fixed kinematics, phases with 0–1 feet on the ground cannot be balanced exactly for a different mass distribution (momentum is not conserved the same way); the residual is reported per iteration and at every joint's torque peak. The link bending peaks all occur at residuals of a few %.
- Validation: on the trained model the inverse dynamics reproduces the forward simulation exactly (torques, ground forces, base balance), and the link wrenches satisfy the hinge-moment identity to machine precision for every design.
- Joint torques include the thesis model's joint damping (a simulation regularizer); its share at each joint's peak is listed. Set `JOINT_DAMPING_OVERRIDE` in config.py to size for a real value.
- Motors are point masses at the joints (the catalog has no dimensions); rotor inertia/gear reflected inertia is the model's armature (0.01 kg·m²), not motor-specific.
- Torque–speed coupling is not checked beyond peak torque and rated speed separately.
- FDM strength: horizontal-print TDS values; parts loaded across layers can retain only ~50–70% (`ANISOTROPY_FACTOR`). Stiffness, buckling, fatigue and joint/connection design are not covered.
