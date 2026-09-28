"""
run.py

One command to take a raw C3D file all the way to trajectory(.npz) and,
optionally, trajectory_augmented(.npz):

    python -m simulation.mocap.run                 # file index 0, no augmentation
    python -m simulation.mocap.run --augment       # + build augmented dataset
    python -m simulation.mocap.run --file-index 2  # pick another C3D file
    python -m simulation.mocap.run --c3d /path/to/file.c3d --augment

It prints a time-consistency report so the single notion of dt/frequency is
visible end to end (TODO_time_consistency_check.md, checkpoints 1-4). Stage 4
(trajectory_generation.py) needs the Loco-MuJoCo env + Hydra and is run
separately afterwards; it now reads the truthful frequency stored here.
"""

import argparse
from pathlib import Path

from simulation.config import pipeline as P
from simulation.mocap import io, kinematics, build, augment


def run(c3d_path, do_augment=False, save_csv=True):
    print("=" * 60)
    print(f"Stage 1  C3D -> markers   : {Path(c3d_path).name}")
    df, point_rate = io.load_c3d(c3d_path)
    print(f"         frames={len(df)}  point_rate={point_rate} Hz  "
          f"duration={len(df) / point_rate:.3f} s")
    if abs(point_rate - P.POINT_RATE) > 1e-6:
        print(f"         NOTE: header rate {point_rate} != config POINT_RATE "
              f"{P.POINT_RATE}; using header value.")

    print("Stage 2  markers -> angles")
    df = kinematics.compute_joint_angles(df)
    if save_csv:
        P.PROC_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = P.PROC_DIR / P.CSV_NAME
        df.to_csv(csv_path, index=False)
        print(f"         saved {csv_path}")

    print("Stage 3  angles -> qpos/qvel (MuJoCo FK)")
    traj = build.angles_to_trajectory(df, frequency=point_rate)
    traj_path = build.save_trajectory(traj)
    print(f"         qpos={traj['qpos'].shape}  qvel={traj['qvel'].shape}")
    print(f"         saved {traj_path}")

    aug_path = None
    if do_augment:
        print("Stage 4  augmentation")
        aug = augment.generate_augmented(traj["qpos"], traj["qvel"], frequency=point_rate)
        aug_path = augment.save_augmented(aug)
        print(f"         cycles={len(aug['meta'])}  qpos={aug['qpos'].shape}")
        print(f"         saved {aug_path}")

    _consistency_report(point_rate, traj["frequency"],
                        aug["frequency"] if do_augment else None)

    print("\nNext: run trajectory_generation.py to produce trajectory_adapted.npz")
    print("=" * 60)
    return traj_path, aug_path


def _consistency_report(header_rate, build_freq, aug_freq):
    print("\n--- Time-consistency report ---")
    print(f"  C3D header point rate     : {header_rate} Hz")
    print(f"  qvel dt used in build     : {1.0 / build_freq:.6f} s  ({build_freq} Hz)")
    print(f"  frequency stored (traj)   : {build_freq} Hz")
    if aug_freq is not None:
        print(f"  frequency stored (augment): {aug_freq} Hz")
    rates = [header_rate, build_freq] + ([aug_freq] if aug_freq is not None else [])
    ok = all(abs(r - header_rate) < 1e-6 for r in rates)
    print(f"  ALL CONSISTENT            : {ok}")
    assert ok, "Frequency mismatch across pipeline stages!"


def _parse_args():
    ap = argparse.ArgumentParser(description="C3D -> trajectory pipeline runner")
    ap.add_argument("--c3d", type=str, default=None,
                    help="Path to a .c3d file (overrides --file-index).")
    ap.add_argument("--file-index", type=int, default=0,
                    help="Index into the sorted C3D_Final/*.c3d list.")
    ap.add_argument("--augment", action="store_true",
                    help="Also build trajectory_augmented.npz.")
    ap.add_argument("--no-csv", action="store_true",
                    help="Do not write ovino_angles.csv.")
    return ap.parse_args()


def main():
    args = _parse_args()
    if args.c3d:
        c3d_path = Path(args.c3d)
    else:
        files = io.list_c3d_files()
        if not files:
            raise FileNotFoundError(f"No .c3d files in {P.C3D_DIR}")
        c3d_path = P.C3D_DIR / files[args.file_index]
    run(c3d_path, do_augment=args.augment, save_csv=not args.no_csv)


if __name__ == "__main__":
    main()
