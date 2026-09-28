"""SenecaBot project launcher — a simple interactive menu over the main entry points.

Run from anywhere:  python main.py
Each task runs in its own subprocess (Hydra apps and MJX/GPU tools want a clean
process; see debug_agent's --record_only note), launched from the repo root.
"""
import subprocess
import sys

from simulation.config import paths

MENU = """
================ SenecaBot — DeepMimic imitation learning ================
 MOCAP -> trajectory pipeline
   1) Run MOCAP pipeline, stages 1-3 (C3D -> trajectory.npz)
   2) Generate training trajectory, stages 4-5 (-> trajectory_adapted.npz)
 Training & evaluation
   3) Train (Hydra; simulation/training/conf.yaml)
   4) Replay a trained agent (eval)
   5) Agent debug figures (-> artifacts/figures/agent_debug/)
   6) Agent scalar diagnostics + composite
 Reference & extras
   7) Play back the reference trajectory (no agent)
   8) Plot trajectory joint angles vs model limits
   9) Scripted walk animation (no RL)
   0) Exit
===========================================================================
"""


def ask(prompt, default=""):
    s = input(prompt).strip()
    return s if s else default


def yes(prompt):
    return ask(f"{prompt} [y/N]: ").lower().startswith("y")


def list_agents(limit=15):
    """Newest saved agents under artifacts/trained_agents/ (incl. curated/)."""
    agents = sorted(paths.TRAINED_AGENTS.rglob("PPOJax_saved.pkl"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    return agents[:limit]


def pick_agent():
    agents = list_agents()
    if not agents:
        print("No saved agents found under", paths.TRAINED_AGENTS)
        return None
    print("\nAvailable agents (newest first):")
    for i, p in enumerate(agents, 1):
        print(f"  {i:2d}) {p.relative_to(paths.TRAINED_AGENTS).parent}")
    s = ask(f"Pick agent [1-{len(agents)}, Enter = 1]: ", "1")
    try:
        return agents[int(s) - 1]
    except (ValueError, IndexError):
        print("Invalid choice.")
        return None


def run(argv):
    print("\n$", " ".join(str(a) for a in argv), "\n")
    subprocess.run([str(a) for a in argv], cwd=paths.PROJECT_ROOT)


def main():
    py = sys.executable
    while True:
        print(MENU)
        choice = ask("Choose an option: ")

        if choice == "1":
            cmd = [py, "-m", "simulation.mocap.run"]
            if yes("Also build the augmented dataset?"):
                cmd.append("--augment")
            run(cmd)
        elif choice == "2":
            cmd = [py, "-m", "simulation.mocap.trajectory.trajectory_generation"]
            if yes("Use the augmented dataset (+use_augmented=true)?"):
                cmd.append("+use_augmented=true")
            run(cmd)
        elif choice == "3":
            steps = ask("Total timesteps [Enter = conf.yaml value]: ")
            cmd = [py, "simulation/training/train.py"]
            if steps:
                cmd.append(f"experiment.total_timesteps={steps}")
            if yes(f"Start training (long GPU run, output -> {paths.TRAINED_AGENTS})?"):
                run(cmd)
        elif choice == "4":
            agent = pick_agent()
            if agent:
                cmd = [py, "simulation/training/eval.py", "--path", agent]
                if yes("Use CPU MuJoCo instead of MJX?"):
                    cmd.append("--use_mujoco")
                cam = ask("Camera (follow / follow_west / Enter = free): ")
                if cam:
                    cmd += ["--camera", cam]
                run(cmd)
        elif choice == "5":
            agent = pick_agent()
            if agent:
                run([py, "simulation/analysis/debug_agent.py", "--path", agent])
        elif choice == "6":
            agent = pick_agent()
            if agent:
                run([py, "simulation/analysis/agent_diagnostics.py", "--path", agent])
        elif choice == "7":
            run([py, "-m", "simulation.mocap.trajectory.trajectory_playback"])
        elif choice == "8":
            run([py, "simulation/analysis/plot_joint_angles.py"])
        elif choice == "9":
            run([py, "scripts/main.py"])
        elif choice == "0":
            break
        else:
            print("Unknown option.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print()
