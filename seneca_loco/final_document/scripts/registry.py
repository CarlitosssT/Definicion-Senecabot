"""
Figure registry — the SINGLE source of truth for every figure in the thesis.

Each entry says, declaratively:
    "Figure <fig_id> comes from <CSV>, made by <plotter>, with <kwargs>."

Benefits:
    * One file lists every figure → trivial traceability.
    * `generate_all.py` iterates this list to regenerate everything.
    * If a figure looks wrong in the PDF, you grep its fig_id here and
      immediately know which CSV and which function to inspect.

To add a figure: append an entry, run `python generate_all.py`, done.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import plotters


@dataclass(frozen=True)
class FigureSpec:
    # Naming triplet (must be unique across the registry)
    category: str
    descriptor: str
    variant: str = "v1"

    # Where the data lives (relative to data/)
    sources: tuple[str, ...] = ()

    # What to call. Function name in `plotters` + kwargs.
    plotter: str = ""
    kwargs: dict = field(default_factory=dict)

    # Free-text — shown in the manifest, useful as a caption draft.
    note: str = ""

    @property
    def fig_id(self) -> str:
        return f"{self.category}__{self.descriptor}__{self.variant}"


# ----------------------------------------------------------------------
# THE REGISTRY
# Edit this list to add / remove figures.
# ----------------------------------------------------------------------
FIGURES: list[FigureSpec] = [
    FigureSpec(
        category="training",
        descriptor="ppo_reward",
        variant="seed_avg",
        sources=("ppo_training_log.csv",),
        plotter="plot_training_curves",
        kwargs=dict(
            x="step", y="reward", seed_col="seed",
            ylabel="Episode reward", xlabel="Training step",
            smooth_window=5,
        ),
        note="Mean ± std PPO reward across 5 seeds for SenecaBot imitation policy.",
    ),
    FigureSpec(
        category="training",
        descriptor="ppo_reward",
        variant="by_method",
        sources=("ppo_training_log.csv",),
        plotter="plot_training_curves",
        kwargs=dict(
            x="step", y="reward",
            seed_col="seed", group_col="method",
            ylabel="Episode reward", xlabel="Training step",
            smooth_window=5,
        ),
        note="PPO reward, comparing imitation variants.",
    ),
    FigureSpec(
        category="trajectory",
        descriptor="hindlimb_joints",
        variant="sim_vs_mocap",
        sources=("mocap_reference.csv", "sim_rollout.csv"),
        plotter="plot_trajectory_comparison",
        kwargs=dict(
            time_col="t",
            dof_cols=("hip_flex", "knee_flex", "ankle_flex"),
            ylabel="Angle [rad]",
        ),
        note="Hindlimb joint trajectories: ovine mocap reference vs SenecaBot sim.",
    ),
    FigureSpec(
        category="error",
        descriptor="tracking_rmse",
        variant="per_joint",
        sources=("tracking_error.csv",),
        plotter="plot_tracking_error",
        kwargs=dict(
            time_col="t",
            error_cols=("hip_flex_err", "knee_flex_err", "ankle_flex_err"),
            ylabel="RMSE [rad]",
        ),
        note="Per-joint tracking error along a 2-second rollout.",
    ),
    FigureSpec(
        category="schematic",
        descriptor="leg_naming",
        variant="v1",
        sources=(),                      # schematic: no data source
        plotter="plot_leg_convention",
        kwargs=dict(),
        note="SenecaBot leg naming convention: generic segments (upper/lower/foot) "
             "and functional joint angles (hip absolute; knee, ankle relative), with "
             "fore/hind anatomical equivalents. Proportions from the hindlimb in "
             "senecabot_loco.xml.",
    ),
    FigureSpec(
        category="schematic",
        descriptor="project_overview",
        variant="v1",
        sources=(),                      # schematic: no data source
        plotter="plot_project_overview",
        kwargs=dict(),
        note="General architecture of the SenecaBot DeepMimic project: ovine MOCAP → "
             "retargeted reference trajectory (seneca_loco/simulation/mocap) → imitation "
             "training (Hydra/PPO on the loco-mujoco-seneca fork) → trained policy → "
             "evaluation. The side panel and ★ marks call out the local fork modifications "
             "(MimicReward rootvel/track_root_xy/qvel_w_exp, PPOJax live-logging & "
             "best-checkpoint, viewer record_camera).",
    ),
    FigureSpec(
        category="error",
        descriptor="validation_distances",
        variant="grid",
        sources=(),                      # reads data/Wandb_stats_5_best_try/ directly
        plotter="plot_validation_distance_grid",
        kwargs=dict(),
        note="Validation tracking distances of the trained agent (5_best_try / "
             "faithful-sunset-317) over training: 5 quantities x 3 distance measures "
             "from the framework MetricsHandler, exported from wandb "
             "(data/Wandb_stats_5_best_try/). Rows = measure (Euclidean / DTW / Discrete "
             "Frechet), cols = quantity (Joint{Position,Velocity}, RelSite{Position,"
             "Orientation,Velocity}); each panel on its own y-scale.",
    ),
    FigureSpec(
        category="training",
        descriptor="policy_entropy",
        variant="v1",
        sources=(),                      # reads data/Wandb_stats_5_best_try/entropy.csv
        plotter="plot_entropy_curve",
        kwargs=dict(),
        note="Policy entropy (Train/Entropy) of the trained agent over env-steps — a "
             "training-monitoring signal (exploration / std-collapse early warning), not a "
             "final-evaluation metric. Source: data/Wandb_stats_5_best_try/entropy.csv.",
    ),
    FigureSpec(
        category="training",
        descriptor="episode_return",
        variant="v1",
        sources=(),                      # reads data/Wandb_stats_5_best_try/MeanEpisodeReturn.csv
        plotter="plot_wandb_metric_curve",
        kwargs=dict(fname="MeanEpisodeReturn.csv", ylabel="mean episode return",
                    title="Mean episode return"),
        note="Mean episode return of the trained agent over env-steps (training-progress "
             "monitor). Source: data/Wandb_stats_5_best_try/MeanEpisodeReturn.csv.",
    ),
    FigureSpec(
        category="training",
        descriptor="episode_length",
        variant="v1",
        sources=(),                      # reads data/Wandb_stats_5_best_try/MeanEpisodeLength.csv
        plotter="plot_wandb_metric_curve",
        kwargs=dict(fname="MeanEpisodeLength.csv", ylabel="mean episode length [steps]",
                    title="Mean episode length"),
        note="Mean episode length of the trained agent over env-steps. NB this is a training-"
             "loop metric, NOT a survival measure (it reads low even when E[L] is high). "
             "Source: data/Wandb_stats_5_best_try/MeanEpisodeLength.csv.",
    ),
    FigureSpec(
        category="training",
        descriptor="metric_for_sweep",
        variant="v1",
        sources=(),                      # reads data/Wandb_stats_5_best_try/Metric_for_sweep.csv
        plotter="plot_metric_for_sweep",
        kwargs=dict(),
        note="The validation 'Metric for Sweep' (Euclidean distance on the relative-site "
             "kinematics site_rpos+site_rrotvec+site_rvel — the PPO/Optuna objective) over "
             "env-steps; logged only on validation updates (sparse). "
             "Source: data/Wandb_stats_5_best_try/Metric_for_sweep.csv.",
    ),
]


def get_plotter(name: str) -> Callable:
    """Resolve a plotter function name to the actual callable."""
    fn = getattr(plotters, name, None)
    if fn is None:
        raise AttributeError(f"plotters has no function '{name}'")
    return fn


def check_unique() -> None:
    """Sanity check: every fig_id must be unique."""
    seen = set()
    for spec in FIGURES:
        if spec.fig_id in seen:
            raise ValueError(f"Duplicate fig_id: {spec.fig_id}")
        seen.add(spec.fig_id)
