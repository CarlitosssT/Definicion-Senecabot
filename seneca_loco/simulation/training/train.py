import os
import sys
import jax
import jax.numpy as jnp
import wandb
from pathlib import Path
from dataclasses import fields
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.utils.metrics import QuantityContainer
from loco_mujoco.utils import MetricsHandler
from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

from simulation.config import paths

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import traceback

@hydra.main(version_base=None, config_path="./", config_name="conf")
def experiment(config: DictConfig):
    try:
        # Accessing the current sweep number
        result_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        
        # Setup wandb
        wandb.login()
        config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        run = wandb.init(project=config.wandb.project, config=config_dict, dir=result_dir)

        # wandb x-axis: the env-step is passed as wandb's INTERNAL step
        # (wandb.log(..., step=env_step) in ppo_jax.py::_live_log and in the n_seeds>1 loop below),
        # so the default "Step" axis IS the true env-step and renders LIVE while training. We do
        # NOT declare a custom step_metric here: a custom x-axis ("global_step") only materialises
        # on the wandb server once a run finishes, so during training the panel shows an empty
        # "no data on global_step" — which made live monitoring look broken even though the data
        # was logging fine (finished runs had full history; running runs did not). This is safe now
        # because validation is streamed live in monotonic env-step order, so every wandb.log uses
        # a non-decreasing step (the old post-hoc, non-monotonic validation forced the custom axis).

        # get task factory
        factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

        # load trajectory
        traj = Trajectory.load(paths.TRAJ_ADAPTED)
        custom_conf = CustomDatasetConf(traj=traj)

        # create env
        env = factory.make(**config.experiment.env_params, custom_dataset_conf=custom_conf)

        # get initial agent configuration
        agent_conf = PPOJax.init_agent_conf(env,config)

        # setup metric handler (optional)
        mh = MetricsHandler(config, env) if config.experiment.validation.active else None

        # build training function
        train_fn = PPOJax.build_train_fn(env, agent_conf, mh=mh)

        # jit and vmap training function
        train_fn = jax.jit(jax.vmap(train_fn)) if config.experiment.n_seeds > 1 else jax.jit(train_fn)

        # get rng keys and run training
        rngs = [jax.random.PRNGKey(i) for i in range(config.experiment.n_seeds+1)]  # create rngs from seed
        rng, _rng = rngs[0], jnp.squeeze(jnp.vstack(rngs[1:]))
        out = train_fn(_rng)

        import time, pickle
        import numpy as np
    
        final_agent_state = out["agent_state"]
        n_seeds = config.experiment.n_seeds

        # ---- pick the BEST checkpoint by validation return (guards against late collapse) ----
        # The library snapshots train states into train_state_buffer at every validation
        # interval; pick the snapshot with the highest validation return rather than the
        # (possibly collapsed) final state. Falls back to final when unavailable.
        agent_state = final_agent_state
        best_info = "final state (no best-checkpoint selection)"
        if config.experiment.validation.active and "train_state_buffer" in out and n_seeds == 1:
            val_m = out["validation_metrics"]
            train_m = out["training_metrics"]
            buffer = out["train_state_buffer"]
            num_updates = int(config.experiment.num_updates)
            vi = int(config.experiment.validation_interval)
            # update indices where validation ran / a snapshot was stored, in order
            val_step_idx = np.array([k for k in range(num_updates) if (k + 1) % vi == 0])
            m = min(int(np.asarray(buffer.n)), len(val_step_idx))
            if m > 0:
                val_ret = np.asarray(val_m.mean_episode_return, dtype=float)[val_step_idx[:m]]
                if np.all(np.isnan(val_ret)):
                    # every validation eval was NaN (e.g. MjWarp non-determinism) -> keep final
                    best_info = "final state (all validation returns are NaN)"
                else:
                    # nan-safe: NaN validation evals must NOT win argmax (np.argmax would pick a
                    # NaN as the max and save a broken checkpoint).
                    best_j = int(np.nanargmax(val_ret))
                    best_ts = jax.tree.map(lambda x: x[best_j], buffer.train_states)
                    agent_state = PPOJax._agent_state(train_state=best_ts)
                    ts_axis = np.asarray(train_m.max_timestep)
                    n_nan = int(np.isnan(val_ret).sum())
                    best_info = (f"best val-return {val_ret[best_j]:.2f} @ step "
                                 f"{int(ts_axis[val_step_idx[best_j]]):,} (checkpoint {best_j + 1}/{m}, "
                                 f"{n_nan} NaN evals skipped); final val-return {val_ret[m - 1]:.2f} "
                                 f"@ step {int(ts_axis[-1]):,}")
        print(f"[checkpoint] {best_info}")
        run.summary["best_checkpoint_info"] = best_info

        # save BEST as the canonical agent (what eval.py / debug_agent.py default to),
        # and keep the FINAL state separately for resuming / debugging.
        save_path = PPOJax.save_agent(result_dir, agent_conf, agent_state)
        run.config.update({"agent_save_path": save_path})
        with open(Path(result_dir) / "PPOJax_final.pkl", "wb") as f:
            pickle.dump(PPOJax.serialize(agent_conf, final_agent_state), f)

        # ---- metric logging ----
        t_start = time.time()
        if not config.experiment.debug:
            if n_seeds == 1:
                # Nothing to log post-hoc for single-seed runs anymore: ALL metrics
                # (Mean Episode Return/Length + Train/Entropy every update, and the full
                # validation block — Metric for Sweep, Validation Info/*, Validation Measures/* —
                # on each validation update) are streamed LIVE from inside training by the
                # io_callback in ppo_jax.py::_update_step::_live_log, each at wandb internal
                # step = env-step. This is deliberate: the old code dumped validation here in a
                # single burst right before wandb.finish(), which was silently lost whenever the
                # wandb sync dropped in the final seconds of a run (observed on 2026-06-09/16-25-18).
                # Streaming acks each point incrementally during the run. The n_seeds > 1 branch
                # below still logs post-hoc, because the live callback is disabled under vmap.
                pass
            else:
                # n_seeds > 1: the live callback is disabled (vmapped train fn), so log everything
                # post-hoc, at wandb internal step = env-step (mean across seeds) — same axis as the
                # single-seed live path, so the default "Step" axis is the true env-step.
                training_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), out["training_metrics"])
                validation_metrics = jax.tree.map(lambda x: jnp.mean(jnp.atleast_2d(x), axis=0), out["validation_metrics"])

                for i in range(len(training_metrics.mean_episode_return)):
                    gstep = int(training_metrics.max_timestep[i])
                    # one combined row per update so train + validation share the env-step
                    data = {"Mean Episode Return": training_metrics.mean_episode_return[i],
                            "Mean Episode Length": training_metrics.mean_episode_length[i]}

                    if (i+1) % config.experiment.validation_interval == 0 and config.experiment.validation.active:
                        data["Validation Info/Mean Episode Return"] = validation_metrics.mean_episode_return[i]
                        data["Validation Info/Mean Episode Length"] = validation_metrics.mean_episode_length[i]

                        # log all measures
                        for field in fields(validation_metrics):
                            attr = getattr(validation_metrics, field.name)
                            if isinstance(attr, QuantityContainer):
                                measure_name = field.name
                                for field_attr in fields(attr):
                                    attr_name = field_attr.name
                                    attr_value = getattr(attr, attr_name)
                                    if attr_value.size > 0:
                                        data[f"Validation Measures/{measure_name}/{attr_name}"] = attr_value[i]

                        # metric used for wandb sweep (optional)
                        ed = validation_metrics.euclidean_distance
                        data["Metric for Sweep"] = ed.site_rpos[i] + ed.site_rrotvec[i] + ed.site_rvel[i]

                    run.log(data, step=gstep)

        print(f"Time taken to log metrics: {time.time() - t_start}s")

        # run the environment with the trained (best) agent to record video.
        # Wrapped so a render/encode/upload failure cannot skip wandb.finish() below: finish()
        # is what flushes the live history and (re)attempts the media upload, so it must always
        # run. The video file is also written to disk under the run dir, so even if the upload is
        # dropped by a flaky end-of-run sync it can be recovered with `wandb sync <run-dir>`.
        try:
            PPOJax.play_policy(env, agent_conf, agent_state, deterministic=True, n_steps=200, n_envs=20, record=True,
                               train_state_seed=0)
            video_file = env.video_file_path
            run.log({"Agent Video": wandb.Video(video_file)})
        except Exception:
            print("[video] recording/logging failed; continuing to wandb.finish()", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

        wandb.finish()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise

if __name__ == "__main__":
    experiment()