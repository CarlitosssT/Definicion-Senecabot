import os
import argparse
from pathlib import Path
from loco_mujoco import TaskFactory
from loco_mujoco.algorithms import PPOJax
from loco_mujoco.trajectory.dataclasses import Trajectory
from loco_mujoco.task_factories.dataset_confs import CustomDatasetConf

from simulation.config import paths

from omegaconf import OmegaConf

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ')

# Set up argument parser
parser = argparse.ArgumentParser(description='Run evaluation with PPOJax.')
parser.add_argument('--path', type=str, required=True, help='Path to the agent pkl file')
parser.add_argument('--use_mujoco', action='store_true', help='Use MuJoCo for evaluation instead of Mjx')
parser.add_argument('--camera', type=str, default=None,
                    help='Name of an XML <camera> to render from (e.g. follow, follow_west). '
                         'Omit for the default free camera (cycle modes with keys when windowed).')
args = parser.parse_args()

# Use the path from command line arguments
path = args.path
agent_conf, agent_state = PPOJax.load_agent(path)
config = agent_conf.config

# get task factory
factory = TaskFactory.get_factory_cls(config.experiment.task_factory.name)

traj = Trajectory.load(paths.TRAJ_ADAPTED)
custom_conf = CustomDatasetConf(traj=traj)


# create env
OmegaConf.set_struct(config, False)  # Allow modifications
config.experiment.env_params["headless"] = False
# Render from a chosen XML camera (locks the viewer to it); None = default free camera.
# Set explicitly so it overrides whatever the saved agent's config carries (older agents
# have no record_camera key at all).
config.experiment.env_params["record_camera"] = args.camera
#config.experiment.env_params["goal_type"] = "GoalTrajMimicv2"   # nicer looking than GoalTrajMimic
env = factory.make(**config.experiment.env_params, custom_dataset_conf=custom_conf)

# Determine which evaluation environment to run
if args.use_mujoco:
    # run eval mujoco
    PPOJax.play_policy_mujoco(env, agent_conf, agent_state, deterministic=False, n_steps=10000, record=True,
                              train_state_seed=0)
else:
    # run eval mjx
    PPOJax.play_policy(env, agent_conf, agent_state, deterministic=False, n_steps=10000, n_envs=1, record=True,
                       train_state_seed=0)
