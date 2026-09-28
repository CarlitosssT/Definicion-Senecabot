import mujoco.mjx
import jax
import numpy as np
from mujoco import MjSpec
import jax.numpy as jnp
import pytest
from copy import deepcopy

from loco_mujoco.core.observations import ObservationType
from loco_mujoco.environments import LocoEnv
from loco_mujoco.core.utils.math import quat_scalarfirst2scalarlast
from scipy.spatial.transform import Rotation as np_R
from jax.scipy.spatial.transform import Rotation as jnp_R
from loco_mujoco.core.utils.math import calculate_relative_site_quatities


from test_conf import DummyHumamoidEnv


# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")

_OBS_ATOL = 1e-6
_OBS_RTOL = 1e-6

DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

OBSERVATION_SPACE = [
    {"obs_name": "name_obs9", "type": "JointVel", "xml_name": "right_hip_y"},
    {"obs_name": "name_obs10", "type": "SiteRot", "xml_name": "left_thigh_site"},
    {"obs_name": "name_obs3", "type": "BodyVel", "xml_name": "left_thigh"},
    {"obs_name": "name_obs4", "type": "BodyVel", "xml_name": "right_shin"},
    {"obs_name": "name_obs5", "type": "BodyVel", "xml_name": "pelvis"},
    {"obs_name": "name_obs7", "type": "JointVel", "xml_name": "left_hip_y"},
    {"obs_name": "name_obs6", "type": "JointPos", "xml_name": "left_hip_y"},
    {"obs_name": "name_obs8", "type": "JointPos", "xml_name": "right_hip_y"},
]


# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_BodyPos(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    body_name1 = "left_shin"
    body_name2 = "right_thigh"

    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "BodyPos", "xml_name": body_name1},
        {"obs_name": "name_obs2", "type": "BodyPos", "xml_name": body_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1",
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    # get the body position from data
    model = mjx_env.get_model()
    body_ids = []
    for name in [body_name1, body_name2]:
        body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    if backend == np:
        # get the body position from Mujoco
        data = mjx_env.get_data()
        body_pos1 = np.array(data.xpos[body_ids[0]])
        body_pos2 = np.array(data.xpos[body_ids[1]])
        gt_obs_mujoco = np.concatenate([body_pos1, body_pos2])

        np.testing.assert_allclose(
            obs,
            gt_obs_mujoco,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the body position from Mjx
        body_pos1 = jnp.array(state.data.xpos[body_ids[0]])
        body_pos2 = jnp.array(state.data.xpos[body_ids[1]])
        gt_obs_mjx = jnp.concatenate([body_pos1, body_pos2])

        # check the observation
        np.testing.assert_allclose(
            obs_mjx,
            gt_obs_mjx,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_BodyRot(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    body_name1 = "left_shin"
    body_name2 = "right_thigh"

    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "BodyRot", "xml_name": body_name1},
        {"obs_name": "name_obs2", "type": "BodyRot", "xml_name": body_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1",
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    # get the body xquat from data
    model = mjx_env.get_model()
    body_ids = []
    for name in [body_name1, body_name2]:
        body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    if backend == np:
        # get the body xquat from Mujoco
        data = mjx_env.get_data()
        body_pos1 = np.array(data.xquat[body_ids[0]])
        body_pos2 = np.array(data.xquat[body_ids[1]])
        gt_obs_mujoco = np.concatenate([body_pos1, body_pos2])

        np.testing.assert_allclose(
            obs,
            gt_obs_mujoco,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the body xquat from Mjx
        body_pos1 = jnp.array(state.data.xquat[body_ids[0]])
        body_pos2 = jnp.array(state.data.xquat[body_ids[1]])
        gt_obs_mjx = jnp.concatenate([body_pos1, body_pos2])

        # check the observation
        np.testing.assert_allclose(
            obs_mjx,
            gt_obs_mjx,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_BodyVel(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    body_name1 = "left_shin"
    body_name2 = "right_thigh"

    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "BodyVel", "xml_name": body_name1},
        {"obs_name": "name_obs2", "type": "BodyVel", "xml_name": body_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1", 
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    # get the body cvel from data
    model = mjx_env.get_model()
    body_ids = []
    for name in [body_name1, body_name2]:
        body_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))

    if backend == np:
        # get the body cvel from Mujoco
        data = mjx_env.get_data()
        body_pos1 = np.array(data.cvel[body_ids[0]])
        body_pos2 = np.array(data.cvel[body_ids[1]])
        gt_obs_mujoco = np.concatenate([body_pos1, body_pos2])

        np.testing.assert_allclose(
            obs,
            gt_obs_mujoco,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the body cvel from Mjx
        body_pos1 = jnp.array(state.data.cvel[body_ids[0]])
        body_pos2 = jnp.array(state.data.cvel[body_ids[1]])
        gt_obs_mjx = jnp.concatenate([body_pos1, body_pos2])

        # check the observation
        np.testing.assert_allclose(
            obs_mjx,
            gt_obs_mjx,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_FreeJointPos(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name = "root"

    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "FreeJointPos", "xml_name": joint_name},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed

    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    # get the joint pos from data
    model = mjx_env.get_model()
    joint_id = []
    joint_id.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
    if backend == np:
        # get the joint pos from Mujoco
        data = mjx_env.get_data()
        gt_joint_pos = np.array(data.qpos[joint_id[0] : joint_id[0] + 7])

        np.testing.assert_allclose(
            obs,
            gt_joint_pos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the joint pos from Mjx
        gt_joint_pos = jnp.array(state.data.qpos[joint_id[0] : joint_id[0] + 7])

        # check the observation
        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_pos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_EntryFromFreeJointPos(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name = "root"
    entry_index = 3
    # specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "EntryFromFreeJointPos",
            "xml_name": joint_name,
            "entry_index": entry_index,
        },
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the joint position from data
    model = mjx_env.get_model()
    joint_ids = []
    joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))

    if backend == np:
        # get the joint position from Mujoco
        data = mjx_env.get_data()
        gt_joint_qpos = np.array(data.qpos[joint_ids[0] + entry_index])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qpos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the joint position from Mjx
        gt_joint_qpos = jnp.array(state.data.qpos[joint_ids[0] + entry_index])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qpos,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_FreeJointPosNoXY(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name = "root"
    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "FreeJointPosNoXY", "xml_name": joint_name},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the body position from data
    model = mjx_env.get_model()
    joint_ids = []
    joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))

    if backend == np:
        # get the joint position from Mujoco
        data = mjx_env.get_data()
        gt_joint_qpos = np.array(data.qpos[joint_ids[0] + 2 : joint_ids[0] + 7])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qpos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the joint position from Mjx
        gt_joint_qpos = jnp.array(state.data.qpos[joint_ids[0] + 2 : joint_ids[0] + 7])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qpos,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_JointPos(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name1 = "right_knee"
    joint_name2 = "left_knee"
    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "JointPos", "xml_name": joint_name1},
        {"obs_name": "name_obs2", "type": "JointPos", "xml_name": joint_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1",
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the body position from data
    model = mjx_env.get_model()
    joint_ids = []
    for name in [joint_name1, joint_name2]:
        joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))

    if backend == np:
        # get the joint position from Mujoco
        data = mjx_env.get_data()
        joint_qpos1 = np.array([data.qpos[joint_ids[0]]])
        joint_qpos2 = np.array([data.qpos[joint_ids[1]]])
        gt_joint_qpos = np.concatenate([joint_qpos1, joint_qpos2])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qpos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the joint position from Mjx
        joint_qpos1 = jnp.array([state.data.qpos[joint_ids[0]]])
        joint_qpos2 = jnp.array([state.data.qpos[joint_ids[1]]])
        gt_joint_qpos = jnp.concatenate([joint_qpos1, joint_qpos2])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qpos,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_JointPosArray(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name1 = "right_knee"
    joint_name2 = "left_knee"
    # specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "JointPosArray",
            "xml_names": [joint_name1, joint_name2],
        },
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the joint position from data
    model = mjx_env.get_model()
    joint_ids = []
    for name in [joint_name1, joint_name2]:
        joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))

    if backend == np:
        # get the joint position from Mujoco
        data = mjx_env.get_data()
        joint_qpos1 = np.array([data.qpos[joint_ids[0]]])
        joint_qpos2 = np.array([data.qpos[joint_ids[1]]])
        gt_joint_qpos = np.concatenate([joint_qpos1, joint_qpos2])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qpos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the joint position from Mjx
        joint_qpos1 = jnp.array([state.data.qpos[joint_ids[0]]])
        joint_qpos2 = jnp.array([state.data.qpos[joint_ids[1]]])
        gt_joint_qpos = jnp.concatenate([joint_qpos1, joint_qpos2])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qpos,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_FreeJointVel(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name = "root"
    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "FreeJointVel", "xml_name": joint_name},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the body position from data
    model = mjx_env.get_model()
    joint_ids = []
    joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))

    if backend == np:
        # get qvel from Mujoco
        data = mjx_env.get_data()
        gt_joint_qvel = np.array(data.qvel[joint_ids[0] : joint_ids[0] + 6])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qvel,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get qvel from Mjx
        gt_joint_qvel = jnp.array(state.data.qvel[joint_ids[0] : joint_ids[0] + 6])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qvel,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_EntryFromFreeJointVel(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name = "root"
    entry_index = 2
    # specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "EntryFromFreeJointVel",
            "xml_name": joint_name,
            "entry_index": entry_index,
        },
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    model = mjx_env.get_model()
    joint_ids = []
    joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))

    if backend == np:
        # get qvel from Mujoco
        data = mjx_env.get_data()
        gt_joint_qvel = np.array(data.qvel[joint_ids[0] + entry_index])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qvel,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get qvel from Mjx
        gt_joint_qvel = jnp.array(state.data.qvel[joint_ids[0]])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qvel,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_JointVel(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name1 = "right_knee"
    joint_name2 = "left_knee"
    # specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "JointVel",
            "xml_name": joint_name1,
        },
        {
            "obs_name": "name_obs2",
            "type": "JointVel",
            "xml_name": joint_name2,
        },
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1", 
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the joint position from data
    model = mjx_env.get_model()
    joint_ids = []
    for name in [joint_name1, joint_name2]:
        joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))

    if backend == np:
        # get joint qvel from Mujoco
        data = mjx_env.get_data()
        joint_qvel1 = np.array([data.qvel[joint_ids[0]]])
        joint_qvel2 = np.array([data.qvel[joint_ids[1]]])
        gt_joint_qvel = np.concatenate([joint_qvel1, joint_qvel2])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qvel,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the joint position from Mjx
        joint_qvel1 = jnp.array([state.data.qvel[joint_ids[0]]])
        joint_qvel2 = jnp.array([state.data.qvel[joint_ids[1]]])
        gt_joint_qvel = jnp.concatenate([joint_qvel1, joint_qvel2])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qvel,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_JointVelArray(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name1 = "right_knee"
    joint_name2 = "left_knee"
    # specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "JointVelArray",
            "xml_names": [joint_name1, joint_name2],
        },
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    model = mjx_env.get_model()
    joint_ids = []
    for name in [joint_name1, joint_name2]:
        joint_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))

    if backend == np:
        data = mjx_env.get_data()
        joint_qvel1 = np.array([data.qvel[joint_ids[0]]])
        joint_qvel2 = np.array([data.qvel[joint_ids[1]]])
        gt_joint_qvel = np.concatenate([joint_qvel1, joint_qvel2])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_joint_qvel,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        joint_qvel1 = jnp.array([state.data.qvel[joint_ids[0]]])
        joint_qvel2 = jnp.array([state.data.qvel[joint_ids[1]]])
        gt_joint_qvel = jnp.concatenate([joint_qvel1, joint_qvel2])

        np.testing.assert_allclose(
            obs_mjx,
            gt_joint_qvel,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_SitePos(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    site_name1 = "torso_site"
    site_name2 = "right_foot_site"
    # specify the observation space
    observation_spec = [
        {"obs_name": "name_obs1", "type": "SitePos", "xml_name": site_name1},
        {"obs_name": "name_obs2", "type": "SitePos", "xml_name": site_name2},
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1", 
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]
    # get the site position from data
    model = mjx_env.get_model()
    site_ids = []
    for name in [site_name1, site_name2]:
        site_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))

    if backend == np:
        # get the site position from Mujoco
        data = mjx_env.get_data()
        site_xpos1 = np.array(data.site_xpos[site_ids[0]])
        site_xpos2 = np.array(data.site_xpos[site_ids[1]])
        gt_site_xpos = np.concatenate([site_xpos1, site_xpos2])

        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_site_xpos,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        # get the site position from Mjx
        site_xpos1 = jnp.array(state.data.site_xpos[site_ids[0]])
        site_xpos2 = jnp.array(state.data.site_xpos[site_ids[1]])
        gt_site_xpos = jnp.concatenate([site_xpos1, site_xpos2])

        np.testing.assert_allclose(
            obs_mjx,
            gt_site_xpos,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_SiteRot(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    site_name1 = "torso_site"
    site_name2 = "right_foot_site"
    # specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "SiteRot",
            "xml_name": site_name1,
        },
        {
            "obs_name": "name_obs2",
            "type": "SiteRot",
            "xml_name": site_name2,
        },
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # specify the name of the actuators of the xml
    action_spec = ["abdomen_y"]  # --> use more motors if needed
    # define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1", 
                                                                                    "name_obs2"]])
    if backend == np:
        # reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    model = mjx_env.get_model()
    site_ids = []
    for name in [site_name1, site_name2]:
        site_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))

    if backend == np:
        data = mjx_env.get_data()
        site_xmat1 = data.site_xmat[site_ids[0]]
        site_xmat2 = data.site_xmat[site_ids[1]]
        gt_site_xmat = np.concatenate([site_xmat1.ravel(), site_xmat2.ravel()])
        # check the observation
        np.testing.assert_allclose(
            obs,
            gt_site_xmat,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        site_xmat1 = state.data.site_xmat[site_ids[0]]
        site_xmat2 = state.data.site_xmat[site_ids[1]]
        gt_site_xmat = jnp.concatenate(
            [site_xmat1.ravel(), site_xmat2.ravel()]
        )  # reshape to 1D

        np.testing.assert_allclose(
            obs_mjx,
            gt_site_xmat,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_ProjectedGravityVector(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)
    joint_name = "root"

    # Specify the observation space
    observation_spec = [
        {
            "obs_name": "name_obs1",
            "type": "ProjectedGravityVector",
            "xml_name": joint_name,
        }
    ]

    observation_spec.extend(OBSERVATION_SPACE)

    # Specify the name of the actuators of the XML
    action_spec = ["abdomen_y"]  # Use more motors if needed

    # Define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # Reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # Reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    model = mjx_env.get_model()
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if backend == np:
        data = mjx_env.get_data()
        quat = data.qpos[joint_id + 3 : joint_id + 7]
        rots = np_R.from_quat(quat_scalarfirst2scalarlast(quat))
        gt_proj_grav = rots.inv().apply(np.array([0, 0, -1]))

        # Check the observation
        np.testing.assert_allclose(
            obs,
            gt_proj_grav,
            err_msg="Mismatch between Mujoco observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )
    else:
        quat = state.data.qpos[joint_id + 3 : joint_id + 7]
        rots = jnp_R.from_quat(quat_scalarfirst2scalarlast(quat))
        gt_proj_grav = rots.inv().apply(jnp.array([0, 0, -1]))

        # Check the observation
        np.testing.assert_allclose(
            obs_mjx,
            gt_proj_grav,
            err_msg="Mismatch between Mjx observation and ground truth",
            atol=_OBS_ATOL,
            rtol=_OBS_RTOL,
        )


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_LastAction(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Specify the observation space
    observation_spec = deepcopy(OBSERVATION_SPACE)
    observation_spec.append({"obs_name": "name_obs1", "type": "LastAction"})

    # Specify the name of the actuators of the XML
    action_spec = ["abdomen_y"]  # Use more motors if needed

    # Define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # Reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]

        assert backend.all(obs == 0), "Initial last action should be zero"
    else:
        # Reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

        assert backend.all(obs_mjx == 0), "Initial last action should be zero"

    # arbitray action to be applied
    action = backend.array([0.1])

    # Take a step in the environment with the chosen action
    if backend == np:
        next_obs, _, _, _, _ = mjx_env.step(action)
        next_obs = next_obs[obs_ind]
    else:
        next_state = mjx_env.mjx_step(state, action)
        next_obs = next_state.observation
        next_obs = next_obs[obs_ind]

    # Step 2: Check if LastAction in new observation matches the applied action
    assert backend.all(
        next_obs == action
    ), "LastAction should match the previous action"


@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_ModelInfo(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Specify the observation space
    observation_spec = deepcopy(OBSERVATION_SPACE)
    model_attributes = ["geom_friction", "geom_solref", "body_ipos",
                        "body_mass", "dof_frictionloss", "dof_damping",
                        "dof_armature"]
    observation_spec.insert(0, {"obs_name": "name_obs1", "type": "ModelInfo",
                             "model_attributes": model_attributes})

    # Specify the name of the actuators of the XML
    action_spec = ["abdomen_y"]

    # Define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # Reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # Reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    model = mjx_env.get_model()

    expected_obs = []
    for attr in model_attributes:
        expected_obs.append(backend.ravel(getattr(model, attr)))

    expected_obs = backend.concatenate(expected_obs)

    if backend == np:
        np.testing.assert_allclose(
            obs, expected_obs, err_msg="Mismatch between Mujoco observation and expected values",
                                                                                        atol=_OBS_ATOL, rtol=_OBS_RTOL
        )
    else:
        np.testing.assert_allclose(
            obs_mjx, expected_obs, err_msg="Mismatch between Mjx observation and expected values",
                                                                                        atol=_OBS_ATOL, rtol=_OBS_RTOL
        )



@pytest.mark.parametrize("backend", ["jax", "numpy"])
def test_RelativeSiteQuantaties(backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    # Specify the observation space
    observation_spec = deepcopy(OBSERVATION_SPACE)
    observation_spec.append({"obs_name": "name_obs1", "type": "RelativeSiteQuantaties"})

    # Specify the name of the actuators of the XML
    action_spec = ["abdomen_y"]  # Use more motors if needed

    # Define a simple Mjx environment
    mjx_env = DummyHumamoidEnv(
        enable_mjx=True,
        actuation_spec=action_spec,
        observation_spec=observation_spec,
        **DEFAULTS,
    )

    backend = jnp if backend == "jax" else np
    # index the correct observation dims
    obs_ind = backend.concatenate([mjx_env.obs_container[name].obs_ind for name in ["name_obs1"]])
    if backend == np:
        # Reset the environment in Mujoco
        obs = mjx_env.reset(key)
        obs = obs[obs_ind]
    else:
        # Reset the environment in Mjx
        state = mjx_env.mjx_reset(key)
        obs_mjx = state.observation
        obs_mjx = obs_mjx[obs_ind]

    model = mjx_env.get_model()
    data = mjx_env.get_data()
    
    # Retrieve relative site IDs and compute expected observations
    site_names = mjx_env.obs_container["name_obs1"].site_names
    rel_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in site_names]
    rel_body_ids = model.site_bodyid[rel_site_ids]
    body_rootid = model.body_rootid
    
    site_rpos, site_rangles, site_rvel = calculate_relative_site_quatities(
        data, np.array(rel_site_ids), rel_body_ids, body_rootid, backend
    )

    expected_obs = backend.concatenate([
        backend.ravel(site_rpos),
        backend.ravel(site_rangles),
        backend.ravel(site_rvel)
    ])

    if backend == np:
        np.testing.assert_allclose(
            obs, expected_obs, err_msg="Mismatch between Mujoco observation and expected values", 
                                                                                        atol=_OBS_ATOL, rtol=_OBS_RTOL
        )
    else:
        np.testing.assert_allclose(
            obs_mjx, expected_obs, err_msg="Mismatch between Mjx observation and expected values", 
                                                                                        atol=_OBS_ATOL, rtol=_OBS_RTOL
        )
