from huggingface_hub import list_repo_files

from loco_mujoco import LocoEnv


def get_numpy_env_names():
    """Generates a list of numpy env_names for testing."""
    np_env_list = []
    for env_name in LocoEnv.registered_envs.keys():
        if "Mjx" not in env_name:
            np_env_list.append(env_name)
    return np_env_list


def get_jax_env_names():
    """Generates a list of jax env_names for testing."""
    jax_env_list = []
    for env_name in LocoEnv.registered_envs.keys():
        if "Mjx" in env_name:
            jax_env_list.append(env_name)
    return jax_env_list


def get_numpy_env_names_default_dataset_conf():
    """Generates a list of numpy env_names for testing (only for envs that have uploaded datasets). """
    repo_id = "robfiras/loco-mujoco-datasets"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    directory = "DefaultDatasets/mocap"  # Adjust to your dataset structure
    np_env_list = set([f.split("/")[-2] for f in all_files if f.startswith(directory)])
    return np_env_list


def get_numpy_env_names_lafan1_dataset_conf():
    """Generates a list of numpy env_names for testing (only for envs that have uploaded datasets). """
    repo_id = "robfiras/loco-mujoco-datasets"
    all_files = list_repo_files(repo_id, repo_type="dataset")
    directory = "Lafan1/mocap"  # Adjust to your dataset structure
    np_env_list = set([f.split("/")[-2] for f in all_files if f.startswith(directory)])
    return np_env_list
