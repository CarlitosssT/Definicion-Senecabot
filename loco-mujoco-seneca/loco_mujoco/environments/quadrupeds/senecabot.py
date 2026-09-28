from typing import List, Tuple
import numpy as np
import mujoco
from mujoco import MjSpec
from pathlib import Path

from loco_mujoco.core import ObservationType
from loco_mujoco.environments.quadrupeds.base_robot_quadruped import BaseRobotQuadruped
from loco_mujoco.core.utils import info_property


class SenecaBot(BaseRobotQuadruped):

    mjx_enabled = True

    XML_PATH = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / 'seneca_loco'
        / 'simulation'
        / 'assets'
        / 'xml'
        / 'senecabot_loco.xml'
    )

    def __init__(self, spec=None, camera_params=None,
                 observation_spec=None, actuation_spec=None, **kwargs):

        if spec is None:
            spec = self.XML_PATH.resolve()

        spec = mujoco.MjSpec.from_file(str(spec)) if not isinstance(spec, MjSpec) else spec

        if observation_spec is None:
            observation_spec = self._get_observation_specification(spec)
        else:
            observation_spec = self.parse_observation_spec(observation_spec)

        if actuation_spec is None:
            actuation_spec = self._get_action_specification(spec)

        if "init_state_type" not in kwargs.keys():
            kwargs["init_state_type"] = "DefaultInitialStateHandler"
            kwargs["init_state_params"] = dict(
                qpos_init=self.init_qpos,
                qvel_init=self.init_qvel
            )

        if camera_params is None:
            camera_params = dict(
                follow=dict(distance=3.5, elevation=-20.0, azimuth=90.0)
            )

        super().__init__(
            spec=spec,
            actuation_spec=actuation_spec,
            observation_spec=observation_spec,
            camera_params=camera_params,
            **kwargs
        )

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        # ACTUALIZACIÓN: xml_name ahora apunta a los joints con sufijo "_joint"
        observation_spec = [
            ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

            ObservationType.JointPos("q_fr_hip",   xml_name="fr_hip_joint"),
            ObservationType.JointPos("q_fr_knee",  xml_name="fr_knee_joint"),
            ObservationType.JointPos("q_fr_ankle", xml_name="fr_ankle_joint"),

            ObservationType.JointPos("q_fl_hip",   xml_name="fl_hip_joint"),
            ObservationType.JointPos("q_fl_knee",  xml_name="fl_knee_joint"),
            ObservationType.JointPos("q_fl_ankle", xml_name="fl_ankle_joint"),

            ObservationType.JointPos("q_br_hip",   xml_name="br_hip_joint"),
            ObservationType.JointPos("q_br_knee",  xml_name="br_knee_joint"),
            ObservationType.JointPos("q_br_ankle", xml_name="br_ankle_joint"),

            ObservationType.JointPos("q_bl_hip",   xml_name="bl_hip_joint"),
            ObservationType.JointPos("q_bl_knee",  xml_name="bl_knee_joint"),
            ObservationType.JointPos("q_bl_ankle", xml_name="bl_ankle_joint"),

            ObservationType.FreeJointVel("dq_root", xml_name="root"),

            ObservationType.JointVel("dq_fr_hip",   xml_name="fr_hip_joint"),
            ObservationType.JointVel("dq_fr_knee",  xml_name="fr_knee_joint"),
            ObservationType.JointVel("dq_fr_ankle", xml_name="fr_ankle_joint"),

            ObservationType.JointVel("dq_fl_hip",   xml_name="fl_hip_joint"),
            ObservationType.JointVel("dq_fl_knee",  xml_name="fl_knee_joint"),
            ObservationType.JointVel("dq_fl_ankle", xml_name="fl_ankle_joint"),

            ObservationType.JointVel("dq_br_hip",   xml_name="br_hip_joint"),
            ObservationType.JointVel("dq_br_knee",  xml_name="br_knee_joint"),
            ObservationType.JointVel("dq_br_ankle", xml_name="br_ankle_joint"),

            ObservationType.JointVel("dq_bl_hip",   xml_name="bl_hip_joint"),
            ObservationType.JointVel("dq_bl_knee",  xml_name="bl_knee_joint"),
            ObservationType.JointVel("dq_bl_ankle", xml_name="bl_ankle_joint"),
        ]
        return observation_spec

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        # ACTUALIZACIÓN: Sin sufijos "_act", empareja exactamente con los motores del XML
        return [
            "fr_hip", "fr_knee", "fr_ankle",
            "fl_hip", "fl_knee", "fl_ankle",
            "br_hip", "br_knee", "br_ankle",
            "bl_hip", "bl_knee", "bl_ankle",
        ]

    @info_property
    def grf_size(self) -> int:
        return 12

    @info_property
    def upper_body_xml_name(self) -> str:
        return self.root_body_name

    @info_property
    def root_body_name(self) -> str:
        return "base"

    @info_property
    def root_free_joint_xml_name(self) -> str:
        return "root"

    @info_property
    def root_height_healthy_range(self) -> Tuple[float, float]:
        # Altura en reposo ≈ 0.6 - 0.04 (mitad torso) - F_HIP_LEN - F_KNEE_LEN - F_ANKLE_LEN
        # ≈ 0.6 - 0.04 - 0.121 - 0.111 - 0.1534 ≈ 0.174 m hasta el suelo
        # Rango saludable: no caer por debajo de 0.3 m ni superar 0.55 m
        return (0.3, 0.55)

    @info_property
    def foot_geom_names(self) -> List[str]:
        # Estos se mantienen igual, apuntan al 'name' del <geom> de los pies
        return ["fr_foot", "fl_foot", "br_foot", "bl_foot"]

    @info_property
    def init_qpos(self) -> np.ndarray:
        # [x, y, z,  quat_w, quat_x, quat_y, quat_z,  12 joints en radianes]
        # Joints en punto medio de sus rangos (en grados), convertidos a radianes
        f_hip   = np.deg2rad(82.5)
        f_knee  = np.deg2rad(110)
        f_ankle = np.deg2rad(27.5)

        b_hip   = np.deg2rad(-60)
        b_knee  = np.deg2rad(-88)
        b_ankle = np.deg2rad(-28)

        return np.array([
            0.0, 0.0, 0.4,        # posición xyz del torso
            1.0, 0.0, 0.0, 0.0,   # quaternion identidad (sin rotación)
            f_hip, f_knee, f_ankle,      # fr
            f_hip, f_knee, f_ankle,      # fl
            b_hip, b_knee, b_ankle,      # br
            b_hip, b_knee, b_ankle,      # bl
        ])

    @info_property
    def init_qvel(self) -> np.ndarray:
        # 6 DOF raíz + 12 joints = 18
        return np.zeros(18)
    
    @info_property
    def sites_for_mimic(self) -> list[str]:
        # Estos coinciden perfectamente con los nombres de los <site> en el XML
        return [
            # Root Joint
            "base_mimic",

            # Hip joint
            "fr_foot_mimic",
            "fl_foot_mimic",
            "br_foot_mimic",
            "bl_foot_mimic",
            
            # knee joint
            "fr_shank_mimic",
            "fl_shank_mimic",
            "br_shank_mimic",
            "bl_shank_mimic",

            # ankle joint
            "fr_thigh_mimic",
            "fl_thigh_mimic",
            "br_thigh_mimic",
            "bl_thigh_mimic"
        ]