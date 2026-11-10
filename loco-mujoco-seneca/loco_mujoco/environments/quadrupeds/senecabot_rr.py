from typing import List
from pathlib import Path

import mujoco
import numpy as np
from mujoco import MjSpec

from loco_mujoco.core import ObservationType
from loco_mujoco.core.utils import info_property
from loco_mujoco.environments.quadrupeds.senecabot import SenecaBot


class SenecaBotRR(SenecaBot):
    """[SENECA LOCAL CHANGE] SenecaBot with RR legs (no hip): tibia + metatarsus driven from the torso through the
    two RR bar mechanisms of each leg (geometry_opt/make_rr_xml.py -> senecabot_loco_rr.xml).

    - Observations: root + knee and ankle position / velocity of every leg (the bar joints follow from them).
    - Actions: the 8 crank motors (``<leg>_rr1``, ``<leg>_rr2``); knee and ankle are passive, closed by the loops.
    - Pair it with trajectory_adapted_rr.npz (simulation/mocap/trajectory/trajectory_generation_rr.py) and
      ``joints_for_mimic`` = root + knee + ankle joints.
    """

    XML_PATH = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
        / 'seneca_loco' / 'simulation' / 'assets' / 'xml' / 'senecabot_loco_rr.xml'
    )

    LEGS = ("fr", "fl", "br", "bl")

    @staticmethod
    def _get_observation_specification(spec: MjSpec) -> List[ObservationType]:
        legs = SenecaBotRR.LEGS
        obs = [ObservationType.FreeJointPosNoXY("q_root", xml_name="root")]
        obs += [ObservationType.JointPos(f"q_{l}_{j}", xml_name=f"{l}_{j}_joint") for l in legs for j in ("knee", "ankle")]
        obs += [ObservationType.FreeJointVel("dq_root", xml_name="root")]
        obs += [ObservationType.JointVel(f"dq_{l}_{j}", xml_name=f"{l}_{j}_joint") for l in legs for j in ("knee", "ankle")]
        return obs

    @staticmethod
    def _get_action_specification(spec: MjSpec) -> List[str]:
        return [f"{l}_rr{k}" for l in SenecaBotRR.LEGS for k in (1, 2)]

    @info_property
    def init_qpos(self) -> np.ndarray:
        # closed reference pose of the model (qpos0) with the torso at the height of the adapted reference
        q = mujoco.MjModel.from_xml_path(str(self.XML_PATH)).qpos0.copy()
        q[:3] = (0.0, 0.0, 0.38)
        return q

    @info_property
    def init_qvel(self) -> np.ndarray:
        return np.zeros(mujoco.MjModel.from_xml_path(str(self.XML_PATH)).nv)
