"""Dedicated Xsens T-pose calibration IK for G1."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy.optimize import least_squares  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from holosoma_retargeting.config_types.data_type import XSENS_DEMO_JOINTS
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XsensHdf5Tpose,
    load_xsens_hdf5_tpose,
    resolve_xsens_hdf5_path,
)


CALIBRATION_POSITION_MAPPING = {
    "Pelvis": "pelvis_contour_link",
    "L5": "torso_link",
    "Head": "head_link",
    "Left Shoulder": "left_shoulder_pitch_link",
    "Right Shoulder": "right_shoulder_pitch_link",
    "Left Upper Arm": "left_shoulder_roll_link",
    "Right Upper Arm": "right_shoulder_roll_link",
    "Left Forearm": "left_elbow_link",
    "Right Forearm": "right_elbow_link",
    "Left Hand": "left_rubber_hand_link",
    "Right Hand": "right_rubber_hand_link",
    "Left Upper Leg": "left_hip_pitch_link",
    "Right Upper Leg": "right_hip_pitch_link",
    "Left Lower Leg": "left_knee_link",
    "Right Lower Leg": "right_knee_link",
    "Left Foot": "left_ankle_intermediate_1_link",
    "Right Foot": "right_ankle_intermediate_1_link",
    "Left Toe": "left_ankle_roll_sphere_5_link",
    "Right Toe": "right_ankle_roll_sphere_5_link",
}

CANDIDATE_ORIENTATION_MAPPING = {
    "L5": "torso_link",
    "Head": "head_link",
    "Left Foot": "left_ankle_intermediate_1_link",
    "Right Foot": "right_ankle_intermediate_1_link",
    "Left Hand": "left_rubber_hand_link",
    "Right Hand": "right_rubber_hand_link",
}

LIMB_AXIS_TARGETS = (
    ("Left Upper Arm", "Left Forearm", "left_shoulder_roll_link", "left_elbow_link", "left_upper_arm"),
    ("Right Upper Arm", "Right Forearm", "right_shoulder_roll_link", "right_elbow_link", "right_upper_arm"),
    ("Left Forearm", "Left Hand", "left_elbow_link", "left_rubber_hand_link", "left_forearm"),
    ("Right Forearm", "Right Hand", "right_elbow_link", "right_rubber_hand_link", "right_forearm"),
    ("Left Upper Leg", "Left Lower Leg", "left_hip_pitch_link", "left_knee_link", "left_thigh"),
    ("Right Upper Leg", "Right Lower Leg", "right_hip_pitch_link", "right_knee_link", "right_thigh"),
    ("Left Lower Leg", "Left Foot", "left_knee_link", "left_ankle_intermediate_1_link", "left_shank"),
    ("Right Lower Leg", "Right Foot", "right_knee_link", "right_ankle_intermediate_1_link", "right_shank"),
    ("Left Foot", "Left Toe", "left_ankle_intermediate_1_link", "left_ankle_roll_sphere_5_link", "left_foot"),
    ("Right Foot", "Right Toe", "right_ankle_intermediate_1_link", "right_ankle_roll_sphere_5_link", "right_foot"),
    ("Pelvis", "L5", "pelvis_contour_link", "torso_link", "torso"),
    ("L5", "Head", "torso_link", "head_link", "head"),
)

SYMMETRY_LINK_PAIRS = (
    ("left_shoulder_roll_link", "right_shoulder_roll_link", "shoulder"),
    ("left_elbow_link", "right_elbow_link", "elbow"),
    ("left_rubber_hand_link", "right_rubber_hand_link", "hand"),
    ("left_hip_pitch_link", "right_hip_pitch_link", "hip"),
    ("left_knee_link", "right_knee_link", "knee"),
    ("left_ankle_intermediate_1_link", "right_ankle_intermediate_1_link", "ankle"),
    ("left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link", "toe"),
)

ARM_COLLINEARITY_TARGETS = (
    ("left_shoulder_roll_link", "left_elbow_link", "left_rubber_hand_link", "left_arm"),
    ("right_shoulder_roll_link", "right_elbow_link", "right_rubber_hand_link", "right_arm"),
)

HAND_VERTICAL_TARGETS = (
    ("left_rubber_hand_link", np.array([0.0, 0.0, 1.0]), "left_hand_vertical"),
    ("right_rubber_hand_link", np.array([0.0, 0.0, 1.0]), "right_hand_vertical"),
)

MIRRORED_JOINT_PAIRS = (
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1.0),
    ("left_elbow_joint", "right_elbow_joint", 1.0),
    ("left_wrist_roll_joint", "right_wrist_roll_joint", -1.0),
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1.0),
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1.0),
)


@dataclass(frozen=True)
class XsensTposeCalibrationConfig:
    """Configuration for dedicated Xsens T-pose calibration IK."""

    robot_type: str = "g1"
    variant: str = "Tpose"
    robot_urdf_file: str | None = None
    default_human_height: float = 1.78
    fps: int = 30
    max_nfev: int = 400
    verbose: int = 1
    base_xy_radius_m: float = 0.5
    base_z_margin_m: float = 0.35
    position_weight: float = 3.0
    head_position_weight: float = 5.0
    axis_weight: float = 4.0
    symmetry_weight: float = 7.0
    straightness_weight: float = 100.0
    torso_upright_weight: float = 5.0
    foot_flat_weight: float = 3.0
    nominal_weight: float = 0.5
    wrist_neutral_weight: float = 2000.0
    arm_collinearity_weight: float = 5000.0
    hand_vertical_weight: float = 500.0
    mirror_joint_weight: float = 12.0
    active_position_error_threshold_m: float = 0.15
    head_position_error_threshold_m: float = 0.10
    head_orientation_offset_threshold_deg: float = 140.0
    candidate_orientation_mapping: dict[str, str] = field(
        default_factory=lambda: dict(CANDIDATE_ORIENTATION_MAPPING)
    )


@dataclass(frozen=True)
class XsensTposeCalibrationResult:
    """Result of Xsens T-pose calibration IK."""

    qpos: np.ndarray
    xsens_tpose_positions_m: np.ndarray
    xsens_tpose_quaternions_wijk: np.ndarray
    candidate_orientation_mapping_names: list[str]
    candidate_robot_link_names: list[str]
    active_orientation_mapping_names: list[str]
    robot_link_names: list[str]
    orientation_offsets_wijk: np.ndarray
    position_error_names: list[str]
    position_errors_m: np.ndarray
    axis_error_names: list[str]
    axis_errors_deg: np.ndarray
    head_candidate_status: str
    solver_cost: float
    solver_success: bool
    variant: str
    scale_factor: float


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_model_path(path: str | Path) -> Path:
    model_path = Path(path)
    if model_path.is_file():
        return model_path
    package_relative = _package_root() / model_path
    if package_relative.is_file():
        return package_relative
    raise FileNotFoundError(f"Robot model not found: {path}")


def _normalize(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        fallback = np.zeros_like(vector)
    return np.asarray(fallback, dtype=float)


def _segment_index(name: str) -> int:
    return XSENS_DEMO_JOINTS.index(name)


def _quat_wijk_to_matrix(quaternions_wijk: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions_wijk, dtype=float)
    xyzw = np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)
    return Rotation.from_quat(xyzw).as_matrix()


def _matrix_to_quat_wijk(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=float)


def compute_canonical_axes(positions_m: np.ndarray) -> np.ndarray:
    """Compute right-handed `[forward, left, up]` axes from Xsens T-pose geometry."""
    world_up = np.array([0.0, 0.0, 1.0])
    left = positions_m[_segment_index("Left Upper Leg")] - positions_m[_segment_index("Right Upper Leg")]
    left[2] = 0.0

    left_foot = positions_m[_segment_index("Left Toe")] - positions_m[_segment_index("Left Foot")]
    right_foot = positions_m[_segment_index("Right Toe")] - positions_m[_segment_index("Right Foot")]
    forward = 0.5 * (left_foot + right_foot)
    forward[2] = 0.0

    forward = _normalize(forward, np.array([1.0, 0.0, 0.0]))
    left = left - np.dot(left, forward) * forward
    left = _normalize(left, np.array([0.0, 1.0, 0.0]))
    up = _normalize(np.cross(forward, left), world_up)
    if float(np.dot(up, world_up)) < 0.0:
        left = -left
        up = -up
    left = _normalize(np.cross(up, forward), np.array([0.0, 1.0, 0.0]))
    return np.vstack([forward, left, up])


def align_and_scale_tpose_targets(
    positions_m: np.ndarray,
    *,
    scale_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale, ground-align, and pelvis-center Xsens T-pose positions."""
    positions = np.asarray(positions_m, dtype=float).copy() * float(scale_factor)
    foot_indices = [
        _segment_index("Left Foot"),
        _segment_index("Right Foot"),
        _segment_index("Left Toe"),
        _segment_index("Right Toe"),
    ]
    positions[:, 2] -= float(np.min(positions[foot_indices, 2]))
    pelvis_xy = positions[_segment_index("Pelvis"), :2].copy()
    positions[:, :2] -= pelvis_xy
    axes = compute_canonical_axes(positions)
    return positions, axes


def limb_axis_error_deg(
    source_proximal: np.ndarray,
    source_distal: np.ndarray,
    target_proximal: np.ndarray,
    target_distal: np.ndarray,
) -> float:
    source_axis = _normalize(np.asarray(source_distal) - np.asarray(source_proximal), np.array([1.0, 0.0, 0.0]))
    target_axis = _normalize(np.asarray(target_distal) - np.asarray(target_proximal), np.array([1.0, 0.0, 0.0]))
    dot = float(np.clip(np.dot(source_axis, target_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def visual_elbow_angle_deg(shoulder_point: np.ndarray, elbow_point: np.ndarray, hand_point: np.ndarray) -> float:
    """Return the angle between shoulder->elbow and elbow->hand visual segments."""
    upper_axis = _normalize(np.asarray(elbow_point) - np.asarray(shoulder_point), np.array([1.0, 0.0, 0.0]))
    forearm_axis = _normalize(np.asarray(hand_point) - np.asarray(elbow_point), upper_axis)
    dot = float(np.clip(np.dot(upper_axis, forearm_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def axis_alignment_error_deg(axis: np.ndarray, target_axis: np.ndarray) -> float:
    """Return the angle between two axes in degrees."""
    axis = _normalize(np.asarray(axis), np.array([1.0, 0.0, 0.0]))
    target_axis = _normalize(np.asarray(target_axis), np.array([1.0, 0.0, 0.0]))
    dot = float(np.clip(np.dot(axis, target_axis), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def symmetry_residual(left_point: np.ndarray, right_point: np.ndarray) -> np.ndarray:
    """Residual for mirror symmetry across the X-Z sagittal plane."""
    left_point = np.asarray(left_point, dtype=float)
    right_point = np.asarray(right_point, dtype=float)
    return np.array(
        [
            left_point[0] - right_point[0],
            left_point[1] + right_point[1],
            left_point[2] - right_point[2],
        ],
        dtype=float,
    )


class _G1CalibrationProblem:
    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        target_positions: np.ndarray,
        target_axes: np.ndarray,
        config: XsensTposeCalibrationConfig,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.target_positions = target_positions
        self.target_axes = target_axes
        self.config = config
        self.actuated_joint_names = [
            model.joint(i).name for i in range(model.njnt) if model.joint(i).name and model.jnt_qposadr[i] >= 7
        ]
        self.joint_name_to_qadr = {
            model.joint(i).name: int(model.jnt_qposadr[i])
            for i in range(model.njnt)
            if model.joint(i).name and model.jnt_qposadr[i] >= 7
        }
        self.body_name_to_id = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(model.nbody)
        }
        self.geom_name_to_id = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)
        }
        self.nominal_qpos = self._initial_qpos()

    def _joint_qpos_index(self, joint_name: str) -> int:
        return self.joint_name_to_qadr[joint_name]

    def _frame_position(self, frame_name: str) -> np.ndarray:
        if frame_name in self.body_name_to_id:
            return self.data.xpos[self.body_name_to_id[frame_name]].copy()
        if frame_name in self.geom_name_to_id:
            return self.data.geom_xpos[self.geom_name_to_id[frame_name]].copy()
        raise KeyError(f"No MuJoCo body or geom named '{frame_name}'")

    def _frame_rotation(self, frame_name: str) -> np.ndarray:
        if frame_name in self.body_name_to_id:
            return self.data.xmat[self.body_name_to_id[frame_name]].reshape(3, 3).copy()
        if frame_name in self.geom_name_to_id:
            return self.data.geom_xmat[self.geom_name_to_id[frame_name]].reshape(3, 3).copy()
        raise KeyError(f"No MuJoCo body or geom named '{frame_name}'")

    def _initial_qpos(self) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=float)
        yaw = float(np.arctan2(self.target_axes[0, 1], self.target_axes[0, 0]))
        qpos[:3] = self.target_positions[_segment_index("Pelvis")]
        qpos[3:7] = _yaw_quat_wxyz(yaw)
        for joint_name, value in {
            "left_shoulder_roll_joint": 1.3,
            "right_shoulder_roll_joint": -1.3,
            "left_elbow_joint": 0.03,
            "right_elbow_joint": 0.03,
        }.items():
            if joint_name in self.joint_name_to_qadr:
                qpos[self._joint_qpos_index(joint_name)] = value
        return qpos

    def x_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        yaw = Rotation.from_quat([qpos[4], qpos[5], qpos[6], qpos[3]]).as_euler("zyx")[0]
        return np.concatenate([qpos[:3], [yaw], qpos[7:]])

    def qpos_from_x(self, x: np.ndarray) -> np.ndarray:
        qpos = np.zeros(self.model.nq, dtype=float)
        qpos[:3] = x[:3]
        qpos[3:7] = _yaw_quat_wxyz(float(x[3]))
        qpos[7:] = x[4:]
        return qpos

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        q0 = self.nominal_qpos
        lower = np.empty(4 + len(self.actuated_joint_names), dtype=float)
        upper = np.empty_like(lower)
        lower[:2] = q0[:2] - self.config.base_xy_radius_m
        upper[:2] = q0[:2] + self.config.base_xy_radius_m
        lower[2] = max(0.2, q0[2] - self.config.base_z_margin_m)
        upper[2] = q0[2] + self.config.base_z_margin_m
        lower[3] = -np.pi
        upper[3] = np.pi

        joint_lowers = []
        joint_uppers = []
        for joint_name in self.actuated_joint_names:
            qadr = self._joint_qpos_index(joint_name)
            joint_id = self.model.jnt_qposadr.tolist().index(qadr)
            joint_lowers.append(float(self.model.jnt_range[joint_id, 0]))
            joint_uppers.append(float(self.model.jnt_range[joint_id, 1]))
        lower[4:] = joint_lowers
        upper[4:] = joint_uppers

        robot_config = RobotConfig(robot_type=self.config.robot_type, robot_urdf_file=self.config.robot_urdf_file)
        for key, value in robot_config.MANUAL_LB.items():
            qadr = int(key)
            if qadr >= 7:
                lower[4 + qadr - 7] = max(lower[4 + qadr - 7], float(value))
        for key, value in robot_config.MANUAL_UB.items():
            qadr = int(key)
            if qadr >= 7:
                upper[4 + qadr - 7] = min(upper[4 + qadr - 7], float(value))
        return lower, upper

    def set_qpos(self, qpos: np.ndarray) -> None:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        qpos = self.qpos_from_x(x)
        self.set_qpos(qpos)
        residuals: list[np.ndarray] = []

        self._append_position_residuals(residuals)
        self._append_axis_residuals(residuals)
        self._append_arm_collinearity_residuals(residuals)
        self._append_symmetry_residuals(residuals)
        self._append_joint_priors(residuals, qpos)
        self._append_orientation_priors(residuals)
        return np.concatenate(residuals)

    def _append_position_residuals(self, residuals: list[np.ndarray]) -> None:
        for xsens_name, link_name in CALIBRATION_POSITION_MAPPING.items():
            weight = self.config.head_position_weight if xsens_name == "Head" else self.config.position_weight
            target = self.target_positions[_segment_index(xsens_name)]
            current = self._frame_position(link_name)
            residuals.append(np.sqrt(weight) * (current - target))

    def _append_axis_residuals(self, residuals: list[np.ndarray]) -> None:
        for source_a, source_b, link_a, link_b, _name in LIMB_AXIS_TARGETS:
            target_axis = _normalize(
                self.target_positions[_segment_index(source_b)] - self.target_positions[_segment_index(source_a)],
                np.array([1.0, 0.0, 0.0]),
            )
            robot_axis = _normalize(self._frame_position(link_b) - self._frame_position(link_a), target_axis)
            residuals.append(np.sqrt(self.config.axis_weight) * (robot_axis - target_axis))

    def _append_arm_collinearity_residuals(self, residuals: list[np.ndarray]) -> None:
        for shoulder_link, elbow_link, hand_link, _name in ARM_COLLINEARITY_TARGETS:
            upper_axis = _normalize(
                self._frame_position(elbow_link) - self._frame_position(shoulder_link),
                np.array([1.0, 0.0, 0.0]),
            )
            forearm_axis = _normalize(
                self._frame_position(hand_link) - self._frame_position(elbow_link),
                upper_axis,
            )
            residuals.append(np.sqrt(self.config.arm_collinearity_weight) * (forearm_axis - upper_axis))

    def _append_symmetry_residuals(self, residuals: list[np.ndarray]) -> None:
        for left_link, right_link, _name in SYMMETRY_LINK_PAIRS:
            residuals.append(
                np.sqrt(self.config.symmetry_weight)
                * symmetry_residual(self._frame_position(left_link), self._frame_position(right_link))
            )

    def _append_joint_priors(self, residuals: list[np.ndarray], qpos: np.ndarray) -> None:
        straight_joint_names = (
            "left_knee_joint",
            "right_knee_joint",
            "left_elbow_joint",
            "right_elbow_joint",
        )
        for joint_name in straight_joint_names:
            residuals.append(np.array([np.sqrt(self.config.straightness_weight) * qpos[self._joint_qpos_index(joint_name)]]))

        foot_flat_joint_names = (
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        )
        for joint_name in foot_flat_joint_names:
            residuals.append(np.array([np.sqrt(self.config.foot_flat_weight) * qpos[self._joint_qpos_index(joint_name)]]))

        waist_joint_names = (
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
        )
        for joint_name in waist_joint_names:
            residuals.append(np.array([np.sqrt(self.config.nominal_weight) * qpos[self._joint_qpos_index(joint_name)]]))

        wrist_joint_names = (
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        )
        for joint_name in wrist_joint_names:
            residuals.append(
                np.array([np.sqrt(self.config.wrist_neutral_weight) * qpos[self._joint_qpos_index(joint_name)]])
            )

        for left_joint, right_joint, sign in MIRRORED_JOINT_PAIRS:
            left_value = qpos[self._joint_qpos_index(left_joint)]
            right_value = qpos[self._joint_qpos_index(right_joint)]
            residuals.append(np.array([np.sqrt(self.config.mirror_joint_weight) * (left_value - sign * right_value)]))

    def _append_orientation_priors(self, residuals: list[np.ndarray]) -> None:
        up = self.target_axes[2]
        torso_up = self._frame_rotation("torso_link")[:, 2]
        residuals.append(np.sqrt(self.config.torso_upright_weight) * (torso_up - up))

        foot_links = ("left_ankle_roll_link", "right_ankle_roll_link")
        for foot_link in foot_links:
            foot_up = self._frame_rotation(foot_link)[:, 2]
            residuals.append(np.sqrt(self.config.foot_flat_weight) * (foot_up - up))

        for hand_link, local_vertical_axis, _name in HAND_VERTICAL_TARGETS:
            hand_vertical = self._frame_rotation(hand_link) @ local_vertical_axis
            residuals.append(np.sqrt(self.config.hand_vertical_weight) * (hand_vertical - up))


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half_yaw = 0.5 * yaw
    return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=float)


def _robot_urdf_for_config(config: XsensTposeCalibrationConfig) -> Path:
    if config.robot_type != "g1":
        raise ValueError("Xsens T-pose calibration currently supports robot_type='g1' only")
    robot_config = RobotConfig(robot_type=config.robot_type, robot_urdf_file=config.robot_urdf_file)
    return _resolve_model_path(robot_config.ROBOT_URDF_FILE)


def _solve_problem(
    problem: _G1CalibrationProblem,
    config: XsensTposeCalibrationConfig,
) -> tuple[np.ndarray, float, bool]:
    x0 = problem.x_from_qpos(problem.nominal_qpos)
    lower, upper = problem.bounds()
    result = least_squares(
        problem.residuals,
        x0,
        bounds=(lower, upper),
        max_nfev=config.max_nfev,
        verbose=config.verbose,
        x_scale="jac",
    )
    qpos = problem.qpos_from_x(result.x)
    return qpos, float(result.cost), bool(result.success)


def _position_errors(
    problem: _G1CalibrationProblem,
    qpos: np.ndarray,
    mapping: dict[str, str],
) -> dict[str, float]:
    problem.set_qpos(qpos)
    errors = {}
    for xsens_name, link_name in mapping.items():
        target = problem.target_positions[_segment_index(xsens_name)]
        current = problem._frame_position(link_name)
        errors[xsens_name] = float(np.linalg.norm(current - target))
    return errors


def _axis_errors(problem: _G1CalibrationProblem, qpos: np.ndarray) -> dict[str, float]:
    problem.set_qpos(qpos)
    errors = {}
    for source_a, source_b, link_a, link_b, name in LIMB_AXIS_TARGETS:
        errors[name] = limb_axis_error_deg(
            problem.target_positions[_segment_index(source_a)],
            problem.target_positions[_segment_index(source_b)],
            problem._frame_position(link_a),
            problem._frame_position(link_b),
        )
    return errors


def _orientation_offsets(
    problem: _G1CalibrationProblem,
    qpos: np.ndarray,
    quaternions_wijk: np.ndarray,
    mapping: dict[str, str],
) -> dict[str, np.ndarray]:
    problem.set_qpos(qpos)
    xsens_rotations = _quat_wijk_to_matrix(quaternions_wijk)
    offsets = {}
    for xsens_name, link_name in mapping.items():
        xsens_idx = _segment_index(xsens_name)
        robot_rotation = problem._frame_rotation(link_name)
        offset = xsens_rotations[xsens_idx].T @ robot_rotation
        offsets[xsens_name] = _matrix_to_quat_wijk(offset)
    return offsets


def _offset_angle_deg(offset_wijk: np.ndarray) -> float:
    matrix = _quat_wijk_to_matrix(np.asarray(offset_wijk).reshape(1, 4))[0]
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(matrix).as_rotvec())))


def evaluate_head_candidate(
    *,
    head_position_error_m: float,
    head_orientation_offset_wijk: np.ndarray,
    config: XsensTposeCalibrationConfig,
) -> tuple[bool, str]:
    """Return whether `Head -> head_link` passes calibration diagnostics."""
    offset_angle_deg = _offset_angle_deg(head_orientation_offset_wijk)
    accepted = (
        head_position_error_m <= config.head_position_error_threshold_m
        and offset_angle_deg <= config.head_orientation_offset_threshold_deg
    )
    status = "accepted" if accepted else "rejected"
    return (
        accepted,
        (
            f"{status}: position_error_m={head_position_error_m:.4f}, "
            f"offset_angle_deg={offset_angle_deg:.1f}"
        ),
    )


def _active_orientation_mapping(
    *,
    position_errors: dict[str, float],
    offsets: dict[str, np.ndarray],
    config: XsensTposeCalibrationConfig,
) -> tuple[dict[str, str], str]:
    active = {}
    head_status = "not evaluated"
    for xsens_name, link_name in config.candidate_orientation_mapping.items():
        if xsens_name == "Head":
            accepted, head_status = evaluate_head_candidate(
                head_position_error_m=position_errors[xsens_name],
                head_orientation_offset_wijk=offsets[xsens_name],
                config=config,
            )
            if accepted:
                active[xsens_name] = link_name
            continue

        if position_errors[xsens_name] <= config.active_position_error_threshold_m:
            active[xsens_name] = link_name
    return active, head_status


def solve_xsens_tpose_calibration(
    hdf5_path: str | Path,
    config: XsensTposeCalibrationConfig | None = None,
) -> XsensTposeCalibrationResult:
    """Solve the dedicated G1 calibration IK for an Xsens HDF5 T-pose."""
    config = config or XsensTposeCalibrationConfig()
    tpose = load_xsens_hdf5_tpose(hdf5_path, variant=config.variant)
    return solve_xsens_tpose_calibration_from_data(tpose, config=config)


def solve_xsens_tpose_calibration_from_data(
    tpose: XsensHdf5Tpose,
    config: XsensTposeCalibrationConfig | None = None,
) -> XsensTposeCalibrationResult:
    """Solve the dedicated G1 calibration IK from already loaded Xsens T-pose data."""
    config = config or XsensTposeCalibrationConfig()
    scale_factor = RobotConfig(robot_type=config.robot_type).ROBOT_HEIGHT / config.default_human_height
    target_positions, target_axes = align_and_scale_tpose_targets(tpose.positions_m, scale_factor=scale_factor)
    model = mujoco.MjModel.from_xml_path(str(_robot_urdf_for_config(config)).replace(".urdf", ".xml"))
    problem = _G1CalibrationProblem(
        model=model,
        target_positions=target_positions,
        target_axes=target_axes,
        config=config,
    )
    qpos, solver_cost, solver_success = _solve_problem(problem, config)

    candidate_mapping = dict(config.candidate_orientation_mapping)
    candidate_position_errors = _position_errors(problem, qpos, candidate_mapping)
    candidate_offsets = _orientation_offsets(problem, qpos, tpose.quaternions_wijk, candidate_mapping)
    active_mapping, head_status = _active_orientation_mapping(
        position_errors=candidate_position_errors,
        offsets=candidate_offsets,
        config=config,
    )
    active_offsets = [candidate_offsets[name] for name in active_mapping]
    all_position_errors = _position_errors(problem, qpos, CALIBRATION_POSITION_MAPPING)
    axis_errors = _axis_errors(problem, qpos)

    return XsensTposeCalibrationResult(
        qpos=qpos.reshape(1, -1),
        xsens_tpose_positions_m=target_positions,
        xsens_tpose_quaternions_wijk=tpose.quaternions_wijk,
        candidate_orientation_mapping_names=list(candidate_mapping.keys()),
        candidate_robot_link_names=list(candidate_mapping.values()),
        active_orientation_mapping_names=list(active_mapping.keys()),
        robot_link_names=list(active_mapping.values()),
        orientation_offsets_wijk=np.asarray(active_offsets, dtype=float).reshape(-1, 4),
        position_error_names=list(all_position_errors.keys()),
        position_errors_m=np.asarray(list(all_position_errors.values()), dtype=float),
        axis_error_names=list(axis_errors.keys()),
        axis_errors_deg=np.asarray(list(axis_errors.values()), dtype=float),
        head_candidate_status=head_status,
        solver_cost=solver_cost,
        solver_success=solver_success,
        variant=tpose.variant,
        scale_factor=scale_factor,
    )


def save_xsens_tpose_calibration(result: XsensTposeCalibrationResult, path: str | Path, fps: int = 30) -> None:
    """Save calibration output as a Viser-compatible NPZ plus diagnostics."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        qpos=result.qpos,
        fps=np.array(fps, dtype=int),
        xsens_tpose_positions_m=result.xsens_tpose_positions_m,
        xsens_tpose_quaternions_wijk=result.xsens_tpose_quaternions_wijk,
        candidate_orientation_mapping_names=np.asarray(result.candidate_orientation_mapping_names, dtype=str),
        candidate_robot_link_names=np.asarray(result.candidate_robot_link_names, dtype=str),
        active_orientation_mapping_names=np.asarray(result.active_orientation_mapping_names, dtype=str),
        robot_link_names=np.asarray(result.robot_link_names, dtype=str),
        orientation_offsets_wijk=result.orientation_offsets_wijk,
        position_error_names=np.asarray(result.position_error_names, dtype=str),
        position_errors_m=result.position_errors_m,
        axis_error_names=np.asarray(result.axis_error_names, dtype=str),
        axis_errors_deg=result.axis_errors_deg,
        head_candidate_status=np.asarray(result.head_candidate_status),
        solver_cost=np.asarray(result.solver_cost, dtype=float),
        solver_success=np.asarray(result.solver_success, dtype=bool),
        variant=np.asarray(result.variant),
        scale_factor=np.asarray(result.scale_factor, dtype=float),
    )


def resolve_xsens_tennis_hdf5_path(data_path: str | Path, task_name: str) -> Path:
    """Resolve an Xsens tennis task name to its HDF5 file."""
    return resolve_xsens_hdf5_path(data_path, task_name)
