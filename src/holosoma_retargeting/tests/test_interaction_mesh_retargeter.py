"""Unit tests for retargeting constraint activation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import cvxpy as cp
import holosoma_retargeting.src.interaction_mesh_retargeter as retargeter_module
import mujoco
import numpy as np
import pytest
from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import (
    FootLockConfig,
    OrientationTrackingConfig,
    RetargeterConfig,
)
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import create_task_constants
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter
from holosoma_retargeting.xsens.orientation_tracking import XsensOrientationTargets

import holosoma_retargeting

MODEL_DIR = Path(holosoma_retargeting.__file__).parent / "models" / "g1"


class _Progress:
    def __init__(self, values):
        self.values = values

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, **_kwargs) -> None:
        pass


def _orientation_targets(
    num_frames: int,
    *,
    orientation_names: list[str] | None = None,
    axis_names: list[str] | None = None,
    include_elbow_bend: bool = False,
) -> XsensOrientationTargets:
    orientation_names = ["Right Hand"] if orientation_names is None else orientation_names
    axis_names = ["right_upper_arm"] if axis_names is None else axis_names
    orientation_count = len(orientation_names)
    axis_count = len(axis_names)
    return XsensOrientationTargets(
        orientation_names=orientation_names,
        orientation_robot_link_names=["right_rubber_hand_link"] * orientation_count,
        orientation_offsets_wijk=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (orientation_count, 1)),
        orientation_target_rotations=np.broadcast_to(
            np.eye(3),
            (num_frames, orientation_count, 3, 3),
        ).copy(),
        axis_names=axis_names,
        axis_xsens_segment_names=["Right Upper Arm"] * axis_count,
        axis_robot_start_link_names=["right_shoulder_yaw_link"] * axis_count,
        axis_robot_end_link_names=["right_elbow_link"] * axis_count,
        axis_robot_local_vectors=np.zeros((axis_count, 3)),
        axis_target_vectors=np.broadcast_to(
            np.array([1.0, 0.0, 0.0]),
            (num_frames, axis_count, 3),
        ).copy(),
        axis_weights=np.ones(axis_count),
        elbow_bend_names=["right_elbow_bend"] if include_elbow_bend else [],
        elbow_bend_robot_shoulder_link_names=(["right_shoulder_yaw_link"] if include_elbow_bend else []),
        elbow_bend_robot_elbow_link_names=["right_elbow_link"] if include_elbow_bend else [],
        elbow_bend_robot_wrist_link_names=(["right_wrist_yaw_link"] if include_elbow_bend else []),
        elbow_bend_target_cosines=(
            np.ones((num_frames, 1), dtype=float) if include_elbow_bend else np.zeros((num_frames, 0), dtype=float)
        ),
        elbow_bend_weights=np.ones(1 if include_elbow_bend else 0, dtype=float),
    )


def _frame_and_bend_arm_targets(
    num_frames: int,
    *,
    include_elbow_bends: bool = True,
    retain_arm_axis: bool = False,
) -> XsensOrientationTargets:
    axis_names = ["left_foot_forward"]
    axis_segments = ["Left Foot"]
    axis_start_links = ["left_ankle_roll_link"]
    axis_end_links = [""]
    axis_local_vectors = [[1.0, 0.0, 0.0]]
    if retain_arm_axis:
        axis_names.append("right_forearm")
        axis_segments.append("Right Forearm")
        axis_start_links.append("right_elbow_link")
        axis_end_links.append("right_wrist_yaw_link")
        axis_local_vectors.append([0.0, 0.0, 0.0])
    elbow_names = ["left_elbow_bend", "right_elbow_bend"] if include_elbow_bends else []
    return XsensOrientationTargets(
        orientation_names=["Left Upper Arm", "Right Upper Arm", "Left Hand", "Right Hand"],
        orientation_robot_link_names=[
            "left_shoulder_yaw_link",
            "right_shoulder_yaw_link",
            "left_rubber_hand_link",
            "right_rubber_hand_link",
        ],
        orientation_offsets_wijk=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (4, 1)),
        orientation_target_rotations=np.broadcast_to(
            np.eye(3),
            (num_frames, 4, 3, 3),
        ).copy(),
        axis_names=axis_names,
        axis_xsens_segment_names=axis_segments,
        axis_robot_start_link_names=axis_start_links,
        axis_robot_end_link_names=axis_end_links,
        axis_robot_local_vectors=np.asarray(axis_local_vectors, dtype=float),
        axis_target_vectors=np.broadcast_to(
            np.array([1.0, 0.0, 0.0]),
            (num_frames, len(axis_names), 3),
        ).copy(),
        axis_weights=np.ones(len(axis_names)),
        elbow_bend_names=elbow_names,
        elbow_bend_robot_shoulder_link_names=(
            ["left_shoulder_yaw_link", "right_shoulder_yaw_link"] if include_elbow_bends else []
        ),
        elbow_bend_robot_elbow_link_names=(["left_elbow_link", "right_elbow_link"] if include_elbow_bends else []),
        elbow_bend_robot_wrist_link_names=(
            ["left_wrist_yaw_link", "right_wrist_yaw_link"] if include_elbow_bends else []
        ),
        elbow_bend_target_cosines=(
            np.ones((num_frames, 2), dtype=float) if include_elbow_bends else np.zeros((num_frames, 0), dtype=float)
        ),
        elbow_bend_weights=np.ones(2 if include_elbow_bends else 0, dtype=float),
    )


def _motion_retargeter(
    *,
    optimization_schedule: str,
    arm_orientation_mode: str = "longitudinal-axes",
) -> InteractionMeshRetargeter:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter.nq = 10
    retargeter.q_a_indices = np.array([0, 1], dtype=int)
    retargeter.object_name = "ground"
    retargeter.smplh_mapped_joint_indices = np.array([0], dtype=int)
    retargeter.orientation_config = OrientationTrackingConfig(arm_mode=arm_orientation_mode)
    retargeter.optimization_schedule = optimization_schedule
    retargeter.orientation_first_iterations = 4
    retargeter.w_nominal_tracking_init = 5.0
    retargeter.nominal_tracking_tau = 10.0
    retargeter.initial_iterations = 3
    retargeter.iterations_per_frame = 2
    retargeter.debug = False
    retargeter.visualize = False
    retargeter._orientation_tracking_errors = lambda *_args: (np.zeros(1), np.zeros(1))
    return retargeter


def _run_stub_motion(
    retargeter: InteractionMeshRetargeter,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    targets: XsensOrientationTargets,
) -> tuple[np.ndarray, list[dict]]:
    calls: list[dict] = []

    def fake_iterate(**kwargs):
        calls.append(kwargs)
        q = np.copy(kwargs["q_n"])
        increment = 1.0 if not kwargs.get("include_position_tracking", True) else 10.0
        q[retargeter.q_a_indices] += increment
        return q, 0.0

    monkeypatch.setattr(retargeter, "iterate", fake_iterate)
    monkeypatch.setattr(retargeter_module, "tqdm", _Progress)
    monkeypatch.setattr(
        retargeter_module,
        "create_interaction_mesh",
        lambda vertices: (vertices, np.zeros((0, 4), dtype=int)),
    )
    monkeypatch.setattr(retargeter_module, "get_adjacency_list", lambda *_args: [])
    monkeypatch.setattr(retargeter_module, "calculate_laplacian_coordinates", lambda vertices, _adj: vertices)

    num_frames = targets.orientation_target_rotations.shape[0]
    result, *_ = retargeter.retarget_motion(
        human_joint_motions=np.zeros((num_frames, 1, 3)),
        object_poses=np.zeros((num_frames, 7)),
        object_poses_augmented=np.zeros((num_frames, 7)),
        object_points_local_demo=np.zeros((1, 3)),
        object_points_local=np.zeros((1, 3)),
        foot_sticking_sequences=[{"left": False, "right": False}] * num_frames,
        q_a_init=np.zeros(2),
        orientation_targets=targets,
        dest_res_path=tmp_path / "retargeted.npz",
    )
    return result, calls


def _retargeter_with_non_penetration(enabled: bool) -> InteractionMeshRetargeter:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter.activate_obj_non_penetration = enabled
    retargeter.q_a_indices = np.array([0, 1], dtype=int)
    retargeter.penetration_tolerance = 1e-3
    return retargeter


def _retargeter_with_collision_pair() -> tuple[InteractionMeshRetargeter, tuple[int, int]]:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="1 1 0.1"/>
            <body name="robot_body" pos="0 0 1">
              <geom name="robot_collision" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    ground_id = model.geom("ground").id
    robot_id = model.geom("robot_collision").id
    pair = (ground_id, robot_id)

    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.collision_detection_threshold = 0.1
    retargeter.object_name = "tracked_object"
    retargeter._geom_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "" for geom_id in range(model.ngeom)
    ]
    retargeter._prefilter_pairs_with_mj_collision = lambda _threshold: {pair}
    retargeter._compute_jacobian_for_contact_relative = Mock(return_value=np.zeros(model.nv))
    retargeter._self_collision_enabled = True
    retargeter._self_collision_windows = None
    retargeter._self_collision_geom_pairs = [pair]
    retargeter._sc_last_vis_frame = -1
    retargeter.visualize = False
    return retargeter, pair


def test_environment_non_penetration_constraints_are_disabled_by_config() -> None:
    retargeter = _retargeter_with_non_penetration(False)

    constraints = retargeter._environment_non_penetration_constraints(
        cp.Variable(2),
        {"ground-contact": np.zeros(2)},
        {"ground-contact": -0.1},
    )

    assert constraints == []


def test_environment_non_penetration_constraints_are_enabled_by_default() -> None:
    retargeter = _retargeter_with_non_penetration(True)

    constraints = retargeter._environment_non_penetration_constraints(
        cp.Variable(2),
        {"ground-contact": np.zeros(2)},
        {"ground-contact": -0.1},
    )

    assert len(constraints) == 1


def test_foot_lock_windows_return_per_window_floor_height() -> None:
    retargeter = object.__new__(InteractionMeshRetargeter)
    retargeter._init_foot_lock(
        FootLockConfig(
            enable=True,
            windows={
                "L_Toe": [(10, 20, 0.12), (30, 40)],
                "R_Toe": [(15, 25, 0.56)],
            },
            z_floor=0.34,
        )
    )

    assert retargeter._is_foot_locked_in_window("left_ankle_link", 15) == pytest.approx(0.12)
    assert retargeter._is_foot_locked_in_window("left_ankle_link", 35) == pytest.approx(0.34)
    assert retargeter._is_foot_locked_in_window("right_ankle_link", 20) == pytest.approx(0.56)
    assert retargeter._is_foot_locked_in_window("left_ankle_link", 25) is None
    assert retargeter._is_foot_locked_in_window("torso_link", 15) is None


def test_optimization_schedule_defaults_to_single_stage() -> None:
    config = RetargeterConfig()

    assert config.optimization_schedule == "single-stage"
    assert config.orientation_first_iterations == 20
    assert config.orientation.arm_mode == "auto"


@pytest.mark.parametrize("iterations", [0, -1])
def test_orientation_first_schedule_rejects_non_positive_iterations(iterations: int) -> None:
    with pytest.raises(ValueError, match="iterations"):
        RetargeterConfig(orientation_first_iterations=iterations)


def test_single_stage_schedule_keeps_one_full_solve_and_warm_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, calls = _run_stub_motion(
        _motion_retargeter(optimization_schedule="single-stage"),
        monkeypatch,
        tmp_path,
        _orientation_targets(2),
    )

    assert len(calls) == 2
    assert all(call.get("include_position_tracking", True) for call in calls)
    assert [call["n_iter"] for call in calls] == [3, 2]
    np.testing.assert_allclose(calls[0]["q_n"][:2], [0.0, 0.0])
    np.testing.assert_allclose(calls[0]["q_t_last"][:2], [0.0, 0.0])
    np.testing.assert_allclose(calls[1]["q_n"][:2], [10.0, 10.0])
    np.testing.assert_allclose(calls[1]["q_t_last"][:2], [10.0, 10.0])
    np.testing.assert_allclose(result[:, :2], [[10.0, 10.0], [20.0, 20.0]])


def test_orientation_first_runs_coarse_then_full_and_propagates_refined_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    targets = _orientation_targets(2)
    result, calls = _run_stub_motion(
        _motion_retargeter(optimization_schedule="orientation-first"),
        monkeypatch,
        tmp_path,
        targets,
    )

    assert len(calls) == 4
    coarse_0, full_0, coarse_1, full_1 = calls
    for coarse in (coarse_0, coarse_1):
        assert coarse["include_position_tracking"] is False
        assert "q_a_nominal" not in coarse
        assert "w_nominal_tracking" not in coarse
        assert coarse["n_iter"] == 4
        assert coarse["orientation_targets"] is targets
    for full in (full_0, full_1):
        assert full.get("include_position_tracking", True) is True
        assert full["q_a_nominal"] is None
        assert full["w_nominal_tracking"] == pytest.approx(5.0)
        assert full["n_iter"] == 4
        assert full["orientation_targets"] is targets

    np.testing.assert_allclose(coarse_0["q_n"][:2], [0.0, 0.0])
    np.testing.assert_allclose(coarse_0["q_t_last"][:2], [0.0, 0.0])
    np.testing.assert_allclose(full_0["q_n"][:2], [1.0, 1.0])
    np.testing.assert_allclose(full_0["q_t_last"][:2], [0.0, 0.0])
    np.testing.assert_allclose(coarse_1["q_n"][:2], [11.0, 11.0])
    np.testing.assert_allclose(coarse_1["q_t_last"][:2], [11.0, 11.0])
    np.testing.assert_allclose(full_1["q_n"][:2], [12.0, 12.0])
    np.testing.assert_allclose(full_1["q_t_last"][:2], [11.0, 11.0])
    np.testing.assert_allclose(result[:, :2], [[11.0, 11.0], [22.0, 22.0]])


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (_orientation_targets(1, orientation_names=[]), "full-orientation"),
        (_orientation_targets(1, axis_names=[]), "segment-axis"),
    ],
)
def test_orientation_first_requires_full_orientation_and_axis_targets(
    targets: XsensOrientationTargets,
    message: str,
    tmp_path: Path,
) -> None:
    retargeter = _motion_retargeter(optimization_schedule="orientation-first")

    with pytest.raises(ValueError, match=message):
        retargeter.retarget_motion(
            human_joint_motions=np.zeros((1, 1, 3)),
            object_poses=np.zeros((1, 7)),
            object_poses_augmented=np.zeros((1, 7)),
            object_points_local_demo=np.zeros((1, 3)),
            object_points_local=np.zeros((1, 3)),
            foot_sticking_sequences=[{"left": False, "right": False}],
            q_a_init=np.zeros(2),
            orientation_targets=targets,
            dest_res_path=tmp_path / "unused.npz",
        )


@pytest.mark.parametrize(
    ("targets", "message"),
    [
        (
            _orientation_targets(1, orientation_names=["Left Hand", "Right Hand"]),
            "full-orientation targets for both upper arms",
        ),
        (
            _frame_and_bend_arm_targets(1, include_elbow_bends=False),
            "elbow-bend targets for both arms",
        ),
        (
            _frame_and_bend_arm_targets(1, retain_arm_axis=True),
            "must replace, rather than supplement",
        ),
    ],
)
def test_frame_and_bend_mode_rejects_incomplete_or_redundant_targets(
    targets: XsensOrientationTargets,
    message: str,
    tmp_path: Path,
) -> None:
    retargeter = _motion_retargeter(
        optimization_schedule="single-stage",
        arm_orientation_mode="frame-and-bend",
    )

    with pytest.raises(ValueError, match=message):
        retargeter.retarget_motion(
            human_joint_motions=np.zeros((1, 1, 3)),
            object_poses=np.zeros((1, 7)),
            object_poses_augmented=np.zeros((1, 7)),
            object_points_local_demo=np.zeros((1, 3)),
            object_points_local=np.zeros((1, 3)),
            foot_sticking_sequences=[{"left": False, "right": False}],
            q_a_init=np.zeros(2),
            orientation_targets=targets,
            dest_res_path=tmp_path / "unused.npz",
        )


def test_frame_and_bend_targets_are_shared_by_both_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    targets = _frame_and_bend_arm_targets(1)
    _, calls = _run_stub_motion(
        _motion_retargeter(
            optimization_schedule="orientation-first",
            arm_orientation_mode="frame-and-bend",
        ),
        monkeypatch,
        tmp_path,
        targets,
    )

    assert len(calls) == 2
    assert calls[0]["include_position_tracking"] is False
    assert calls[1].get("include_position_tracking", True) is True
    assert calls[0]["orientation_targets"] is targets
    assert calls[1]["orientation_targets"] is targets


def test_coarse_solve_excludes_position_tracking_but_keeps_orientation_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_urdf = str(MODEL_DIR / "g1_29dof.urdf")
    constants = create_task_constants(
        robot_config=RobotConfig(robot_type="g1", robot_urdf_file=robot_urdf),
        motion_data_config=MotionDataConfig(data_format="xsens", robot_type="g1"),
        task_config=TaskConfig(),
        task_type="robot_only",
    )
    retargeter = InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=None,
        activate_foot_sticking=False,
        activate_obj_non_penetration=False,
    )
    q = retargeter.robot_model.qpos0.copy()
    retargeter.robot_data.qpos[:] = q
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    _, hand_rotation, _ = retargeter._frame_pose("right_rubber_hand_link")
    upper_arm_axis, _ = retargeter._axis_jacobian("right_shoulder_yaw_link", "right_elbow_link")
    targets = _orientation_targets(1)
    targets.orientation_target_rotations[0, 0] = hand_rotation
    targets.axis_target_vectors[0, 0] = upper_arm_axis

    orientation_term_counts: list[int] = []
    problem_shapes: list[tuple[int, int]] = []
    original_orientation_terms = retargeter._orientation_tracking_objective_terms
    original_problem = retargeter_module.cp.Problem

    def tracked_orientation_terms(*args, **kwargs):
        terms = original_orientation_terms(*args, **kwargs)
        orientation_term_counts.append(len(terms))
        return terms

    def fail_if_position_tracking_is_built(*_args, **_kwargs):
        raise AssertionError("coarse solve constructed the positional objective")

    def tracked_problem(objective, constraints):
        objective_expression = objective.args[0]
        problem_shapes.append((len(objective_expression.args), len(constraints)))
        return original_problem(objective, constraints)

    monkeypatch.setattr(retargeter, "_orientation_tracking_objective_terms", tracked_orientation_terms)
    monkeypatch.setattr(retargeter_module, "calculate_laplacian_matrix", fail_if_position_tracking_is_built)
    monkeypatch.setattr(retargeter_module.cp, "Problem", tracked_problem)

    _, cost = retargeter.solve_single_iteration(
        q_locked=q,
        q_a_n_last=q[retargeter.q_a_indices],
        q_t_last=q,
        target_laplacian=np.zeros((0, 3)),
        adj_list=[],
        obj_pts_local=np.zeros((0, 3)),
        foot_sticking={"left": False, "right": False},
        orientation_targets=targets,
        include_position_tracking=False,
    )

    assert orientation_term_counts == [2]
    # Q-diagonal regularization + temporal smoothness + full orientation + axis.
    # The three retained constraints are lower limits, upper limits, and step size.
    assert problem_shapes == [(4, 3)]
    assert np.isfinite(cost)


def test_elbow_bend_remains_active_without_position_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    robot_urdf = str(MODEL_DIR / "g1_29dof.urdf")
    constants = create_task_constants(
        robot_config=RobotConfig(robot_type="g1", robot_urdf_file=robot_urdf),
        motion_data_config=MotionDataConfig(data_format="xsens", robot_type="g1"),
        task_config=TaskConfig(),
        task_type="robot_only",
    )
    retargeter = InteractionMeshRetargeter(
        task_constants=constants,
        object_urdf_path=None,
        activate_foot_sticking=False,
        activate_obj_non_penetration=False,
    )
    q = retargeter.robot_model.qpos0.copy()
    targets = _orientation_targets(1, include_elbow_bend=True)
    retargeter.robot_data.qpos[:] = q
    mujoco.mj_forward(retargeter.robot_model, retargeter.robot_data)
    _, hand_rotation, _ = retargeter._frame_pose("right_rubber_hand_link")
    upper_arm_axis, _ = retargeter._axis_jacobian("right_shoulder_yaw_link", "right_elbow_link")
    elbow_cosine, _ = retargeter._elbow_bend_cosine_jacobian(
        "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link"
    )
    targets.orientation_target_rotations[0, 0] = hand_rotation
    targets.axis_target_vectors[0, 0] = upper_arm_axis
    targets.elbow_bend_target_cosines[0, 0] = elbow_cosine

    orientation_term_counts: list[int] = []
    original_orientation_terms = retargeter._orientation_tracking_objective_terms

    def tracked_orientation_terms(*args, **kwargs):
        terms = original_orientation_terms(*args, **kwargs)
        orientation_term_counts.append(len(terms))
        return terms

    def fail_if_position_tracking_is_built(*_args, **_kwargs):
        raise AssertionError("coarse solve constructed the positional objective")

    monkeypatch.setattr(retargeter, "_orientation_tracking_objective_terms", tracked_orientation_terms)
    monkeypatch.setattr(retargeter_module, "calculate_laplacian_matrix", fail_if_position_tracking_is_built)

    _, cost = retargeter.solve_single_iteration(
        q_locked=q,
        q_a_n_last=q[retargeter.q_a_indices],
        q_t_last=q,
        target_laplacian=np.zeros((0, 3)),
        adj_list=[],
        obj_pts_local=np.zeros((0, 3)),
        foot_sticking={"left": False, "right": False},
        orientation_targets=targets,
        include_position_tracking=False,
    )

    assert orientation_term_counts == [3]
    assert np.isfinite(cost)


def test_collision_queries_reject_cutoff_equal_results(monkeypatch: pytest.MonkeyPatch) -> None:
    retargeter, _pair = _retargeter_with_collision_pair()

    def cutoff_result(_model, _data, _geom_a, _geom_b, cutoff, fromto):
        fromto[:] = 0.0
        return cutoff

    monkeypatch.setattr(mujoco, "mj_geomDistance", cutoff_result)

    environment_js, environment_phis = retargeter._update_jacobians_and_phis_from_q(np.zeros(retargeter.robot_model.nq))
    self_js, self_phis = retargeter._compute_self_collision_constraints(frame_idx=0)

    assert environment_js == {}
    assert environment_phis == {}
    assert self_js == {}
    assert self_phis == {}
    retargeter._compute_jacobian_for_contact_relative.assert_not_called()


def test_collision_queries_accept_results_strictly_below_cutoff(monkeypatch: pytest.MonkeyPatch) -> None:
    retargeter, pair = _retargeter_with_collision_pair()
    distance = retargeter.collision_detection_threshold - 0.01

    def in_range_result(_model, _data, _geom_a, _geom_b, _cutoff, fromto):
        fromto[:] = [0.0, 0.0, 0.0, 0.0, 0.0, distance]
        return distance

    monkeypatch.setattr(mujoco, "mj_geomDistance", in_range_result)

    environment_js, environment_phis = retargeter._update_jacobians_and_phis_from_q(np.zeros(retargeter.robot_model.nq))
    self_js, self_phis = retargeter._compute_self_collision_constraints(frame_idx=0)

    assert set(environment_js) == {pair}
    assert environment_phis[pair] == pytest.approx(distance)
    self_key = ("self", *pair)
    assert set(self_js) == {self_key}
    assert self_phis[self_key] == pytest.approx(distance)
    assert retargeter._compute_jacobian_for_contact_relative.call_count == 2
