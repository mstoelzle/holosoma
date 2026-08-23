"""Tennis-specific frame selection layered over the generic retargeting solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

from holosoma_retargeting.config_types.retargeter import TennisRacketTrackingConfig
from holosoma_retargeting.transformation_utils import rotation_as_wxyz
from holosoma_retargeting.xsens.tennis_racket import (
    TennisRacketMotion,
    TennisRacketTargets,
    achieved_tennis_racket_pose,
    choose_tennis_racket_symmetry_branch,
    decide_filtered_tennis_racket_tracking,
    tennis_racket_target_error_rad,
)

OrientationOverride = tuple[int, np.ndarray] | None
FrameSolver = Callable[..., tuple[np.ndarray, float]]


@dataclass(frozen=True)
class TennisRacketFrameSolution:
    """Selected solve and diagnostics needed by the generic frame loop."""

    q: np.ndarray
    cost: float
    orientation_override: OrientationOverride
    symmetry_branch: int
    tracking_state: str


@dataclass(frozen=True)
class _BranchTrial:
    branch: int
    q: np.ndarray
    cost: float
    target_error_rad: float
    wrist_margin_rad: float


class TennisRacketFrameTracker:
    """Own racket branch continuity, feasibility hysteresis, and result collection."""

    _WRIST_JOINT_NAMES = (
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )

    def __init__(
        self,
        config: TennisRacketTrackingConfig,
        targets: TennisRacketTargets,
        *,
        right_hand_target_index: int,
        robot_model: mujoco.MjModel,
        robot_data: mujoco.MjData,
        active_qpos_indices: np.ndarray,
        active_lower_limits: np.ndarray,
        active_upper_limits: np.ndarray,
    ) -> None:
        self.config = config
        self.targets = targets
        self.right_hand_target_index = right_hand_target_index
        self.robot_model = robot_model
        self.robot_data = robot_data
        self._wrist_limits = self._resolve_wrist_limits(
            active_qpos_indices,
            active_lower_limits,
            active_upper_limits,
        )
        self._active = False
        self._reentry_streak = 0
        self._previous_branch: int | None = None
        self._positions: list[np.ndarray] = []
        self._quaternions: list[np.ndarray] = []
        self._states: list[str] = []
        self._branches: list[int] = []
        self._target_errors: list[float] = []
        self._wrist_margins: list[float] = []

    def _resolve_wrist_limits(
        self,
        active_qpos_indices: np.ndarray,
        active_lower_limits: np.ndarray,
        active_upper_limits: np.ndarray,
    ) -> tuple[tuple[int, float, float], ...]:
        limits: list[tuple[int, float, float]] = []
        for joint_name in self._WRIST_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                raise ValueError(f"Tennis-racket tracking requires G1 joint '{joint_name}'")
            qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
            active_index = np.flatnonzero(active_qpos_indices == qpos_index)
            if active_index.size != 1:
                raise ValueError(f"Tennis-racket tracking requires unlocked joint '{joint_name}'")
            index = int(active_index[0])
            limits.append(
                (
                    qpos_index,
                    float(active_lower_limits[index]),
                    float(active_upper_limits[index]),
                )
            )
        return tuple(limits)

    def _achieved_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        return achieved_tennis_racket_pose(
            self.robot_model,
            self.robot_data,
            self.targets.attachment,
        )

    def _wrist_limit_margin(self, q: np.ndarray) -> float:
        return float(
            min(min(float(q[index]) - lower, upper - float(q[index])) for index, lower, upper in self._wrist_limits)
        )

    def _orientation_override(self, frame_index: int, branch: int) -> tuple[int, np.ndarray]:
        return (
            self.right_hand_target_index,
            self.targets.candidate_hand_rotations[frame_index, branch],
        )

    def _trial_branch(
        self,
        frame_index: int,
        branch: int,
        solve_frame: FrameSolver,
    ) -> _BranchTrial | None:
        try:
            trial_q, trial_cost = solve_frame(
                orientation_rotation_override=self._orientation_override(frame_index, branch)
            )
        except RuntimeError:
            return None
        if not np.isfinite(trial_q).all() or not np.isfinite(trial_cost):
            return None
        _, achieved_rotation, _ = self._achieved_pose(trial_q)
        return _BranchTrial(
            branch=branch,
            q=trial_q,
            cost=trial_cost,
            target_error_rad=tennis_racket_target_error_rad(
                achieved_rotation,
                self.targets.candidate_racket_rotations[frame_index],
            ),
            wrist_margin_rad=self._wrist_limit_margin(trial_q),
        )

    def solve_frame(
        self,
        frame_index: int,
        current_q: np.ndarray,
        solve_frame: FrameSolver,
    ) -> TennisRacketFrameSolution:
        """Select hand fallback or a feasible symmetry-aware racket solve."""

        if self.config.mode == "hand":
            q, cost = solve_frame(orientation_rotation_override=None)
            return TennisRacketFrameSolution(q, cost, None, -1, "hand")

        _, _, current_hand_rotation = self._achieved_pose(current_q)
        preferred_branch = choose_tennis_racket_symmetry_branch(
            current_hand_rotation,
            self.targets.candidate_hand_rotations[frame_index],
            preferred_branch=self._previous_branch,
        )
        branch_order = (preferred_branch, 1 - preferred_branch)
        if self.config.mode == "racket":
            trial = self._trial_branch(frame_index, preferred_branch, solve_frame)
            if trial is None:
                raise RuntimeError(f"Tennis-racket solve failed at frame {frame_index}")
            self._previous_branch = preferred_branch
            return TennisRacketFrameSolution(
                trial.q,
                trial.cost,
                self._orientation_override(frame_index, preferred_branch),
                preferred_branch,
                "racket",
            )

        source_deviation = float(self.targets.source_origin_deviation_m[frame_index])
        detach_threshold = (
            self.config.detach_exit_threshold_m if self._active else self.config.detach_reentry_threshold_m
        )
        if source_deviation > detach_threshold:
            decision = decide_filtered_tennis_racket_tracking(
                self.config,
                active=self._active,
                feasible_streak=self._reentry_streak,
                source_origin_deviation_m=source_deviation,
                solve_succeeded=False,
                target_error_rad=float("inf"),
                wrist_limit_margin_rad=float("-inf"),
            )
            self._active = decision.active
            self._reentry_streak = decision.feasible_streak
            q, cost = solve_frame(orientation_rotation_override=None)
            return TennisRacketFrameSolution(q, cost, None, -1, decision.state)

        residual_limit = self.config.feasible_exit_error_rad if self._active else self.config.feasible_entry_error_rad
        accepted_trial: _BranchTrial | None = None
        best_failure: _BranchTrial | None = None
        for branch in branch_order:
            trial = self._trial_branch(frame_index, branch, solve_frame)
            if trial is None:
                continue
            if best_failure is None or trial.target_error_rad < best_failure.target_error_rad:
                best_failure = trial
            if (
                trial.target_error_rad <= residual_limit
                and trial.wrist_margin_rad >= self.config.min_wrist_limit_margin_rad
            ):
                accepted_trial = trial
                break

        decision_source = accepted_trial or best_failure
        decision = decide_filtered_tennis_racket_tracking(
            self.config,
            active=self._active,
            feasible_streak=self._reentry_streak,
            source_origin_deviation_m=source_deviation,
            solve_succeeded=decision_source is not None,
            target_error_rad=(float("inf") if decision_source is None else decision_source.target_error_rad),
            wrist_limit_margin_rad=(float("-inf") if decision_source is None else decision_source.wrist_margin_rad),
        )
        self._active = decision.active
        self._reentry_streak = decision.feasible_streak
        if decision.use_racket:
            assert accepted_trial is not None
            self._previous_branch = accepted_trial.branch
            return TennisRacketFrameSolution(
                accepted_trial.q,
                accepted_trial.cost,
                self._orientation_override(frame_index, accepted_trial.branch),
                accepted_trial.branch,
                decision.state,
            )
        q, cost = solve_frame(orientation_rotation_override=None)
        return TennisRacketFrameSolution(q, cost, None, -1, decision.state)

    def record_frame(
        self,
        frame_index: int,
        solution: TennisRacketFrameSolution,
    ) -> None:
        """Record achieved motion and diagnostics for one selected frame solve."""

        position, achieved_rotation, _ = self._achieved_pose(solution.q)
        self._positions.append(position)
        self._quaternions.append(rotation_as_wxyz(Rotation.from_matrix(achieved_rotation)))
        self._states.append(solution.tracking_state)
        self._branches.append(solution.symmetry_branch)
        self._target_errors.append(
            tennis_racket_target_error_rad(
                achieved_rotation,
                self.targets.candidate_racket_rotations[frame_index],
            )
        )
        self._wrist_margins.append(self._wrist_limit_margin(solution.q))

    @property
    def recorded_frame_count(self) -> int:
        """Number of accepted frames currently held by this tracker."""

        return len(self._positions)

    def checkpoint_filter_state(self) -> tuple[bool, int, int | None]:
        """Return the temporal filter state required for an exact continuation."""

        return self._active, self._reentry_streak, self._previous_branch

    def restore_checkpoint(
        self,
        motion: TennisRacketMotion,
        *,
        active: bool,
        reentry_streak: int,
        previous_branch: int | None,
    ) -> None:
        """Restore accepted motion diagnostics and temporal filter state."""

        if motion.tracking_mode != self.config.mode:
            raise ValueError("Checkpoint tennis-racket tracking mode does not match the current retargeting run")
        current_attachment = self.targets.attachment
        saved_attachment = motion.attachment
        attachment_arrays = (
            "position_m",
            "quaternion_wxyz",
            "longitudinal_axis_local",
            "palm_bounds_min_m",
            "palm_bounds_max_m",
            "palm_contact_bounds_min_m",
            "palm_contact_bounds_max_m",
        )
        if saved_attachment.hand_link != current_attachment.hand_link or any(
            not np.array_equal(getattr(saved_attachment, name), getattr(current_attachment, name))
            for name in attachment_arrays
        ):
            raise ValueError("Checkpoint tennis-racket attachment does not match the current retargeting run")
        frame_count = motion.position_m.shape[0]
        expected_source_deviation = np.asarray(self.targets.source_origin_deviation_m[:frame_count], dtype=float)
        if not np.array_equal(motion.source_origin_deviation_m, expected_source_deviation):
            raise ValueError("Checkpoint tennis-racket targets do not match the current retargeting run")
        if reentry_streak < 0:
            raise ValueError("Checkpoint racket reentry streak must be nonnegative")
        if previous_branch not in (None, 0, 1):
            raise ValueError("Checkpoint previous racket branch must be None, 0, or 1")

        self._positions = [row.copy() for row in motion.position_m]
        self._quaternions = [row.copy() for row in motion.quaternion_wxyz]
        self._states = [str(value) for value in motion.tracking_state]
        self._branches = [int(value) for value in motion.symmetry_branch]
        self._target_errors = [float(value) for value in motion.target_error_rad]
        self._wrist_margins = [float(value) for value in motion.min_wrist_limit_margin_rad]
        self._active = active
        self._reentry_streak = reentry_streak
        self._previous_branch = previous_branch

    def build_motion(self) -> TennisRacketMotion:
        """Build the first-class saved racket result after all frames are recorded."""

        frame_count = self.recorded_frame_count
        return TennisRacketMotion(
            position_m=np.asarray(self._positions, dtype=float),
            quaternion_wxyz=np.asarray(self._quaternions, dtype=float),
            tracking_state=np.asarray(self._states, dtype=str),
            symmetry_branch=np.asarray(self._branches, dtype=np.int8),
            target_error_rad=np.asarray(self._target_errors, dtype=float),
            source_origin_deviation_m=np.asarray(self.targets.source_origin_deviation_m[:frame_count], dtype=float),
            min_wrist_limit_margin_rad=np.asarray(self._wrist_margins, dtype=float),
            attachment=self.targets.attachment,
            tracking_mode=self.config.mode,
        )


def create_tennis_racket_frame_tracker(
    config: TennisRacketTrackingConfig,
    targets: TennisRacketTargets | None,
    *,
    frame_count: int,
    orientation_names: Sequence[str] | None,
    joint_limits_enforced: bool,
    robot_model: mujoco.MjModel,
    robot_data: mujoco.MjData,
    active_qpos_indices: np.ndarray,
    active_lower_limits: np.ndarray,
    active_upper_limits: np.ndarray,
) -> TennisRacketFrameTracker | None:
    """Validate optional racket inputs and create the isolated frame tracker."""

    if targets is None:
        if config.mode != "hand":
            raise ValueError(f"Tennis-racket mode '{config.mode}' requires RightHandSword data")
        return None
    if targets.frame_count != frame_count:
        raise ValueError(
            "Tennis-racket target frame count does not match motion frame count: "
            f"{targets.frame_count} vs {frame_count}"
        )
    if orientation_names is None:
        raise ValueError("Tennis-racket targets require orientation-aware retargeting")
    if "Right Hand" not in orientation_names:
        raise ValueError("Tennis-racket tracking requires a calibrated 'Right Hand' orientation target")
    if config.mode == "filtered" and not joint_limits_enforced:
        raise ValueError("Filtered tennis-racket tracking requires retargeter.activate_joint_limits=True")
    return TennisRacketFrameTracker(
        config,
        targets,
        right_hand_target_index=orientation_names.index("Right Hand"),
        robot_model=robot_model,
        robot_data=robot_data,
        active_qpos_indices=active_qpos_indices,
        active_lower_limits=active_lower_limits,
        active_upper_limits=active_upper_limits,
    )


__all__ = [
    "TennisRacketFrameSolution",
    "TennisRacketFrameTracker",
    "create_tennis_racket_frame_tracker",
]
