"""Configuration types for retargeter settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

ArmOrientationTrackingMode: TypeAlias = Literal["auto", "off", "longitudinal-axes", "frame-and-bend"]
"""Available arm-orientation target formulations.

``auto`` preserves the context-dependent Xsens-to-G1 default and is resolved
before the retargeter is constructed.
"""

OptimizationSchedule: TypeAlias = Literal["single-stage", "orientation-first"]
"""Available per-frame optimization schedules."""


@dataclass(frozen=True)
class FootLockConfig:
    """Configuration for explicit frame-range based foot locking constraints."""

    enable: bool = False
    """Whether to enforce explicit frame-range based foot locking constraints."""

    windows: dict[str, list[tuple[int, int] | tuple[int, int, float]]] | None = None
    """Per-foot inclusive frame windows for locking.
    Each window is (start, end) or (start, end, z_floor).
    If z_floor is given per-window, it overrides the global z_floor for that window.
    Example: {"L_Toe": [(30, 60, 0.15)], "R_Toe": [(10, 20), (80, 95, 0.30)]}"""

    z_floor: float = 0.0
    """Default floor height used by Z pinning constraints (overridden by per-window z)."""

    tolerance: float = 5e-3
    """Tolerance for Z floor pinning constraints."""


@dataclass(frozen=True)
class SelfCollisionConfig:
    """Configuration for self-collision avoidance constraints."""

    enable: bool = False
    """Whether to enforce self-collision constraints."""

    pairs: list[tuple[str, str]] = field(default_factory=list)
    """Body name pairs to check for self-collision.
    Example: [("left_elbow_link", "left_knee_link"), ("left_wrist_yaw_link", "left_knee_link")]"""

    windows: list[tuple[int, int]] | None = None
    """Inclusive frame windows during which self-collision is enforced.
    If None, enforced on all frames.
    Example: [(50, 120)] means only enforce on frames 50..120."""

    tolerance: float = 0.02
    """Minimum distance (meters) to maintain between body pairs."""


@dataclass(frozen=True)
class OrientationTrackingConfig:
    """Configuration for Xsens orientation tracking."""

    arm_mode: ArmOrientationTrackingMode = "auto"
    """Arm-orientation target formulation.

    ``auto`` enables ``longitudinal-axes`` for the supported Xsens-to-G1
    morphology-adaptation path and otherwise resolves to ``off``.
    ``frame-and-bend`` tracks full upper-arm frames, elbow bend, and hand
    orientations instead of absolute forearm directions.
    """

    calibration_path: Path | None = None
    """Path to the Xsens T-pose calibration artifact used for orientation/axis correspondences."""

    orientation_weight: float = 2.0
    """Weight for full segment orientation tracking residuals."""

    axis_weight: float = 5.0
    """Global weight for segment-axis direction tracking residuals."""

    orientation_error_clip_rad: float = 0.7
    """Maximum rotation-vector magnitude used by orientation residuals."""

    def __post_init__(self) -> None:
        if self.arm_mode not in {"auto", "off", "longitudinal-axes", "frame-and-bend"}:
            raise ValueError(f"Unsupported arm-orientation mode: {self.arm_mode}")

    @property
    def is_enabled(self) -> bool:
        """Whether the resolved configuration supplies orientation objectives."""

        return self.arm_mode in {"longitudinal-axes", "frame-and-bend"}


@dataclass(frozen=True)
class RetargeterConfig:
    """Configuration for retargeter parameters.

    These parameters control the retargeting optimization process.
    """

    q_a_init_idx: int = -7
    """Index in robot's configuration where optimization variables start.
    -7: starts from floating base, -3: starts from translation of floating base,
    0: starts from actuated DOF, 12: starts from waist, 15: starts from left shoulder"""

    activate_joint_limits: bool = True
    """Whether to enforce joint limits during retargeting."""

    activate_obj_non_penetration: bool = True
    """Whether to enforce ground/object non-penetration constraints."""

    activate_foot_sticking: bool = True
    """Whether to enforce foot sticking constraints."""

    penetration_tolerance: float = 0.001
    """Tolerance for penetration when enforcing non-penetration constraints."""

    foot_sticking_tolerance: float = 1e-3
    """Tolerance for foot sticking constraints in x, y."""

    foot_lock: FootLockConfig = field(default_factory=FootLockConfig)
    """Configuration for explicit frame-range based foot locking."""

    step_size: float = 0.2
    """Trust region for each SQP iteration."""

    initial_iterations: int = 50
    """SQP iterations used to solve the first motion frame."""

    iterations_per_frame: int = 10
    """SQP iterations used for each subsequent frame; raise for sparse keyframes."""

    visualize: bool = False
    """Whether to visualize the retargeting process."""

    debug: bool = False
    """Whether to enable debug mode."""

    self_collision: SelfCollisionConfig = field(default_factory=SelfCollisionConfig)
    """Configuration for self-collision avoidance."""

    orientation: OrientationTrackingConfig = field(default_factory=OrientationTrackingConfig)
    """Xsens orientation-target formulation and weights."""

    optimization_schedule: OptimizationSchedule = "single-stage"
    """Per-frame solve schedule."""

    orientation_first_iterations: int = 20
    """Coarse-stage iterations and minimum final-stage budget when orientation-first."""

    w_nominal_tracking_init: float = 5.0
    """Initial weight for nominal tracking cost."""

    nominal_tracking_tau: float = 1e6
    """Time constant for the nominal tracking cost."""

    def __post_init__(self) -> None:
        if self.optimization_schedule not in {"single-stage", "orientation-first"}:
            raise ValueError(f"Unsupported optimization schedule: {self.optimization_schedule}")
        if self.orientation_first_iterations <= 0:
            raise ValueError("orientation_first_iterations must be positive")
