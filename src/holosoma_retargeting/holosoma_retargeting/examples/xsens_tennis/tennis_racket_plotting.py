"""Shared Matplotlib geometry for tennis-racket pose visualizations."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

__all__ = ["draw_racket_pose", "racket_local_lines", "transform_racket_points"]


def racket_local_lines() -> tuple[np.ndarray, ...]:
    """Return a simple metric racket model aligned with local longitudinal +X."""

    shaft = np.array([[-0.09, 0.0, 0.0], [0.24, 0.0, 0.0]])
    theta = np.linspace(0.0, 2.0 * np.pi, 80)
    hoop = np.column_stack(
        [
            0.415 + 0.175 * np.cos(theta),
            np.zeros(theta.size),
            0.135 * np.sin(theta),
        ]
    )
    throat_left = np.array([[0.18, 0.0, 0.0], [0.27, 0.0, 0.075]])
    throat_right = np.array([[0.18, 0.0, 0.0], [0.27, 0.0, -0.075]])
    return shaft, hoop, throat_left, throat_right


def transform_racket_points(
    points: np.ndarray,
    origin: np.ndarray,
    rotation: Rotation | np.ndarray,
) -> np.ndarray:
    """Transform local racket geometry into the displayed coordinate frame."""

    racket_rotation = rotation if isinstance(rotation, Rotation) else Rotation.from_matrix(rotation)
    return racket_rotation.apply(np.asarray(points, dtype=float)) + np.asarray(origin, dtype=float)


def draw_racket_pose(
    axis: Axes,
    origin: np.ndarray,
    rotation: Rotation | np.ndarray,
    *,
    color: str,
    linestyle: str,
    label: str,
    alpha: float,
) -> None:
    """Draw a shaft, throat, and hoop for one racket pose."""

    for line_index, local_line in enumerate(racket_local_lines()):
        world_line = transform_racket_points(local_line, origin, rotation)
        axis.plot(
            world_line[:, 0],
            world_line[:, 1],
            world_line[:, 2],
            color=color,
            linestyle=linestyle,
            linewidth=2.8 if line_index == 0 else 2.0,
            alpha=alpha,
            label=label if line_index == 0 else None,
        )
