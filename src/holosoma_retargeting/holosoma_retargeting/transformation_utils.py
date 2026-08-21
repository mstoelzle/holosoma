"""Shared scalar-first quaternion, rotation-matrix, and rigid-transform conversions."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

__all__ = [
    "normalize_quaternions_wxyz",
    "normalize_vector",
    "position_quaternion_from_transform",
    "quaternion_conjugate",
    "quaternion_multiply",
    "rotate_vector",
    "rotate_vectors",
    "rotation_as_wxyz",
    "rotation_matrices_as_wxyz",
    "rotation_matrices_from_wxyz",
    "rotations_from_wxyz",
    "transform_from_position_quaternion",
    "transform_point",
    "transform_points",
]


def normalize_quaternions_wxyz(
    quaternions_wxyz: np.ndarray,
    *,
    canonical: bool = True,
) -> np.ndarray:
    """Normalize scalar-first quaternion(s), optionally choosing a nonnegative scalar part."""

    values = np.asarray(quaternions_wxyz, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError(f"Expected scalar-first quaternion array ending in 4, got {values.shape}")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(norms <= 1e-12):
        raise ValueError("Quaternions must contain finite, nonzero values")
    normalized = values / norms
    if not canonical:
        return normalized
    signs = np.where(normalized[..., :1] < 0.0, -1.0, 1.0)
    return normalized * signs


def normalize_vector(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Normalize one vector, returning a deterministic fallback when it is degenerate."""

    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm > 1e-9:
        return value / norm
    if fallback is None:
        return np.zeros_like(value)
    return np.asarray(fallback, dtype=float)


def quaternion_conjugate(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Return normalized scalar-first quaternion conjugates without changing their signs."""

    values = normalize_quaternions_wxyz(quaternions_wxyz, canonical=False)
    result = values.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left_wxyz: np.ndarray, right_wxyz: np.ndarray) -> np.ndarray:
    """Multiply broadcast-compatible scalar-first rotation quaternions."""

    left = normalize_quaternions_wxyz(left_wxyz, canonical=False)
    right = normalize_quaternions_wxyz(right_wxyz, canonical=False)
    try:
        left, right = np.broadcast_arrays(left, right)
    except ValueError as error:
        raise ValueError(
            f"Quaternion arrays are not broadcast-compatible: {left.shape} and {right.shape}"
        ) from error
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    product = np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )
    return normalize_quaternions_wxyz(product, canonical=False)


def rotate_vectors(quaternions_wxyz: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate broadcast-compatible vectors by scalar-first quaternions."""

    quaternions = normalize_quaternions_wxyz(quaternions_wxyz, canonical=False)
    values = np.asarray(vectors, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3:
        raise ValueError(f"Expected vector array ending in 3, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Vectors must contain finite values")
    try:
        vector_part, values = np.broadcast_arrays(quaternions[..., 1:], values)
        scalar_part = np.broadcast_to(quaternions[..., :1], vector_part.shape[:-1] + (1,))
    except ValueError as error:
        raise ValueError(
            f"Quaternion and vector arrays are not broadcast-compatible: {quaternions.shape} and {values.shape}"
        ) from error
    twice_cross = 2.0 * np.cross(vector_part, values)
    return values + scalar_part * twice_cross + np.cross(vector_part, twice_cross)


def rotate_vector(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one vector by one scalar-first quaternion."""

    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    value = np.asarray(vector, dtype=float)
    if quaternion.shape != (4,) or value.shape != (3,):
        raise ValueError("Quaternion and vector must have shapes (4,) and (3,)")
    return rotate_vectors(quaternion, value)


def transform_points(
    position_m: np.ndarray,
    quaternion_wxyz: np.ndarray,
    points_m: np.ndarray,
) -> np.ndarray:
    """Apply one or more broadcast-compatible position/quaternion transforms to points."""

    positions = np.asarray(position_m, dtype=float)
    if positions.ndim < 1 or positions.shape[-1] != 3 or not np.isfinite(positions).all():
        raise ValueError(f"Expected finite position array ending in 3, got {positions.shape}")
    rotated = rotate_vectors(quaternion_wxyz, points_m)
    try:
        return positions + rotated
    except ValueError as error:
        raise ValueError(
            f"Position and rotated-point arrays are not broadcast-compatible: {positions.shape} and {rotated.shape}"
        ) from error


def transform_point(
    position_m: np.ndarray,
    quaternion_wxyz: np.ndarray,
    point_m: np.ndarray,
) -> np.ndarray:
    """Apply one position/quaternion transform to one point."""

    position = np.asarray(position_m, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    point = np.asarray(point_m, dtype=float)
    if position.shape != (3,) or quaternion.shape != (4,) or point.shape != (3,):
        raise ValueError("Position, quaternion, and point must have shapes (3,), (4,), and (3,)")
    return transform_points(position, quaternion, point)


def rotations_from_wxyz(quaternions_wxyz: np.ndarray) -> Rotation:
    """Build one or more SciPy rotations from scalar-first quaternion(s)."""

    values = normalize_quaternions_wxyz(quaternions_wxyz)
    if values.ndim > 2:
        raise ValueError(f"SciPy Rotation requires shape [4] or [N, 4], got {values.shape}")
    return Rotation.from_quat(values[..., [1, 2, 3, 0]])


def rotation_as_wxyz(rotation: Rotation, *, canonical: bool = True) -> np.ndarray:
    """Return scalar-first quaternion(s) from one or more SciPy rotations."""

    xyzw = np.asarray(rotation.as_quat(), dtype=float)
    return normalize_quaternions_wxyz(xyzw[..., [3, 0, 1, 2]], canonical=canonical)


def rotation_matrices_from_wxyz(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Convert scalar-first quaternion arrays with arbitrary leading dimensions to matrices."""

    values = normalize_quaternions_wxyz(quaternions_wxyz)
    leading_shape = values.shape[:-1]
    xyzw = values[..., [1, 2, 3, 0]].reshape(-1, 4)
    return Rotation.from_quat(xyzw).as_matrix().reshape(leading_shape + (3, 3))


def rotation_matrices_as_wxyz(
    rotation_matrices: np.ndarray,
    *,
    canonical: bool = True,
) -> np.ndarray:
    """Convert rotation-matrix arrays with arbitrary leading dimensions to scalar-first quaternions."""

    matrices = np.asarray(rotation_matrices, dtype=float)
    if matrices.ndim < 2 or matrices.shape[-2:] != (3, 3) or not np.isfinite(matrices).all():
        raise ValueError(f"Expected finite rotation matrices ending in [3, 3], got {matrices.shape}")
    leading_shape = matrices.shape[:-2]
    rotations = Rotation.from_matrix(matrices.reshape(-1, 3, 3))
    return rotation_as_wxyz(rotations, canonical=canonical).reshape(leading_shape + (4,))


def transform_from_position_quaternion(
    position_m: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> np.ndarray:
    """Construct a homogeneous transform from a position and scalar-first quaternion."""

    position = np.asarray(position_m, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError(f"Expected a finite position with shape [3], got {position.shape}")
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotations_from_wxyz(quaternion_wxyz).as_matrix()
    transform[:3, 3] = position
    return transform


def position_quaternion_from_transform(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract a position and canonical scalar-first quaternion from a transform."""

    value = np.asarray(transform, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"Expected a finite homogeneous transform with shape [4, 4], got {value.shape}")
    return value[:3, 3].copy(), rotation_as_wxyz(Rotation.from_matrix(value[:3, :3]))
