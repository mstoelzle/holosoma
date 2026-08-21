"""Tests for shared scalar-first rotation and transform conversions."""

from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.transformation_utils import (
    normalize_quaternions_wxyz,
    position_quaternion_from_transform,
    quaternion_conjugate,
    quaternion_multiply,
    rotate_vector,
    rotate_vectors,
    rotation_as_wxyz,
    rotation_matrices_as_wxyz,
    rotation_matrices_from_wxyz,
    rotations_from_wxyz,
    transform_from_position_quaternion,
    transform_point,
    transform_points,
)
from scipy.spatial.transform import Rotation


def test_scalar_first_rotation_conversions_support_single_and_batched_values() -> None:
    rotations = Rotation.from_euler("xyz", [[10.0, 20.0, 30.0], [-25.0, 5.0, 170.0]], degrees=True)
    quaternions = rotation_as_wxyz(rotations)
    assert quaternions.shape == (2, 4)
    assert np.all(quaternions[:, 0] >= 0.0)
    np.testing.assert_allclose(
        (rotations_from_wxyz(quaternions).inv() * rotations).magnitude(),
        0.0,
        atol=1e-12,
    )
    single = rotation_as_wxyz(rotations[0])
    assert single.shape == (4,)
    np.testing.assert_allclose(rotations_from_wxyz(single).as_matrix(), rotations[0].as_matrix())


def test_rotation_matrix_conversions_preserve_arbitrary_leading_dimensions() -> None:
    quaternions = np.zeros((2, 3, 4), dtype=float)
    quaternions[..., 0] = -1.0
    matrices = rotation_matrices_from_wxyz(quaternions)
    assert matrices.shape == (2, 3, 3, 3)
    np.testing.assert_allclose(matrices, np.broadcast_to(np.eye(3), matrices.shape), atol=1e-12)
    restored = rotation_matrices_as_wxyz(matrices)
    np.testing.assert_allclose(restored[..., 0], 1.0)
    np.testing.assert_allclose(restored[..., 1:], 0.0, atol=1e-12)


def test_quaternion_normalization_rejects_invalid_values() -> None:
    np.testing.assert_allclose(normalize_quaternions_wxyz([-2.0, 0.0, 0.0, 0.0]), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        normalize_quaternions_wxyz([-2.0, 0.0, 0.0, 0.0], canonical=False),
        [-1.0, 0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="nonzero"):
        normalize_quaternions_wxyz(np.zeros(4))
    with pytest.raises(ValueError, match="ending in 4"):
        normalize_quaternions_wxyz(np.zeros((2, 3)))


def test_position_quaternion_transform_round_trip() -> None:
    position = np.array([0.4, -0.2, 1.3])
    quaternion = rotation_as_wxyz(Rotation.from_euler("zyx", [45.0, -10.0, 25.0], degrees=True))
    transform = transform_from_position_quaternion(position, quaternion)
    restored_position, restored_quaternion = position_quaternion_from_transform(transform)
    np.testing.assert_allclose(restored_position, position)
    np.testing.assert_allclose(restored_quaternion, quaternion)


def test_quaternion_conjugate_and_multiply_support_broadcast_batches() -> None:
    left_rotations = Rotation.from_euler("xyz", [[10.0, 20.0, 30.0], [-45.0, 5.0, 80.0]], degrees=True)
    right_rotation = Rotation.from_euler("zyx", [25.0, -15.0, 5.0], degrees=True)
    left = 3.0 * rotation_as_wxyz(left_rotations, canonical=False)
    right = -2.0 * rotation_as_wxyz(right_rotation, canonical=False)

    product = quaternion_multiply(left, right)
    expected = left_rotations * right_rotation
    np.testing.assert_allclose(
        rotation_matrices_from_wxyz(product),
        expected.as_matrix(),
        atol=1e-12,
    )
    identity = quaternion_multiply(left, quaternion_conjugate(left))
    np.testing.assert_allclose(
        rotation_matrices_from_wxyz(identity),
        np.broadcast_to(np.eye(3), (2, 3, 3)),
        atol=1e-15,
    )


def test_vector_rotation_supports_broadcasting_and_matches_scipy() -> None:
    rng = np.random.default_rng(8146)
    quaternions = rng.normal(size=(17, 1, 4))
    quaternions *= rng.uniform(0.1, 4.0, size=(17, 1, 1))
    vectors = rng.normal(size=(1, 23, 3))

    actual = rotate_vectors(quaternions, vectors)
    expected = np.stack(
        [rotations_from_wxyz(quaternions[frame, 0]).apply(vectors[0]) for frame in range(17)]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(rotate_vector(quaternions[3, 0], vectors[0, 7]), expected[3, 7])


def test_point_transforms_support_scalar_and_broadcast_inputs() -> None:
    quaternion = rotation_as_wxyz(Rotation.from_euler("z", 90.0, degrees=True))
    position = np.array([1.0, 2.0, 3.0])
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, -1.0]])

    np.testing.assert_allclose(
        transform_points(position, quaternion, points),
        [[1.0, 3.0, 3.0], [-1.0, 2.0, 2.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(transform_point(position, quaternion, points[0]), [1.0, 3.0, 3.0], atol=1e-12)


@pytest.mark.parametrize(
    ("function", "arguments", "message"),
    [
        (quaternion_conjugate, (np.zeros(4),), "nonzero"),
        (quaternion_multiply, (np.ones(4), np.ones(3)), "ending in 4"),
        (rotate_vector, (np.ones(4), np.ones(4)), "shapes"),
        (rotate_vectors, (np.ones(4), np.array([np.nan, 0.0, 0.0])), "finite"),
        (transform_point, (np.ones(4), np.ones(4), np.ones(3)), "shapes"),
    ],
)
def test_quaternion_and_point_helpers_reject_invalid_inputs(function, arguments, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        function(*arguments)
