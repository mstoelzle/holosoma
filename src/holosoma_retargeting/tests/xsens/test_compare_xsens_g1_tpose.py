"""Tests for Xsens and G1 T-pose comparison."""

from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.examples.xsens_tennis.compare_xsens_g1_tpose import (
    body_poses_from_xsens_pose,
    orientation_correspondence_body_names,
    side_by_side_offsets,
    tree_vertical_bounds,
)
from holosoma_retargeting.kinematics import KinematicPose
from holosoma_retargeting.xsens.g1_kinematic_reduction import (
    build_g1_proportioned_xsens_tree,
    extract_g1_anthropometry,
)


def test_side_by_side_offsets_preserve_requested_column_order() -> None:
    assert side_by_side_offsets(2.0) == (-2.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="positive"):
        side_by_side_offsets(0.0)


def test_tree_vertical_bounds_include_procedural_meshes() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())
    minimum_z, maximum_z = tree_vertical_bounds(model)

    body_z = np.asarray([body.reference_pose.translation_m[2] for body in model.bodies])
    assert minimum_z < float(body_z.min())
    assert maximum_z > float(body_z.max())
    # The G1 is roughly 1.3 m tall.  This catches accidental deletion of the
    # pelvis-to-hip anchor or the collapsed hip-cluster extent.
    assert maximum_z - minimum_z > 1.25


def test_g1_xsens_render_pose_preserves_supplied_global_orientations() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())
    source_names = tuple(str(body.metadata["xsens:sourceSegmentName"]) for body in model.bodies)
    positions = np.asarray([body.reference_pose.translation_m for body in model.bodies], dtype=float)
    quaternions = np.tile(np.array([0.5, 0.5, 0.5, 0.5]), (len(model.bodies), 1))

    body_poses = body_poses_from_xsens_pose(
        model,
        KinematicPose(source_names, positions, quaternions),
    )

    assert set(body_poses) == {body.name for body in model.bodies}
    for pose in body_poses.values():
        np.testing.assert_allclose(pose.rotation_wxyz, [0.5, 0.5, 0.5, 0.5])


def test_orientation_correspondence_selection_uses_xsens_semantics() -> None:
    model = build_g1_proportioned_xsens_tree(extract_g1_anthropometry())

    selected = orientation_correspondence_body_names(model, ["L5", "Left Hand", "Right Hand"])

    assert selected == {"L5", "LeftHand", "RightHand"}
