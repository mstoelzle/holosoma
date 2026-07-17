"""Tests for the Xsens retargeting morphology-selection seam."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from holosoma_retargeting.config_types.retargeting import XsensMorphologyConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import XSENS_BODY_SEGMENT_NAMES, XsensHdf5Motion
from holosoma_retargeting.examples import robot_retarget
from holosoma_retargeting.kinematics import KinematicMotion


def _motion() -> XsensHdf5Motion:
    body_count = len(XSENS_BODY_SEGMENT_NAMES)
    quaternions = np.zeros((1, body_count, 4), dtype=float)
    quaternions[..., 0] = 1.0
    return XsensHdf5Motion(
        positions_m=np.arange(body_count * 3, dtype=float).reshape(1, body_count, 3),
        times_s=np.array([0.0]),
        stream_name="body_position_xyz_m",
        segment_names=list(XSENS_BODY_SEGMENT_NAMES),
        source_indices=list(range(body_count)),
        quaternions_wijk=quaternions,
        orientation_stream_name="body_orientation_quaternion_wijk",
    )


def test_xsens_morphology_defaults_to_g1_proportioned_dynamic_grounding() -> None:
    config = XsensMorphologyConfig()

    assert config.mode == "g1_proportioned"
    assert config.grounding == "match_lowest_soles"
    assert config.preserve_joint_offsets is False
    assert config.g1_model_path is None


def test_direct_mode_preserves_raw_positions_and_human_scale() -> None:
    motion = _motion()

    positions, scale = robot_retarget.prepare_xsens_motion_for_retargeting(
        motion,
        hdf5_path=Path("recording.hdf5"),
        direct_scale=0.75,
        robot_config=RobotConfig(robot_type="g1"),
        morphology_config=XsensMorphologyConfig(mode="direct"),
    )

    np.testing.assert_array_equal(positions, motion.positions_m)
    assert not np.shares_memory(positions, motion.positions_m)
    assert scale == 0.75


def test_g1_mode_uses_robot_xml_and_disables_uniform_rescaling(monkeypatch) -> None:
    motion = _motion()
    adapted_positions = motion.positions_m + 10.0
    captured: dict[str, object] = {}

    def fake_adapt(input_motion, **kwargs):
        captured.update(kwargs)
        return KinematicMotion(
            tuple(input_motion.segment_names),
            adapted_positions,
            input_motion.quaternions_wijk,
            input_motion.times_s,
        )

    monkeypatch.setattr(robot_retarget, "adapt_xsens_motion_to_g1", fake_adapt)
    positions, scale = robot_retarget.prepare_xsens_motion_for_retargeting(
        motion,
        hdf5_path=Path("recording.hdf5"),
        direct_scale=0.75,
        robot_config=RobotConfig(robot_type="g1", robot_urdf_file="models/g1/custom.urdf"),
        morphology_config=XsensMorphologyConfig(),
    )

    np.testing.assert_array_equal(positions, adapted_positions)
    assert scale == 1.0
    assert captured["g1_model_path"] == Path("models/g1/custom.xml")
    assert captured["grounding"] == "match_lowest_soles"


def test_g1_mode_rejects_unsupported_task_or_robot() -> None:
    with pytest.raises(ValueError, match="robot_only"):
        robot_retarget.validate_xsens_morphology_selection(
            task_type="object_interaction",
            data_format="xsens",
            robot="g1",
            config=XsensMorphologyConfig(),
        )
    with pytest.raises(ValueError, match="robot='g1'"):
        robot_retarget.validate_xsens_morphology_selection(
            task_type="robot_only",
            data_format="xsens",
            robot="t1",
            config=XsensMorphologyConfig(),
        )

    robot_retarget.validate_xsens_morphology_selection(
        task_type="robot_only",
        data_format="xsens",
        robot="t1",
        config=XsensMorphologyConfig(mode="direct"),
    )
