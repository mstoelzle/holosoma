from __future__ import annotations

import h5py
import numpy as np

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.data_utils.xsens_hdf5 import (
    XSENS_BODY_SEGMENT_NAMES,
    load_xsens_hdf5_positions,
    load_xsens_hdf5_tpose,
    resolve_xsens_hdf5_path,
)


def _headings(segment_names: list[str]) -> list[str]:
    return [f"{segment_name} ({axis})" for segment_name in segment_names for axis in ("x", "y", "z")]


def _write_stream(
    hdf5_file: h5py.File,
    stream_name: str,
    data: np.ndarray,
    times_s: np.ndarray,
    segment_names: list[str] | None = None,
) -> None:
    stream_group = hdf5_file.require_group("xsens-segments").create_group(stream_name)
    stream_group.create_dataset("data", data=data)
    stream_group.create_dataset("time_s", data=times_s.reshape(-1, 1))
    if segment_names is not None:
        stream_group.attrs["Data headings"] = str(_headings(segment_names))


def test_motion_data_config_registers_xsens_for_g1() -> None:
    cfg = MotionDataConfig(data_format="xsens", robot_type="g1")

    assert cfg.resolved_demo_joints == XSENS_BODY_SEGMENT_NAMES
    assert cfg.toe_names == ["Left Toe", "Right Toe"]
    assert cfg.default_human_height == 1.78
    assert cfg.target_fps == 30.0
    assert cfg.frame_start == 0
    assert cfg.max_frames is None
    assert cfg.resolved_joints_mapping["Pelvis"] == "pelvis_contour_link"
    assert cfg.resolved_joints_mapping["Right Hand"] == "right_rubber_hand_link"


def test_loader_prefers_meter_stream_and_uses_headings_to_ignore_extra_segments(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(5, dtype=float) / 60.0
    segment_names = XSENS_BODY_SEGMENT_NAMES[:11] + ["Prop1"] + XSENS_BODY_SEGMENT_NAMES[11:]
    positions_m = np.zeros((5, len(segment_names), 3), dtype=float)
    for frame_idx in range(positions_m.shape[0]):
        for segment_idx in range(positions_m.shape[1]):
            positions_m[frame_idx, segment_idx] = [segment_idx, 100 + frame_idx, 200 + segment_idx]

    ignored_positions_cm = np.full((5, len(XSENS_BODY_SEGMENT_NAMES), 3), 999.0, dtype=float)

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "position_cm", ignored_positions_cm, times_s, XSENS_BODY_SEGMENT_NAMES)
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, segment_names)

    motion = load_xsens_hdf5_positions(hdf5_path, target_fps=30.0)

    assert motion.stream_name == "body_position_xyz_m"
    assert motion.positions_m.shape == (3, len(XSENS_BODY_SEGMENT_NAMES), 3)
    assert motion.times_s.tolist() == [times_s[0], times_s[2], times_s[4]]
    assert motion.segment_names == XSENS_BODY_SEGMENT_NAMES

    left_shoulder_source_idx = segment_names.index("Left Shoulder")
    left_shoulder_body_idx = XSENS_BODY_SEGMENT_NAMES.index("Left Shoulder")
    expected = positions_m[2, left_shoulder_source_idx]
    np.testing.assert_allclose(motion.positions_m[1, left_shoulder_body_idx], expected)


def test_loader_uses_segment_names_body_metadata_and_drops_sword_segment(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(3, dtype=float) / 30.0
    metadata_segment_names = [name.replace(" ", "") for name in XSENS_BODY_SEGMENT_NAMES] + ["RightHandSword"]
    positions_m = np.zeros((3, len(metadata_segment_names), 3), dtype=float)
    left_toe_source_idx = metadata_segment_names.index("LeftToe")
    positions_m[0, left_toe_source_idx] = [1.0, 2.0, 3.0]
    positions_m[0, -1] = [999.0, 999.0, 999.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        stream_group = hdf5_file.require_group("xsens-segments").create_group("body_position_xyz_m")
        stream_group.create_dataset("data", data=positions_m)
        stream_group.create_dataset("time_s", data=times_s.reshape(-1, 1))
        stream_group.attrs["segment_names_body"] = str(metadata_segment_names)

    motion = load_xsens_hdf5_positions(hdf5_path, target_fps=30.0)

    left_toe_body_idx = XSENS_BODY_SEGMENT_NAMES.index("Left Toe")
    assert motion.positions_m.shape == (3, len(XSENS_BODY_SEGMENT_NAMES), 3)
    assert motion.source_indices[-1] == left_toe_source_idx
    np.testing.assert_allclose(motion.positions_m[0, left_toe_body_idx], [1.0, 2.0, 3.0])
    assert not np.any(motion.positions_m == 999.0)


def test_loader_falls_back_to_cm_stream_without_headings(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.h5"
    times_s = np.arange(3, dtype=float) / 30.0
    positions_cm = np.zeros((3, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    positions_cm[0, 0] = [100.0, 200.0, 300.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "position_cm", positions_cm, times_s)

    motion = load_xsens_hdf5_positions(hdf5_path, target_fps=30.0)

    assert motion.stream_name == "position_cm"
    assert motion.source_indices == list(range(len(XSENS_BODY_SEGMENT_NAMES)))
    np.testing.assert_allclose(motion.positions_m[0, 0], [1.0, 3.0, 2.0])


def test_loader_applies_frame_window_after_time_sampling(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_sample.hdf5"
    times_s = np.arange(7, dtype=float) / 60.0
    positions_m = np.zeros((7, len(XSENS_BODY_SEGMENT_NAMES), 3), dtype=float)
    for frame_idx in range(positions_m.shape[0]):
        positions_m[frame_idx, 0] = [frame_idx, 0.0, 0.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        _write_stream(hdf5_file, "body_position_xyz_m", positions_m, times_s, XSENS_BODY_SEGMENT_NAMES)

    motion = load_xsens_hdf5_positions(hdf5_path, target_fps=30.0, frame_start=1, max_frames=2)

    assert motion.times_s.tolist() == [times_s[2], times_s[4]]
    np.testing.assert_allclose(motion.positions_m[:, 0, 0], [2.0, 4.0])


def test_resolve_xsens_hdf5_path_accepts_task_stems_and_explicit_files(tmp_path) -> None:
    hdf5_path = tmp_path / "tennis_motion.hdf5"
    hdf5_path.touch()

    assert resolve_xsens_hdf5_path(tmp_path, "tennis_motion") == hdf5_path
    assert resolve_xsens_hdf5_path(tmp_path, "tennis_motion.hdf5") == hdf5_path


def test_tpose_loader_selects_body_segments_and_ignores_extra_segment(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_tpose.hdf5"
    segment_names = XSENS_BODY_SEGMENT_NAMES[:5] + ["RightHandSword"] + XSENS_BODY_SEGMENT_NAMES[5:]
    positions_m = np.zeros((len(segment_names), 3), dtype=float)
    quaternions_wijk = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (len(segment_names), 1))
    head_source_idx = segment_names.index("Head")
    positions_m[head_source_idx] = [1.0, 2.0, 3.0]
    positions_m[segment_names.index("RightHandSword")] = [999.0, 999.0, 999.0]

    with h5py.File(hdf5_path, "w") as hdf5_file:
        group = hdf5_file.create_group("xsens-segments-tpose")
        group.attrs["segment_names_body"] = str(segment_names)
        group.create_dataset("body_position_Tpose_xyz_m", data=positions_m)
        group.create_dataset("body_orientation_Tpose_quaternion_wijk", data=quaternions_wijk)

    tpose = load_xsens_hdf5_tpose(hdf5_path)

    assert tpose.variant == "Tpose"
    assert tpose.positions_m.shape == (len(XSENS_BODY_SEGMENT_NAMES), 3)
    assert tpose.quaternions_wijk.shape == (len(XSENS_BODY_SEGMENT_NAMES), 4)
    assert tpose.segment_names == XSENS_BODY_SEGMENT_NAMES
    assert tpose.source_indices[XSENS_BODY_SEGMENT_NAMES.index("Head")] == head_source_idx
    np.testing.assert_allclose(tpose.positions_m[XSENS_BODY_SEGMENT_NAMES.index("Head")], [1.0, 2.0, 3.0])
    assert not np.any(tpose.positions_m == 999.0)
    np.testing.assert_allclose(np.linalg.norm(tpose.quaternions_wijk, axis=1), 1.0)


def test_tpose_loader_errors_on_missing_variant(tmp_path) -> None:
    hdf5_path = tmp_path / "xsens_tpose.hdf5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_group("xsens-segments-tpose")

    try:
        load_xsens_hdf5_tpose(hdf5_path, variant="Tpose")
    except KeyError as exc:
        assert "body_position_Tpose_xyz_m" in str(exc)
    else:
        raise AssertionError("Expected a KeyError for missing T-pose datasets")
