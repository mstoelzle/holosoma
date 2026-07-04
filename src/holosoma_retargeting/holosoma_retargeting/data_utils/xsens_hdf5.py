"""Utilities for loading ActionNet-style Xsens HDF5 segment data."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]


XSENS_DEVICE_NAME = "xsens-segments"
XSENS_TPOSE_DEVICE_NAME = "xsens-segments-tpose"
XSENS_POSITION_STREAM_NAMES = ("body_position_xyz_m", "position_cm")
XSENS_ORIENTATION_STREAM_NAME = "body_orientation_quaternion_wijk"
XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

XSENS_BODY_SEGMENT_NAMES = [
    "Pelvis",
    "L5",
    "L3",
    "T12",
    "T8",
    "Neck",
    "Head",
    "Right Shoulder",
    "Right Upper Arm",
    "Right Forearm",
    "Right Hand",
    "Left Shoulder",
    "Left Upper Arm",
    "Left Forearm",
    "Left Hand",
    "Right Upper Leg",
    "Right Lower Leg",
    "Right Foot",
    "Right Toe",
    "Left Upper Leg",
    "Left Lower Leg",
    "Left Foot",
    "Left Toe",
]


@dataclass(frozen=True)
class XsensHdf5Motion:
    """Loaded Xsens segment positions in retargeting coordinates."""

    positions_m: np.ndarray
    times_s: np.ndarray
    stream_name: str
    segment_names: list[str]
    source_indices: list[int]
    quaternions_wijk: np.ndarray | None = None
    orientation_stream_name: str | None = None


@dataclass(frozen=True)
class XsensHdf5Tpose:
    """Loaded Xsens static T-pose in retargeting coordinates."""

    positions_m: np.ndarray
    quaternions_wijk: np.ndarray
    variant: str
    segment_names: list[str]
    source_indices: list[int]


def _import_h5py():
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("Loading Xsens HDF5 files requires the optional dependency 'h5py'.") from exc
    return h5py


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def parse_data_headings(data_headings_attr: Any) -> list[str] | None:
    """Parse an ActionNet `Data headings` attribute into heading strings."""
    if data_headings_attr is None:
        return None

    data_headings_attr = _decode_attr(data_headings_attr)
    if isinstance(data_headings_attr, np.ndarray):
        data_headings_attr = data_headings_attr.tolist()
    if isinstance(data_headings_attr, list):
        return [str(item) for item in data_headings_attr]
    if not isinstance(data_headings_attr, str):
        return None

    parsed = ast.literal_eval(data_headings_attr)
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def segment_names_from_headings(headings: list[str] | None) -> list[str] | None:
    """Extract segment names from flattened `Segment (axis)` headings."""
    if headings is None:
        return None
    if len(headings) % 3 != 0:
        return None

    segment_names: list[str] = []
    for i in range(0, len(headings), 3):
        names = []
        for heading in headings[i : i + 3]:
            if "(" in heading:
                names.append(heading.rsplit("(", 1)[0].strip())
            else:
                names.append(heading.strip())
        if len(set(names)) != 1:
            return None
        segment_names.append(names[0])
    return segment_names


def parse_segment_names(segment_names_attr: Any) -> list[str] | None:
    """Parse an ActionNet `segment_names_body` attribute into segment names."""
    if segment_names_attr is None:
        return None

    segment_names_attr = _decode_attr(segment_names_attr)
    if isinstance(segment_names_attr, np.ndarray):
        segment_names_attr = segment_names_attr.tolist()
    if isinstance(segment_names_attr, list):
        return [str(item) for item in segment_names_attr]
    if not isinstance(segment_names_attr, str):
        return None

    parsed = ast.literal_eval(segment_names_attr)
    if not isinstance(parsed, list):
        return None
    return [str(item) for item in parsed]


def _segment_names_from_attrs(hdf5_obj: Any) -> list[str] | None:
    headings = parse_data_headings(hdf5_obj.attrs.get("Data headings"))
    segment_names = segment_names_from_headings(headings)
    if segment_names is None:
        segment_names = parse_segment_names(hdf5_obj.attrs.get("segment_names_body"))
    return segment_names


def transform_xsens_y_up_to_retargeting_z_up(positions: np.ndarray) -> np.ndarray:
    """Convert Xsens coordinates `[x, y_up, z]` to retargeting `[x, y, z_up]`."""
    return positions[..., [0, 2, 1]]


def transform_xsens_stream_to_retargeting(positions: np.ndarray, stream_name: str) -> np.ndarray:
    """Convert a known Xsens position stream into retargeting coordinates."""
    if stream_name == "body_position_xyz_m":
        return positions
    return transform_xsens_y_up_to_retargeting_z_up(positions)


def _quat_wijk_to_matrix(quaternions_wijk: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions_wijk, dtype=float)
    xyzw = np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)
    return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(q.shape[:-1] + (3, 3))


def _matrix_to_quat_wijk(rotations: np.ndarray) -> np.ndarray:
    rotations = np.asarray(rotations, dtype=float)
    xyzw = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(rotations.shape[:-2] + (4,))
    return np.stack([xyzw[..., 3], xyzw[..., 0], xyzw[..., 1], xyzw[..., 2]], axis=-1)


def transform_xsens_orientation_stream_to_retargeting(
    quaternions_wijk: np.ndarray,
    stream_name: str,
) -> np.ndarray:
    """Convert Xsens world-frame orientations to the position stream's retargeting convention."""
    if stream_name == "body_position_xyz_m":
        return quaternions_wijk
    rotations = _quat_wijk_to_matrix(quaternions_wijk)
    basis = XSENS_Y_UP_TO_RETARGETING_Z_UP_MATRIX
    return _matrix_to_quat_wijk(basis @ rotations @ basis.T)


def _get_position_stream(hdf5_file: Any) -> tuple[str, Any]:
    if XSENS_DEVICE_NAME not in hdf5_file:
        raise KeyError(f"No {XSENS_DEVICE_NAME} group found in the HDF5 file")

    streams = hdf5_file[XSENS_DEVICE_NAME]
    for stream_name in XSENS_POSITION_STREAM_NAMES:
        if stream_name not in streams:
            continue
        stream_group = streams[stream_name]
        if "data" in stream_group and "time_s" in stream_group:
            return stream_name, stream_group

    expected = ", ".join(XSENS_POSITION_STREAM_NAMES)
    raise KeyError(f"No Xsens segment position stream found. Expected one of: {expected}")


def _get_orientation_stream(hdf5_file: Any) -> Any:
    if XSENS_DEVICE_NAME not in hdf5_file:
        raise KeyError(f"No {XSENS_DEVICE_NAME} group found in the HDF5 file")
    streams = hdf5_file[XSENS_DEVICE_NAME]
    if XSENS_ORIENTATION_STREAM_NAME not in streams:
        raise KeyError(f"No Xsens segment orientation stream found: {XSENS_ORIENTATION_STREAM_NAME}")
    stream_group = streams[XSENS_ORIENTATION_STREAM_NAME]
    if "data" not in stream_group or "time_s" not in stream_group:
        raise KeyError(f"Xsens orientation stream is missing required 'data' or 'time_s': {XSENS_ORIENTATION_STREAM_NAME}")
    return stream_group


def _reshape_position_data(position_data: np.ndarray, segment_names: list[str] | None) -> np.ndarray:
    if position_data.ndim == 3 and position_data.shape[-1] == 3:
        return position_data
    if position_data.ndim == 2 and position_data.shape[1] % 3 == 0:
        return position_data.reshape(position_data.shape[0], -1, 3)

    if segment_names is None:
        raise ValueError(f"Unsupported Xsens position data shape: {position_data.shape}")
    expected_flat_width = len(segment_names) * 3
    if position_data.ndim == 2 and position_data.shape[1] == expected_flat_width:
        return position_data.reshape(position_data.shape[0], len(segment_names), 3)

    raise ValueError(f"Unsupported Xsens position data shape: {position_data.shape}")


def _reshape_static_position_data(position_data: np.ndarray, segment_names: list[str] | None) -> np.ndarray:
    if position_data.ndim == 2 and position_data.shape[-1] == 3:
        return position_data
    if position_data.ndim == 3 and position_data.shape[-1] == 3 and position_data.shape[0] == 1:
        return position_data[0]
    if position_data.ndim == 1 and position_data.shape[0] % 3 == 0:
        return position_data.reshape(-1, 3)

    if segment_names is None:
        raise ValueError(f"Unsupported Xsens static position data shape: {position_data.shape}")
    expected_flat_width = len(segment_names) * 3
    if position_data.ndim == 1 and position_data.shape[0] == expected_flat_width:
        return position_data.reshape(len(segment_names), 3)

    raise ValueError(f"Unsupported Xsens static position data shape: {position_data.shape}")


def _reshape_quaternion_data(quaternion_data: np.ndarray, segment_names: list[str] | None) -> np.ndarray:
    if quaternion_data.ndim == 3 and quaternion_data.shape[-1] == 4:
        return quaternion_data
    if quaternion_data.ndim == 2 and quaternion_data.shape[-1] == 4:
        return quaternion_data
    if quaternion_data.ndim == 3 and quaternion_data.shape[-1] == 4 and quaternion_data.shape[0] == 1:
        return quaternion_data[0]
    if quaternion_data.ndim == 1 and quaternion_data.shape[0] % 4 == 0:
        return quaternion_data.reshape(-1, 4)
    if quaternion_data.ndim == 2 and quaternion_data.shape[1] % 4 == 0:
        return quaternion_data.reshape(quaternion_data.shape[0], -1, 4)

    if segment_names is None:
        raise ValueError(f"Unsupported Xsens quaternion data shape: {quaternion_data.shape}")
    expected_flat_width = len(segment_names) * 4
    if quaternion_data.ndim == 2 and quaternion_data.shape[1] == expected_flat_width:
        return quaternion_data.reshape(quaternion_data.shape[0], len(segment_names), 4)
    if quaternion_data.ndim == 1 and quaternion_data.shape[0] == expected_flat_width:
        return quaternion_data.reshape(len(segment_names), 4)

    raise ValueError(f"Unsupported Xsens quaternion data shape: {quaternion_data.shape}")


def _normalize_quaternions_wijk(quaternions: np.ndarray) -> np.ndarray:
    quat_norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    return quaternions / np.maximum(quat_norms, 1e-12)


def _normalize_segment_name(name: str) -> str:
    return "".join(name.lower().split())


def _body_segment_indices(segment_names: list[str] | None, n_segments: int) -> tuple[list[str], list[int]]:
    if segment_names is None:
        if n_segments < len(XSENS_BODY_SEGMENT_NAMES):
            raise ValueError(
                f"Expected at least {len(XSENS_BODY_SEGMENT_NAMES)} Xsens body segments, got {n_segments}"
            )
        if n_segments == len(XSENS_BODY_SEGMENT_NAMES) + 1:
            # Common tennis recording layout: body segments plus RightHandSword after the body list.
            return XSENS_BODY_SEGMENT_NAMES, list(range(len(XSENS_BODY_SEGMENT_NAMES)))
        return XSENS_BODY_SEGMENT_NAMES, list(range(len(XSENS_BODY_SEGMENT_NAMES)))

    normalized_to_index = {_normalize_segment_name(name): i for i, name in enumerate(segment_names)}
    missing = [name for name in XSENS_BODY_SEGMENT_NAMES if _normalize_segment_name(name) not in normalized_to_index]
    if missing:
        raise ValueError(f"Xsens stream is missing body segments: {missing}")
    source_indices = [normalized_to_index[_normalize_segment_name(name)] for name in XSENS_BODY_SEGMENT_NAMES]
    return XSENS_BODY_SEGMENT_NAMES, source_indices


def _unit_scale_for_stream(stream_name: str) -> float:
    if stream_name.endswith("_m"):
        return 1.0
    if stream_name.endswith("_cm"):
        return 0.01
    return 1.0


def sample_indices_by_time(times_s: np.ndarray, target_fps: float | None) -> np.ndarray:
    """Sample monotonically increasing timestamps at `target_fps` using previous-frame hold."""
    times_s = np.asarray(times_s, dtype=float).reshape(-1)
    if times_s.size == 0:
        raise ValueError("Xsens stream has no timestamps")
    if target_fps is None:
        return np.arange(times_s.size)
    if target_fps <= 0:
        raise ValueError("target_fps must be positive")

    period_s = 1.0 / target_fps
    sample_times_s = np.arange(times_s[0], times_s[-1] + period_s * 0.5, period_s)
    indices = np.searchsorted(times_s, sample_times_s, side="right") - 1
    indices = np.clip(indices, 0, times_s.size - 1)
    return np.unique(indices)


def _slice_sample_indices(sample_indices: np.ndarray, frame_start: int, max_frames: int | None) -> np.ndarray:
    if frame_start < 0:
        raise ValueError("frame_start must be non-negative")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")

    sample_indices = sample_indices[frame_start:]
    if max_frames is not None:
        sample_indices = sample_indices[:max_frames]
    if sample_indices.size == 0:
        raise ValueError("Selected Xsens frame window is empty")
    return sample_indices


def load_xsens_hdf5_positions(
    path: str | Path,
    target_fps: float | None = 30.0,
    frame_start: int = 0,
    max_frames: int | None = None,
) -> XsensHdf5Motion:
    """Load Xsens body segment positions from an ActionNet-style HDF5 file."""
    return load_xsens_hdf5_motion(
        path,
        target_fps=target_fps,
        frame_start=frame_start,
        max_frames=max_frames,
        include_orientations=False,
    )


def load_xsens_hdf5_motion(
    path: str | Path,
    target_fps: float | None = 30.0,
    frame_start: int = 0,
    max_frames: int | None = None,
    include_orientations: bool = False,
) -> XsensHdf5Motion:
    """Load Xsens body segment positions and, optionally, segment orientations."""
    h5py = _import_h5py()
    path = Path(path)
    quaternions: np.ndarray | None = None

    with h5py.File(path, "r") as hdf5_file:
        stream_name, stream_group = _get_position_stream(hdf5_file)
        segment_names = _segment_names_from_attrs(stream_group)

        positions = np.asarray(stream_group["data"], dtype=float)
        times_s = np.asarray(stream_group["time_s"], dtype=float).reshape(-1)
        orientation_segment_names = None
        orientation_times_s = None
        if include_orientations:
            orientation_group = _get_orientation_stream(hdf5_file)
            orientation_segment_names = _segment_names_from_attrs(orientation_group)
            quaternions = np.asarray(orientation_group["data"], dtype=float)
            orientation_times_s = np.asarray(orientation_group["time_s"], dtype=float).reshape(-1)

    positions = _reshape_position_data(positions, segment_names)
    selected_segment_names, source_indices = _body_segment_indices(segment_names, positions.shape[1])
    positions = positions[:, source_indices, :] * _unit_scale_for_stream(stream_name)
    positions = transform_xsens_stream_to_retargeting(positions, stream_name)

    sample_indices = sample_indices_by_time(times_s, target_fps)
    sample_indices = _slice_sample_indices(sample_indices, frame_start, max_frames)
    if include_orientations:
        if quaternions is None or orientation_times_s is None:
            raise RuntimeError("Internal error: orientation stream was not loaded")
        quaternions = _reshape_quaternion_data(quaternions, orientation_segment_names)
        selected_orientation_names, orientation_source_indices = _body_segment_indices(
            orientation_segment_names, quaternions.shape[1]
        )
        if selected_orientation_names != selected_segment_names:
            raise ValueError("Xsens position and orientation body segment selections differ")
        if source_indices != orientation_source_indices:
            raise ValueError("Xsens position and orientation source segment indices differ")
        if orientation_times_s.shape != times_s.shape or not np.allclose(orientation_times_s, times_s):
            raise ValueError("Xsens position and orientation streams must share timestamps for orientation tracking")
        quaternions = _normalize_quaternions_wijk(quaternions[:, orientation_source_indices, :])
        quaternions = transform_xsens_orientation_stream_to_retargeting(quaternions, stream_name)

    return XsensHdf5Motion(
        positions_m=positions[sample_indices],
        times_s=times_s[sample_indices],
        stream_name=stream_name,
        segment_names=selected_segment_names,
        source_indices=source_indices,
        quaternions_wijk=None if quaternions is None else quaternions[sample_indices],
        orientation_stream_name=XSENS_ORIENTATION_STREAM_NAME if include_orientations else None,
    )


def load_xsens_hdf5_tpose(path: str | Path, variant: str = "Tpose") -> XsensHdf5Tpose:
    """Load Xsens static T-pose body segment positions and orientations.

    The ActionNet tennis files store T-pose data as direct datasets under
    `xsens-segments-tpose`, not as time series. Quaternions are returned in the
    source `w, i, j, k` order.
    """
    h5py = _import_h5py()
    path = Path(path)
    position_dataset_name = f"body_position_{variant}_xyz_m"
    quaternion_dataset_name = f"body_orientation_{variant}_quaternion_wijk"

    with h5py.File(path, "r") as hdf5_file:
        if XSENS_TPOSE_DEVICE_NAME not in hdf5_file:
            raise KeyError(f"No {XSENS_TPOSE_DEVICE_NAME} group found in the HDF5 file")

        tpose_group = hdf5_file[XSENS_TPOSE_DEVICE_NAME]
        if position_dataset_name not in tpose_group:
            raise KeyError(f"No Xsens T-pose position dataset found: {position_dataset_name}")
        if quaternion_dataset_name not in tpose_group:
            raise KeyError(f"No Xsens T-pose quaternion dataset found: {quaternion_dataset_name}")

        position_dataset = tpose_group[position_dataset_name]
        quaternion_dataset = tpose_group[quaternion_dataset_name]
        segment_names = _segment_names_from_attrs(position_dataset)
        if segment_names is None:
            segment_names = _segment_names_from_attrs(quaternion_dataset)
        if segment_names is None:
            segment_names = _segment_names_from_attrs(tpose_group)
        if segment_names is None and XSENS_DEVICE_NAME in hdf5_file:
            xsens_streams = hdf5_file[XSENS_DEVICE_NAME]
            for stream_name in XSENS_POSITION_STREAM_NAMES:
                if stream_name in xsens_streams:
                    segment_names = _segment_names_from_attrs(xsens_streams[stream_name])
                    if segment_names is not None:
                        break

        positions = np.asarray(position_dataset, dtype=float)
        quaternions = np.asarray(quaternion_dataset, dtype=float)

    positions = _reshape_static_position_data(positions, segment_names)
    quaternions = _reshape_quaternion_data(quaternions, segment_names)

    if positions.shape[0] != quaternions.shape[0]:
        raise ValueError(
            "Xsens T-pose position/quaternion segment counts differ: "
            f"{positions.shape[0]} vs {quaternions.shape[0]}"
        )

    selected_segment_names, source_indices = _body_segment_indices(segment_names, positions.shape[0])
    positions = positions[source_indices]
    quaternions = quaternions[source_indices]
    quaternions = _normalize_quaternions_wijk(quaternions)

    return XsensHdf5Tpose(
        positions_m=positions,
        quaternions_wijk=quaternions,
        variant=variant,
        segment_names=selected_segment_names,
        source_indices=source_indices,
    )


def resolve_xsens_hdf5_path(data_path: str | Path, task_name: str) -> Path:
    """Resolve a task name to an Xsens HDF5 file under `data_path`."""
    data_path = Path(data_path)
    candidate = data_path / task_name
    if candidate.is_file() and candidate.suffix.lower() in {".hdf5", ".h5"}:
        return candidate

    for suffix in (".hdf5", ".h5"):
        candidate_with_suffix = data_path / f"{task_name}{suffix}"
        if candidate_with_suffix.is_file():
            return candidate_with_suffix

    raise FileNotFoundError(f"Xsens HDF5 data file not found for task '{task_name}' in {data_path}")
