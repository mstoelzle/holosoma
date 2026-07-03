"""Utilities for loading ActionNet-style Xsens HDF5 segment positions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


XSENS_DEVICE_NAME = "xsens-segments"
XSENS_POSITION_STREAM_NAMES = ("body_position_xyz_m", "position_cm")

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


def transform_xsens_y_up_to_retargeting_z_up(positions: np.ndarray) -> np.ndarray:
    """Convert Xsens coordinates `[x, y_up, z]` to retargeting `[x, y, z_up]`."""
    return positions[..., [0, 2, 1]]


def transform_xsens_stream_to_retargeting(positions: np.ndarray, stream_name: str) -> np.ndarray:
    """Convert a known Xsens position stream into retargeting coordinates."""
    if stream_name == "body_position_xyz_m":
        return positions
    return transform_xsens_y_up_to_retargeting_z_up(positions)


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
    h5py = _import_h5py()
    path = Path(path)

    with h5py.File(path, "r") as hdf5_file:
        stream_name, stream_group = _get_position_stream(hdf5_file)
        headings = parse_data_headings(stream_group.attrs.get("Data headings"))
        segment_names = segment_names_from_headings(headings)
        if segment_names is None:
            segment_names = parse_segment_names(stream_group.attrs.get("segment_names_body"))

        positions = np.asarray(stream_group["data"], dtype=float)
        times_s = np.asarray(stream_group["time_s"], dtype=float).reshape(-1)

    positions = _reshape_position_data(positions, segment_names)
    selected_segment_names, source_indices = _body_segment_indices(segment_names, positions.shape[1])
    positions = positions[:, source_indices, :] * _unit_scale_for_stream(stream_name)
    positions = transform_xsens_stream_to_retargeting(positions, stream_name)

    sample_indices = sample_indices_by_time(times_s, target_fps)
    sample_indices = _slice_sample_indices(sample_indices, frame_start, max_frames)
    return XsensHdf5Motion(
        positions_m=positions[sample_indices],
        times_s=times_s[sample_indices],
        stream_name=stream_name,
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
