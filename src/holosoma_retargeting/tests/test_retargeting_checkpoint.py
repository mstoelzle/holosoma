"""Tests for retargeting checkpoint persistence and validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from holosoma_retargeting.retargeting_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    atomic_savez,
    checkpoint_path_for_result,
    checkpoint_payload,
    load_retargeting_checkpoint,
)
from holosoma_retargeting.xsens.tennis_racket import (
    TennisRacketMotion,
    load_tennis_racket_attachment,
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _payload(*, frame_count: int = 2, total_frames: int = 5) -> dict[str, np.ndarray]:
    return checkpoint_payload(
        total_frames=total_frames,
        qpos=np.arange(frame_count * 4, dtype=float).reshape(frame_count, 4),
        cost=1.25,
        orientation_errors_rad=np.empty((0,), dtype=float),
        axis_errors_deg=np.empty((0,), dtype=float),
        racket_motion=None,
    )


def test_checkpoint_path_is_a_separate_npz_sidecar() -> None:
    assert checkpoint_path_for_result("results/sequence.npz") == Path("results/sequence.checkpoint.npz")
    assert checkpoint_path_for_result("results/sequence") == Path("results/sequence.checkpoint.npz")


def test_atomic_savez_replaces_complete_archive(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoint.npz"
    atomic_savez(destination, {"value": np.asarray(1)})
    atomic_savez(destination, {"value": np.asarray(2)})

    assert int(_load(destination)["value"]) == 2
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_savez_retains_previous_archive_after_failed_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "checkpoint.npz"
    atomic_savez(destination, {"value": np.asarray(1)})

    def fail_savez(*_args, **_kwargs) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("holosoma_retargeting.retargeting_checkpoint.np.savez", fail_savez)
    with pytest.raises(OSError, match="simulated write failure"):
        atomic_savez(destination, {"value": np.asarray(2)})

    assert int(_load(destination)["value"]) == 1
    assert list(tmp_path.glob(".*.tmp")) == []


def test_checkpoint_round_trip_without_optional_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "motion.checkpoint.npz"
    atomic_savez(path, _payload())

    checkpoint = load_retargeting_checkpoint(
        path,
        total_frames=5,
        nq=4,
        has_orientation_targets=False,
        orientation_error_count=0,
        axis_error_count=0,
        racket_tracking_mode=None,
    )

    assert checkpoint.completed_frames == 2
    assert checkpoint.cost == 1.25
    np.testing.assert_array_equal(checkpoint.qpos, np.arange(8, dtype=float).reshape(2, 4))
    assert checkpoint.racket_motion is None


def test_checkpoint_preserves_zero_width_orientation_histories(tmp_path: Path) -> None:
    payload = _payload()
    payload["orientation_errors_rad"] = np.empty((2, 0), dtype=float)
    payload["axis_errors_deg"] = np.empty((2, 0), dtype=float)
    path = tmp_path / "empty-targets.checkpoint.npz"
    atomic_savez(path, payload)

    checkpoint = load_retargeting_checkpoint(
        path,
        total_frames=5,
        nq=4,
        has_orientation_targets=True,
        orientation_error_count=0,
        axis_error_count=0,
        racket_tracking_mode=None,
    )

    assert checkpoint.orientation_errors_rad.shape == (2, 0)
    assert checkpoint.axis_errors_deg.shape == (2, 0)


def test_checkpoint_round_trip_with_diagnostics_and_racket(tmp_path: Path) -> None:
    frame_count = 2
    racket_motion = TennisRacketMotion(
        position_m=np.zeros((frame_count, 3)),
        quaternion_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (frame_count, 1)),
        tracking_state=np.array(["racket", "reentry_hysteresis"]),
        symmetry_branch=np.array([0, -1]),
        target_error_rad=np.array([0.1, 0.2]),
        source_origin_deviation_m=np.array([0.0, 0.01]),
        min_wrist_limit_margin_rad=np.array([0.2, 0.3]),
        attachment=load_tennis_racket_attachment(),
        tracking_mode="filtered",
    )
    payload = checkpoint_payload(
        total_frames=4,
        qpos=np.zeros((frame_count, 6)),
        cost=0.5,
        orientation_errors_rad=np.zeros((frame_count, 3)),
        axis_errors_deg=np.zeros((frame_count, 2)),
        racket_motion=racket_motion,
        racket_active=True,
        racket_reentry_streak=2,
        racket_previous_branch=0,
    )
    path = tmp_path / "racket.checkpoint.npz"
    atomic_savez(path, payload)

    checkpoint = load_retargeting_checkpoint(
        path,
        total_frames=4,
        nq=6,
        has_orientation_targets=True,
        orientation_error_count=3,
        axis_error_count=2,
        racket_tracking_mode="filtered",
    )

    assert checkpoint.racket_motion is not None
    assert checkpoint.racket_active
    assert checkpoint.racket_reentry_streak == 2
    assert checkpoint.racket_previous_branch == 0
    np.testing.assert_array_equal(checkpoint.racket_motion.symmetry_branch, [0, -1])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.__setitem__(
                "checkpoint_schema_version",
                np.asarray(CHECKPOINT_SCHEMA_VERSION + 1),
            ),
            "Unsupported checkpoint schema version",
        ),
        (
            lambda payload: payload.__setitem__("checkpoint_total_frames", np.asarray(6)),
            "does not match current motion",
        ),
        (
            lambda payload: payload.__setitem__("qpos", np.zeros((2, 3))),
            "Checkpoint qpos must have shape",
        ),
        (
            lambda payload: payload.__setitem__("qpos", np.full((2, 4), np.nan)),
            "Checkpoint qpos must contain only finite values",
        ),
    ],
)
def test_checkpoint_rejects_incompatible_state(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _payload()
    mutation(payload)
    path = tmp_path / "invalid.checkpoint.npz"
    atomic_savez(path, payload)

    with pytest.raises(ValueError, match=message):
        load_retargeting_checkpoint(
            path,
            total_frames=5,
            nq=4,
            has_orientation_targets=False,
            orientation_error_count=0,
            axis_error_count=0,
            racket_tracking_mode=None,
        )


def test_checkpoint_rejects_corrupt_archive(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.checkpoint.npz"
    path.write_bytes(b"not an npz archive")

    with pytest.raises(ValueError, match="Could not load retargeting checkpoint"):
        load_retargeting_checkpoint(
            path,
            total_frames=5,
            nq=4,
            has_orientation_targets=False,
            orientation_error_count=0,
            axis_error_count=0,
            racket_tracking_mode=None,
        )
