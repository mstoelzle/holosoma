"""Integration tests for checkpointing in the frame-by-frame retargeting loop."""

from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace

import numpy as np
import pytest
from holosoma_retargeting.retargeting_checkpoint import (
    atomic_savez,
    checkpoint_path_for_result,
    checkpoint_payload,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import InteractionMeshRetargeter


def _retargeter(*, fail_at_frame: int | None = None) -> InteractionMeshRetargeter:
    retargeter = InteractionMeshRetargeter.__new__(InteractionMeshRetargeter)
    retargeter.nq = 10
    retargeter.q_a_indices = np.arange(3, dtype=int)
    retargeter.orientation_config = SimpleNamespace(
        is_enabled=False,
        tennis_racket=SimpleNamespace(mode="hand"),
    )
    retargeter.optimization_schedule = "single-stage"
    retargeter.activate_joint_limits = True
    retargeter.object_name = "ground"
    retargeter.smplh_mapped_joint_indices = np.arange(2, dtype=int)
    retargeter.w_nominal_tracking_init = 1.0
    retargeter.nominal_tracking_tau = 10.0
    retargeter.initial_iterations = 1
    retargeter.iterations_per_frame = 1
    retargeter.debug = False
    retargeter.visualize = False
    retargeter.iterated_frames = []

    def iterate(**kwargs):
        frame_idx = kwargs["frame_idx"]
        retargeter.iterated_frames.append(frame_idx)
        if frame_idx == fail_at_frame:
            raise KeyboardInterrupt("simulated interruption")
        q = np.asarray(kwargs["q_n"], dtype=float).copy()
        q[0] = frame_idx + 1.0
        return q, float(frame_idx + 0.5)

    retargeter.iterate = iterate
    return retargeter


def _inputs(frame_count: int = 5) -> dict[str, object]:
    object_poses = np.zeros((frame_count, 7), dtype=float)
    object_poses[:, 3] = 1.0
    return {
        "human_joint_motions": np.zeros((frame_count, 2, 3), dtype=float),
        "object_poses": object_poses.copy(),
        "object_poses_augmented": object_poses.copy(),
        "object_points_local_demo": np.zeros((1, 3), dtype=float),
        "object_points_local": np.zeros((1, 3), dtype=float),
        "foot_sticking_sequences": [{} for _ in range(frame_count)],
        "q_a_init": np.zeros(3, dtype=float),
    }


@pytest.fixture(autouse=True)
def _stub_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "holosoma_retargeting.src.interaction_mesh_retargeter.create_interaction_mesh",
        lambda vertices: (vertices, np.array([[0, 0, 0, 0]], dtype=int)),
    )
    monkeypatch.setattr(
        "holosoma_retargeting.src.interaction_mesh_retargeter.get_adjacency_list",
        lambda _tetrahedra, vertex_count: [[] for _ in range(vertex_count)],
    )
    monkeypatch.setattr(
        "holosoma_retargeting.src.interaction_mesh_retargeter.calculate_laplacian_coordinates",
        lambda vertices, _adjacency: np.zeros_like(vertices),
    )


def test_interrupted_run_flushes_and_resumes_exactly(tmp_path) -> None:
    inputs = _inputs()
    resumed_path = tmp_path / "resumed.npz"

    interrupted = _retargeter(fail_at_frame=3)
    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        interrupted.retarget_motion(
            **inputs,
            dest_res_path=resumed_path,
            checkpoint_interval_frames=2,
        )

    checkpoint_path = checkpoint_path_for_result(resumed_path)
    assert checkpoint_path.exists()
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        assert checkpoint["qpos"].shape == (3, 10)

    resumed = _retargeter()
    resumed_result = resumed.retarget_motion(
        **inputs,
        dest_res_path=resumed_path,
        checkpoint_interval_frames=2,
        resume=True,
    )[0]
    assert resumed.iterated_frames == [3, 4]
    assert not checkpoint_path.exists()

    uninterrupted_path = tmp_path / "uninterrupted.npz"
    uninterrupted = _retargeter()
    uninterrupted_result = uninterrupted.retarget_motion(
        **inputs,
        dest_res_path=uninterrupted_path,
    )[0]

    np.testing.assert_array_equal(resumed_result, uninterrupted_result)
    with ExitStack() as stack:
        resumed_archive = stack.enter_context(np.load(resumed_path, allow_pickle=False))
        uninterrupted_archive = stack.enter_context(np.load(uninterrupted_path, allow_pickle=False))
        np.testing.assert_array_equal(resumed_archive["qpos"], uninterrupted_archive["qpos"])
        np.testing.assert_array_equal(
            resumed_archive["orientation_errors_rad"],
            uninterrupted_archive["orientation_errors_rad"],
        )


def test_resume_with_missing_checkpoint_starts_from_zero(tmp_path) -> None:
    retargeter = _retargeter()

    retargeter.retarget_motion(
        **_inputs(frame_count=2),
        dest_res_path=tmp_path / "new.npz",
        checkpoint_interval_frames=1,
        resume=True,
    )

    assert retargeter.iterated_frames == [0, 1]


@pytest.mark.parametrize("resume", [False, True])
def test_existing_checkpoint_is_ignored_or_finalized_without_solving(
    tmp_path,
    resume: bool,
) -> None:
    result_path = tmp_path / "complete.npz"
    checkpoint_path = checkpoint_path_for_result(result_path)
    saved_qpos = np.zeros((2, 10), dtype=float)
    saved_qpos[:, 0] = [10.0, 20.0]
    atomic_savez(
        checkpoint_path,
        checkpoint_payload(
            total_frames=2,
            qpos=saved_qpos,
            cost=1.5,
            orientation_errors_rad=np.empty((0,), dtype=float),
            axis_errors_deg=np.empty((0,), dtype=float),
            racket_motion=None,
        ),
    )
    retargeter = _retargeter()

    result = retargeter.retarget_motion(
        **_inputs(frame_count=2),
        dest_res_path=result_path,
        checkpoint_interval_frames=1,
        resume=resume,
    )[0]

    if resume:
        assert retargeter.iterated_frames == []
        np.testing.assert_array_equal(result, saved_qpos)
    else:
        assert retargeter.iterated_frames == [0, 1]
        assert not np.array_equal(result, saved_qpos)
    assert not checkpoint_path.exists()


def test_resume_requires_enabled_checkpointing(tmp_path) -> None:
    with pytest.raises(ValueError, match="resume requires checkpoint_interval_frames to be positive"):
        _retargeter().retarget_motion(
            **_inputs(frame_count=1),
            dest_res_path=tmp_path / "result.npz",
            checkpoint_interval_frames=0,
            resume=True,
        )
