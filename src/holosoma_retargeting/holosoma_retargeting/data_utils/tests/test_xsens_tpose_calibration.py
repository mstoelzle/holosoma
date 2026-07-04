from __future__ import annotations

import mujoco
import numpy as np

from holosoma_retargeting.data_utils.xsens_hdf5 import XSENS_BODY_SEGMENT_NAMES, XsensHdf5Tpose
from holosoma_retargeting.src.xsens_tpose_calibration import (
    XsensTposeCalibrationConfig,
    align_and_scale_tpose_targets,
    axis_alignment_error_deg,
    compute_canonical_axes,
    evaluate_head_candidate,
    limb_axis_error_deg,
    solve_xsens_tpose_calibration_from_data,
    symmetry_residual,
    visual_elbow_angle_deg,
)


def _symmetric_tpose_positions() -> np.ndarray:
    coords = {name: np.zeros(3, dtype=float) for name in XSENS_BODY_SEGMENT_NAMES}
    coords.update(
        {
            "Pelvis": np.array([0.0, 0.0, 0.86]),
            "L5": np.array([0.0, 0.0, 0.97]),
            "L3": np.array([0.0, 0.0, 1.04]),
            "T12": np.array([0.0, 0.0, 1.12]),
            "T8": np.array([0.0, 0.0, 1.22]),
            "Neck": np.array([0.0, 0.0, 1.36]),
            "Head": np.array([0.0, 0.0, 1.48]),
            "Left Shoulder": np.array([0.0, 0.16, 1.28]),
            "Right Shoulder": np.array([0.0, -0.16, 1.28]),
            "Left Upper Arm": np.array([0.0, 0.25, 1.28]),
            "Right Upper Arm": np.array([0.0, -0.25, 1.28]),
            "Left Forearm": np.array([0.0, 0.52, 1.28]),
            "Right Forearm": np.array([0.0, -0.52, 1.28]),
            "Left Hand": np.array([0.0, 0.74, 1.28]),
            "Right Hand": np.array([0.0, -0.74, 1.28]),
            "Left Upper Leg": np.array([0.0, 0.09, 0.84]),
            "Right Upper Leg": np.array([0.0, -0.09, 0.84]),
            "Left Lower Leg": np.array([-0.02, 0.09, 0.46]),
            "Right Lower Leg": np.array([-0.02, -0.09, 0.46]),
            "Left Foot": np.array([-0.05, 0.09, 0.08]),
            "Right Foot": np.array([-0.05, -0.09, 0.08]),
            "Left Toe": np.array([0.12, 0.09, 0.0]),
            "Right Toe": np.array([0.12, -0.09, 0.0]),
        }
    )
    return np.asarray([coords[name] for name in XSENS_BODY_SEGMENT_NAMES], dtype=float)


def test_align_and_scale_tpose_targets_ground_aligns_and_centers() -> None:
    positions = _symmetric_tpose_positions()
    positions[:, :2] += np.array([2.0, -3.0])

    aligned, axes = align_and_scale_tpose_targets(positions, scale_factor=2.0)

    np.testing.assert_allclose(aligned[XSENS_BODY_SEGMENT_NAMES.index("Pelvis"), :2], [0.0, 0.0])
    foot_indices = [XSENS_BODY_SEGMENT_NAMES.index(name) for name in ("Left Foot", "Right Foot", "Left Toe", "Right Toe")]
    assert np.min(aligned[foot_indices, 2]) == 0.0
    np.testing.assert_allclose(np.cross(axes[0], axes[1]), axes[2], atol=1e-8)
    assert axes[2, 2] > 0.0


def test_compute_canonical_axes_is_right_handed() -> None:
    axes = compute_canonical_axes(_symmetric_tpose_positions())

    np.testing.assert_allclose(axes[0], [1.0, 0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(axes[1], [0.0, 1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(axes[2], [0.0, 0.0, 1.0], atol=1e-8)


def test_symmetry_and_limb_axis_residual_helpers() -> None:
    np.testing.assert_allclose(symmetry_residual([1.0, 2.0, 3.0], [1.0, -2.0, 3.0]), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(symmetry_residual([1.1, 2.0, 3.0], [1.0, -2.0, 3.0]), [0.1, 0.0, 0.0])
    assert limb_axis_error_deg([0, 0, 0], [1, 0, 0], [2, 2, 2], [3, 2, 2]) < 1e-8


def test_head_candidate_diagnostics_accept_and_reject() -> None:
    cfg = XsensTposeCalibrationConfig(verbose=0)

    accepted, status = evaluate_head_candidate(
        head_position_error_m=0.05,
        head_orientation_offset_wijk=np.array([1.0, 0.0, 0.0, 0.0]),
        config=cfg,
    )
    assert accepted
    assert status.startswith("accepted")

    rejected, status = evaluate_head_candidate(
        head_position_error_m=0.2,
        head_orientation_offset_wijk=np.array([1.0, 0.0, 0.0, 0.0]),
        config=cfg,
    )
    assert not rejected
    assert status.startswith("rejected")


def test_g1_tpose_calibration_smoke_on_synthetic_symmetric_tpose() -> None:
    positions = _symmetric_tpose_positions()
    quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (len(XSENS_BODY_SEGMENT_NAMES), 1))
    tpose = XsensHdf5Tpose(
        positions_m=positions,
        quaternions_wijk=quaternions,
        variant="Tpose",
        segment_names=XSENS_BODY_SEGMENT_NAMES,
        source_indices=list(range(len(XSENS_BODY_SEGMENT_NAMES))),
    )

    result = solve_xsens_tpose_calibration_from_data(
        tpose,
        config=XsensTposeCalibrationConfig(max_nfev=8, verbose=0),
    )

    assert result.qpos.shape == (1, 36)
    assert np.isfinite(result.qpos).all()
    assert result.head_candidate_status.startswith(("accepted", "rejected"))

    model = mujoco.MjModel.from_xml_path(
        "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.xml"
    )
    data = mujoco.MjData(model)
    data.qpos[:] = result.qpos[0]
    mujoco.mj_forward(model, data)
    left_hand = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand_link")]
    right_hand = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_rubber_hand_link")]
    np.testing.assert_allclose(left_hand[[0, 2]], right_hand[[0, 2]], atol=0.2)
    np.testing.assert_allclose(left_hand[1], -right_hand[1], atol=0.2)

    left_knee = result.qpos[0, 10]
    right_knee = result.qpos[0, 16]
    assert abs(left_knee) < 0.5
    assert abs(right_knee) < 0.5
    for wrist_qpos_idx in (26, 27, 28, 33, 34, 35):
        assert abs(result.qpos[0, wrist_qpos_idx]) < 0.35

    up = np.array([0.0, 0.0, 1.0])
    for side in ("left", "right"):
        shoulder = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_shoulder_roll_link")]
        elbow = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_elbow_link")]
        hand = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rubber_hand_link")]
        assert visual_elbow_angle_deg(shoulder, elbow, hand) < 12.0
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_rubber_hand_link")
        hand_vertical = data.xmat[hand_id].reshape(3, 3)[:, 2]
        assert axis_alignment_error_deg(hand_vertical, up) < 8.0
