from __future__ import annotations

import numpy as np
import pytest
from holosoma_retargeting.config_types.viser import ViserConfig, XsensViserConfig
from holosoma_retargeting.config_values.viser import get_default_xsens_viser_config
from holosoma_retargeting.src.viser_utils import resolve_frame_times
from holosoma_retargeting.viser_player import (
    compute_ground_plane_bounds,
    resolve_actor_modes,
    resolve_xsens_actor_offsets,
)


def test_xsens_options_are_not_part_of_global_viser_config() -> None:
    base = ViserConfig()
    xsens = XsensViserConfig(actor_modes=("xsens", "g1_xsens"), xsens_hdf5="motion.hdf5")

    assert not hasattr(base, "actor_modes")
    assert not hasattr(base, "xsens_hdf5")
    assert xsens.actor_modes == ("xsens", "g1_xsens")
    assert xsens.xsens_hdf5 == "motion.hdf5"
    assert isinstance(get_default_xsens_viser_config(), XsensViserConfig)


def test_actor_modes_are_composable_unique_and_canonically_ordered() -> None:
    assert resolve_actor_modes(("g1_xsens", "xsens", "g1_xsens")) == ("xsens", "g1_xsens")


def test_all_actor_alias_expands_to_every_actor() -> None:
    assert resolve_actor_modes(("all",)) == ("robot", "xsens", "g1_xsens")
    assert resolve_actor_modes(("xsens", "all")) == ("robot", "xsens", "g1_xsens")


def test_actor_modes_reject_empty_and_unknown_selections() -> None:
    with pytest.raises(ValueError, match="at least one actor"):
        resolve_actor_modes(())
    with pytest.raises(ValueError, match="Unknown actor modes"):
        resolve_actor_modes(("not_an_actor",))


def test_g1_xsens_composition_offset_only_separates_the_paired_avatars() -> None:
    single = resolve_xsens_actor_offsets(("g1_xsens",), (1.5, 0.0, 0.0))
    paired = resolve_xsens_actor_offsets(("xsens", "g1_xsens"), (1.5, -0.25, 0.0))

    np.testing.assert_array_equal(single["g1_xsens"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(paired["xsens"], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(paired["g1_xsens"], [1.5, -0.25, 0.0])


def test_g1_xsens_composition_offset_requires_finite_xyz() -> None:
    with pytest.raises(ValueError, match="three finite xyz"):
        resolve_xsens_actor_offsets(("xsens", "g1_xsens"), (1.0, 2.0))
    with pytest.raises(ValueError, match="three finite xyz"):
        resolve_xsens_actor_offsets(("xsens", "g1_xsens"), (np.nan, 0.0, 0.0))


def test_robot_ground_bounds_include_base_and_tracked_object() -> None:
    robot_dof = 2
    qpos = np.zeros((2, 7 + robot_dof + 7))
    qpos[:, 0:2] = [[10.0, -2.0], [14.0, 6.0]]
    qpos[:, -7:-5] = [[8.0, -5.0], [20.0, 9.0]]

    bounds = compute_ground_plane_bounds(
        qpos=qpos,
        robot_dof=robot_dof,
        contains_object_in_qpos=True,
        padding_m=1.0,
    )

    np.testing.assert_allclose(bounds.center_xy, [14.0, 2.0])
    assert bounds.width == 14.0
    assert bounds.height == 16.0


def test_xsens_and_g1_xsens_ground_bounds_use_all_segment_positions() -> None:
    positions = np.array(
        [
            [[-2.0, 1.0, 0.8], [1.0, 3.0, 1.7]],
            [[4.0, 9.0, 0.8], [2.0, 5.0, 1.7]],
        ]
    )

    bounds = compute_ground_plane_bounds(xsens_positions_m=positions, padding_m=1.0)

    np.testing.assert_allclose(bounds.center_xy, [1.0, 5.0])
    assert bounds.width == 8.0
    assert bounds.height == 10.0


def test_combined_ground_bounds_cover_both_data_sources() -> None:
    qpos = np.zeros((2, 7))
    qpos[:, 0:2] = [[-10.0, 2.0], [-8.0, 4.0]]
    positions = np.array([[[3.0, -6.0, 1.0]], [[7.0, -2.0, 1.0]]])

    bounds = compute_ground_plane_bounds(
        qpos=qpos,
        xsens_positions_m=positions,
        padding_m=0.5,
    )

    np.testing.assert_allclose(bounds.center_xy, [-1.5, -1.0])
    assert bounds.width == 18.0
    assert bounds.height == 11.0


def test_ground_bounds_cover_composed_xsens_actor_offsets() -> None:
    positions = np.array([[[0.0, 2.0, 1.0]], [[1.0, 4.0, 1.0]]])
    combined_positions = np.concatenate(
        [positions, positions + np.array([1.5, -0.5, 0.0])[None, None, :]],
        axis=1,
    )

    bounds = compute_ground_plane_bounds(xsens_positions_m=combined_positions, padding_m=0.5)

    np.testing.assert_allclose(bounds.center_xy, [1.25, 2.75])
    assert bounds.width == 3.5
    assert bounds.height == 3.5


def test_explicit_grid_extents_act_only_as_minimums() -> None:
    qpos = np.zeros((1, 7))

    bounds = compute_ground_plane_bounds(
        qpos=qpos,
        padding_m=1.0,
        minimum_width_m=8.0,
        minimum_height_m=6.0,
    )

    np.testing.assert_array_equal(bounds.center_xy, [0.0, 0.0])
    assert bounds.width == 8.0
    assert bounds.height == 6.0


def test_ground_bounds_require_finite_motion_positions() -> None:
    with pytest.raises(ValueError, match="finite horizontal positions"):
        compute_ground_plane_bounds(qpos=np.full((1, 7), np.nan))


def test_frame_times_use_fps_when_timestamps_are_unavailable() -> None:
    np.testing.assert_allclose(
        resolve_frame_times(4, initial_fps=2.0),
        [0.0, 0.5, 1.0, 1.5],
    )


def test_frame_times_preserve_irregular_timestamp_correspondence() -> None:
    np.testing.assert_allclose(
        resolve_frame_times(
            3,
            initial_fps=30.0,
            frame_times_s=np.array([100.0, 100.01, 100.035]),
        ),
        [0.0, 0.01, 0.035],
    )
