"""Tests for inferred retargeting output directories."""

from pathlib import Path

import pytest
from holosoma_retargeting.config_types.retargeting import RetargetingConfig
from holosoma_retargeting.examples.robot_retarget import determine_save_dir, validate_config


def test_save_dir_is_inferred_from_input_dataset() -> None:
    result = determine_save_dir(
        None,
        results_root=Path("demo_results"),
        robot="g1",
        task_type="robot_only",
        data_path=Path("demo_data/xsens_tennis"),
        data_format="xsens",
    )

    assert result == Path("demo_results/g1/robot_only/xsens_tennis")


def test_parallel_save_dir_uses_parallel_root() -> None:
    result = determine_save_dir(
        None,
        results_root=Path("demo_results_parallel"),
        robot="g1",
        task_type="robot_only",
        data_path=Path("demo_data/lafan"),
        data_format="lafan",
    )

    assert result == Path("demo_results_parallel/g1/robot_only/lafan")


def test_existing_dataset_output_names_are_preserved() -> None:
    result = determine_save_dir(
        None,
        results_root=Path("demo_results"),
        robot="g1",
        task_type="object_interaction",
        data_path=Path("demo_data/OMOMO_new"),
        data_format="smplh",
    )

    assert result == Path("demo_results/g1/object_interaction/omomo")


def test_explicit_save_dir_takes_precedence() -> None:
    result = determine_save_dir(
        Path("custom/results"),
        results_root=Path("demo_results"),
        robot="g1",
        task_type="robot_only",
        data_path=Path("demo_data/xsens_tennis"),
        data_format="xsens",
    )

    assert result == Path("custom/results")


def test_checkpointing_defaults_to_100_frames_with_explicit_resume() -> None:
    config = RetargetingConfig()

    assert config.checkpoint_interval_frames == 100
    assert not config.resume


def test_resume_rejects_disabled_checkpointing() -> None:
    with pytest.raises(ValueError, match="resume requires checkpoint_interval_frames to be positive"):
        validate_config(RetargetingConfig(checkpoint_interval_frames=0, resume=True))
