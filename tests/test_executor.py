"""Tests for trajectory handoff, commands, and canonical outputs."""

# ruff: noqa: D103

import argparse
import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pyarrow as pa
import pytest
import dora_openarm_actions_executor.main as executor

from dora_openarm_actions_executor.main import (
    QPOS_TYPE,
    BiquadLowpass,
    OneEuroLowpass,
    _apply_command,
    _next_input,
    _nonnegative_float,
    _put_latest,
    _qpos_output,
)
from dora_openarm_actions_executor.trajectory import (
    _blend_trajectories,
    _upsample_trajectory,
)


def _command_event(command):
    return {"value": pa.array([command])}


def test_pchip_upsampler_preserves_policy_knots():
    positions = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)

    output = _upsample_trajectory(positions, 100_000_000, 50_000_000)

    np.testing.assert_allclose(output[::2], positions, atol=1e-6)


def test_upsampling_includes_final_pose_without_extrapolation():
    positions = np.array([[0.0], [0.1], [0.2]], dtype=np.float32)

    output = _upsample_trajectory(positions, 33_333_333, 4_000_000)

    assert len(output) == 18
    np.testing.assert_array_equal(output[[0, -1]], positions[[0, -1]])


def test_blend_starts_at_pending_position_and_ends_on_new_chunk():
    previous = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    current = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)

    blended, count = _blend_trajectories(previous, current, 100_000_000)

    np.testing.assert_allclose(blended, [[1.0], [11.0], [30.0], [40.0]])
    assert count == 3


def test_blend_limits_transition_to_configured_duration():
    previous = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
    current = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)

    blended, count = _blend_trajectories(
        previous, current, step_interval_ns=100_000_000, duration_ns=200_000_000
    )

    np.testing.assert_allclose(blended, [[1.0], [11.0], [30.0], [40.0]])
    assert count == 3


def test_blend_duration_rejects_negative_values():
    with pytest.raises(argparse.ArgumentTypeError):
        _nonnegative_float("-1")


def test_lowpass_reset_starts_at_new_pose():
    lowpass = BiquadLowpass(fs=250.0, fc=15.0)
    lowpass.step(np.array([0.0, 0.0], dtype=np.float32))
    new_pose = np.array([1.0, -1.0], dtype=np.float32)

    lowpass.reset_state(new_pose)

    np.testing.assert_allclose(lowpass.step(new_pose), new_pose, atol=1e-6)


def test_one_euro_fixed_cutoff_and_adaptive_response():
    fixed = OneEuroLowpass(250, min_cutoff=5, beta=0)
    adaptive = OneEuroLowpass(250, min_cutoff=5, beta=0.5)
    for lowpass in (fixed, adaptive):
        np.testing.assert_array_equal(lowpass.step([0.0]), [0.0])
    alpha = 1 / (1 + 250 / (2 * np.pi * 5))
    np.testing.assert_allclose(fixed.step([1.0]), [alpha], atol=1e-7)
    assert alpha < adaptive.step([1.0])[0] < 1
    values = np.array([adaptive.step([1.0])[0] for _ in range(100)])
    assert np.all(np.diff(values) >= 0) and np.all(values <= 1)


def test_one_euro_joint_independence_and_reset():
    lowpass = OneEuroLowpass(250, beta=0.5)
    initial = np.array([0.0, 2.0], dtype=np.float32)
    lowpass.step(initial)
    lowpass.step([1.0, 2.0])
    assert lowpass.filtered[1] == 2 and lowpass.derivative[1] == 0
    np.testing.assert_array_equal(initial, [0.0, 2.0])
    lowpass.reset_state([3.0, -1.0])
    np.testing.assert_array_equal(lowpass.derivative, [0.0, 0.0])
    np.testing.assert_array_equal(lowpass.step([3.0, -1.0]), [3.0, -1.0])


@pytest.mark.parametrize(
    "environment,options,expected",
    [
        ({}, [], (15.0, 0.5)),
        ({"ACTION_FILTER_CUTOFF_HZ": "8", "ACTION_FILTER_Q": "0.6"}, [], (8.0, 0.6)),
        (
            {"ACTION_FILTER_CUTOFF_HZ": "8", "ACTION_FILTER_Q": "0.6"},
            ["--filter-cutoff-hz", "12", "--filter-q", "0.7"],
            (12.0, 0.7),
        ),
    ],
)
def test_filter_settings_cli_overrides_environment(
    monkeypatch, environment, options, expected
):
    for name in ("ACTION_FILTER_CUTOFF_HZ", "ACTION_FILTER_Q"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    run = AsyncMock()
    monkeypatch.setattr(executor, "_main_async", run)
    monkeypatch.setattr("sys.argv", ["executor", "--upsample", "--filter", *options])
    executor.main()
    assert run.await_args.kwargs["filter_cutoff_hz"] == expected[0]
    assert run.await_args.kwargs["filter_q"] == expected[1]


@pytest.mark.parametrize(
    "use_env,use_cli,expected",
    [
        (False, False, (1.0, 0.0, 1.0)),
        (True, False, (3.0, 0.1, 2.0)),
        (True, True, (5.0, 0.5, 1.0)),
    ],
)
def test_one_euro_cli_and_environment(monkeypatch, use_env, use_cli, expected):
    names = ("MIN_CUTOFF_HZ", "BETA", "D_CUTOFF_HZ")
    for name, value in zip(names, (3, 0.1, 2)):
        name = "ACTION_ONE_EURO_" + name
        monkeypatch.delenv(name, raising=False)
        if use_env:
            monkeypatch.setenv(name, str(value))
    options = (
        [
            "--one-euro-min-cutoff-hz",
            "5",
            "--one-euro-beta",
            "0.5",
            "--one-euro-d-cutoff-hz",
            "1",
        ]
        if use_cli
        else []
    )
    run = AsyncMock()
    monkeypatch.setattr(executor, "_main_async", run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "executor",
            "--upsample",
            "--filter",
            "--filter-mode",
            "one-euro",
            "--filter-cutoff-hz",
            "999",
            *options,
        ],
    )
    executor.main()
    assert run.await_args.kwargs["filter_mode"] == "one-euro"
    assert (
        tuple(run.await_args.kwargs["one_euro_" + name.lower()] for name in names)
        == expected
    )


@pytest.mark.parametrize(
    "options",
    [
        ["--filter-q", "0"],
        ["--filter-q", "inf"],
        ["--filter-cutoff-hz", "nan"],
        ["--filter-cutoff-hz", "125"],
        ["--control-hz", "0"],
        ["--one-euro-min-cutoff-hz", "0"],
        ["--one-euro-beta", "-1"],
        ["--one-euro-d-cutoff-hz", "nan"],
    ],
)
def test_invalid_filter_settings_fail_before_node_start(monkeypatch, options):
    run = AsyncMock()
    monkeypatch.setattr(executor, "_main_async", run)
    monkeypatch.setattr("sys.argv", ["executor", "--upsample", "--filter", *options])
    with pytest.raises(SystemExit) as error:
        executor.main()
    assert error.value.code == 2
    run.assert_not_called()


@pytest.mark.parametrize(
    "environment,options,expected",
    [
        (None, [], True),
        ("false", [], False),
        ("true", ["--no-filter-grippers"], False),
        ("false", ["--filter-grippers"], True),
    ],
)
def test_filter_grippers_cli_and_environment(
    monkeypatch, environment, options, expected
):
    monkeypatch.delenv("ACTION_FILTER_GRIPPERS", raising=False)
    if environment is not None:
        monkeypatch.setenv("ACTION_FILTER_GRIPPERS", environment)
    run = AsyncMock()
    monkeypatch.setattr(executor, "_main_async", run)
    monkeypatch.setattr("sys.argv", ["executor", "--upsample", "--filter", *options])
    executor.main()
    assert run.await_args.kwargs["filter_grippers"] is expected


def test_latest_queue_replaces_pending_action():
    queue = asyncio.Queue(maxsize=1)
    _put_latest(queue, "old")

    _put_latest(queue, "new")

    assert queue.qsize() == 1
    assert queue.get_nowait() == "new"


def test_command_has_priority_and_clears_pending_action():
    action_queue = asyncio.Queue(maxsize=1)
    command_queue = asyncio.Queue(maxsize=1)
    action_queue.put_nowait("stale action")
    stop_event = _command_event("stop")
    command_queue.put_nowait(stop_event)

    event_id, event = asyncio.run(_next_input(action_queue, command_queue))

    assert event_id == "command"
    assert event is stop_event
    assert _apply_command(event, action_queue) is False
    assert action_queue.empty()


def test_start_command_enables_executor():
    action_queue = asyncio.Queue(maxsize=1)

    enabled = _apply_command(_command_event("start"), action_queue)

    assert enabled is True


def test_qpos_output_uses_canonical_arm_payload():
    output = _qpos_output(np.array([1.0, 2.0], dtype=np.float32))

    assert output.type == QPOS_TYPE
    assert output.to_pylist() == [{"qpos": [1.0, 2.0]}]
