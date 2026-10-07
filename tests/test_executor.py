"""Tests for trajectory handoff, commands, and canonical outputs."""

# ruff: noqa: D103

import argparse
import asyncio

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_actions_executor.main import (
    QPOS_TYPE,
    BiquadLowpass,
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
