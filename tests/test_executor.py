"""Tests for trajectory handoff, commands, and canonical outputs."""

# ruff: noqa: D103

import asyncio

import numpy as np
import pyarrow as pa

from dora_openarm_actions_executor.main import (
    QPOS_TYPE,
    BiquadLowpass,
    HermiteUpsampler,
    _apply_command,
    _blend_trajectories,
    _next_input,
    _put_latest,
    _qpos_output,
    _remaining_policy_trajectory,
)


def _command_event(command):
    return {"value": pa.array([command])}


def test_hermite_upsampler_preserves_policy_knots():
    positions = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    upsampler = HermiteUpsampler(chunk_hz=10.0, horizon_sec=0.2)

    output = upsampler.upsample(positions, [0.0, 0.1, 0.2])

    np.testing.assert_allclose(output, positions, atol=1e-6)


def test_remaining_trajectory_keeps_policy_period_from_last_sent_point():
    positions = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    upsampler = HermiteUpsampler(chunk_hz=1.0, horizon_sec=3.0)

    remaining = _remaining_policy_trajectory(
        upsampler,
        positions,
        last_sent_position=np.array([0.5], dtype=np.float32),
        last_sent_time=0.5,
    )

    np.testing.assert_allclose(remaining, [[0.5], [1.5], [2.5]], atol=1e-6)


def test_blend_starts_at_previous_sent_position_and_ends_on_new_chunk():
    previous = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    current = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)

    blended = _blend_trajectories(previous, current)

    np.testing.assert_allclose(blended, [[1.0], [11.0], [30.0], [40.0]])


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
