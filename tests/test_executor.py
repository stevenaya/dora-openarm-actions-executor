"""Tests for trajectory interpolation, blending, and reset behavior."""

# ruff: noqa: D103

import asyncio

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_actions_executor.main import (
    ACTION_TYPE,
    QPOS_TYPE,
    BiquadLowpass,
    TrajectoryInterpolator,
    _blend_trajectories,
    _parse_action,
    _put_action,
    _qpos_output,
    _remaining_policy_trajectory,
)


def _action_event(positions, *, reset=False):
    return {
        "metadata": {
            "interval": 100_000_000,
            "cutoff_hz": 15.0,
            "reset": reset,
        },
        "value": pa.array(positions, type=ACTION_TYPE),
    }


def test_parse_action_uses_canonical_policy_payload():
    chunk = _parse_action(_action_event([[0.0], [1.0]], reset=True))

    assert chunk.reset is True
    assert chunk.interval_ns == 100_000_000
    np.testing.assert_allclose(chunk.positions, [[0.0], [1.0]])


def test_reset_action_discards_queued_chunks():
    queue = asyncio.Queue(maxsize=3)
    _put_action(queue, _action_event([[0.0], [1.0]]))
    _put_action(queue, _action_event([[1.0], [2.0]]))

    reset_event = _action_event([[10.0], [11.0]], reset=True)
    _put_action(queue, reset_event)

    assert queue.qsize() == 1
    assert queue.get_nowait() is reset_event


@pytest.mark.parametrize("method", ["hermite", "pchip"])
def test_interpolator_preserves_policy_knots(method):
    positions = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    interpolator = TrajectoryInterpolator(positions, 0.1, method)

    output = interpolator.sample([0.0, 0.1, 0.2])

    np.testing.assert_allclose(output, positions, atol=1e-6)


def test_remaining_trajectory_keeps_policy_period_from_last_sent_point():
    positions = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    interpolator = TrajectoryInterpolator(positions, 1.0, "hermite")

    remaining = _remaining_policy_trajectory(
        interpolator,
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
    lowpass = BiquadLowpass(sample_hz=250.0, cutoff_hz=15.0)
    lowpass.step(np.array([0.0, 0.0], dtype=np.float32))
    new_pose = np.array([1.0, -1.0], dtype=np.float32)

    lowpass.reset_state(new_pose)

    np.testing.assert_allclose(lowpass.step(new_pose), new_pose, atol=1e-6)


def test_qpos_output_uses_only_canonical_arm_payload():
    output = _qpos_output(np.array([1.0, 2.0], dtype=np.float32))

    assert output.type == QPOS_TYPE
    assert output.to_pylist() == [{"qpos": [1.0, 2.0]}]
