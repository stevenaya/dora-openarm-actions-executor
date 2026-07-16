# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Execute policy action chunks on OpenArm."""

import argparse
import asyncio
import os
import time
from dataclasses import dataclass

import dora
import numpy as np
import pyarrow as pa


ELEMENTS_PER_ARM = 8
ACTION_TYPE = pa.list_(pa.float32())
QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
INTERPOLATION_METHODS = {"hermite", "pchip"}
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


class TrajectoryInterpolator:
    """Interpolate a position trajectory on its policy time axis."""

    def __init__(self, positions, interval_s, method):
        """Build an interpolator for one action chunk."""
        if method not in INTERPOLATION_METHODS:
            raise ValueError(f"Unsupported interpolation method: {method}")
        if interval_s <= 0:
            raise ValueError("Action interval must be positive")

        self.positions = np.asarray(positions, dtype=np.float64)
        if self.positions.ndim != 2 or len(self.positions) == 0:
            raise ValueError("Action chunk must be a non-empty 2D array")

        self.interval_s = float(interval_s)
        self.method = method
        self.times = np.arange(len(self.positions), dtype=np.float64) * self.interval_s
        self.slopes = self._compute_slopes()

    @property
    def horizon_s(self):
        """Return the trajectory duration."""
        return float(self.times[-1])

    @staticmethod
    def _edge_slope(h0, h1, slope0, slope1):
        slope = ((2.0 * h0 + h1) * slope0 - h0 * slope1) / (h0 + h1)
        opposite_sign = np.sign(slope) != np.sign(slope0)
        too_steep = (np.sign(slope0) != np.sign(slope1)) & (
            np.abs(slope) > 3.0 * np.abs(slope0)
        )
        slope[opposite_sign] = 0.0
        slope[~opposite_sign & too_steep] = 3.0 * slope0[~opposite_sign & too_steep]
        return slope

    def _hermite_slopes(self):
        if len(self.positions) == 1:
            return np.zeros_like(self.positions)

        secants = np.diff(self.positions, axis=0) / self.interval_s
        slopes = np.zeros_like(self.positions)
        slopes[0] = secants[0]
        slopes[-1] = secants[-1]
        for index in range(1, len(self.positions) - 1):
            slope = 0.5 * (secants[index - 1] + secants[index])
            slope[secants[index - 1] * secants[index] <= 0.0] = 0.0
            slopes[index] = slope
        return slopes

    def _pchip_slopes(self):
        if len(self.positions) <= 2:
            return self._hermite_slopes()

        intervals = np.diff(self.times)
        secants = np.diff(self.positions, axis=0) / intervals[:, None]
        slopes = np.zeros_like(self.positions)

        previous_interval = intervals[:-1, None]
        next_interval = intervals[1:, None]
        previous_secant = secants[:-1]
        next_secant = secants[1:]
        same_sign = previous_secant * next_secant > 0.0
        weight1 = 2.0 * next_interval + previous_interval
        weight2 = next_interval + 2.0 * previous_interval
        with np.errstate(divide="ignore", invalid="ignore"):
            interior = (weight1 + weight2) / (
                weight1 / previous_secant + weight2 / next_secant
            )
        slopes[1:-1] = np.where(same_sign, interior, 0.0)
        slopes[0] = self._edge_slope(
            intervals[0],
            intervals[1],
            secants[0],
            secants[1],
        )
        slopes[-1] = self._edge_slope(
            intervals[-1],
            intervals[-2],
            secants[-1],
            secants[-2],
        )
        return slopes

    def _compute_slopes(self):
        if self.method == "pchip":
            return self._pchip_slopes()
        return self._hermite_slopes()

    def sample(self, sample_times):
        """Sample positions at times measured from the chunk start."""
        sample_times = np.asarray(sample_times, dtype=np.float64)
        if len(self.positions) == 1:
            return np.repeat(
                self.positions,
                len(sample_times),
                axis=0,
            ).astype(np.float32)

        sample_times = np.clip(sample_times, 0.0, self.horizon_s)
        indices = np.searchsorted(self.times, sample_times, side="right") - 1
        indices = np.clip(indices, 0, len(self.times) - 2)

        t0 = self.times[indices]
        t1 = self.times[indices + 1]
        duration = t1 - t0
        phase = (sample_times - t0) / duration

        h00 = 2 * phase**3 - 3 * phase**2 + 1
        h10 = (phase**3 - 2 * phase**2 + phase) * duration
        h01 = -2 * phase**3 + 3 * phase**2
        h11 = (phase**3 - phase**2) * duration

        return (
            h00[:, None] * self.positions[indices]
            + h10[:, None] * self.slopes[indices]
            + h01[:, None] * self.positions[indices + 1]
            + h11[:, None] * self.slopes[indices + 1]
        ).astype(np.float32)


class BiquadLowpass:
    """Smooth position commands with a Tustin biquad low-pass filter."""

    def __init__(self, sample_hz, cutoff_hz, quality=0.707):
        """Create a low-pass filter."""
        sample_hz = float(sample_hz)
        cutoff_hz = float(cutoff_hz)
        quality = float(quality)
        if not 0.0 < cutoff_hz < sample_hz / 2.0:
            raise ValueError("Filter cutoff must be between zero and Nyquist")

        angular_frequency = 2 * np.pi * cutoff_hz / sample_hz
        cosine = np.cos(angular_frequency)
        alpha = np.sin(angular_frequency) / (2 * quality)
        a0 = 1 + alpha
        self.b0 = ((1 - cosine) / 2) / a0
        self.b1 = (1 - cosine) / a0
        self.b2 = ((1 - cosine) / 2) / a0
        self.a1 = (-2 * cosine) / a0
        self.a2 = (1 - alpha) / a0
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

    def reset_state(self, position):
        """Initialize filter memory at a position without a startup transient."""
        position = np.asarray(position, dtype=np.float32)
        self.x1 = position.copy()
        self.x2 = position.copy()
        self.y1 = position.copy()
        self.y2 = position.copy()

    def step(self, position):
        """Filter one position sample."""
        position = np.asarray(position, dtype=np.float32)
        if self.x1 is None:
            self.reset_state(position)
        output = (
            self.b0 * position
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, position
        self.y2, self.y1 = self.y1, output
        return output.astype(np.float32)


@dataclass
class _ActionChunk:
    positions: np.ndarray
    interval_ns: int
    cutoff_hz: float
    reset: bool

    @property
    def interval_s(self):
        return self.interval_ns / 1_000_000_000


def _parse_action(event):
    value = event["value"]
    if value.type != ACTION_TYPE or len(value) == 0:
        raise ValueError("Actions must be a non-empty list<float32> array")

    width = len(value[0])
    if width == 0 or any(len(value[index]) != width for index in range(len(value))):
        raise ValueError("Every action in a chunk must have the same non-zero width")

    interval_ns = int(event["metadata"]["interval"])
    if interval_ns <= 0:
        raise ValueError("Action interval must be positive")

    return _ActionChunk(
        positions=value.values.to_numpy().reshape(len(value), width),
        interval_ns=interval_ns,
        cutoff_hz=float(event["metadata"].get("cutoff_hz", 15.0)),
        reset=bool(event["metadata"].get("reset", False)),
    )


def _blend_trajectories(previous, current):
    if previous is None or len(previous) == 0:
        return current

    count = min(len(previous), len(current))
    output = current.copy()
    weights = np.linspace(1.0, 0.0, count, dtype=np.float32)[:, None]
    output[:count] = previous[:count] * weights + current[:count] * (1.0 - weights)
    return output


def _remaining_policy_trajectory(
    interpolator,
    last_sent_position,
    last_sent_time,
):
    if last_sent_position is None or last_sent_time is None:
        return None

    future_times = np.arange(
        last_sent_time + interpolator.interval_s,
        interpolator.horizon_s + 1e-12,
        interpolator.interval_s,
    )
    if len(future_times) == 0:
        return last_sent_position[None, :]

    return np.concatenate(
        [
            last_sent_position[None, :],
            interpolator.sample(future_times),
        ]
    )


def _control_trajectory(interpolator, use_upsample, control_hz):
    if not use_upsample:
        return (
            interpolator.times,
            interpolator.positions.astype(np.float32),
            int(interpolator.interval_s * 1_000_000_000),
        )

    control_interval_s = 1.0 / float(control_hz)
    duration_s = max(interpolator.horizon_s, interpolator.interval_s)
    sample_times = np.arange(0.0, duration_s + 1e-12, control_interval_s)
    return (
        sample_times,
        interpolator.sample(sample_times),
        int(control_interval_s * 1_000_000_000),
    )


def _clear_queue(queue):
    while not queue.empty():
        queue.get_nowait()


def _put_latest(queue, event):
    _clear_queue(queue)
    queue.put_nowait(event)


def _put_action(queue, event):
    if bool(event["metadata"].get("reset", False)):
        _clear_queue(queue)
    if queue.maxsize > 0:
        while queue.full():
            queue.get_nowait()
    queue.put_nowait(event)


async def _next_input(action_queue, command_queue):
    if not command_queue.empty():
        return "command", command_queue.get_nowait()
    if not action_queue.empty():
        return "actions", action_queue.get_nowait()

    action_task = asyncio.create_task(action_queue.get())
    command_task = asyncio.create_task(command_queue.get())
    done, pending = await asyncio.wait(
        {action_task, command_task},
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if command_task in done:
        if action_task in done:
            action_task.result()
        return "command", command_task.result()
    return "actions", action_task.result()


def _apply_command(event, action_queue):
    command = event["value"][0].as_py()
    if command in START_COMMANDS:
        enabled = True
    elif command in STOP_COMMANDS:
        enabled = False
    else:
        return None

    _clear_queue(action_queue)
    print(
        f"actions-executor command={command}: reset state, enabled={enabled}",
        flush=True,
    )
    return enabled


def _split_positions(position, arms):
    expected_width = ELEMENTS_PER_ARM * len(arms)
    if len(position) != expected_width:
        raise ValueError(
            f"Expected {expected_width} action values for {arms}, got {len(position)}"
        )

    positions = {}
    offset = 0
    for arm in arms:
        positions[arm] = position[offset : offset + ELEMENTS_PER_ARM]
        offset += ELEMENTS_PER_ARM
    return positions


def _qpos_output(position):
    return pa.array(
        [{"qpos": np.asarray(position, dtype=np.float32)}],
        type=QPOS_TYPE,
    )


def _send_positions(node, position, arms):
    timestamp = time.time_ns()
    for arm, arm_position in _split_positions(position, arms).items():
        node.send_output(
            f"move_position_{arm}",
            _qpos_output(arm_position),
            {"timestamp": timestamp},
        )


async def _main_executor(
    node,
    action_queue,
    command_queue,
    arms,
    use_upsample,
    use_filter,
    control_hz,
    interpolation,
):
    if use_filter and not use_upsample:
        print("Filter requires upsampling; disabling filter.", flush=True)
        use_filter = False

    enabled = False
    remaining_trajectory = None
    lowpass = None

    while True:
        event_type, event = await _next_input(action_queue, command_queue)
        if event_type == "command":
            new_enabled = _apply_command(event, action_queue)
            if new_enabled is not None:
                enabled = new_enabled
                remaining_trajectory = None
                lowpass = None
            continue
        if not enabled:
            continue

        chunk = _parse_action(event)
        if chunk.reset:
            remaining_trajectory = None

        positions = _blend_trajectories(
            remaining_trajectory,
            chunk.positions,
        )
        remaining_trajectory = None
        interpolator = TrajectoryInterpolator(
            positions,
            chunk.interval_s,
            interpolation,
        )
        sample_times, control_positions, step_interval_ns = _control_trajectory(
            interpolator,
            use_upsample,
            control_hz,
        )

        if use_filter and lowpass is None:
            lowpass = BiquadLowpass(control_hz, chunk.cutoff_hz)
        if chunk.reset and lowpass is not None:
            print("Resetting trajectory and filter state.", flush=True)
            lowpass.reset_state(positions[0])

        last_sent_position = None
        last_sent_time = None
        next_send_ns = time.monotonic_ns()

        for sample_time, raw_position in zip(
            sample_times,
            control_positions,
            strict=True,
        ):
            sleep_ns = next_send_ns - time.monotonic_ns()
            if sleep_ns > 0:
                await asyncio.sleep(sleep_ns / 1_000_000_000)

            if not command_queue.empty():
                command = command_queue.get_nowait()
                new_enabled = _apply_command(command, action_queue)
                if new_enabled is not None:
                    enabled = new_enabled
                    remaining_trajectory = None
                    lowpass = None
                    break

            if not action_queue.empty():
                remaining_trajectory = _remaining_policy_trajectory(
                    interpolator,
                    last_sent_position,
                    last_sent_time,
                )
                break

            position = raw_position
            if lowpass is not None:
                position = lowpass.step(position)
            _send_positions(node, position, arms)
            last_sent_position = np.asarray(position, dtype=np.float32).copy()
            last_sent_time = float(sample_time)
            next_send_ns += step_interval_ns


async def _main_dora(node, action_queue, command_queue, executor_task):
    while True:
        event = await asyncio.to_thread(node.next)
        if event["type"] != "INPUT":
            break

        if event["id"] == "actions":
            _put_action(action_queue, event)
        elif event["id"] == "command":
            _put_latest(command_queue, event)

    executor_task.cancel()


async def _main_async(
    arms,
    use_upsample,
    use_filter,
    control_hz,
    action_queue_size,
    interpolation,
):
    node = dora.Node()
    action_queue = asyncio.Queue(maxsize=action_queue_size)
    command_queue = asyncio.Queue(maxsize=1)
    executor_task = asyncio.create_task(
        _main_executor(
            node,
            action_queue,
            command_queue,
            arms,
            use_upsample,
            use_filter,
            control_hz,
            interpolation,
        )
    )
    dora_task = asyncio.create_task(
        _main_dora(node, action_queue, command_queue, executor_task)
    )

    try:
        await executor_task
    except asyncio.CancelledError:
        pass
    await dora_task


def main():
    """Execute policy action chunks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="Comma-separated arm sides",
    )
    parser.add_argument(
        "--upsample",
        action="store_true",
        help="Upsample policy actions to the motor control rate",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Low-pass filter upsampled motor commands",
    )
    parser.add_argument(
        "--control-hz",
        default=250.0,
        type=float,
        help="Motor control frequency",
    )
    parser.add_argument(
        "--interpolation",
        default=os.getenv("ACTION_INTERPOLATION", "hermite"),
        choices=sorted(INTERPOLATION_METHODS),
        help="Trajectory interpolation method",
    )
    parser.add_argument(
        "--action-queue-size",
        default=int(os.getenv("ACTION_QUEUE_SIZE", "1")),
        type=int,
        help="Buffered action chunks; 0 means unbounded",
    )
    args = parser.parse_args()

    arms = args.arms.split(",")
    if not arms or len(set(arms)) != len(arms):
        raise ValueError("--arms must contain unique arm sides")
    if any(arm not in {"right", "left"} for arm in arms):
        raise ValueError("--arms must contain only 'right' and/or 'left'")
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive")
    if args.action_queue_size < 0:
        raise ValueError("--action-queue-size must be non-negative")

    asyncio.run(
        _main_async(
            arms,
            args.upsample,
            args.filter,
            args.control_hz,
            args.action_queue_size,
            args.interpolation,
        )
    )


if __name__ == "__main__":
    main()
