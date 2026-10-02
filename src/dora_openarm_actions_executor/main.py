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

"""Node to execute timestamped actions."""

import argparse
import asyncio
import dora
import os
import numpy as np
import pyarrow as pa
from scipy.interpolate import PchipInterpolator
import time


QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


def _nonnegative_float(value):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _upsample_trajectory(positions, interval_ns, step_interval_ns):
    """Use shape-preserving PCHIP on a fixed grid that includes the final pose."""
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) == 1:
        return positions
    knot_ns = np.arange(len(positions), dtype=np.int64) * interval_ns
    horizon_ns = knot_ns[-1]
    steps = (horizon_ns + step_interval_ns - 1) // step_interval_ns
    sample_ns = np.arange(steps + 1, dtype=np.int64) * step_interval_ns
    # Keep the command period fixed; sample past the horizon holds the final pose.
    sample_ns = np.minimum(sample_ns, horizon_ns)
    interpolator = PchipInterpolator(
        knot_ns / 1e9, positions, axis=0, extrapolate=False
    )
    output = interpolator(sample_ns / 1e9).astype(np.float32)
    output[-1] = positions[-1]
    return output


# Tustin bilinear transform based biquad low-pass filter
class BiquadLowpass:
    """Biquad low-pass filter for smoothing outputs."""

    def __init__(self, fs, fc, Q=0.5):
        """Initialize the biquad low-pass filter with sampling frequency, cutoff frequency, and Q factor."""
        fs = float(fs)
        fc = float(fc)
        Q = float(Q)
        w0 = 2 * np.pi * fc / fs
        cosw0 = np.cos(w0)
        alpha = np.sin(w0) / (2 * Q)
        a0 = 1 + alpha
        self.b0 = ((1 - cosw0) / 2) / a0
        self.b1 = (1 - cosw0) / a0
        self.b2 = ((1 - cosw0) / 2) / a0
        self.a1 = (-2 * cosw0) / a0
        self.a2 = (1 - alpha) / a0
        self.x1 = None
        self.x2 = None
        self.y1 = None
        self.y2 = None

    def reset_state(self, initial_x):
        """Reset the filter state with the initial input value."""
        initial_x = np.asarray(initial_x, dtype=np.float32)
        self.x1 = initial_x.copy()
        self.x2 = initial_x.copy()
        self.y1 = initial_x.copy()
        self.y2 = initial_x.copy()

    def step(self, x):
        """Apply one step of the biquad low-pass filter to the input x."""
        x = np.asarray(x, dtype=np.float32)
        if self.x1 is None:
            self.reset_state(x)
        y = (
            self.b0 * x
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2
        )
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y.astype(np.float32)


def _blend_trajectories(previous, current, step_interval_ns, duration_ns=None):
    if previous is None or len(previous) == 0 or duration_ns == 0:
        return current, 0

    count = min(len(previous), len(current))
    if duration_ns is not None:
        steps = (duration_ns + step_interval_ns - 1) // step_interval_ns
        count = min(count, steps + 1)
    output = current.copy()
    weights = np.linspace(1.0, 0.0, count, dtype=np.float32)[:, None]
    output[:count] = previous[:count] * weights + current[:count] * (1.0 - weights)
    return output, count


def _clear_queue(queue):
    while not queue.empty():
        queue.get_nowait()


def _put_latest(queue, event):
    _clear_queue(queue)
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


def _qpos_output(position):
    return pa.array(
        [{"qpos": np.asarray(position, dtype=np.float32)}],
        type=QPOS_TYPE,
    )


async def _main_executor(
    node,
    action_queue,
    command_queue,
    arms,
    use_upsample,
    use_filter,
    control_hz,
    blend_duration_ms,
):
    if not use_upsample and use_filter:
        print(
            "Warning: upsample is False, but filter is True. Forcing filter to False."
        )
        use_filter = False

    blend_description = (
        "full remaining trajectory"
        if blend_duration_ms is None
        else f"{blend_duration_ms:g} ms"
    )
    print(f"actions-executor trajectory blend: {blend_description}", flush=True)
    duration_ns = (
        None if blend_duration_ms is None else int(round(blend_duration_ms * 1e6))
    )

    enabled = False
    canceled_positions = None
    canceled_chunk_id = None
    lowpass = None

    while True:
        event_id, event = await _next_input(action_queue, command_queue)
        if event_id == "command":
            new_enabled = _apply_command(event, action_queue)
            if new_enabled is not None:
                enabled = new_enabled
                canceled_positions = None
                canceled_chunk_id = None
                lowpass = None
            continue
        if not enabled:
            continue

        interval = event["metadata"]["interval"]
        chunk_id = event["metadata"].get("chunk_id")
        # Filter cutoff frequency is 15 Hz by default, which is a common choice for robotic arm control to balance smoothness and responsiveness.
        cutoff = event["metadata"].get("cutoff_hz", 15)
        n_positions = len(event["value"])
        pos_shape = len(event["value"][0])
        reset = event["metadata"].get("reset", False)
        positions = event["value"].values.to_numpy().reshape(n_positions, pos_shape)

        # Build each new chunk on the execution grid before mixing trajectories.
        if use_upsample:
            step_interval_ns = int(1e9 / control_hz)
            loop_positions = _upsample_trajectory(positions, interval, step_interval_ns)

            if use_filter and lowpass is None:
                lowpass = BiquadLowpass(fs=control_hz, fc=cutoff)
        else:
            loop_positions = positions
            step_interval_ns = interval

        # On a reset, these actions are the first of a new episode, so drop any
        # trajectory carried over from the previous one instead of blending it.
        if reset:
            print("Resetting trajectory, discarding any previous trajectory.")
            canceled_positions = None
            canceled_chunk_id = None
            if lowpass is not None:
                lowpass.reset_state(loop_positions[0])

        # The old buffer contains only unsent, not-yet-filtered control points.
        blended_chunk_id = canceled_chunk_id
        loop_positions, blend_steps = _blend_trajectories(
            canceled_positions,
            loop_positions,
            step_interval_ns,
            duration_ns,
        )
        blend_duration_ns = max(0, blend_steps - 1) * step_interval_ns
        # Retain the existing recorder field as a rounded policy-span equivalent.
        blend_policy_points = (
            1 + (blend_duration_ns + interval - 1) // interval if blend_steps else 0
        )
        canceled_positions = None
        canceled_chunk_id = None

        # send motor command
        base_time = time.time_ns() - step_interval_ns

        for i_step, raw_position in enumerate(loop_positions):
            next_base_time = base_time + step_interval_ns
            sleep_time = next_base_time - time.time_ns()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time / 1e9)
            base_time = next_base_time

            # If there is a new event, cancel the current event.
            if not command_queue.empty():
                break
            if not action_queue.empty():
                canceled_positions = loop_positions[i_step:]
                canceled_chunk_id = chunk_id
                break

            position = raw_position
            # Conditionally apply low-pass filter
            if use_filter and lowpass is not None:
                position = lowpass.step(position)

            timestamp = time.time_ns()
            output_metadata = {"timestamp": timestamp}
            if chunk_id is not None:
                output_metadata["chunk_id"] = chunk_id
                output_metadata["blend_policy_points"] = blend_policy_points
                output_metadata["blend_duration_ns"] = blend_duration_ns
                received_timestamp = event["metadata"].get(
                    "executor_received_timestamp_ns"
                )
                if received_timestamp is not None:
                    output_metadata["executor_received_timestamp_ns"] = (
                        received_timestamp
                    )
                if blended_chunk_id is not None and i_step < blend_steps:
                    output_metadata["blended_chunk_id"] = blended_chunk_id
            offset = 0
            n_elements = 8  # 7 joints + 1 gripper
            if "right" in arms:
                right_position = position[offset : offset + n_elements]
                offset += n_elements
            else:
                right_position = None
            if "left" in arms:
                left_position = position[offset : offset + n_elements]
                offset += n_elements
            else:
                left_position = None
            if right_position is not None:
                node.send_output(
                    "move_position_right",
                    _qpos_output(right_position),
                    output_metadata,
                )
            if left_position is not None:
                node.send_output(
                    "move_position_left",
                    _qpos_output(left_position),
                    output_metadata,
                )


async def _main_dora(node, action_queue, command_queue, executor_task):
    while True:
        event = await asyncio.to_thread(node.next)
        if event["type"] != "INPUT":
            break

        # Main process
        if event["id"] == "actions":
            event["metadata"] = dict(event["metadata"])
            event["metadata"]["executor_received_timestamp_ns"] = time.time_ns()
            _put_latest(action_queue, event)
        elif event["id"] == "command":
            _put_latest(command_queue, event)
    executor_task.cancel()


async def _main_async(
    arms,
    use_upsample,
    use_filter,
    control_hz,
    blend_duration_ms,
):
    node = dora.Node()
    action_queue = asyncio.Queue(maxsize=1)
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
            blend_duration_ms,
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
    """Execute timestamped actions."""
    parser = argparse.ArgumentParser(description="Execute timestamped actions")
    parser.add_argument(
        "--arms",
        default=os.getenv("ARMS", "right,left"),
        help="The used arms: 'right,left' (default), 'right' or 'left'",
        type=str,
    )
    parser.add_argument(
        "--upsample",
        action="store_true",
        help="Whether to upsample the actions using shape-preserving PCHIP",
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Whether to apply low-pass filter to the upsampled actions (only works if `upsample` is set)",
    )
    parser.add_argument(
        "--control-hz",
        default=250.0,
        type=float,
        help="motor control frequency (Hz)",
    )
    parser.add_argument(
        "--blend-duration-ms",
        default=os.getenv("ACTION_BLEND_DURATION_MS"),
        type=_nonnegative_float,
        help="handoff duration in milliseconds (unset: full overlap; 0: no blend)",
    )

    args = parser.parse_args()
    arms = args.arms.split(",")

    asyncio.run(
        _main_async(
            arms,
            use_upsample=args.upsample,
            use_filter=args.filter,
            control_hz=args.control_hz,
            blend_duration_ms=args.blend_duration_ms,
        )
    )


if __name__ == "__main__":
    main()
