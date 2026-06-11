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
from dataclasses import dataclass
import dora
import os
import numpy as np
import pyarrow as pa
import time


ELEMENTS_PER_ARM = 8  # 7 joints + 1 gripper
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "success", "fail", "cancel", "quit"}


# PCHIP interpolation for upsampling
class PchipUpsampler:
    """Upsamples coarse trajectory chunks using PCHIP interpolation."""

    def __init__(self, chunk_hz, horizon_sec):
        """Initialize the upsampler with the chunk frequency and horizon."""
        self.chunk_hz = float(chunk_hz)
        self.horizon_sec = float(horizon_sec)
        self.dt_chunk = 1.0 / self.chunk_hz
        self.t_chunk = np.arange(0.0, self.horizon_sec + 1e-12, self.dt_chunk)

    @staticmethod
    def _edge_slope(h0, h1, s0, s1):
        slope = ((2.0 * h0 + h1) * s0 - h0 * s1) / (h0 + h1)
        opposite_sign = np.sign(slope) != np.sign(s0)
        too_steep = (np.sign(s0) != np.sign(s1)) & (
            np.abs(slope) > 3.0 * np.abs(s0)
        )
        slope[opposite_sign] = 0.0
        slope[~opposite_sign & too_steep] = 3.0 * s0[~opposite_sign & too_steep]
        return slope

    @staticmethod
    def _compute_slopes(t, y):
        h = np.diff(t)
        secants = np.diff(y, axis=0) / h[:, None]
        slopes = np.zeros_like(y)

        if len(y) == 1:
            return slopes
        if len(y) == 2:
            slopes[0] = secants[0]
            slopes[1] = secants[0]
            return slopes

        h_prev = h[:-1, None]
        h_next = h[1:, None]
        s_prev = secants[:-1]
        s_next = secants[1:]
        same_sign = s_prev * s_next > 0.0
        w1 = 2.0 * h_next + h_prev
        w2 = h_next + 2.0 * h_prev

        with np.errstate(divide="ignore", invalid="ignore"):
            interior = (w1 + w2) / (w1 / s_prev + w2 / s_next)
        slopes[1:-1] = np.where(same_sign, interior, 0.0)
        slopes[0] = PchipUpsampler._edge_slope(h[0], h[1], secants[0], secants[1])
        slopes[-1] = PchipUpsampler._edge_slope(
            h[-1], h[-2], secants[-1], secants[-2]
        )
        return slopes

    def upsample(self, y_chunk, t_eval):
        """Upsample the given chunk of trajectory to the specified evaluation times."""
        y = np.asarray(y_chunk, dtype=np.float64)
        if y.shape[0] != len(self.t_chunk):
            raise ValueError(
                f"Expected chunk length {len(self.t_chunk)}, got {y.shape[0]}"
            )

        slopes = self._compute_slopes(self.t_chunk, y)
        idx = np.searchsorted(self.t_chunk, t_eval, side="right") - 1
        idx = np.clip(idx, 0, len(self.t_chunk) - 2)

        t0 = self.t_chunk[idx]
        t1 = self.t_chunk[idx + 1]
        h = t1 - t0
        u = (t_eval - t0) / h

        h00 = 2 * u**3 - 3 * u**2 + 1
        h10 = (u**3 - 2 * u**2 + u) * h
        h01 = -2 * u**3 + 3 * u**2
        h11 = (u**3 - u**2) * h

        y0 = y[idx]
        y1 = y[idx + 1]
        m0 = slopes[idx]
        m1 = slopes[idx + 1]

        return (
            h00[:, None] * y0
            + h10[:, None] * m0
            + h01[:, None] * y1
            + h11[:, None] * m1
        ).astype(np.float32)


# Tustin bilinear transform based biquad low-pass filter
class BiquadLowpass:
    """Biquad low-pass filter for smoothing outputs."""

    def __init__(self, fs, fc, Q=0.707):
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


@dataclass
class _ActionChunk:
    positions: np.ndarray
    interval_ns: int
    cutoff_hz: float


@dataclass
class _ProcessorState:
    upsampler: PchipUpsampler | None = None
    lowpass: BiquadLowpass | None = None
    dynamic_chunk_hz: float | None = None
    target_interval_ns: int | None = None
    target_interval_s: float | None = None
    t_eval: np.ndarray | None = None


def _parse_event(event):
    metadata = event["metadata"]
    interval_ns = metadata["interval"]
    cutoff_hz = metadata.get("cutoff_hz", 15)
    n_positions = len(event["value"])
    pos_shape = len(event["value"][0])
    positions = event["value"].values.to_numpy().reshape(n_positions, pos_shape)
    return _ActionChunk(
        positions=positions,
        interval_ns=interval_ns,
        cutoff_hz=cutoff_hz,
    )


def _build_or_validate_processors(chunk, state, use_upsample, use_filter, control_hz):
    if not use_upsample or state.upsampler is not None:
        return state

    state.dynamic_chunk_hz = 1e9 / chunk.interval_ns
    horizon_sec = (len(chunk.positions) - 1) / state.dynamic_chunk_hz

    state.upsampler = PchipUpsampler(
        chunk_hz=state.dynamic_chunk_hz,
        horizon_sec=horizon_sec,
    )

    state.target_interval_s = 1.0 / float(control_hz)
    state.target_interval_ns = int(state.target_interval_s * 1e9)
    state.t_eval = np.arange(0.0, horizon_sec + 1e-9, state.target_interval_s)

    if use_filter:
        state.lowpass = BiquadLowpass(fs=control_hz, fc=chunk.cutoff_hz)

    return state


def _blend_remaining(remaining_raw_positions, next_positions):
    if remaining_raw_positions is None:
        return next_positions

    n = len(remaining_raw_positions)
    overlapped_positions = next_positions[:n]
    weights = np.linspace(1, 0, n, dtype=np.float32).reshape(n, 1)
    blended = remaining_raw_positions * weights + overlapped_positions * (1 - weights)
    return np.concatenate([blended, next_positions[n:]])


def _clear_queue(queue):
    while not queue.empty():
        queue.get_nowait()


def _command_value(event):
    return event["value"][0].as_py()


async def _get_next_input(action_queue, command_queue):
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
            # A command wins races against an action; the action may be stale.
            action_task.result()
        return "command", command_task.result()
    return "actions", action_task.result()


def _apply_command(event, action_queue):
    command = _command_value(event)
    _clear_queue(action_queue)
    enabled = command in START_COMMANDS
    if command in START_COMMANDS | STOP_COMMANDS:
        print(f"actions-executor command={command}: reset state, enabled={enabled}", flush=True)
    return enabled, None, _ProcessorState()


def _split_arm_positions(position, arms):
    offset = 0
    if "right" in arms:
        right_position = position[offset : offset + ELEMENTS_PER_ARM]
        offset += ELEMENTS_PER_ARM
    else:
        right_position = None
    if "left" in arms:
        left_position = position[offset : offset + ELEMENTS_PER_ARM]
    else:
        left_position = None
    return right_position, left_position


def _send_arm_outputs(node, right_position, left_position, timestamp):
    if right_position is not None:
        if left_position is None:
            node.send_output(
                "move_position_right",
                right_position,
                {"timestamp": timestamp},
            )
        else:
            node.send_output(
                "move_position_right",
                pa.StructArray.from_arrays(
                    [right_position, left_position],
                    names=("new_position", "other_arm_position"),
                ),
                {"timestamp": timestamp},
            )
    if left_position is not None:
        if right_position is None:
            node.send_output(
                "move_position_left",
                left_position,
                {"timestamp": timestamp},
            )
        else:
            node.send_output(
                "move_position_left",
                pa.StructArray.from_arrays(
                    [left_position, right_position],
                    names=("new_position", "other_arm_position"),
                ),
                {"timestamp": timestamp},
            )


async def _main_executor(
    node, action_queue, command_queue, arms, use_upsample, use_filter, control_hz
):
    if not use_upsample and use_filter:
        print(
            "Warning: upsample is False, but filter is True. Forcing filter to False."
        )
        use_filter = False

    remaining_raw_positions = None
    processor_state = _ProcessorState()
    enabled = False

    while True:
        event_id, event = await _get_next_input(action_queue, command_queue)
        if event_id == "command":
            enabled, remaining_raw_positions, processor_state = _apply_command(
                event, action_queue
            )
            continue
        if not enabled:
            continue

        chunk = _parse_event(event)
        processor_state = _build_or_validate_processors(
            chunk,
            processor_state,
            use_upsample,
            use_filter,
            control_hz,
        )

        positions = _blend_remaining(remaining_raw_positions, chunk.positions)
        remaining_raw_positions = None

        # Conditionally upsample
        if use_upsample:
            loop_positions = processor_state.upsampler.upsample(
                positions, processor_state.t_eval
            )
            step_interval_ns = processor_state.target_interval_ns
            step_interval_s = processor_state.target_interval_s
        else:
            loop_positions = positions
            step_interval_ns = chunk.interval_ns
            step_interval_s = chunk.interval_ns / 1e9

        # send motor command
        base_time = time.monotonic_ns() - step_interval_ns

        for i_step, position in enumerate(loop_positions):
            # Conditionally apply low-pass filter
            if use_filter and processor_state.lowpass is not None:
                position = processor_state.lowpass.step(position)

            next_base_time = base_time + step_interval_ns
            sleep_time = next_base_time - time.monotonic_ns()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time / 1e9)

            base_time = next_base_time
            timestamp = time.time_ns()

            if not command_queue.empty():
                event = command_queue.get_nowait()
                enabled, remaining_raw_positions, processor_state = _apply_command(
                    event, action_queue
                )
                break

            # If there is a new event, cancel the current event.
            if not action_queue.empty():
                if use_upsample:
                    consumed_time_s = i_step * step_interval_s
                    consumed_raw_steps = int(
                        consumed_time_s * processor_state.dynamic_chunk_hz
                    )
                else:
                    # If not upsampling, i_step corresponds directly to the original steps
                    consumed_raw_steps = i_step
                remaining_raw_positions = positions[consumed_raw_steps:]
                break

            right_position, left_position = _split_arm_positions(position, arms)
            _send_arm_outputs(node, right_position, left_position, timestamp)


def _put_latest(queue, event):
    _clear_queue(queue)
    queue.put_nowait(event)


async def _main_dora(node, action_queue, command_queue, executor_task):
    while True:
        event = await asyncio.to_thread(node.next)
        if event["type"] != "INPUT":
            break

        event_id = event["id"]
        if event_id == "actions":
            _put_latest(action_queue, event)
        elif event_id == "command":
            _put_latest(command_queue, event)
    executor_task.cancel()


async def _main_async(arms, use_upsample, use_filter, control_hz):
    node = dora.Node()
    action_queue = asyncio.Queue(maxsize=1)
    command_queue = asyncio.Queue(maxsize=1)
    executor_task = asyncio.create_task(
        _main_executor(node, action_queue, command_queue, arms, use_upsample, use_filter, control_hz)
    )
    dora_task = asyncio.create_task(_main_dora(node, action_queue, command_queue, executor_task))

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
        help="Whether to upsample the actions",
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

    args = parser.parse_args()
    arms = args.arms.split(",")

    asyncio.run(
        _main_async(
            arms,
            use_upsample=args.upsample,
            use_filter=args.filter,
            control_hz=args.control_hz,
        )
    )


if __name__ == "__main__":
    main()
