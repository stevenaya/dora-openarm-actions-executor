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
import time


QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])


# Hermite cubic spline interpolation for upsampling
class HermiteUpsampler:
    """Upsamples coarse trajectory chunks using cubic Hermite spline interpolation."""

    def __init__(self, chunk_hz, horizon_sec):
        """Initialize the upsampler with the chunk frequency and horizon."""
        self.chunk_hz = float(chunk_hz)
        self.horizon_sec = float(horizon_sec)
        self.dt_chunk = 1.0 / self.chunk_hz
        self.t_chunk = np.arange(0.0, self.horizon_sec + 1e-12, self.dt_chunk)

    @staticmethod
    def _compute_slopes(t, y):
        s = np.diff(y, axis=0) / np.diff(t)[:, None]
        m = np.zeros_like(y)
        m[0] = s[0]
        m[-1] = s[-1]
        for i in range(1, len(y) - 1):
            prod = s[i - 1] * s[i]
            mask = prod <= 0.0
            mi = 0.5 * (s[i - 1] + s[i])
            mi[mask] = 0.0
            m[i] = mi
        return m

    def upsample(self, y_chunk, t_eval):
        """Upsample the given chunk of trajectory to the specified evaluation times."""
        y = np.asarray(y_chunk, dtype=np.float32)
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


async def _main_executor(node, events, arms, use_upsample, use_filter, control_hz):
    def blend(canceled_positions, next_positions):
        n = len(canceled_positions)
        overlapped_positions = next_positions[:n]
        weights = np.linspace(1, 0, n, dtype=np.float32).reshape(n, 1)
        blended = canceled_positions * weights + overlapped_positions * (1 - weights)
        return blended, n

    if not use_upsample and use_filter:
        print(
            "Warning: upsample is False, but filter is True. Forcing filter to False."
        )
        use_filter = False

    canceled_positions = None

    upsampler = None
    lowpass = None
    dynamic_chunk_hz = None
    target_interval_ns = None
    target_interval_s = None
    t_eval = None

    while True:
        event = await events.get()
        interval = event["metadata"]["interval"]
        # Filter cutoff frequency is 15 Hz by default, which is a common choice for robotic arm control to balance smoothness and responsiveness.
        cutoff = event["metadata"].get("cutoff_hz", 15)
        n_positions = len(event["value"])
        pos_shape = len(event["value"][0])
        reset = event["metadata"].get("reset", False)
        positions = event["value"].values.to_numpy().reshape(n_positions, pos_shape)

        # Initialize upsampler and low-pass filter if needed
        if use_upsample and upsampler is None:
            dynamic_chunk_hz = 1e9 / interval
            horizon_sec = (n_positions - 1) / dynamic_chunk_hz

            upsampler = HermiteUpsampler(
                chunk_hz=dynamic_chunk_hz, horizon_sec=horizon_sec
            )

            target_interval_s = 1.0 / float(control_hz)
            target_interval_ns = int(target_interval_s * 1e9)

            t_eval = np.arange(0.0, horizon_sec + 1e-9, target_interval_s)

            if use_filter:
                lowpass = BiquadLowpass(fs=control_hz, fc=cutoff)

        # On a reset, these actions are the first of a new episode, so drop any
        # trajectory carried over from the previous one instead of blending it.
        if reset:
            print("Resetting trajectory, discarding any previous trajectory.")
            canceled_positions = None
            # Also re-initialize the low-pass filter to the new episode's first
            # pose. Otherwise its retained state pulls the first samples toward
            # the previous episode's final pose, causing a jerk/ramp at start.
            # positions[0] equals the first upsampled sample (Hermite at t=0).
            if lowpass is not None:
                lowpass.reset_state(positions[0])

        # blend trajectory
        if canceled_positions is not None:
            blended, n = blend(canceled_positions, positions)
            positions = np.concatenate([blended, positions[n:]])
            canceled_positions = None

        # Conditionally upsample
        if use_upsample:
            loop_positions = upsampler.upsample(positions, t_eval)
            step_interval_ns = target_interval_ns
            step_interval_s = target_interval_s
        else:
            loop_positions = positions
            step_interval_ns = interval
            step_interval_s = interval / 1e9

        # send motor command
        base_time = time.time_ns() - step_interval_ns

        for i_step, position in enumerate(loop_positions):
            # Conditionally apply low-pass filter
            if use_filter and lowpass is not None:
                position = lowpass.step(position)

            next_base_time = base_time + step_interval_ns
            sleep_time = next_base_time - time.time_ns()
            if sleep_time > 0:
                await asyncio.sleep(sleep_time / 1e9)

            base_time = next_base_time
            timestamp = time.time_ns()

            # If there is a new event, cancel the current event.
            if not events.empty():
                if use_upsample:
                    consumed_time_s = i_step * step_interval_s
                    consumed_raw_steps = int(consumed_time_s * dynamic_chunk_hz)
                else:
                    # If not upsampling, i_step corresponds directly to the original steps
                    consumed_raw_steps = i_step
                canceled_positions = positions[consumed_raw_steps:]
                break

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
                    pa.array([{"qpos": right_position}], type=QPOS_TYPE),
                    {"timestamp": timestamp},
                )
            if left_position is not None:
                node.send_output(
                    "move_position_left",
                    pa.array([{"qpos": left_position}], type=QPOS_TYPE),
                    {"timestamp": timestamp},
                )


async def _main_dora(node, events, executor_task):
    while True:
        if node.is_empty():
            await asyncio.sleep(0.1)
            continue
        event = node.next()
        if event["type"] != "INPUT":
            break

        # Main process
        await events.put(event)
    executor_task.cancel()


async def _main_async(arms, use_upsample, use_filter, control_hz):
    node = dora.Node()
    events = asyncio.Queue()
    executor_task = asyncio.create_task(
        _main_executor(node, events, arms, use_upsample, use_filter, control_hz)
    )
    dora_task = asyncio.create_task(_main_dora(node, events, executor_task))

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
