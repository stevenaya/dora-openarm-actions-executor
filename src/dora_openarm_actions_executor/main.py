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
from collections import deque
import os
import time

import dora
import numpy as np
import pyarrow as pa
from scipy.signal import sosfiltfilt

from dora_openarm_actions_executor.trajectory import Chunk, TrajectoryScheduler

QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])
START_COMMANDS = {"start"}
STOP_COMMANDS = {"stop", "intervene", "quit"}


def _nonnegative_float(value):
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


def _positive_float(value):
    parsed = _nonnegative_float(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


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


class OneEuroLowpass:
    """Per-joint One Euro filter on the fixed control-rate grid."""

    def __init__(self, fs, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        """Configure the base cutoff and speed-dependent cutoff increase."""
        self.fs, self.min_cutoff, self.beta = float(fs), float(min_cutoff), float(beta)
        self.derivative_alpha = self._alpha(float(d_cutoff))
        self.filtered = self.derivative = None

    def _alpha(self, cutoff):
        return 1.0 / (1.0 + self.fs / (2.0 * np.pi * cutoff))

    def reset_state(self, initial_x):
        """Start at the new reference without retaining its predecessor's speed."""
        self.filtered = np.array(initial_x, dtype=np.float64, copy=True)
        self.derivative = np.zeros_like(self.filtered)

    def step(self, x):
        """Update only for an emitted control point, independently per joint."""
        x = np.asarray(x, dtype=np.float64)
        if self.filtered is None:
            self.reset_state(x)
            return self.filtered.astype(np.float32)
        # Match the reference algorithm's derivative from the previous filtered value.
        velocity = (x - self.filtered) * self.fs
        self.derivative += self.derivative_alpha * (velocity - self.derivative)
        cutoff = self.min_cutoff + self.beta * np.abs(self.derivative)
        self.filtered += self._alpha(cutoff) * (x - self.filtered)
        return self.filtered.astype(np.float32)


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
    try:
        done, _ = await asyncio.wait(
            {action_task, command_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for task in (action_task, command_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(action_task, command_task, return_exceptions=True)

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


def _zero_phase_commands(positions, lowpass, history):
    """Filter known future points with raw past context, never feeding outputs back."""
    context = np.concatenate((np.asarray(history), positions)) if history else positions
    sos = [[lowpass.b0, lowpass.b1, lowpass.b2, 1.0, lowpass.a1, lowpass.a2]]
    filtered = sosfiltfilt(sos, context, axis=0, padlen=min(9, len(context) - 1))
    return filtered[len(history):].astype(np.float32)


async def _main_executor(
    node,
    action_queue,
    command_queue,
    arms,
    use_upsample,
    use_filter,
    control_hz,
    blend_duration_ms,
    mode="blend",
    filter_mode="causal",
    filter_cutoff_hz=15.0,
    filter_q=0.5,
    one_euro_min_cutoff_hz=1.0,
    one_euro_beta=0.0,
    one_euro_d_cutoff_hz=1.0,
    filter_grippers=True,
):
    if not use_upsample and use_filter:
        print("Warning: filter requires upsample; disabling filter.", flush=True)
        use_filter = False
    filter_settings = (
        f"min_cutoff_hz={one_euro_min_cutoff_hz}, beta={one_euro_beta}, "
        f"d_cutoff_hz={one_euro_d_cutoff_hz}"
        if filter_mode == "one-euro"
        else f"filter_cutoff_hz={filter_cutoff_hz}, filter_q={filter_q}"
    )
    print(
        f"actions-executor mode={mode}, blend_duration_ms={blend_duration_ms}, "
        f"filter={filter_mode if use_filter else 'off'}, "
        f"{filter_settings}, filter_grippers={filter_grippers}",
        flush=True,
    )
    duration_ns = None if blend_duration_ms is None else round(blend_duration_ms * 1e6)
    control_interval = int(1e9 / control_hz)
    scheduler = TrajectoryScheduler()
    enabled = False
    lowpass = plan = None
    filter_history = filtered_positions = None
    publish_plan = False

    def send_plan(value):
        metadata = (
            {}
            if scheduler.attempt_id is None
            else {"episode_attempt_id": scheduler.attempt_id}
        )
        node.send_output("execution_plan", pa.array([value]), metadata)

    def feedback(metadata):
        # A rejection still acknowledges the sample, even with no adopted plan yet.
        nonlocal plan
        value = dict(plan) if plan is not None else {"chunk_id": "", "positions": []}
        value["sample_chunk_id"] = metadata.get("chunk_id")
        for source, target in (
            ("executor_received_timestamp_ns", "received_timestamp_ns"),
            ("inference_started_timestamp_ns", "inference_started_timestamp_ns"),
        ):
            value.pop(target, None)
            if source in metadata:
                value[target] = metadata[source]
        send_plan(value)
        if plan is not None:
            plan = value

    def reject(chunk, reason):
        print(f"Dropped chunk={chunk.metadata.get('chunk_id')}: {reason}", flush=True)
        if reason != "episode changed" and "based_on_chunk_id" in chunk.metadata:
            feedback(chunk.metadata)

    while True:
        if (
            not command_queue.empty()
            or not action_queue.empty()
            or (scheduler.pending is None and scheduler.next_timestamp_ns is None)
        ):
            event_id, event = await _next_input(action_queue, command_queue)
            if event_id == "command":
                new_enabled = _apply_command(event, action_queue)
                if new_enabled is not None:
                    enabled = new_enabled
                    scheduler.clear(event.get("metadata", {}))
                    lowpass = None
                    filter_history = filtered_positions = None
                    if plan is not None:
                        send_plan(None)
                    plan = None
                    publish_plan = False
                continue
            if not enabled:
                continue
            chunk = Chunk.from_event(event)
            reason = scheduler.submit(chunk)
            if reason:
                reject(chunk, reason)

        now_ns = time.time_ns()
        chunk, reason = scheduler.take(now_ns)
        if reason:
            reject(chunk, reason)
        elif chunk is not None:
            step_ns = control_interval if use_upsample else chunk.interval_ns
            scheduler.adopt(chunk, now_ns, step_ns, mode, duration_ns, use_upsample)
            if use_filter and lowpass is None:
                if filter_mode == "one-euro":
                    lowpass = OneEuroLowpass(
                        control_hz, one_euro_min_cutoff_hz, one_euro_beta, one_euro_d_cutoff_hz
                    )
                else:
                    lowpass = BiquadLowpass(fs=control_hz, fc=filter_cutoff_hz, Q=filter_q)
                if filter_mode == "zero-phase":
                    filter_history = deque(maxlen=max(9, int(3 * control_hz / filter_cutoff_hz)))
            if chunk.metadata.get("reset") and lowpass is not None:
                lowpass.reset_state(scheduler.active.positions[0])
                if filter_history is not None:
                    filter_history.clear()
            if lowpass is not None and filter_mode == "zero-phase":
                filtered_positions = _zero_phase_commands(
                    scheduler.active.positions, lowpass, filter_history
                )
            publish_plan = plan is not None or "based_on_chunk_id" in chunk.metadata
            command_metadata = {}
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id is not None:
                blend_ns = max(0, scheduler.blend_steps - 1) * step_ns
                policy_interval = chunk.metadata["interval"]
                command_metadata.update(
                    chunk_id=chunk_id,
                    blend_duration_ns=blend_ns,
                    blend_policy_points=(
                        1 + (blend_ns + policy_interval - 1) // policy_interval
                        if scheduler.blend_steps
                        else 0
                    ),
                )
                if "executor_received_timestamp_ns" in chunk.metadata:
                    command_metadata["executor_received_timestamp_ns"] = chunk.metadata[
                        "executor_received_timestamp_ns"
                    ]

        next_ns = scheduler.next_timestamp_ns
        if next_ns is not None and now_ns >= next_ns:
            active, i_step = scheduler.active, scheduler.cursor
            if i_step == 0 and publish_plan:
                plan = {
                    "chunk_id": active.metadata.get("chunk_id", ""),
                    "start_timestamp_ns": active.start_ns,
                    "interval_ns": active.interval_ns,
                    # RTC consumes pre-filter targets; filtering the prior again causes drift.
                    "positions": active.positions.tolist(),
                }
                feedback(active.metadata)
            position = active.positions[i_step]
            if lowpass is not None:
                if filter_mode == "zero-phase":
                    filter_history.append(position.copy())
                    position = filtered_positions[i_step]
                else:
                    position = lowpass.step(position)
                if not filter_grippers:
                    position[7::8] = active.positions[i_step, 7::8]
            output_metadata = {**command_metadata, "timestamp": time.time_ns()}
            if (
                scheduler.blended_chunk_id is not None
                and i_step < scheduler.blend_steps
            ):
                output_metadata["blended_chunk_id"] = scheduler.blended_chunk_id
            offset = 0
            for arm in ("right", "left"):
                if arm in arms:
                    node.send_output(
                        f"move_position_{arm}",
                        _qpos_output(position[offset : offset + 8]),
                        output_metadata,
                    )
                    offset += 8
            scheduler.cursor += 1

        deadlines = []
        if scheduler.next_timestamp_ns is not None:
            deadlines.append(scheduler.next_timestamp_ns)
        if scheduler.pending is not None:
            deadlines.append(scheduler.pending.takeover_ns)
        if deadlines:
            delay_ns = max(0, min(deadlines) - time.time_ns())
            await asyncio.sleep(min(delay_ns, control_interval) / 1e9)


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
    mode="blend",
    filter_mode="causal",
    filter_cutoff_hz=15.0,
    filter_q=0.5,
    one_euro_min_cutoff_hz=1.0,
    one_euro_beta=0.0,
    one_euro_d_cutoff_hz=1.0,
    filter_grippers=True,
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
            mode,
            filter_mode,
            filter_cutoff_hz,
            filter_q,
            one_euro_min_cutoff_hz,
            one_euro_beta,
            one_euro_d_cutoff_hz,
            filter_grippers,
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
        "--filter-grippers",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ACTION_FILTER_GRIPPERS", "true").strip().lower()
        in ("1", "true", "yes", "on"),
        help="filter grippers too; --no-filter-grippers keeps their pre-filter targets (default: enabled)",
    )
    parser.add_argument(
        "--filter-mode",
        choices=("causal", "zero-phase", "one-euro"),
        default=os.getenv("ACTION_FILTER_MODE", "causal"),
        help="with --filter: causal/zero-phase biquad, or adaptive one-euro",
    )
    parser.add_argument(
        "--filter-cutoff-hz",
        type=_positive_float,
        default=os.getenv("ACTION_FILTER_CUTOFF_HZ", "15"),
        help="biquad cutoff parameter in Hz; ignores action cutoff_hz metadata (default: 15)",
    )
    parser.add_argument(
        "--filter-q",
        type=_positive_float,
        default=os.getenv("ACTION_FILTER_Q", "0.5"),
        help="biquad Q factor (default: 0.5)",
    )
    parser.add_argument(
        "--one-euro-min-cutoff-hz",
        type=_positive_float,
        default=os.getenv("ACTION_ONE_EURO_MIN_CUTOFF_HZ", "1"),
        help="One Euro base cutoff (Hz, default: 1)",
    )
    parser.add_argument(
        "--one-euro-beta",
        type=_nonnegative_float,
        default=os.getenv("ACTION_ONE_EURO_BETA", "0"),
        help="One Euro speed coefficient; zero disables adaptation (default: 0)",
    )
    parser.add_argument(
        "--one-euro-d-cutoff-hz",
        type=_positive_float,
        default=os.getenv("ACTION_ONE_EURO_D_CUTOFF_HZ", "1"),
        help="One Euro derivative cutoff (Hz, default: 1)",
    )
    parser.add_argument(
        "--control-hz",
        default=250.0,
        type=_positive_float,
        help="motor control frequency (Hz)",
    )
    parser.add_argument(
        "--blend-duration-ms",
        default=os.getenv("ACTION_BLEND_DURATION_MS"),
        type=_nonnegative_float,
        help="handoff duration in milliseconds (unset: full overlap; 0: no blend)",
    )

    parser.add_argument(
        "--mode",
        choices=("blend", "switch"),
        default=os.getenv("ACTION_EXECUTION_MODE", "blend"),
        help="handoff operation; both modes honor optional action target timestamps",
    )

    args = parser.parse_args()
    if (args.filter and args.upsample and args.filter_mode != "one-euro"
            and args.filter_cutoff_hz >= args.control_hz / 2):
        parser.error("--filter-cutoff-hz must be less than half --control-hz")
    arms = args.arms.split(",")

    asyncio.run(
        _main_async(
            arms,
            use_upsample=args.upsample,
            use_filter=args.filter,
            control_hz=args.control_hz,
            blend_duration_ms=args.blend_duration_ms,
            mode=args.mode,
            filter_mode=args.filter_mode,
            filter_cutoff_hz=args.filter_cutoff_hz,
            filter_q=args.filter_q,
            one_euro_min_cutoff_hz=args.one_euro_min_cutoff_hz,
            one_euro_beta=args.one_euro_beta,
            one_euro_d_cutoff_hz=args.one_euro_d_cutoff_hz,
            filter_grippers=args.filter_grippers,
        )
    )


if __name__ == "__main__":
    main()
