"""Common trajectory timing and handoff, independent of the policy algorithm."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.interpolate import PchipInterpolator


def _upsample_trajectory(positions, interval_ns, step_interval_ns, offset_ns=0):
    """Use shape-preserving PCHIP on a fixed grid that includes the final pose."""
    positions = np.asarray(positions, dtype=np.float32)
    if len(positions) == 1:
        return positions
    knot_ns = np.arange(len(positions), dtype=np.int64) * interval_ns
    horizon_ns = knot_ns[-1]
    steps = max(0, (horizon_ns - offset_ns + step_interval_ns - 1) // step_interval_ns)
    sample_ns = offset_ns + np.arange(steps + 1, dtype=np.int64) * step_interval_ns
    # Keep the command period fixed; sample past the horizon holds the final pose.
    sample_ns = np.minimum(sample_ns, horizon_ns)
    interpolator = PchipInterpolator(
        knot_ns / 1e9, positions, axis=0, extrapolate=False
    )
    output = interpolator(sample_ns / 1e9).astype(np.float32)
    output[-1] = positions[-1]
    return output


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


@dataclass
class Chunk:
    """Parsed input or an adopted, pre-filter control trajectory."""

    positions: np.ndarray
    interval_ns: int
    start_ns: int
    metadata: dict
    takeover_ns: int = 0
    max_lateness_ns: int | None = None
    owner: object = None

    @classmethod
    def from_event(cls, event):
        """Read optional timing once; ordinary timestamps are not action targets."""
        value, metadata = event["value"], event["metadata"]
        interval = int(metadata["interval"])
        origin = metadata.get("action_origin_timestamp_ns")
        first = (
            int(origin) + int(metadata.get("action_window_start", 0)) * interval
            if origin
            else int(metadata.get("takeover_timestamp_ns") or 0)
        )
        return cls(
            value.values.to_numpy().reshape(len(value), len(value[0])),
            interval,
            first,
            metadata,
            int(metadata.get("takeover_timestamp_ns") or first),
            metadata.get("max_lateness_ns"),
            event,
        )


class TrajectoryScheduler:
    """One active trajectory and one pending chunk for both handoff modes."""

    def __init__(self):
        """Start without an active episode or trajectory."""
        self.active = self.pending = None
        self.attempt_id = None
        self.cursor = self.blend_steps = 0
        self.blended_chunk_id = None

    def clear(self, metadata):
        """Discard the current and pending trajectories on a lifecycle command."""
        self.active = self.pending = None
        self.attempt_id = metadata.get("episode_attempt_id")
        self.cursor = self.blend_steps = 0
        self.blended_chunk_id = None

    def submit(self, chunk):
        """Fence Start's first chunk, then accept task changes from the ordered action stream."""
        metadata = chunk.metadata
        attempt = metadata.get("episode_attempt_id")
        if (
            self.active is None and self.attempt_id is not None
            and attempt != self.attempt_id
        ):
            return "episode changed"
        if attempt is not None and attempt != self.attempt_id:
            metadata = {**metadata, "reset": True}
            chunk = replace(chunk, metadata=metadata)
        current_id = (
            self.active.metadata.get("chunk_id", "") if self.active is not None else ""
        )
        if (
            "based_on_chunk_id" in metadata
            and not metadata.get("reset")
            and metadata["based_on_chunk_id"] != current_id
        ):
            return "execution plan changed"
        self.pending = chunk
        return None

    def take(self, now_ns):
        """Return a due chunk or a rejection; an invalid result leaves the old plan active."""
        chunk = self.pending
        if chunk is None or now_ns < chunk.takeover_ns:
            return None, None
        self.pending = None
        if (
            chunk.takeover_ns
            and chunk.max_lateness_ns is not None
            and now_ns > chunk.takeover_ns + chunk.max_lateness_ns
        ):
            return chunk, "late beyond tolerance"
        if (
            chunk.start_ns
            and now_ns > chunk.start_ns + (len(chunk.positions) - 1) * chunk.interval_ns
        ):
            return chunk, "selected window expired"
        return chunk, None

    def adopt(self, chunk, now_ns, step_ns, mode, duration_ns, upsample):
        """Align raw trajectories, then apply only the selected handoff operation."""
        offset = max(0, now_ns - chunk.start_ns) if chunk.start_ns else 0
        current = (
            _upsample_trajectory(chunk.positions, chunk.interval_ns, step_ns, offset)
            if upsample or chunk.start_ns
            else chunk.positions
        )
        old, previous = self.active, None
        if (
            mode == "blend"
            and old is not None
            and self.cursor < len(old.positions)
            and not chunk.metadata.get("reset")
        ):
            if chunk.start_ns:
                old_offset = max(0, now_ns - old.start_ns)
                if old_offset <= (len(old.positions) - 1) * old.interval_ns:
                    previous = _upsample_trajectory(
                        old.positions, old.interval_ns, step_ns, old_offset
                    )
            else:
                previous = old.positions[self.cursor :]
                if old.interval_ns != step_ns:
                    previous = _upsample_trajectory(previous, old.interval_ns, step_ns)
        current, self.blend_steps = _blend_trajectories(
            previous, current, step_ns, duration_ns
        )
        self.blended_chunk_id = (
            old.metadata.get("chunk_id") if self.blend_steps else None
        )
        self.active = replace(
            chunk, positions=current, interval_ns=step_ns, start_ns=now_ns
        )
        attempt = chunk.metadata.get("episode_attempt_id")
        if attempt is not None:
            self.attempt_id = attempt
        self.cursor = 0

    @property
    def next_timestamp_ns(self):
        """Next unsent command time, or None while holding the final command."""
        if self.active is None or self.cursor >= len(self.active.positions):
            return None
        return self.active.start_ns + self.cursor * self.active.interval_ns
