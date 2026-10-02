# dora-openarm-actions-executor

A [Dora](https://dora-rs.ai/) node that executes joint-position action chunks for OpenArm.

## Usage

```yaml
- id: actions-executor
  path: dora-openarm-actions-executor
  args: "--upsample --filter --control-hz 250 --blend-duration-ms 100"
  inputs:
    actions: policy-server/actions
    command: evaluation-ui/arm_command
  outputs:
    - move_position_right
    - move_position_left
```

The executor starts disabled. Send `start` to enable output;
`stop`, `intervene`, and `quit` disable output and clear pending actions,
trajectory, filter, and provenance state. These commands do not disable motors:
wire the arm/driver lifecycle separately.

`actions` is an Arrow `list<float32>` array of shape `[T, D]`: 8 values per
arm (7 joints and a gripper), right arm then left for bimanual input. Metadata
requires `interval` in nanoseconds between policy points; optional fields are
`cutoff_hz` (default 15 Hz), `reset`, and `chunk_id`. Each arm output is a
length-one struct `[{"qpos": [...]}]` containing `list<float32>`.

## Execution

The reader and executor run separately. Action and command queues each retain
only their newest pending input, and command processing takes priority.

```text
new chunk -> optional PCHIP -> blend with unsent old points
          -> optional low-pass per emitted point -> split arms and send
```

PCHIP is rebuilt for each chunk, preserves per-joint shape, and does not force
endpoint velocities to zero. The fixed-period grid covers the horizon; its last
sample holds the exact final pose without extrapolation. Already sent points
are not replayed. The optional biquad uses `Q=0.5` to reduce ringing compared
with `0.707`, with a slower response at the same frequency parameter.
Interpolation guarantees apply before blending/filtering, not to physical motion.

Ordinary handoffs retain filter state. `reset=true` discards the old remainder
and initializes the filter at the new first pose. Filter coefficients come from
the first chunk after a lifecycle command; changing `cutoff_hz` alone does not
rebuild them.

| Option | Default | Meaning |
| --- | --- | --- |
| `--arms` / `ARMS` | `right,left` | Active arms. |
| `--upsample` | Off | PCHIP on the control-rate grid; adds a SciPy dependency. |
| `--filter` | Off | Biquad after blend; requires upsampling. |
| `--control-hz` | `250` | Target output rate while a trajectory has pending points. |
| `--blend-duration-ms` / `ACTION_BLEND_DURATION_MS` | Unset | Blend full overlap; `0` disables blend. |

Blend duration rounds up to an execution period and is clipped by available
overlap. Without upsampling, the execution period is the policy interval.
The old `--blend-max-steps` / `ACTION_BLEND_MAX_STEPS` options are removed:
`N` policy points formerly spanned `(N - 1) * interval`, so four points at
30 Hz correspond to about 100 ms. Blending now follows interpolation, so the
numerical path is different.

## Metadata

Outputs always carry a wall-clock nanosecond `timestamp`. With input `chunk_id`,
they also carry that ID, `executor_received_timestamp_ns`, and the actual
nominal `blend_duration_ns`. `blended_chunk_id` identifies the immediately
preceding chunk during the explicit blend window, including its endpoints.
The legacy `blend_policy_points` remains a rounded policy-span equivalent, not
a count of mixed control points. Recorders must explicitly support
`blend_duration_ns` to persist it; arbitrary input metadata is not forwarded.
These fields describe published commands, not driver acceptance, actual motion,
or complete historical blend weights.

## Limitations

- Single-point chunks emit once, not continuously at `control_hz`; their filter
  advances once. With blending enabled and a one-point overlap, the old point has weight 1 and can
  replace the new chunk's sole target. Continuous single-target tracking is not
  implemented. A finished chunk has no remainder for later handoff blending.
- Latest-only queues can replace pending commands or a `reset=true` chunk.
  An unknown command interrupts playback without changing lifecycle state.
- Input shape/period validation and coordinated task-failure shutdown remain
  limited; a blocked reader can outlive an executor failure. Scheduling uses
  wall time and is not hard real-time. No sensor-time latency compensation or
  extra velocity/acceleration limits are added here.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
