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
new chunk -> wait/align when target times are supplied -> PCHIP as needed
          -> blend or switch -> optional low-pass -> split arms and send
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
| `--mode` / `ACTION_EXECUTION_MODE` | `blend` | `blend` mixes trajectories; `switch` directly replaces them. Both support target times. |
| `--arms` / `ARMS` | `right,left` | Active arms. |
| `--upsample` | Off | PCHIP on the control-rate grid; adds a SciPy dependency. |
| `--filter` | Off | Biquad after blend; requires upsampling. |
| `--filter-mode` / `ACTION_FILTER_MODE` | `causal` | Per-point causal filter or experimental `zero-phase` chunk filtering. |
| `--control-hz` | `250` | Target output rate while a trajectory has pending points. |
| `--blend-duration-ms` / `ACTION_BLEND_DURATION_MS` | Unset | Blend full overlap; `0` disables blend. |

Blend duration rounds up to an execution period and is clipped by available
overlap. Without upsampling, the execution period is the policy interval.
The old `--blend-max-steps` / `ACTION_BLEND_MAX_STEPS` options are removed:
`N` policy points formerly spanned `(N - 1) * interval`, so four points at
30 Hz correspond to about 100 ms. Blending now follows interpolation, so the
numerical path is different.

`zero-phase` runs SciPy `sosfiltfilt` on the already available control-rate chunk,
preceded by up to three cutoff periods of previously sent **raw** targets. It does
not wait for future observations. Output is filtered only once, after handoff/blend;
history and RTC feedback never contain filtered outputs. Start/Stop/reset clear
the filter context. Forward-backward filtering squares the magnitude response,
so the same `cutoff_hz` is not an identical smoothing strength to `causal`.
Finite chunk boundaries and replanning can still introduce command discontinuities;
zero phase within a chunk is not a guarantee of seamless real-world handoffs.
Recorded-chunk replay showed larger handoff jumps in `zero-phase`; use `causal`
as the baseline when evaluating the pre-filter feedback fix.

## Target Times and Feedback

Without target times, both modes adopt immediately from the first supplied point.
With `action_origin_timestamp_ns`, the first point is at
`origin + action_window_start * interval`. `takeover_timestamp_ns` optionally
specifies when adoption starts; alone it specifies the first point's target time.
Zero/missing times mean immediate execution. Generation/ordinary timestamps are
not action targets. Early chunks wait while the current trajectory continues;
late chunks skip expired points. `max_lateness_ns` optionally bounds lateness,
and entirely expired windows are rejected without interrupting the old trajectory.

Timed blending samples both pre-filter trajectories on the same control grid.
Weights start at the actual handoff and use the existing blend duration. Filtering
is applied once, after mixing. `reset=true` prevents mixing with the old trajectory.

`based_on_chunk_id` declares a dependency on the active plan (empty for bootstrap)
and enables `execution_plan` feedback. Add that output to the dataflow and connect
it to the policy node. Both modes publish the post-blend, **pre-filter** plan on adoption and
acknowledge rejected dependent samples; no per-tick progress messages are sent.
Start/Stop clear feedback. This is a target plan, not the driver's limited command
or physical joint motion. Feeding filtered output back into another filter is avoided.
Legacy streams require neither a dependency field nor an `execution_plan` output.
RTC is model-side; its predictions may use either executor mode.

After Start, the first action must match the attempt supplied by that command.
Once a trajectory is active, an action with a new attempt ID is an in-place task
switch: it replaces pending work and is treated as a reset chunk. The active
trajectory continues until the new chunk is due. Feedback uses the adopted chunk's
attempt ID. This assumes one ordered policy output source whose transport discards
old-generation results; no extra task-control input is required.

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
  Unknown commands leave playback and lifecycle state unchanged.
- Input shape/period validation and coordinated task-failure shutdown remain
  limited; a blocked reader can outlive an executor failure. Scheduling uses
  wall time and is not hard real-time. No sensor-time latency compensation or
  extra velocity/acceleration limits are added here.

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
