# Trajectory execution, handoff, and provenance

This document describes the current executor, building on:

1. `5b492aa` — canonical qpos trajectory execution and preemption
2. `b185e7a` — bounded trajectory blending
3. `92837a4` — action-chunk provenance

The current handoff upsamples each new chunk before blending it with unsent
execution points, and bounds the transition by time rather than policy-point
count. The historical commits below introduced the original interfaces;
`b022fb8` had already introduced the canonical qpos envelope.

| Commit | Main behavior change | External interface change |
|---|---|---|
| `5b492aa` | Latest-only action scheduling, command preemption, and stateful handoff from the last published qpos. | Adds the `command` input as a behavioral requirement. The qpos value schema is unchanged from `b022fb8`. |
| `b185e7a` | Limits how many policy points participate in a handoff blend. | Adds `--blend-max-steps` and `ACTION_BLEND_MAX_STEPS`; stream values and metadata are unchanged. |
| `92837a4` | Correlates emitted motor-command targets with the current and immediately preceding policy chunks. | Accepts optional input `chunk_id` metadata and adds optional provenance fields to output metadata; value schemas are unchanged. |

## Motivation

The previous executor had four related problems:

- A single FIFO could accumulate obsolete policy chunks, while control commands
  could not reliably preempt trajectory playback.
- A chunk interrupted between policy points reconstructed its remainder from a
  floored integer policy index rather than from the last command actually sent. Filtering
  made the difference larger and could produce a discontinuous handoff.
- Blending over the entire remaining trajectory could delay adoption of a new
  policy result for too long.
- Once two chunks were blended, downstream recording could not determine which
  policy chunks contributed to an executed command or when a chunk reached the
  executor.

The resulting pipeline is:

```text
Dora actions input ──> latest action queue ───────────────┐
                                                          v
Dora command input ──> latest command queue ──> executor state machine
                                                          |
                                                          v
                            optional PCHIP upsampling of new chunk
                                                          |
                                      blend with unsent old control points
                                                          |
                                      optional biquad low-pass filter
                                                          |
                                      split right/left qpos and publish
```

## External interface

### `actions` input

The input value is an Arrow `list<float32>` array representing a trajectory of
shape `[T, D]`:

- `D = 8` for one active arm.
- `D = 16` for `--arms right,left`.
- A per-arm vector is seven joint positions followed by one gripper position.
- For two arms, the ordering is right-arm 8D followed by left-arm 8D.
- Values must already be joint-space qpos. Workspace/EEF targets require an IK
  conversion before this node.

Input metadata:

| Field | Required | Meaning |
|---|---:|---|
| `interval` | yes | Nanoseconds between policy points. |
| `cutoff_hz` | no | Low-pass frequency parameter; defaults to 15 Hz. With `Q=0.5`, this is not the -3 dB corner frequency. |
| `reset` | no | Marks the first chunk of a new episode; defaults to `false`. |
| `chunk_id` | no | Identifier for the inference chunk, propagated as provenance. |

Other input metadata is not automatically copied to motor-command outputs.

### `command` input

The value is an Arrow string array whose first element is interpreted as the
command:

| Command | Effect |
|---|---|
| `start` | Enable action execution. |
| `stop` | Disable action execution. |
| `intervene` | Disable action execution. |
| `quit` | Disable action execution. |
| any other value | Make no lifecycle state change. During playback it still causes the current chunk to stop at the next control iteration. |

The executor starts disabled. A dataflow must connect this input and send
`start`; otherwise incoming actions are consumed without producing outputs.
Every recognized command also clears the pending action and resets trajectory,
upsampling, filtering, and provenance state.

### Outputs

The node publishes `move_position_right` and/or `move_position_left`, depending
on `--arms`. Each value is a length-one canonical Arrow struct:

```text
struct<qpos: list<float32>>

[{"qpos": [joint_1, ..., joint_7, gripper]}]
```

Each output contains only that arm's 8D command. It does not contain
`other_arm_position`, velocity, torque, temperature, or a workspace pose.

Output metadata:

| Field | Emission | Meaning |
|---|---|---|
| `timestamp` | always | Wall-clock nanoseconds immediately before publishing the command. |
| `chunk_id` | when the current input has one | Identifier of the new/current chunk. |
| `executor_received_timestamp_ns` | when `chunk_id` is present | Wall-clock time at which `node.next()` returned the input to Python. The executor overwrites an upstream field with the same name. |
| `blend_duration_ns` | when `chunk_id` is present | Actual nominal span of the explicit blend window on the execution grid. |
| `blend_policy_points` | when `chunk_id` is present | Legacy policy-span equivalent: `ceil(blend_duration_ns / interval) + 1` when blending, otherwise zero. Not the actual number of mixed control points. |
| `blended_chunk_id` | when both chunks have IDs and the output is in the blend window | Identifier of the immediately preceding chunk whose remainder is being blended. |

Both arm outputs for the same control step receive the same metadata.

Example metadata while chunk `B` is taking over from chunk `A`:

```python
{
    "timestamp": 1788422400123456789,
    "chunk_id": "B",
    "executor_received_timestamp_ns": 1788422400100000000,
    "blend_policy_points": 3,
    "blend_duration_ns": 64000000,
    "blended_chunk_id": "A",
}
```

### Expected provenance integration

The intended end-to-end correlation is:

```text
policy server assigns chunk_id and generation time
    -> local policy bridge preserves metadata and adds interval/reset
    -> executor adds receive/blend metadata
    -> dora-openarm forwards metadata for commands accepted by the driver
       and adds executed_timestamp on latest_command
    -> recorder joins policy chunks and accepted commands by chunk_id
```

The executor does not forward arbitrary fields such as the policy generation
timestamp. A recorder that needs the complete timeline must also
subscribe to the original policy-chunk stream and join the two records by
`chunk_id`. Recording the driver's `latest_command`, rather than the executor's
raw output, also distinguishes a published target from a command the driver
actually accepted.

## `5b492aa`: preemptible trajectory execution

### Latest-only queues and command priority

The Dora reader and trajectory executor are separated into two asynchronous
tasks. Blocking `node.next()` runs through `asyncio.to_thread()`, removing the
old 100 ms polling delay from input detection and preemption.

Actions and commands use separate `asyncio.Queue(maxsize=1)` instances.
`_put_latest()` removes a pending item before adding a new one, so the executor
works on the newest available action instead of replaying a backlog. When both
queues have data, `_next_input()` chooses the command.

During trajectory playback, the executor checks for a pending command or action
before filtering and sending the next control point:

- A command stops the current playback at the next control iteration; recognized
  command processing then clears all execution state.
- A new action captures the interrupted trajectory's remainder and hands off to
  the new chunk.

### Retaining the interrupted trajectory

When a new action is detected before sending control point `i`, the executor
retains `loop_positions[i:]`. These are the old chunk's unsent execution points,
already interpolated and possibly blended by an earlier handoff, but not yet
low-pass filtered. The last published point is not prepended or replayed.

Every new chunk is independently upsampled using its own length and `interval`.
SciPy's PCHIP supplies shape-preserving, first-derivative-continuous interpolation
within each multi-point chunk, without forcing all endpoint velocities to zero.
Knot and sample times are built from integer nanosecond intervals. The output
grid covers the horizon at the first control tick at or beyond its end; sample
times are clamped to the horizon and the final pose is copied exactly. There is
no extrapolation and no shortened final control period. A single-point chunk
has one output point. This does not impose velocity or acceleration limits,
and the optional low-pass filter still changes the trajectory and adds lag.
It is then mixed with the retained remainder on the common control-rate grid.
Without upsampling, the remainder consists of unsent input points instead.
A chunk interrupted before its first output can contribute its whole pending
trajectory. A finished chunk has no pending remainder and does not blend into
a later chunk.

### Blend calculation

Let `P` be the unsent previous remainder and `C` the new execution-grid chunk. The
overlap count is initially:

```text
n = min(len(P), len(C))
```

For `i` from zero through `n - 1`, the executor applies a linear crossfade:

```text
w_i = linspace(1, 0, n)[i]
output_i = w_i * P_i + (1 - w_i) * C_i
```

The first blended point is the old trajectory's next unsent point. For `n >= 2`,
the final point is entirely from the new chunk. With a one-point overlap, the
single weight is `[1]`, so that point comes from the old remainder. Outside the
blend window, the new chunk is unchanged; old points are never appended after
the new chunk ends.

There is no interpolation after blending. Each point passes through the optional
low-pass filter once, immediately before it is split by arm and published.
Ordinary handoffs preserve filter state; reset starts the filter at the new
chunk's first point. Grid alignment is relative to the next output, not to sensor
timestamps, and does not compensate for inference latency.

The biquad uses `Q=0.5`, the critically damped analog-prototype setting, rather
than the previous `0.707`. This reduces ringing but is slower at the same
`cutoff_hz` value; the default frequency parameter remains 15 Hz. No automatic
frequency retuning or additional filtering stage is introduced.

### Episode reset

For an action with `reset=true`, the executor discards the previous remainder
and its provenance. The pre-existing reset behavior also initializes an active
biquad filter to the new chunk's first pose so the new episode is not pulled
toward the previous episode's last pose.

### Canonical output helper

`_qpos_output()` casts each arm command to `float32` and wraps it in the
canonical qpos struct. The schema itself was already introduced by `b022fb8`;
this commit centralizes construction but does not introduce another wire-format
change.

## Duration-based blending

Using the full overlap can make a new policy result take too long to control the
robot. Configure a target transition duration, quantized to the execution grid:

```text
n = min(len(P), len(C), ceil(duration_ns / step_interval_ns) + 1)
```

Configuration is available through either:

```text
--blend-duration-ms 100
ACTION_BLEND_DURATION_MS=100
```

The value must be finite and non-negative. Leaving it unset preserves full
overlap; zero bypasses blending. The actual nominal blend span is:

```text
(n - 1) * execution interval
```

At 250 Hz, a 100 ms transition uses 26 points including both endpoints, provided
both trajectories have enough points. Non-integral durations round up by less
than one execution period; shorter overlap ends the transition earlier.
Without upsampling, the execution interval is the policy interval.

The old point-count option and environment variable are removed. To preserve
approximately the old time span, replace `N` policy points with
`(N - 1) * policy_interval_ms`; e.g. four points at 30 Hz become about 100 ms.
The numerical path still changes because blending now follows interpolation.

## `92837a4`: chunk provenance

This commit makes emitted commands correlatable with policy chunks and recorded
inference logs.

When an `actions` event returns from `node.next()`, `_main_dora()` copies its
metadata and records `executor_received_timestamp_ns`. Copying prevents the
executor from mutating metadata owned by the incoming event.

If a chunk is interrupted, its `chunk_id` is stored with the unsent
remainder. When the next chunk is consumed:

1. Its own ID becomes output `chunk_id`.
2. The previous ID becomes `blended_chunk_id` during the transition window.
3. `_blend_trajectories()` returns the mixed trajectory and execution-point count.
   Its time span is exposed as `blend_duration_ns`; `blend_policy_points` remains
   a rounded policy-span equivalent for compatibility with existing recorders.
4. The cached previous positions and ID are cleared after being consumed.

Recognized commands and episode resets also clear the cached previous ID,
preventing lineage from crossing lifecycle or episode boundaries.

When the current and previous chunks both have IDs, the transition metadata
window is marked directly by execution-point index:

```text
mark blended_chunk_id while i_step < blend_steps
```

The marked window includes the pure-old and pure-new endpoints. The duration is
a grid span, not measured wall-clock playback latency. Existing recorders that
only store `blend_policy_points` retain that estimate; persisting the exact
duration requires adding `blend_duration_ns` to their schema separately.

## Configuration summary

| CLI option | Default | Meaning |
|---|---:|---|
| `--arms` | `right,left` | Active arms and input-vector interpretation. |
| `--upsample` | off | Enable shape-preserving PCHIP interpolation with endpoint holding. |
| `--filter` | off | Enable the biquad low-pass filter; forced off without upsampling. |
| `--control-hz` | `250` | Output frequency when upsampling. |
| `--blend-duration-ms` | unset | Target handoff duration in milliseconds; zero disables blending. |

## Known limitations and review items

### Lifecycle compatibility

- Initial state is disabled, so legacy action-only dataflows silently consume
  actions without executing them. Every deployment must wire `command` and send
  `start`, or the implementation needs a backward-compatible default.
- `stop`, `intervene`, and `quit` stop new outputs but do not publish an explicit
  hold command. From the executor's perspective, downstream behavior depends on
  whether the same command is independently wired to the arm/driver node.
- `quit` only disables this executor; it does not itself terminate the node.
- The action-only examples `Openarm-GR00T/open_eval/dataflow-n17.yaml` and
  `dataflow-n17-local.yaml` do not wire `command` and therefore produce no motor
  outputs with this lifecycle behavior.

### Latest-only loss semantics

- Latest-only behavior is intentional for actions, but an unconsumed
  `reset=true` first chunk can be replaced by the next chunk. In that case the
  executor may retain trajectory or filter state across episode boundaries.
- Commands are also latest-only. A later command can replace an unconsumed
  safety-relevant command such as `stop` or `quit`.
- If action and command waits complete together, command wins and the already
  retrieved action is discarded. This can also discard the first action around
  `start`.
- Any queued command interrupts the playback loop before command validation. An
  unknown command is then ignored by `_apply_command()`, but the interrupted
  trajectory has already been abandoned without preserving its remainder.
- Chunks replaced in the queue or dropped while disabled do not produce an
  execution record.
- Handoff state exists only while a chunk is actively being interrupted. If a
  chunk finishes and the next chunk arrives later, the last sent point is not
  retained across the wait and the next chunk starts with no blend.

### Input validation and trajectory configuration

- The implementation does not validate Arrow shape, finite values, or exact
  width before splitting. A malformed trajectory can be emitted with a short
  arm vector, silently truncated when too long, or rejected later by a
  consumer.
- Empty actions fail when reading the first row. Ragged rows are not rejected
  with a targeted error. Single-point chunks bypass PCHIP; blend and filter
  behavior still applies to that point.
- `interval`, `--control-hz`, and `cutoff_hz` are not validated as positive,
  finite values. The command array is not checked for an element before index
  zero is read, and `--arms` does not enforce the documented choices.
- The upsampler and evaluation grid are rebuilt for every new chunk, including
  changes in length or `interval`. Filter coefficients still come from the first
  chunk after a recognized command; changing `cutoff_hz` mid-session does not
  rebuild the filter.
- Even without upsampling, blending an old remainder against a new chunk assumes
  compatible policy cadence; differing input intervals are not reconciled.
- `reset=true` clears the remainder and resets filter state, but it does not
  reconstruct filter coefficients.

### Task lifetime and timing

- `asyncio.to_thread(node.next)` may leave a worker blocked in `node.next()` if
  the executor fails. The reader and executor tasks are not supervised as a
  single failure domain, so one task can wait indefinitely after the other
  exits.
- Control scheduling and provenance timestamps use `time.time_ns()`. Wall-clock
  adjustments can affect sleep duration; a monotonic clock is preferable for
  control scheduling even if wall time remains useful for correlation.

### Provenance boundaries

- Provenance fields are only emitted when the current chunk has `chunk_id`.
  `chunk_id` is neither type-checked nor checked for uniqueness.
- `executor_received_timestamp_ns` means Python receive time, not policy
  generation time, queue-consumption time, publish time, or confirmed driver
  execution time.
- `blend_duration_ns` and the legacy `blend_policy_points` equivalent are repeated
  on every output from the current chunk. Neither represents measured latency.
- For `n >= 2`, the final blend point is still marked with `blended_chunk_id`,
  even though its explicit weight from the previous chunk is zero.
- Only the immediately preceding chunk is recorded. If an already blended chunk
  is interrupted again, numerical influence from older chunks is not represented
  as a full lineage.
- `blended_chunk_id` describes the explicit execution-point blend window, not
  exact numerical ancestry. The stateful low-pass filter can carry earlier
  influence beyond that window. An already blended remainder can also contain
  older chunks' contributions.
- Only selected fields are forwarded. Upstream generation timestamps and other
  arbitrary metadata must be joined from the policy-chunk record using
  `chunk_id`.
- A policy chunk can exist without accepted-command fields when it was replaced,
  cleared, dropped while disabled, or rejected downstream. Provenance records do
  not imply successful execution.

### Test coverage

The repository keeps compact helper tests for interpolation knots/endpoints,
blend duration, filter reset, queues, and qpos output. More extensive numerical
and mocked-coroutine checks were run outside the repository; they are not a
persistent regression suite. Neither set verifies real Dora transport, driver
acceptance, hardware timing, or all command/reset races.
The repository CI does not currently execute pytest.
