# Trajectory execution, handoff, and provenance

This document describes the behavior introduced by the three consecutive
commits ending at `92837a4`:

1. `5b492aa` — canonical qpos trajectory execution and preemption
2. `b185e7a` — bounded trajectory blending
3. `92837a4` — action-chunk provenance

It documents the implementation as it exists at `92837a4`. The parent commit,
`b022fb8`, had already changed each arm output to the canonical qpos envelope.
The three commits covered here retain that value schema; they primarily change
execution semantics, process configuration, and output metadata.

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
                  reconstruct old remainder -> blend new policy chunk
                                                          |
                                      optional Hermite upsampling
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
| `cutoff_hz` | no | Low-pass cutoff; defaults to 15 Hz. |
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
| `blend_policy_points` | when `chunk_id` is present | Number of coarse policy points used for explicit handoff blending; zero means no blend. |
| `blended_chunk_id` | when both chunks have IDs and the output is in the blend window | Identifier of the immediately preceding chunk whose remainder is being blended. |

Both arm outputs for the same control step receive the same metadata.

Example metadata while chunk `B` is taking over from chunk `A`:

```python
{
    "timestamp": 1788422400123456789,
    "chunk_id": "B",
    "executor_received_timestamp_ns": 1788422400100000000,
    "blend_policy_points": 3,
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

### Reconstructing the interrupted trajectory

The handoff anchor is the last qpos actually published, after optional low-pass
filtering. This avoids jumping back to a raw policy point selected by a floored
integer index.

With Hermite upsampling enabled, `_remaining_policy_trajectory()` builds a
coarse, policy-rate remainder:

```text
[last qpos actually sent,
 old spline at last_sent_time + one policy interval,
 old spline at last_sent_time + two policy intervals,
 ...]
```

If no future policy time remains, the remainder contains only the last sent
qpos. Without upsampling, the remainder is:

```text
[last qpos actually sent,
 unsent coarse points from the currently executing, possibly blended trajectory...]
```

If the new action arrives before any point from the old chunk was sent, there is
no anchor and therefore no blend.

### Blend calculation

Let `P` be the reconstructed previous remainder and `C` the new chunk. The
overlap count is initially:

```text
n = min(len(P), len(C))
```

For `i` from zero through `n - 1`, the executor applies a linear crossfade:

```text
w_i = linspace(1, 0, n)[i]
output_i = w_i * P_i + (1 - w_i) * C_i
```

Consequently, the first coarse blended point is exactly the last command already
sent. For `n >= 2`, the final coarse point in the blend window is exactly the new
chunk. For `n = 1`, NumPy produces the single weight `[1]`, so that point is only
the old anchor. These statements apply before another Hermite interpolation and
low-pass pass; the first published command is not guaranteed to be numerically
identical when filtering is enabled. The rest of the coarse output is a copy of
the new chunk; an old remainder is never appended after the new chunk ends.

Blending happens at the policy rate, before Hermite upsampling. The resulting
trajectory is then upsampled to `--control-hz`, optionally filtered, split by
arm, and serialized as qpos.

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

## `b185e7a`: bounded blending

Using the full overlap can make a new policy result take too long to control the
robot. This commit adds an optional upper bound:

```text
n = min(len(P), len(C), blend_max_steps)
```

Configuration is available through either:

```text
--blend-max-steps N
ACTION_BLEND_MAX_STEPS=N
```

`N` must be a positive integer. Leaving it unset preserves full-overlap
blending.

The unit is policy points, not control-rate samples. The explicit coarse
blend-window span is:

```text
(n - 1) * input interval
```

For example, three points from a 30 Hz policy span about 67 ms and may produce
many commands after upsampling to 250 Hz. The anchor counts as the first policy
point. The edge cases are therefore:

- `n = 1`: replace the new chunk's first point with the old anchor. If the new
  chunk has more points, execution switches to its second point; for a one-point
  chunk, no newly predicted value is published.
- `n = 2`: the two weights are exactly 1 and 0, with no intermediate crossfade
  point at policy resolution.
- `n >= 3`: at least one policy point has a fractional mix.

The executor prints the selected full or bounded mode at startup.

## `92837a4`: chunk provenance

This commit makes emitted commands correlatable with policy chunks and recorded
inference logs.

When an `actions` event returns from `node.next()`, `_main_dora()` copies its
metadata and records `executor_received_timestamp_ns`. Copying prevents the
executor from mutating metadata owned by the incoming event.

If a chunk is interrupted, its `chunk_id` is stored with the reconstructed
remainder. When the next chunk is consumed:

1. Its own ID becomes output `chunk_id`.
2. The previous ID becomes `blended_chunk_id` during the transition window.
3. `_blend_trajectories()` returns both the trajectory and the actual coarse
   blend count, exposed as `blend_policy_points`.
4. The cached previous positions and ID are cleared after being consumed.

Recognized commands and episode resets also clear the cached previous ID,
preventing lineage from crossing lifecycle or episode boundaries.

When the current and previous chunks both have IDs, the transition metadata
window is calculated as:

```text
blend_end_s = (blend_policy_points - 1) * input_interval_s
mark blended_chunk_id while output_time_s <= blend_end_s
```

This converts a policy-point count into the output timeline when upsampling is
enabled. The comparison includes both endpoints.

## Configuration summary

| CLI option | Default | Meaning |
|---|---:|---|
| `--arms` | `right,left` | Active arms and input-vector interpretation. |
| `--upsample` | off | Enable cubic Hermite interpolation. |
| `--filter` | off | Enable the biquad low-pass filter; forced off without upsampling. |
| `--control-hz` | `250` | Output frequency when upsampling. |
| `--blend-max-steps` | unset | Maximum number of policy points in a handoff blend. |

## Known limitations and review items at `92837a4`

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

### Input validation and cached trajectory configuration

- The implementation does not validate Arrow shape, finite values, or exact
  width before splitting. A malformed trajectory can be emitted with a short
  arm vector, silently truncated when too long, or rejected later by a
  consumer.
- Empty actions fail when reading the first row. With `--upsample`, a one-point
  trajectory also fails because Hermite slope construction requires at least
  two policy points. Ragged rows are not rejected with a targeted error.
- `interval`, `--control-hz`, and `cutoff_hz` are not validated as positive,
  finite values. The command array is not checked for an element before index
  zero is read, and `--arms` does not enforce the documented choices.
- The upsampler, evaluation grid, and filter coefficients are initialized from
  the first accepted chunk after a recognized command. A later change in
  `interval`, chunk length, or `cutoff_hz` does not rebuild them. A length change
  can raise an upsampler shape error; interval or cutoff changes can be applied
  inconsistently.
- Even without upsampling, blending an old remainder against a new chunk assumes
  compatible policy cadence; differing input intervals are not reconciled.
- `reset=true` clears the remainder and resets filter state, but it does not
  reconstruct the upsampler or filter coefficients.

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
- `blend_policy_points` is repeated on every output from the current chunk and
  counts coarse policy points, not the number of upsampled output commands.
- For `n >= 2`, the inclusive comparison keeps any output sample that lands
  exactly on the final coarse blend endpoint marked with `blended_chunk_id`,
  even though its explicit linear weight from the previous chunk is zero.
- Only the immediately preceding chunk is recorded. If an already blended chunk
  is interrupted again, numerical influence from older chunks is not represented
  as a full lineage.
- `blended_chunk_id` describes the explicit coarse-point blend window, not exact
  numerical ancestry. Hermite tangents and a stateful low-pass filter can carry
  earlier influence beyond the marked window, so the metadata cannot be used to
  reconstruct per-sample blend weights.
- Only selected fields are forwarded. Upstream generation timestamps and other
  arbitrary metadata must be joined from the policy-chunk record using
  `chunk_id`.
- A policy chunk can exist without accepted-command fields when it was replaced,
  cleared, dropped while disabled, or rejected downstream. Provenance records do
  not imply successful execution.

### Test coverage

At `92837a4`, ten helper-level tests cover Hermite knot preservation, remainder
reconstruction, blend behavior and count, the blend limit, filter reset,
latest-only action replacement, command priority, start, and qpos serialization.

There is no end-to-end executor test for scheduling, stop/reset races, output
metadata, provenance-window boundaries, dynamic chunk parameters, or Dora task
shutdown. The qpos helper test also does not assert the required per-arm width.
The repository CI does not currently execute pytest.
