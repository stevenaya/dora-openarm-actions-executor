# dora-openarm-actions-executor

A [Dora](https://dora-rs.ai/) node that executes timestamped actions for OpenArm.

## Trajectory handoff

With `--upsample`, each new chunk is first interpolated with SciPy PCHIP onto the `--control-hz`
grid, then blended point-by-point with the old trajectory's unsent control
points. Already published points are not replayed. The optional low-pass filter
runs once after blending; episode reset discards the old remainder and resets
the filter.
PCHIP preserves the per-joint input shape without interpolation overshoot and
does not force endpoint velocities to zero. The fixed-period grid reaches or
passes the chunk horizon; its final sample holds the exact final pose rather
than extrapolating. A single-point chunk passes through unchanged by upsampling.
These properties apply before blending/filtering. The optional biquad low-pass
uses `Q=0.5` instead of `0.707` to reduce ringing, at the cost of a slower response
with the same frequency parameter. `cutoff_hz` still defaults to 15 Hz; the
filter remains after blending and retains its state across ordinary handoffs.

Use `--blend-duration-ms 100` (or `ACTION_BLEND_DURATION_MS=100`) for a 100 ms
transition. An unset duration blends the full overlap; `0` disables blending.
Durations round up to an execution period and are clipped by the available
overlap. Without upsampling, the execution period is the input policy interval.
The former `--blend-max-steps` / `ACTION_BLEND_MAX_STEPS` options are removed.

Outputs with `chunk_id` include the actual nominal `blend_duration_ns`.
`blended_chunk_id` marks the explicit transition window. The legacy
`blend_policy_points` field remains a rounded policy-span equivalent for existing
recorders, not a count of points mixed before interpolation.

## Documentation

- [Trajectory execution, handoff, and provenance](docs/trajectory-execution.md)

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

Copyright 2026 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
