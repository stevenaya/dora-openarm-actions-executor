"""Shared handoff tests, without a robot or Dora daemon."""
# ruff: noqa: D103

import asyncio
import time

import numpy as np
import pyarrow as pa
import pytest

from dora_openarm_actions_executor.main import (
    BiquadLowpass,
    _main_executor,
    _zero_phase_commands,
)
from dora_openarm_actions_executor.trajectory import Chunk, TrajectoryScheduler, _upsample_trajectory


def event(values=(0, 1, 2), **metadata):
    return {
        "value": pa.array(
            [[float(value)] * 16 for value in values], type=pa.list_(pa.float32())
        ),
        "metadata": {"interval": 100, "chunk_id": "new", **metadata},
    }


@pytest.mark.parametrize("mode", ["blend", "switch"])
@pytest.mark.parametrize("timed", [False, True])
def test_same_control_grid_for_both_handoff_modes(mode, timed):
    scheduler = TrajectoryScheduler()
    old = Chunk.from_event(event((0, 1, 2, 3, 4), chunk_id="old"))
    scheduler.adopt(old, 1000, 50, mode, 100, True)
    scheduler.cursor = 3
    metadata = {"action_origin_timestamp_ns": 1050} if timed else {"timestamp": 10**18}
    new = Chunk.from_event(event((11, 21, 31), **metadata))
    assert new.start_ns == (1050 if timed else 0)
    scheduler.submit(new)
    assert scheduler.take(1150) == (new, None)
    scheduler.adopt(new, 1150, 50, mode, 100, True)
    result = scheduler.active.positions[:, 0]
    if mode == "blend":
        np.testing.assert_allclose(
            result[:3], [1.5, 14 if timed else 9, 31 if timed else 21]
        )
        assert scheduler.blended_chunk_id == "old"
    else:
        assert result[0] == (21 if timed else 11)
        assert scheduler.blend_steps == 0 and scheduler.blended_chunk_id is None


def test_optional_references_early_late_and_expired_leave_old_trajectory_active():
    scheduler = TrajectoryScheduler()
    scheduler.clear({"episode_attempt_id": "a"})
    old = Chunk.from_event(event(chunk_id="old"))
    scheduler.adopt(old, 1000, 50, "switch", None, True)
    new = Chunk.from_event(
        event(
            takeover_timestamp_ns=1100,
            based_on_chunk_id="old",
            max_lateness_ns=20,
            episode_attempt_id="a",
        )
    )
    assert scheduler.submit(new) is None
    assert scheduler.take(1099) == (None, None)
    assert scheduler.take(1115) == (new, None)
    scheduler.submit(new)
    assert scheduler.take(1121)[1] == "late beyond tolerance"
    expired = Chunk.from_event(event((1, 2), action_origin_timestamp_ns=1000))
    scheduler.submit(expired)
    assert scheduler.take(1200)[1] == "selected window expired"
    assert scheduler.active.metadata["chunk_id"] == "old" and scheduler.cursor == 0
    assert (
        scheduler.submit(Chunk.from_event(event(based_on_chunk_id="stale")))
        == "execution plan changed"
    )
    assert scheduler.submit(Chunk.from_event(event(episode_attempt_id="b", based_on_chunk_id=""))) is None
    new, reason = scheduler.take(1200)
    assert reason is None and new.metadata["reset"]
    scheduler.adopt(new, 1200, 50, "blend", None, True)
    assert scheduler.attempt_id == "b" and scheduler.blend_steps == 0
    scheduler.clear({"episode_attempt_id": "c"})
    assert scheduler.active is scheduler.pending is None
    assert scheduler.submit(Chunk.from_event(event(episode_attempt_id="b", reset=True))) == "episode changed"


def test_blend_uses_raw_trajectory():
    scheduler = TrajectoryScheduler()
    old = Chunk.from_event(event((0, 1, 2, 3, 4), chunk_id="old"))
    scheduler.adopt(old, 1000, 50, "blend", 100, True)
    scheduler.cursor = 3
    scheduler.adopt(Chunk.from_event(event((5, 6, 7))), 1150, 50, "blend", 100, True)
    assert (
        scheduler.active.positions[0, 0] == 1.5
    )  # Raw old point, not its filtered preview.
    scheduler.adopt(
        Chunk.from_event(event((9, 10), reset=True)), 1160, 50, "blend", 100, True
    )
    assert scheduler.active.positions[0, 0] == 9 and scheduler.blend_steps == 0


@pytest.mark.parametrize("mode", ["blend", "switch"])
@pytest.mark.parametrize("timed", [False, True])
def test_executor_handoff_feedback_and_stop(mode, timed):
    async def run():
        actions, commands = asyncio.Queue(maxsize=1), asyncio.Queue()
        done = asyncio.Event()
        plans, sent = [], []
        target = None
        queued = False

        class Node:
            def send_output(self, name, value, metadata):
                nonlocal queued, target
                if name == "execution_plan":
                    plan = value[0].as_py()
                    if plan is None:
                        return
                    plans.append(plan)
                    assert "inference_started_timestamp_ns" not in plan
                elif name == "move_position_right":
                    sent.append((metadata, value[0].as_py()["qpos"]))
                    if not queued:
                        queued = True
                        target = time.time_ns() + 25_000_000 if timed else 0
                        extra = (
                            {
                                "action_origin_timestamp_ns": target,
                                "based_on_chunk_id": "first",
                            }
                            if timed
                            else {}
                        )
                        actions.put_nowait(
                            event(
                                (1,) * 20,
                                interval=10_000_000,
                                chunk_id="second",
                                **extra,
                            )
                        )
                    if metadata.get("chunk_id") == "second" and sent[-1][1][0] >= 0.999:
                        commands.put_nowait(
                            {"value": pa.array(["stop"]), "metadata": {}}
                        )
                        done.set()

        commands.put_nowait({"value": pa.array(["start"]), "metadata": {}})
        task = asyncio.create_task(
            _main_executor(
                Node(),
                actions,
                commands,
                ["right", "left"],
                True,
                False,
                250,
                12,
                mode=mode,
            )
        )
        try:
            await asyncio.sleep(0)
            extra = {"based_on_chunk_id": ""} if timed else {}
            actions.put_nowait(
                event((0,) * 20, interval=10_000_000, chunk_id="first", **extra)
            )
            await asyncio.wait_for(done.wait(), 2)
            await asyncio.sleep(0.01)
            second = [(meta, pos) for meta, pos in sent if meta["chunk_id"] == "second"]
            assert second[0][1][0] == (0 if mode == "blend" else 1)
            assert ("blended_chunk_id" in second[0][0]) == (mode == "blend")
            if timed:
                assert len(plans) == 2 and second[0][0]["timestamp"] >= target
                assert plans[1]["positions"][0][0] == (0 if mode == "blend" else 1)
                assert sum(meta["chunk_id"] == "first" for meta, _ in sent) >= 2
            else:
                assert not plans  # Legacy streams need no execution_plan output.
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_stop_clears_a_future_handoff():
    async def run():
        actions, commands = asyncio.Queue(maxsize=1), asyncio.Queue()
        started, stopped = asyncio.Event(), asyncio.Event()
        ids = []

        class Node:
            def send_output(self, name, value, metadata):
                if name == "execution_plan":
                    if value[0].as_py() is None:
                        stopped.set()
                    else:
                        started.set()
                elif name == "move_position_right":
                    ids.append(metadata["chunk_id"])

        commands.put_nowait({"value": pa.array(["start"]), "metadata": {}})
        task = asyncio.create_task(
            _main_executor(
                Node(),
                actions,
                commands,
                ["right"],
                True,
                False,
                250,
                None,
                mode="switch",
            )
        )
        try:
            await asyncio.sleep(0)
            actions.put_nowait(
                event(
                    (0,) * 20, interval=10_000_000, chunk_id="old", based_on_chunk_id=""
                )
            )
            await asyncio.wait_for(started.wait(), 1)
            actions.put_nowait(
                event(
                    (1,) * 20,
                    interval=10_000_000,
                    based_on_chunk_id="old",
                    takeover_timestamp_ns=time.time_ns() + 10_000_000_000,
                )
            )
            await asyncio.sleep(0.01)
            commands.put_nowait({"value": pa.array(["stop"]), "metadata": {}})
            await asyncio.wait_for(stopped.wait(), 1)
            assert set(ids) == {"old"}
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())


def test_zero_phase_has_no_interior_phase_shift_or_repeated_filtering():
    lowpass = BiquadLowpass(250, 15)
    t = np.arange(400) / 250
    positions = np.sin(2 * np.pi * 2 * t)[:, None].astype(np.float32)
    original = positions.copy()
    expected = _zero_phase_commands(positions, lowpass, [])
    parts = []
    for start in range(0, 250, 50):
        future = _zero_phase_commands(positions[start:], lowpass,
                                      list(positions[max(0, start - 50):start]))
        parts.append(future[:50])
    np.testing.assert_allclose(np.concatenate(parts), expected[:250], atol=2e-6)
    np.testing.assert_array_equal(positions, original)
    assert lowpass.x1 is lowpass.y1 is None
    correlations = [np.corrcoef(expected[60:340,0],positions[60+s:340+s,0])[0,1] for s in range(-10,11)]
    assert np.argmax(correlations) == 10
    for count in (1, 2):
        np.testing.assert_allclose(_zero_phase_commands(np.ones((count,16)),lowpass,[]),1)


@pytest.mark.parametrize("filter_mode", ["causal", "zero-phase"])
def test_feedback_stays_raw_while_final_output_is_filtered(filter_mode):
    async def run():
        actions, commands = asyncio.Queue(maxsize=1), asyncio.Queue()
        done, plans, sent = asyncio.Event(), [], []

        class Node:
            def send_output(self, name, value, metadata):
                if name == "execution_plan":
                    plans.append(value[0].as_py())
                elif name == "move_position_right":
                    sent.append(value[0].as_py()["qpos"])
                    if len(sent) == 8:
                        done.set()

        commands.put_nowait({"value": pa.array(["start"]), "metadata": {}})
        task = asyncio.create_task(_main_executor(Node(), actions, commands, ["right"],
            True, True, 250, None, mode="switch", filter_mode=filter_mode))
        try:
            await asyncio.sleep(0)
            incoming = event((0, 1, 1, 0, 0), interval=33_333_333, reset=True, based_on_chunk_id="")
            actions.put_nowait(incoming)
            await asyncio.wait_for(done.wait(), 2)
            raw = _upsample_trajectory(Chunk.from_event(incoming).positions,33_333_333,4_000_000)
            np.testing.assert_array_equal(plans[0]["positions"],raw)
            assert not np.allclose(np.asarray(sent)[:,0],raw[:len(sent),0])
        finally:
            task.cancel()
            await asyncio.gather(task,return_exceptions=True)

    asyncio.run(run())


def test_task_switch_clears_pending_old_chunk_without_arm_restart(capsys):
    async def run():
        actions, commands = asyncio.Queue(maxsize=1), asyncio.Queue()
        first, accepted, restarted, next_accepted = (asyncio.Event() for _ in range(4))
        outputs, plans = [], []

        class Node:
            def send_output(self, name, value, metadata):
                if name == "execution_plan":
                    plans.append((value[0].as_py(), metadata))
                    if value[0].as_py() is None and metadata.get("episode_attempt_id") == "c":
                        restarted.set()
                elif name == "move_position_right":
                    outputs.append(metadata["chunk_id"])
                    {"a": first, "b": accepted, "c": next_accepted}[metadata["chunk_id"]].set()

        commands.put_nowait({"id": "command", "value": pa.array(["start"]),
                             "metadata": {"episode_attempt_id": "a"}})
        task = asyncio.create_task(_main_executor(Node(), actions, commands, ["right"],
            True, False, 250, None, mode="blend"))
        try:
            await asyncio.sleep(0)
            actions.put_nowait(event((0,)*100, interval=10_000_000, chunk_id="a",
                                    based_on_chunk_id="", reset=True, episode_attempt_id="a"))
            await asyncio.wait_for(first.wait(), 1)
            actions.put_nowait(event(chunk_id="old-pending", based_on_chunk_id="a",
                                    episode_attempt_id="a", takeover_timestamp_ns=time.time_ns()+40_000_000))
            await asyncio.sleep(.01)
            actions.put_nowait(event((1,)*20, interval=10_000_000, chunk_id="b",
                                    based_on_chunk_id="", episode_attempt_id="b",
                                    takeover_timestamp_ns=time.time_ns()+150_000_000))
            await asyncio.sleep(.06)
            assert set(outputs) == {"a"}  # Keep the active plan, never adopt old pending work.
            await asyncio.wait_for(accepted.wait(), 1)
            assert set(outputs) == {"a", "b"}
            assert plans[-1][0]["chunk_id"] == "b" and plans[-1][1]["episode_attempt_id"] == "b"
            assert plans[-1][0]["positions"][0][0] == 1  # New task is reset, not blended with A.
            commands.put_nowait({"value": pa.array(["start"]), "metadata": {"episode_attempt_id": "c"}})
            await asyncio.wait_for(restarted.wait(), 1)
            count = len(outputs)
            actions.put_nowait(event(chunk_id="old-late", based_on_chunk_id="", reset=True,
                                    episode_attempt_id="b"))
            await asyncio.sleep(.01)
            assert len(outputs) == count
            actions.put_nowait(event((2,)*20, interval=10_000_000, chunk_id="c",
                                    based_on_chunk_id="", reset=True, episode_attempt_id="c"))
            await asyncio.wait_for(next_accepted.wait(), 1)
            assert plans[-1][0]["chunk_id"] == "c" and plans[-1][1]["episode_attempt_id"] == "c"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run())
    assert "old-late: episode changed" in capsys.readouterr().out
