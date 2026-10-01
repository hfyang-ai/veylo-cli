from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from veylo.agent import PlanExecuteAgent
from veylo.agent.orchestrator import ExecutionStep, StepStatus
from veylo.checkpoint import CheckpointRecord, CheckpointStore, new_run_id
from veylo.config import CheckpointConfig, VeyloConfig, load_config
from veylo.plan import ExecutionPlan, Task, TaskStatus, TaskType
from veylo.tools import ToolRegistry, get_builtin_tools


class FakeClient:
    model_name = "fake-model"
    provider_name = "fake-provider"
    max_context_window = 1000

    async def chat(self, messages, tools, *, system_prompt):  # noqa: ARG002
        yield {"type": "text_delta", "text": "{}"}
        yield {"type": "message_end", "stop_reason": "end_turn"}


def _record(**kwargs: Any) -> CheckpointRecord:
    defaults: dict[str, Any] = {
        "run_id": new_run_id("plan"),
        "mode": "plan",
        "message": "demo goal",
        "cwd": "/tmp/demo",
    }
    defaults.update(kwargs)
    return CheckpointRecord(**defaults)


def _plan_with_tasks() -> ExecutionPlan:
    plan = ExecutionPlan(id="plan_1", goal="demo")
    plan.add_task(Task("task_1", "read a", TaskType.FILE_READ))
    plan.add_task(Task("task_2", "summarize", TaskType.ANALYSIS, ["task_1"]))
    return plan


# --- storage layer ---


def test_store_saves_and_loads_latest_state(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    run_id = new_run_id("plan")

    store.save(_record(run_id=run_id, status="running", progress={"completed": 0, "total": 2}))
    store.save(_record(run_id=run_id, status="completed", progress={"completed": 2, "total": 2}))

    loaded = store.load(run_id)
    assert loaded is not None
    assert loaded.status == "completed"
    assert loaded.progress["completed"] == 2
    assert loaded.message == "demo goal"


def test_store_lists_most_recent_first(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    store.save(_record(run_id="plan-older", updated_at="2026-01-01T00:00:00"))
    store.save(_record(run_id="plan-newer", updated_at="2026-02-01T00:00:00"))

    listed = store.list()
    assert [item.run_id for item in listed] == ["plan-newer", "plan-older"]


def test_resumable_excludes_finished_runs(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    store.save(_record(run_id="plan-done", status="completed"))
    store.save(_record(run_id="plan-stopped", status="interrupted"))
    store.save(_record(run_id="plan-live", status="running"))

    assert {item.run_id for item in store.resumable()} == {"plan-stopped", "plan-live"}


def test_store_delete_and_clean(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    store.save(_record(run_id="plan-a"))
    store.save(_record(run_id="plan-b"))

    assert store.delete("plan-a") is True
    assert store.delete("plan-a") is False
    assert store.load("plan-a") is None
    assert store.clean() == 1
    assert store.list() == []


def test_corrupt_line_does_not_break_load(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    run_id = new_run_id("plan")
    store.save(_record(run_id=run_id, status="running"))
    with (store.root / f"{run_id}.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")

    loaded = store.load(run_id)
    assert loaded is not None
    assert loaded.status == "running"


# --- pruning ---


def _seed(store: CheckpointStore, run_id: str, *, status: str, updated_at: str) -> None:
    """Write a checkpoint file directly with a fixed timestamp.

    Bypasses ``save()`` (which always stamps ``updated_at`` with the current
    time) so prune tests can control ordering and age deterministically.
    """
    record = _record(run_id=run_id, status=status)
    record.created_at = updated_at
    record.updated_at = updated_at
    payload = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
    (store.root / f"{run_id}.jsonl").write_text(payload, encoding="utf-8")


def test_prune_caps_completed_runs(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    for i in range(5):
        _seed(store, f"done-{i}", status="completed", updated_at=f"2026-01-0{i + 1}T00:00:00")

    removed = store.prune(max_completed=2)

    assert removed == 3
    assert {r.run_id for r in store.list()} == {"done-3", "done-4"}


def test_prune_caps_unfinished_runs(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    for i in range(4):
        _seed(store, f"int-{i}", status="interrupted", updated_at=f"2026-02-0{i + 1}T00:00:00")

    store.prune(max_resumable=2)

    assert {r.run_id for r in store.list()} == {"int-2", "int-3"}


def test_prune_ttl_reaps_expired_unfinished(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    _seed(store, "stale-running", status="running", updated_at="2025-12-01T00:00:00")
    _seed(store, "recent-interrupted", status="interrupted", updated_at="2026-01-07T00:00:00")

    removed = store.prune(ttl=timedelta(days=7), now=datetime(2026, 1, 8, tzinfo=UTC))

    assert removed == 1
    assert {r.run_id for r in store.list()} == {"recent-interrupted"}


def test_prune_ttl_keeps_unparseable_timestamp(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    _seed(store, "weird", status="interrupted", updated_at="not-a-timestamp")

    removed = store.prune(ttl=timedelta(days=7), now=datetime(2026, 1, 8, tzinfo=UTC))

    assert removed == 0
    assert {r.run_id for r in store.list()} == {"weird"}


def test_prune_ignores_time_without_ttl(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    _seed(store, "old-running", status="running", updated_at="2020-01-01T00:00:00")

    removed = store.prune()  # no ttl → age is irrelevant

    assert removed == 0
    assert {r.run_id for r in store.list()} == {"old-running"}


def test_prune_returns_zero_when_nothing_to_remove(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    _seed(store, "one", status="completed", updated_at="2026-01-01T00:00:00")
    _seed(store, "two", status="interrupted", updated_at="2026-01-02T00:00:00")

    removed = store.prune(max_completed=5, max_resumable=5)

    assert removed == 0
    assert len(store.list()) == 2


def test_prune_with_config_drives_cleanup(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    for i in range(5):
        _seed(store, f"done-{i}", status="completed", updated_at=f"2026-01-0{i + 1}T00:00:00")

    config = VeyloConfig(checkpoint=CheckpointConfig(max_completed=2))
    removed = store.prune_with_config(config)

    assert removed == 3
    assert {r.run_id for r in store.list()} == {"done-3", "done-4"}


def test_prune_with_config_applies_ttl(tmp_path) -> None:
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    _seed(store, "stale-running", status="running", updated_at="2025-12-01T00:00:00")
    _seed(store, "recent", status="interrupted", updated_at="2026-01-07T00:00:00")

    config = VeyloConfig(checkpoint=CheckpointConfig(ttl_seconds=7 * 24 * 3600))
    removed = store.prune_with_config(config, now=datetime(2026, 1, 8, tzinfo=UTC))

    assert removed == 1
    assert {r.run_id for r in store.list()} == {"recent"}


# --- prune triggered at run completion ---


def test_completed_checkpoint_triggers_prune(tmp_path, monkeypatch) -> None:
    """Saving a completed checkpoint prunes old completed runs (via config)."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    for i in range(5):
        _seed(store, f"old-done-{i}", status="completed", updated_at=f"2026-01-0{i + 1}T00:00:00")

    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(
        project_root=tmp_path, overrides={"checkpoint": {"max_completed": 2}}
    )
    config.policy.hitl_mode = "never"
    agent = PlanExecuteAgent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        checkpoint_store=store,
    )
    plan = _plan_with_tasks()
    plan.tasks["task_1"].mark_completed("done")
    plan.tasks["task_2"].mark_completed("done")
    agent.run_id = new_run_id("plan")
    agent._run_message = "demo goal"

    agent._save_checkpoint(status="completed", plan=plan)

    # Newest two completed survive: the just-finished run + old-done-4.
    remaining = {r.run_id for r in store.list()}
    assert remaining == {agent.run_id, "old-done-4"}


def test_running_checkpoint_does_not_prune(tmp_path, monkeypatch) -> None:
    """A non-completed checkpoint save must not trigger pruning."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    for i in range(5):
        _seed(store, f"old-done-{i}", status="completed", updated_at=f"2026-01-0{i + 1}T00:00:00")

    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(
        project_root=tmp_path, overrides={"checkpoint": {"max_completed": 2}}
    )
    config.policy.hitl_mode = "never"
    agent = PlanExecuteAgent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        checkpoint_store=store,
    )
    plan = _plan_with_tasks()
    agent.run_id = new_run_id("plan")
    agent._run_message = "demo goal"

    agent._save_checkpoint(status="running", plan=plan)

    # Nothing pruned: all 5 seed records + the running one remain.
    remaining = {r.run_id for r in store.list()}
    assert remaining == {
        "old-done-0",
        "old-done-1",
        "old-done-2",
        "old-done-3",
        "old-done-4",
        agent.run_id,
    }


# --- serialization ---


def test_execution_plan_round_trips_through_dict() -> None:
    plan = _plan_with_tasks()
    plan.tasks["task_1"].mark_completed("file contents")
    plan.tasks["task_2"].mark_failed("boom")
    plan.mark_started()

    restored = ExecutionPlan.from_dict(plan.to_dict())

    assert restored.id == "plan_1"
    assert restored.goal == "demo"
    assert restored.status == plan.status
    assert restored.execution_order() == ["task_1", "task_2"]
    assert restored.tasks["task_1"].status == TaskStatus.COMPLETED
    assert restored.tasks["task_1"].result == "file contents"
    assert restored.tasks["task_2"].status == TaskStatus.FAILED
    assert restored.tasks["task_2"].error == "boom"
    assert restored.tasks["task_2"].dependencies == ["task_1"]


def test_execution_step_round_trips_through_dict() -> None:
    step = ExecutionStep("step_1", "implement", "ANALYSIS", ["step_0"], result="done")
    step = step.with_result("done")

    restored = ExecutionStep.from_dict(step.to_dict())

    assert restored.id == "step_1"
    assert restored.description == "implement"
    assert restored.dependencies == ["step_0"]
    assert restored.status == StepStatus.COMPLETED
    assert restored.result == "done"


# --- resuming an interrupted run ---


def test_resuming_a_plan_resets_only_inflight_tasks(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    plan = _plan_with_tasks()
    # task_1 finished before the interruption; task_2 was mid-flight.
    plan.tasks["task_1"].mark_completed("done")
    plan.tasks["task_2"].mark_started()
    record = _record(
        run_id="plan-interrupted",
        status="interrupted",
        cwd=str(tmp_path),
        state={"plan": plan.to_dict()},
        progress={"completed": 1, "total": 2},
    )

    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    store.save(record)
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    agent = PlanExecuteAgent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        checkpoint_store=store,
    )

    events = asyncio.run(_take(agent.resume(record), 1))

    assert events[0]["type"] == "text_delta"
    text = str(events[0]["text"])
    assert "已完成 1/2" in text
    assert "重置 1 个中断任务" in text


def test_resume_without_state_reports_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    record = _record(run_id="plan-empty", status="interrupted", cwd=str(tmp_path))
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    agent = PlanExecuteAgent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=load_config(project_root=tmp_path),
        cwd=str(tmp_path),
        checkpoint_store=CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints"),
    )

    events = asyncio.run(_take(agent.resume(record), 1))

    assert events[0]["type"] == "error"


def test_plan_agent_writes_resumable_checkpoint(tmp_path, monkeypatch) -> None:
    """A run saved mid-flight shows up as resumable with its progress."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    store = CheckpointStore(str(tmp_path), root=tmp_path / "checkpoints")
    registry = ToolRegistry()
    registry.register_all(get_builtin_tools())
    config = load_config(project_root=tmp_path)
    config.policy.hitl_mode = "never"
    agent = PlanExecuteAgent(
        llm_client=FakeClient(),
        tool_registry=registry,
        config=config,
        cwd=str(tmp_path),
        checkpoint_store=store,
    )
    plan = _plan_with_tasks()
    plan.tasks["task_1"].mark_completed("done")
    plan.tasks["task_2"].mark_started()

    agent.run_id = new_run_id("plan")
    agent._run_message = "demo goal"
    agent._save_checkpoint(status="running", plan=plan)

    saved = store.resumable()
    assert len(saved) == 1
    assert saved[0].mode == "plan"
    assert saved[0].progress == {"total": 2, "completed": 1, "failed": 0}
    assert saved[0].state["plan"]["tasks"]["task_1"]["status"] == "COMPLETED"


async def _take(generator, count: int) -> list[dict[str, Any]]:
    """Consume the first *count* events, then close the generator."""
    events: list[dict[str, Any]] = []
    async for event in generator:
        events.append(event)
        if len(events) >= count:
            break
    await generator.aclose()
    return events
