from __future__ import annotations

import asyncio
from typing import Any

from veylo.agent import PlanExecuteAgent
from veylo.agent.orchestrator import ExecutionStep, StepStatus
from veylo.checkpoint import CheckpointRecord, CheckpointStore, new_run_id
from veylo.config import load_config
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
