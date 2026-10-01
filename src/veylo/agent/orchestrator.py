from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from veylo.agent.query import query
from veylo.checkpoint import CheckpointRecord, CheckpointStore, new_run_id
from veylo.config import VeyloConfig
from veylo.llm.base import LlmClient
from veylo.prompt import PromptAssembler
from veylo.skill import SkillContextBuffer
from veylo.snapshot import SnapshotService
from veylo.tools.registry import ToolRegistry
from veylo.types import Message, Usage


class AgentRole(StrEnum):
    PLANNER = "PLANNER"
    WORKER = "WORKER"
    REVIEWER = "REVIEWER"


class AgentMessageType(StrEnum):
    TASK = "TASK"
    RESULT = "RESULT"
    FEEDBACK = "FEEDBACK"
    APPROVAL = "APPROVAL"
    REJECTION = "REJECTION"
    ERROR = "ERROR"


class StepStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentRunMode(StrEnum):
    REACT = "react"
    PLAN = "plan"


@dataclass(slots=True)
class AgentMessage:
    from_agent: str
    from_role: AgentRole | None
    content: str
    type: AgentMessageType
    usage: Usage = field(default_factory=Usage)
    turns: int = 0

    @classmethod
    def task(cls, from_agent: str, content: str) -> AgentMessage:
        return cls(from_agent, None, content, AgentMessageType.TASK)

    @classmethod
    def result(
        cls,
        from_agent: str,
        role: AgentRole,
        content: str,
        usage: Usage | None = None,
        turns: int = 0,
    ) -> AgentMessage:
        return cls(from_agent, role, content, AgentMessageType.RESULT, usage or Usage(), turns)

    @classmethod
    def error(
        cls,
        from_agent: str,
        role: AgentRole,
        content: str,
        usage: Usage | None = None,
        turns: int = 0,
    ) -> AgentMessage:
        return cls(from_agent, role, content, AgentMessageType.ERROR, usage or Usage(), turns)


@dataclass(slots=True)
class ExecutionStep:
    id: str
    description: str
    type: str
    dependencies: list[str]
    mode: str = AgentRunMode.REACT.value
    result: str = ""
    status: StepStatus = StepStatus.PENDING

    def with_result(self, result: str) -> ExecutionStep:
        return replace(self, result=result, status=StepStatus.COMPLETED)

    def with_failed(self, result: str) -> ExecutionStep:
        return replace(self, result=result, status=StepStatus.FAILED)

    def started(self) -> ExecutionStep:
        return replace(self, status=StepStatus.RUNNING)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "dependencies": list(self.dependencies),
            "mode": self.mode,
            "result": self.result,
            "status": str(self.status),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionStep:
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            type=data.get("type", "ANALYSIS"),
            dependencies=list(data.get("dependencies", [])),
            mode=data.get("mode", AgentRunMode.REACT.value),
            result=data.get("result", ""),
            status=StepStatus(data.get("status", StepStatus.PENDING.value)),
        )


class SubAgent:
    def __init__(
        self,
        *,
        name: str,
        role: AgentRole,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: VeyloConfig,
        cwd: str,
        approval_callback=None,
        skill_context_buffer: SkillContextBuffer | None = None,
        default_mode: str = AgentRunMode.REACT.value,
        max_plan_depth: int = 1,
    ):
        self.name = name
        self.role = role
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.skill_context_buffer = skill_context_buffer or SkillContextBuffer()
        self.default_mode = _normalize_worker_mode(default_mode)
        self.max_plan_depth = max(1, max_plan_depth)
        self.history: list[Message] = []

    async def execute(
        self,
        task: AgentMessage,
        context: str = "",
        *,
        mode: str | None = None,
        plan_depth: int = 0,
    ) -> AgentMessage:
        content = f"{context}\n\nCurrent task:\n{task.content}".strip() if context else task.content
        if self.role == AgentRole.WORKER:
            selected_mode = _normalize_worker_mode(mode or self.default_mode)
            if selected_mode == AgentRunMode.PLAN:
                return await self._execute_plan(content, plan_depth=plan_depth)
            return await self._execute_worker(content)
        return await self._execute_without_tools(content)

    async def review(self, original_task: str, execution_result: str) -> AgentMessage:
        return await self.execute(
            AgentMessage.task(
                "orchestrator",
                f"Original task:\n{original_task}\n\nExecution result:\n{execution_result}",
            )
        )

    def clear_history(self) -> None:
        self.history = []
        self.skill_context_buffer.clear()

    async def _execute_worker(self, content: str) -> AgentMessage:
        text = ""
        tool_results: list[str] = []
        usage = Usage()
        turns = 0
        try:
            async for event in query(
                llm_client=self.llm_client,
                tool_registry=self.tool_registry,
                system_prompt=self._system_prompt(),
                user_message=content,
                history=self.history,
                cwd=self.cwd,
                config=self.config,
                approval_callback=self.approval_callback,
                skill_context_buffer=self.skill_context_buffer,
                max_turns=8,
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "tool_result":
                    tool_results.append(str(event.get("result") or ""))
                elif event.get("type") == "done":
                    self.history = list(event.get("messages") or [])
                    usage = usage + Usage.from_mapping(event.get("usage") or {})
                    turns += int(event.get("total_turns") or 0)
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc), usage, turns)
        result = text.strip() or "\n".join(item for item in tool_results if item).strip()
        return AgentMessage.result(self.name, self.role, result, usage, turns)

    async def _execute_plan(self, content: str, *, plan_depth: int) -> AgentMessage:
        if plan_depth >= self.max_plan_depth:
            return AgentMessage.error(
                self.name,
                self.role,
                f"nested Plan depth limit ({self.max_plan_depth}) reached",
            )
        from veylo.agent.plan_execute import PlanExecuteAgent

        agent = PlanExecuteAgent(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
        )
        text = ""
        usage = Usage()
        turns = 0
        try:
            async for event in agent.run(content):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "done":
                    usage = usage + Usage.from_mapping(event.get("usage") or {})
                    turns += int(event.get("total_turns") or 0)
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc), usage, turns)
        self.history = [
            Message(role="user", content=content),
            Message(role="assistant", content=text),
        ]
        return AgentMessage.result(self.name, self.role, text.strip(), usage, turns)

    async def _execute_without_tools(self, content: str) -> AgentMessage:
        text = ""
        usage = Usage()
        messages = [*self.history, Message(role="user", content=content)]
        try:
            async for event in self.llm_client.chat(
                messages,
                [],
                system_prompt=self._system_prompt(),
            ):
                if event.get("type") == "text_delta":
                    text += str(event.get("text") or "")
                elif event.get("type") == "usage":
                    usage = usage + Usage.from_mapping(event.get("usage") or {})
                elif event.get("type") == "error":
                    raise event["error"]
        except Exception as exc:  # noqa: BLE001
            return AgentMessage.error(self.name, self.role, str(exc), usage, 1)
        self.history = [*messages, Message(role="assistant", content=text)]
        return AgentMessage.result(self.name, self.role, text, usage, 1)

    def _system_prompt(self) -> str:
        base = PromptAssembler(
            config=self.config,
            cwd=self.cwd,
            tool_names=self.tool_registry.list_names(),
            model=self.llm_client.model_name,
            provider=self.llm_client.provider_name,
        ).build_static()
        role_prompt = {
            AgentRole.PLANNER: (
                "You are the Planner in a multi-agent workflow. Return only JSON with a "
                "steps array. Each step needs id, description, type, dependencies, and optional "
                'mode ("react" or "plan"). Use plan only when a worker needs its own nested DAG.'
            ),
            AgentRole.WORKER: (
                "You are the Worker in a multi-agent workflow. Execute only the assigned "
                "step. Use tools when needed and return the concrete result."
            ),
            AgentRole.REVIEWER: (
                "You are the Reviewer in a multi-agent workflow. Return JSON only: "
                '{"approved": true|false, "summary": "...", "issues": []}.'
            ),
        }[self.role]
        return f"{base}\n\n{role_prompt}\nAgent name: {self.name}"


class AgentOrchestrator:
    max_retries_per_step = 2

    def __init__(
        self,
        *,
        llm_client: LlmClient,
        tool_registry: ToolRegistry,
        config: VeyloConfig,
        cwd: str,
        approval_callback=None,
        worker_count: int = 2,
        default_worker_mode: str = AgentRunMode.REACT.value,
        checkpoint_store: CheckpointStore | None = None,
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.config = config
        self.cwd = cwd
        self.approval_callback = approval_callback
        self.default_worker_mode = _normalize_worker_mode(default_worker_mode)
        self.planner = self._subagent("planner", AgentRole.PLANNER)
        self.workers = [
            self._subagent(f"worker-{index}", AgentRole.WORKER)
            for index in range(1, max(1, worker_count) + 1)
        ]
        self.reviewer = self._subagent("reviewer", AgentRole.REVIEWER)
        self.history: list[Message] = []
        self.total_usage = Usage()
        self.total_turns = 0
        self._checkpoint_store = checkpoint_store
        self.run_id: str | None = None
        self._run_message = ""

    @property
    def checkpoint_store(self) -> CheckpointStore:
        if self._checkpoint_store is None:
            self._checkpoint_store = CheckpointStore(self.cwd)
        return self._checkpoint_store

    def _save_checkpoint(
        self,
        *,
        status: str,
        steps: list[ExecutionStep] | None = None,
        error: str = "",
    ) -> None:
        """Persist step progress; never let a storage failure break the run."""
        if not self.run_id:
            return
        record = CheckpointRecord(
            run_id=self.run_id,
            mode="team",
            message=self._run_message,
            cwd=self.cwd,
            status=status,
            state={"steps": [step.to_dict() for step in steps]} if steps else {},
            usage=self.total_usage.to_dict(),
            turns=self.total_turns,
            error=error,
            progress=_steps_progress(steps or []),
        )
        with suppress(Exception):
            self.checkpoint_store.save(record)
            # A run just finished; opportunistically prune old checkpoints.
            if status == "completed":
                self.checkpoint_store.prune_with_config(self.config)

    async def run(self, message: str) -> AsyncIterator[dict[str, Any]]:
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-turn")
        final_text = ""
        self.total_usage = Usage()
        self.total_turns = 0
        self._run_message = message
        self.run_id = new_run_id("team")
        steps: list[ExecutionStep] = []
        try:
            yield {"type": "text_delta", "text": "Phase 1: planner\n\n"}
            plan_result = await self.planner.execute(
                AgentMessage.task("orchestrator", f"Create an execution plan for:\n{message}")
            )
            self.total_usage = self.total_usage + plan_result.usage
            self.total_turns += plan_result.turns
            self.planner.clear_history()
            if plan_result.type == AgentMessageType.ERROR:
                raise RuntimeError(f"planner failed: {plan_result.content}")
            steps = self.parse_plan(plan_result.content)
            if not steps:
                raise ValueError(f"planner output could not be parsed:\n{plan_result.content}")
            self._save_checkpoint(status="running", steps=steps)
            yield {"type": "text_delta", "text": self.summarize_steps(steps) + "\n"}
            yield {"type": "text_delta", "text": "Phase 2: workers and reviewer\n\n"}
            for event in await self._execute_steps(
                steps, lambda text: {"type": "text_delta", "text": text}
            ):
                yield event
            final_text = self.build_final_result(steps)
            yield {"type": "text_delta", "text": final_text}
            self.history = [
                Message(role="user", content=message),
                Message(role="assistant", content=final_text),
            ]
            self._save_checkpoint(status="completed", steps=steps)
        except Exception as exc:  # noqa: BLE001
            self._save_checkpoint(status="interrupted", steps=steps, error=str(exc))
            yield {"type": "error", "error": exc}
            return
        done: dict[str, Any] = {
            "type": "done",
            "total_turns": self.total_turns,
            "total_tokens": self.total_usage.total_tokens,
            "usage": self.total_usage.to_dict(),
            "messages": self.history,
        }
        costs = _calculate_costs(self.llm_client, self.total_usage)
        if costs:
            done["cost"] = costs
        yield done

    async def resume(self, record: CheckpointRecord) -> AsyncIterator[dict[str, Any]]:
        """Continue an interrupted multi-agent run from its last checkpoint."""
        raw_steps = record.state.get("steps")
        if not raw_steps:
            yield {"type": "error", "error": ValueError("checkpoint 不包含步骤状态")}
            return
        steps = [ExecutionStep.from_dict(item) for item in raw_steps]
        reset = _reset_inflight_steps(steps)
        progress = _steps_progress(steps)
        snapshot = SnapshotService(self.cwd)
        with suppress(Exception):
            snapshot.create("pre-resume")
        self.run_id = record.run_id
        self._run_message = record.message
        self.total_usage = Usage.from_mapping(record.usage or {})
        self.total_turns = record.turns
        try:
            yield {
                "type": "text_delta",
                "text": (
                    f"从检查点恢复：{record.run_id}\n"
                    f"已完成 {progress['completed']}/{progress['total']}，"
                    f"重置 {reset} 个中断步骤。\n\n"
                ),
            }
            for event in await self._execute_steps(
                steps, lambda text: {"type": "text_delta", "text": text}
            ):
                yield event
            final_text = self.build_final_result(steps)
            yield {"type": "text_delta", "text": final_text}
            self.history = [
                Message(role="user", content=record.message),
                Message(role="assistant", content=final_text),
            ]
            self._save_checkpoint(status="completed", steps=steps)
        except Exception as exc:  # noqa: BLE001
            self._save_checkpoint(status="interrupted", steps=steps, error=str(exc))
            yield {"type": "error", "error": exc}
            return
        done: dict[str, Any] = {
            "type": "done",
            "total_turns": self.total_turns,
            "total_tokens": self.total_usage.total_tokens,
            "usage": self.total_usage.to_dict(),
            "messages": self.history,
        }
        costs = _calculate_costs(self.llm_client, self.total_usage)
        if costs:
            done["cost"] = costs
        yield done

    async def _execute_steps(
        self,
        steps: list[ExecutionStep],
        event_factory,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        retry_count: dict[str, int] = {}
        worker_queue: asyncio.Queue[SubAgent] = asyncio.Queue()
        for worker in self.workers:
            worker_queue.put_nowait(worker)

        while True:
            executable = self.get_executable_steps(steps)
            if not executable:
                break
            if len(executable) > 1:
                events.append(
                    event_factory(
                        f"Parallel batch: {', '.join(step.id for step in executable)}\n\n"
                    )
                )
            await asyncio.gather(
                *(
                    self._run_step_with_worker_queue(
                        step,
                        steps,
                        retry_count,
                        worker_queue,
                    )
                    for step in executable
                )
            )
        return events

    async def _run_step_with_worker_queue(
        self,
        step: ExecutionStep,
        steps: list[ExecutionStep],
        retry_count: dict[str, int],
        worker_queue: asyncio.Queue[SubAgent],
    ) -> None:
        worker = await worker_queue.get()
        try:
            reviewer = self._subagent(f"reviewer-{step.id}", AgentRole.REVIEWER)
            await self._run_step(step, steps, retry_count, worker, reviewer)
        finally:
            worker.clear_history()
            worker_queue.put_nowait(worker)

    async def _run_step(
        self,
        step: ExecutionStep,
        steps: list[ExecutionStep],
        retry_count: dict[str, int],
        worker: SubAgent,
        reviewer: SubAgent,
    ) -> None:
        self._update_step(steps, step.id, step.started())
        context = self.build_step_context(steps, step)
        task_msg = AgentMessage.task("orchestrator", step.description)
        result = await worker.execute(task_msg, context, mode=step.mode)
        self.total_usage = self.total_usage + result.usage
        self.total_turns += result.turns
        if result.type == AgentMessageType.ERROR or not result.content.strip():
            self._update_step(steps, step.id, step.with_failed(result.content or "empty result"))
            return

        accepted_result = result.content
        review = await reviewer.review(step.description, accepted_result)
        self.total_usage = self.total_usage + review.usage
        self.total_turns += review.turns
        reviewer.clear_history()
        approved = self.parse_review_approval(review.content)
        issues = self.parse_review_issues(review.content)
        retries = retry_count.get(step.id, 0)
        while not approved and retries < self.max_retries_per_step:
            retries += 1
            retry_count[step.id] = retries
            retry_context = context + f"\n\nReviewer rejected the previous result:\n{issues}"
            retry_result = await worker.execute(task_msg, retry_context, mode=step.mode)
            self.total_usage = self.total_usage + retry_result.usage
            self.total_turns += retry_result.turns
            if retry_result.type == AgentMessageType.ERROR or not retry_result.content.strip():
                issues = retry_result.content or "empty retry result"
                continue
            accepted_result = retry_result.content
            retry_review = await reviewer.review(step.description, accepted_result)
            self.total_usage = self.total_usage + retry_review.usage
            self.total_turns += retry_review.turns
            reviewer.clear_history()
            approved = self.parse_review_approval(retry_review.content)
            issues = self.parse_review_issues(retry_review.content)

        if not approved:
            self._update_step(
                steps,
                step.id,
                step.with_failed(f"review rejected after {retries} retries: {issues}"),
            )
            return
        self._update_step(steps, step.id, step.with_result(accepted_result))

    def parse_plan(self, plan_json: str) -> list[ExecutionStep]:
        try:
            data = _parse_json_object(plan_json)
        except (json.JSONDecodeError, ValueError):
            return []
        nodes = data.get("steps") or data.get("tasks") or []
        if not isinstance(nodes, list) or not nodes:
            return []
        id_mapping: dict[str, str] = {}
        steps: list[ExecutionStep] = []
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict):
                continue
            original_id = str(node.get("id") or f"step_{index}")
            new_id = f"step_{index}"
            id_mapping[original_id] = new_id
            steps.append(
                ExecutionStep(
                    id=new_id,
                    description=str(node.get("description") or original_id),
                    type=str(node.get("type") or "COMMAND"),
                    dependencies=[],
                    mode=_normalize_worker_mode(
                        str(
                            node.get("mode")
                            or (
                                AgentRunMode.PLAN.value
                                if str(node.get("type") or "").upper() == "PLAN"
                                else self.default_worker_mode
                            )
                        )
                    ),
                )
            )
        for index, node in enumerate(nodes, start=1):
            if not isinstance(node, dict) or index > len(steps):
                continue
            raw_deps = node.get("dependencies") or []
            if not isinstance(raw_deps, list):
                continue
            steps[index - 1].dependencies = [
                id_mapping.get(str(dep), str(dep)) for dep in raw_deps if str(dep)
            ]
        return steps

    def get_executable_steps(self, steps: list[ExecutionStep]) -> list[ExecutionStep]:
        status = {step.id: step.status for step in steps}
        return [
            step
            for step in steps
            if step.status == StepStatus.PENDING
            and all(status.get(dep) == StepStatus.COMPLETED for dep in step.dependencies)
        ]

    def parse_review_approval(self, review_content: str | None) -> bool:
        if not review_content:
            return False
        try:
            data = _parse_json_object(review_content)
            if "approved" not in data:
                return False
            return bool(data.get("approved"))
        except (json.JSONDecodeError, ValueError):
            lower = review_content.lower()
            negative = ["未通过", "不通过", "不合格", "有问题", '"approved": false']
            positive = ["通过", "合格", '"approved": true']
            if any(item in lower for item in negative):
                return False
            return any(item in lower for item in positive)

    def parse_review_issues(self, review_content: str | None) -> str:
        if not review_content:
            return ""
        try:
            data = _parse_json_object(review_content)
        except (json.JSONDecodeError, ValueError):
            return "review rejected the result"
        for key in ("issues", "suggestions"):
            value = data.get(key)
            if isinstance(value, list) and value:
                return "\n".join(f"- {item}" for item in value)
        return str(data.get("summary") or "review rejected the result")

    def build_step_context(self, steps: list[ExecutionStep], current_step: ExecutionStep) -> str:
        lines = ["Overall task context:"]
        for step in steps:
            if step.id in current_step.dependencies and step.status == StepStatus.COMPLETED:
                lines.append(f"[{step.id}] {step.description}")
                if step.result:
                    lines.append(f"Result: {_preview(step.result, 500)}")
        return "\n".join(lines)

    def summarize_steps(self, steps: list[ExecutionStep]) -> str:
        lines = ["Execution plan:"]
        for step in steps:
            deps = ", ".join(step.dependencies) if step.dependencies else "none"
            lines.append(
                f"- [{step.id}] {step.description} ({step.type}, mode: {step.mode}, deps: {deps})"
            )
        return "\n".join(lines)

    def build_final_result(self, steps: list[ExecutionStep]) -> str:
        all_completed = all(step.status == StepStatus.COMPLETED for step in steps)
        failed = any(step.status == StepStatus.FAILED for step in steps)
        if all_completed:
            header = "Multi-Agent task completed."
        elif failed:
            header = "Multi-Agent task did not fully complete; failed steps remain."
        else:
            header = "Multi-Agent task partially completed; pending steps remain."
        lines = [header, "", "Execution summary:"]
        for step in steps:
            icon = {
                StepStatus.COMPLETED: "COMPLETED",
                StepStatus.FAILED: "FAILED",
                StepStatus.PENDING: "PENDING",
                StepStatus.RUNNING: "RUNNING",
            }[step.status]
            lines.append(f"- [{step.id}] {icon}: {step.description}")
            if step.result:
                lines.append(f"  Result: {_preview(step.result)}")
        return "\n".join(lines) + "\n"

    def _subagent(self, name: str, role: AgentRole) -> SubAgent:
        return SubAgent(
            name=name,
            role=role,
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
            config=self.config,
            cwd=self.cwd,
            approval_callback=self.approval_callback,
            # Every sub-agent owns its buffer. Sharing it lets concurrent workers consume each
            # other's loaded Skill instructions.
            skill_context_buffer=SkillContextBuffer(),
            default_mode=self.default_worker_mode,
        )

    def _update_step(
        self,
        steps: list[ExecutionStep],
        step_id: str,
        updated: ExecutionStep,
    ) -> None:
        for index, step in enumerate(steps):
            if step.id == step_id:
                steps[index] = updated
                break
        # Land progress on every step transition (start / complete / fail) so an
        # interruption only loses the step in flight.
        self._save_checkpoint(status="running", steps=steps)


def _parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", text or "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("empty JSON")
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def _steps_progress(steps: list[ExecutionStep]) -> dict[str, int]:
    return {
        "total": len(steps),
        "completed": sum(1 for step in steps if step.status == StepStatus.COMPLETED),
        "failed": sum(1 for step in steps if step.status == StepStatus.FAILED),
    }


def _reset_inflight_steps(steps: list[ExecutionStep]) -> int:
    """Re-queue steps left RUNNING by an interrupted run.

    A checkpoint written between "step started" and "step finished" leaves the
    step RUNNING, but nothing is driving it any more, so it must go back to
    PENDING to be picked up again.
    """
    reset = 0
    for step in steps:
        if step.status == StepStatus.RUNNING:
            step.status = StepStatus.PENDING
            reset += 1
    return reset


def _preview(text: str, max_len: int = 160) -> str:
    value = (text or "").replace("\r\n", "\n").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


def _normalize_worker_mode(mode: str) -> str:
    value = str(mode or AgentRunMode.REACT.value).strip().lower()
    if value not in {item.value for item in AgentRunMode}:
        raise ValueError("worker mode must be react or plan")
    return value


def _calculate_costs(llm_client: LlmClient, usage: Usage) -> dict[str, Any]:
    calculator = getattr(llm_client, "calculate_cost", None)
    if not callable(calculator):
        return {}
    result: dict[str, Any] = {}
    for currency in ("usd", "cny"):
        try:
            result[currency] = calculator(usage, currency=currency).to_dict()
        except (KeyError, TypeError, ValueError):
            continue
    return result
