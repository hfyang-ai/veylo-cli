"""checkpoint/service.py — Persistent execution progress for resume.

``SnapshotService`` copies the whole project tree so files can be rolled back.
This module stores the *other* half of recovery: **where the run got to**. A
checkpoint is a small JSON record (mode, plan/step states, accumulated usage)
so an interrupted Plan-and-Execute or Multi-Agent run can continue from the
last finished step instead of re-running — and re-paying for — everything.

Records are appended as JSONL, one line per save. The last line of a run's file
is its current state, so a crash mid-write cannot corrupt earlier state and
recovery is just "read the last line".
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from veylo.config import VeyloConfig

# Statuses that mean "this run is worth resuming".
RESUMABLE_STATUSES = ("running", "interrupted")


def new_run_id(mode: str) -> str:
    """Build a unique, human-sortable run id, e.g. ``plan-20261001T020000-a1b2c3``."""
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{mode}-{stamp}-{uuid4().hex[:6]}"


@dataclass(slots=True)
class CheckpointRecord:
    """One saved point in a run's execution."""

    run_id: str
    mode: str
    message: str
    cwd: str
    status: str = "running"
    created_at: str = ""
    updated_at: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    turns: int = 0
    error: str = ""
    progress: dict[str, Any] = field(default_factory=dict)

    @property
    def resumable(self) -> bool:
        return self.status in RESUMABLE_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "message": self.message,
            "cwd": self.cwd,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state,
            "usage": self.usage,
            "turns": self.turns,
            "error": self.error,
            "progress": self.progress,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointRecord:
        return cls(
            run_id=data["run_id"],
            mode=data.get("mode", ""),
            message=data.get("message", ""),
            cwd=data.get("cwd", ""),
            status=data.get("status", "running"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            state=data.get("state") or {},
            usage=data.get("usage") or {},
            turns=int(data.get("turns", 0)),
            error=data.get("error", ""),
            progress=data.get("progress") or {},
        )


class CheckpointStore:
    """Append-only checkpoint storage rooted per project."""

    def __init__(self, project_root: str | Path, root: Path | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        digest = hashlib.sha256(str(self.project_root).encode("utf-8")).hexdigest()[:16]
        self.root = (
            Path(root) if root is not None else (Path.home() / ".veylo" / "checkpoints" / digest)
        )
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.jsonl"

    def save(self, record: CheckpointRecord) -> CheckpointRecord:
        """Append the record's current state. Returns the record for chaining."""
        now = datetime.now(UTC).isoformat()
        record.updated_at = now
        if not record.created_at:
            record.created_at = now
        payload = json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
        with self._path(record.run_id).open("a", encoding="utf-8") as handle:
            handle.write(payload)
        return record

    def load(self, run_id: str) -> CheckpointRecord | None:
        """Return the latest state for a run, or ``None`` if missing/corrupt."""
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except OSError:
            return None
        if not lines:
            return None
        # Walk backwards so a half-written trailing line (crash mid-write) falls
        # back to the last intact state instead of losing the run.
        for line in reversed(lines):
            try:
                return CheckpointRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return None

    def list(self, limit: int = 20) -> list[CheckpointRecord]:
        """List runs, most recently updated first."""
        records: list[CheckpointRecord] = []
        for path in self.root.glob("*.jsonl"):
            record = self.load(path.stem)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.updated_at, reverse=True)
        return records[:limit]

    def resumable(self, limit: int = 20) -> list[CheckpointRecord]:
        """List only runs that stopped before finishing."""
        return [item for item in self.list(limit=10_000) if item.resumable][:limit]

    def delete(self, run_id: str) -> bool:
        path = self._path(run_id)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def clean(self) -> int:
        """Remove every checkpoint for this project. Returns the count removed."""
        removed = 0
        for path in list(self.root.glob("*.jsonl")):
            path.unlink()
            removed += 1
        return removed

    def prune(
        self,
        *,
        max_completed: int = 2,
        max_resumable: int = 10,
        ttl: timedelta | None = None,
        now: datetime | None = None,
    ) -> int:
        """Drop checkpoints that are no longer worth keeping.

        Finished runs are capped at ``max_completed`` and unfinished runs
        (``running``/``interrupted``) at ``max_resumable``, always dropping the
        oldest first. When ``ttl`` is given, any unfinished run whose
        ``updated_at`` is older than ``ttl`` is also removed — this reaps
        ``running`` records left behind by a crashed process that never got to
        mark itself finished.

        ``now`` is injectable for tests; defaults to the current UTC time.
        Returns the number of records removed.
        """
        now = now or datetime.now(UTC)
        records = self.list(limit=10_000)
        completed = [record for record in records if record.status == "completed"]
        unfinished = [record for record in records if record.status in RESUMABLE_STATUSES]

        doomed: set[str] = set()
        doomed.update(record.run_id for record in completed[max_completed:])

        survivors: list[CheckpointRecord] = []
        for record in unfinished:
            if ttl is not None and _is_expired(record, ttl, now):
                doomed.add(record.run_id)
                continue
            survivors.append(record)
        doomed.update(record.run_id for record in survivors[max_resumable:])

        removed = 0
        for run_id in doomed:
            if self.delete(run_id):
                removed += 1
        return removed

    def prune_with_config(self, config: VeyloConfig, *, now: datetime | None = None) -> int:
        """Prune using the retention policy from ``config.checkpoint``.

        Convenience wrapper so callers can drive cleanup from the same
        ``~/.veylo/config.json`` that configures the rest of Veylo.
        """
        policy = config.checkpoint
        return self.prune(
            max_completed=policy.max_completed,
            max_resumable=policy.max_resumable,
            ttl=policy.ttl,
            now=now,
        )


def _is_expired(record: CheckpointRecord, ttl: timedelta, now: datetime) -> bool:
    """True if ``record.updated_at`` is older than ``ttl`` relative to ``now``.

    Unparseable timestamps are kept (conservative: never drop something we
    cannot safely age out).
    """
    try:
        updated = datetime.fromisoformat(record.updated_at)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    return now - updated > ttl
