from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "node_modules", "dist", "build", "target", "__pycache__"}


@dataclass(slots=True)
class SnapshotRecord:
    id: str
    phase: str
    created_at: str
    path: Path


class SnapshotService:
    """Snapshot the project tree so files can be rolled back.

    Snapshots are stored incrementally. Each snapshot hard-links files whose
    fingerprint (size + mtime) is unchanged from the previous snapshot and only
    copies the files that actually changed. The per-snapshot cost therefore
    scales with the number of *changed* files, not the size of the project.

    Manifests live outside the snapshot directory (``root/manifests/``) so a
    snapshot directory contains only project files and can be restored with a
    plain full copy.
    """

    def __init__(self, project_root: str | Path):
        self.project_root = Path(project_root).resolve()
        digest = hashlib.sha256(str(self.project_root).encode("utf-8")).hexdigest()[:16]
        self.root = Path.home() / ".veylo" / "snapshots" / digest
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.jsonl"
        self._manifest_dir = self.root / "manifests"
        self._manifest_dir.mkdir(parents=True, exist_ok=True)

    def create(self, phase: str) -> SnapshotRecord:
        snapshot_id = f"{phase}_{datetime.now(UTC).strftime('%Y%m%d%H%M%S%f')}"
        target = self.root / snapshot_id
        target.mkdir(parents=True, exist_ok=True)
        base = self._latest_base()
        manifest: dict[str, list[int]] = {}
        self._copy_incremental(self.project_root, target, base, manifest)
        self._write_manifest(snapshot_id, manifest)
        record = SnapshotRecord(
            id=snapshot_id,
            phase=phase,
            created_at=datetime.now(UTC).isoformat(),
            path=target,
        )
        self._append_index(record)
        return record

    def list(self, limit: int = 20) -> list[SnapshotRecord]:
        records = [
            SnapshotRecord(
                id=item["id"],
                phase=item["phase"],
                created_at=item["created_at"],
                path=Path(item["path"]),
            )
            for item in self._read_index()
        ]
        return records[-limit:][::-1]

    def restore(self, snapshot_ref: str) -> SnapshotRecord:
        records = self.list(limit=200)
        record = None
        if snapshot_ref.isdigit():
            index = int(snapshot_ref) - 1
            if 0 <= index < len(records):
                record = records[index]
        else:
            record = next((item for item in records if item.id == snapshot_ref), None)
        if not record:
            raise ValueError(f"snapshot not found: {snapshot_ref}")
        self.create("pre-restore")
        self._restore_tree(record.path, self.project_root)
        return record

    def clean(self) -> int:
        count = len(self.list(limit=10_000))
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest_dir.mkdir(parents=True, exist_ok=True)
        return count

    # ------------------------------------------------------------------
    # Internal: index
    # ------------------------------------------------------------------

    def _read_index(self) -> list[dict]:
        if not self.index_path.exists():
            return []
        items = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return items

    def _append_index(self, record: SnapshotRecord) -> None:
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "id": record.id,
                        "phase": record.phase,
                        "created_at": record.created_at,
                        "path": str(record.path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ------------------------------------------------------------------
    # Internal: manifest (one JSON per snapshot, stored outside the tree)
    # ------------------------------------------------------------------

    def _manifest_path(self, snapshot_id: str) -> Path:
        return self._manifest_dir / f"{snapshot_id}.json"

    def _write_manifest(self, snapshot_id: str, manifest: dict[str, list[int]]) -> None:
        with self._manifest_path(snapshot_id).open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False)

    def _latest_base(self) -> tuple[Path | None, dict[str, tuple[int, int]]]:
        """Find the newest snapshot that has an intact manifest to link against."""
        for item in reversed(self._read_index()):
            base_dir = Path(item["path"])
            manifest_file = self._manifest_path(item["id"])
            if not base_dir.is_dir() or not manifest_file.is_file():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, ValueError):
                continue
            return base_dir, {key: tuple(value) for key, value in data.items()}
        return None, {}

    # ------------------------------------------------------------------
    # Internal: tree copy (incremental create vs full restore)
    # ------------------------------------------------------------------

    def _copy_incremental(
        self,
        source: Path,
        target: Path,
        base: tuple[Path | None, dict[str, tuple[int, int]]],
        manifest: dict[str, list[int]],
    ) -> None:
        base_dir, base_manifest = base
        for item in source.iterdir():
            if _skip(item):
                continue
            destination = target / item.name
            if item.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                self._copy_incremental(item, destination, base, manifest)
            elif item.is_file():
                relative = item.relative_to(self.project_root).as_posix()
                fingerprint = _fingerprint(item)
                base_file = base_dir / relative if base_dir else None
                if (
                    base_file is not None
                    and base_file.is_file()
                    and base_manifest.get(relative) == fingerprint
                    and _hardlink(base_file, destination)
                ):
                    manifest[relative] = list(fingerprint)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)
                manifest[relative] = list(fingerprint)

    def _copy_tree(self, source: Path, target: Path) -> None:
        for item in source.iterdir():
            if _skip(item):
                continue
            destination = target / item.name
            if item.is_dir():
                shutil.copytree(item, destination, ignore=_ignore)
            elif item.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, destination)

    def _restore_tree(self, source: Path, target: Path) -> None:
        for item in target.iterdir():
            if _skip(item):
                continue
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        self._copy_tree(source, target)


def _skip(path: Path) -> bool:
    return path.name in SKIP_DIRS


def _ignore(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name in SKIP_DIRS}


def _fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def _hardlink(source: Path, destination: Path) -> bool:
    """Hard-link ``source`` to ``destination``; ``False`` if the FS forbids it."""
    try:
        os.link(source, destination)
        return True
    except OSError:
        return False
