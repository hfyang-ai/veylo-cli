from __future__ import annotations

from veylo.snapshot import SnapshotService


def test_snapshot_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    file_path = project / "note.txt"
    file_path.write_text("before", encoding="utf-8")

    service = SnapshotService(project)
    first = service.create("pre-turn")
    file_path.write_text("after", encoding="utf-8")

    restored = service.restore(first.id)

    assert restored.id == first.id
    assert file_path.read_text(encoding="utf-8") == "before"


def test_snapshot_incremental_hardlinks_unchanged_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.txt").write_text("aaa", encoding="utf-8")
    (project / "b.txt").write_text("bbb", encoding="utf-8")

    service = SnapshotService(project)
    first = service.create("pre-turn")
    (project / "a.txt").write_text("changed", encoding="utf-8")
    second = service.create("pre-turn")

    # a.txt 变了 → 独立副本，内容更新
    assert (second.path / "a.txt").read_text(encoding="utf-8") == "changed"
    assert (first.path / "a.txt").stat().st_ino != (second.path / "a.txt").stat().st_ino
    # b.txt 没变 → 硬链接，与上一份快照共享 inode
    assert (first.path / "b.txt").stat().st_ino == (second.path / "b.txt").stat().st_ino


def test_snapshot_is_isolated_from_later_edits(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.txt").write_text("v1", encoding="utf-8")

    service = SnapshotService(project)
    first = service.create("pre-turn")
    (project / "a.txt").write_text("v2", encoding="utf-8")
    service.create("pre-turn")

    # 源文件被改后，旧快照仍保留原内容（硬链接只指向快照副本，不是源文件）
    assert (first.path / "a.txt").read_text(encoding="utf-8") == "v1"


def test_snapshot_incremental_drops_deleted_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.txt").write_text("aaa", encoding="utf-8")
    (project / "b.txt").write_text("bbb", encoding="utf-8")

    service = SnapshotService(project)
    service.create("pre-turn")
    (project / "a.txt").unlink()
    second = service.create("pre-turn")

    assert not (second.path / "a.txt").exists()
    assert (second.path / "b.txt").exists()


def test_snapshot_directory_holds_only_project_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    project = tmp_path / "project"
    project.mkdir()
    (project / "a.txt").write_text("aaa", encoding="utf-8")

    service = SnapshotService(project)
    record = service.create("pre-turn")

    # manifest 必须存在快照目录之外，否则 restore 会把它复制回项目
    names = {item.name for item in record.path.iterdir()}
    assert names == {"a.txt"}
