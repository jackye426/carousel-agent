"""studio_persist path resolution (Windows-friendly)."""

from __future__ import annotations

from pathlib import Path

from carousel_agents.ui.studio_persist import resolve_runstate_path


def test_resolve_relative_to_cwd(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "run.json"
    f.write_text('{"schema_version": 3}', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert resolve_runstate_path("run.json", cwd=tmp_path) == f.resolve()


def test_resolve_strips_quotes(tmp_path: Path, monkeypatch) -> None:
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    p = resolve_runstate_path(f'"{str(f.resolve())}"', cwd=tmp_path)
    assert p.is_file()
