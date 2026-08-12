from __future__ import annotations

from pathlib import Path

import pytest

from alphazero_gomoku.cli import main
from alphazero_gomoku.release import audit_repository, validate_tag

REQUIRED_FILES = (
    "README.md",
    "README.zh-CN.md",
    "models/README.md",
    "benchmarks/README.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "SECURITY.md",
    "SUPPORT.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "Dockerfile",
    ".github/workflows/container.yml",
    ".github/workflows/release.yml",
    "docs/ROADMAP.md",
)


def _write(root: Path, relative_path: str, content: str = "present\n") -> None:
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")


@pytest.fixture
def release_tree(tmp_path: Path) -> Path:
    for relative_path in REQUIRED_FILES:
        _write(tmp_path, relative_path)
    _write(tmp_path, "pyproject.toml", '[project]\nversion = "1.0.0"\n')
    _write(tmp_path, "README.md", "[简体中文](README.zh-CN.md)\n")
    _write(tmp_path, "README.zh-CN.md", "[English](README.md)\n")
    _write(tmp_path, "models/sample.model", "model\n")
    _write(tmp_path, "models/README.md", "`sample.model`\n")
    _write(tmp_path, "benchmarks/smoke.json", "{}\n")
    _write(tmp_path, "reports/arena.json", "{}\n")
    _write(tmp_path, "benchmarks/README.md", "`smoke.json`\n`reports/arena.json`\n")
    _write(tmp_path, "CHANGELOG.md", "## [1.0.0] - 2026-08-12\n")
    _write(tmp_path, "CITATION.cff", "version: 1.0.0\n")
    _write(tmp_path, "docs/ROADMAP.md", "## 8. Open-source release\n\n**Status: complete.**\n")
    return tmp_path


def test_audit_reports_missing_required_files(tmp_path: Path) -> None:
    problems = audit_repository(tmp_path)

    assert "missing required file: README.md" in problems
    assert "missing required file: .github/workflows/release.yml" in problems


def test_audit_reports_version_mismatch(release_tree: Path) -> None:
    _write(release_tree, "pyproject.toml", '[project]\nversion = "0.9.0"\n')

    assert "pyproject version must be 1.0.0, found 0.9.0" in audit_repository(release_tree)


def test_audit_reports_unlisted_model(release_tree: Path) -> None:
    _write(release_tree, "models/extra.model", "model\n")

    assert "model card does not list: models/extra.model" in audit_repository(release_tree)


def test_audit_reports_unlisted_benchmark_evidence(release_tree: Path) -> None:
    _write(release_tree, "reports/unlisted.json", "{}\n")

    assert (
        "benchmark index does not list: reports/unlisted.json" in audit_repository(release_tree)
    )


def test_audit_reports_broken_language_navigation(release_tree: Path) -> None:
    _write(release_tree, "README.zh-CN.md", "没有英文入口。\n")

    assert "README.zh-CN.md must link to README.md" in audit_repository(release_tree)


def test_validate_tag_matches_version() -> None:
    assert validate_tag("1.0.0", "v1.0.0") == []
    assert validate_tag("1.0.0", "v1.0.1") == [
        "tag v1.0.1 does not match package version 1.0.0"
    ]


def test_valid_release_tree_passes_audit(release_tree: Path) -> None:
    assert audit_repository(release_tree) == []


def test_release_check_cli_reports_all_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["release-check", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert result == 1
    assert "Release readiness check failed:" in captured.err
    assert "missing required file: README.md" in captured.err
    assert "missing required file: Dockerfile" in captured.err


def test_release_check_cli_reports_success(
    release_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(["release-check", "--root", str(release_tree)])

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert "Release readiness check passed for 1.0.0." in captured.out
