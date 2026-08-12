"""Release-readiness checks for the repository."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


RELEASE_VERSION = "1.0.0"
REQUIRED_FILES = (
    "pyproject.toml",
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


def project_version(root: Path) -> str | None:
    """Return the package version declared by ``pyproject.toml`` when readable."""
    metadata = root / "pyproject.toml"
    if not metadata.is_file():
        return None
    try:
        with metadata.open("rb") as stream:
            value = tomllib.load(stream)["project"]["version"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError):
        return None
    return value if isinstance(value, str) else None


def validate_tag(version: str, tag: str) -> list[str]:
    """Validate that a semantic release tag names ``version`` exactly."""
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        return [f"tag must match vMAJOR.MINOR.PATCH, found {tag}"]
    expected = f"v{version}"
    if tag != expected:
        return [f"tag {tag} does not match package version {version}"]
    return []


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def audit_repository(root: Path) -> list[str]:
    """Return every release-readiness violation found under ``root``."""
    root = root.resolve()
    problems = [
        f"missing required file: {relative}"
        for relative in REQUIRED_FILES
        if not (root / relative).is_file()
    ]

    version = project_version(root)
    if (root / "pyproject.toml").is_file() and version != RELEASE_VERSION:
        found = version or "unreadable"
        problems.append(
            f"pyproject version must be {RELEASE_VERSION}, found {found}"
        )

    english_readme = _read_text(root / "README.md")
    chinese_readme = _read_text(root / "README.zh-CN.md")
    if english_readme and "README.zh-CN.md" not in english_readme:
        problems.append("README.md must link to README.zh-CN.md")
    if chinese_readme and "README.md" not in chinese_readme:
        problems.append("README.zh-CN.md must link to README.md")

    model_card = _read_text(root / "models" / "README.md")
    if model_card:
        for model in sorted((root / "models").glob("*.model*")):
            if model.name not in model_card:
                problems.append(f"model card does not list: models/{model.name}")

    benchmark_index = _read_text(root / "benchmarks" / "README.md")
    if benchmark_index:
        evidence = sorted((root / "benchmarks").glob("*.json"))
        evidence.extend(sorted((root / "reports").glob("*.json")))
        for report in evidence:
            relative = report.relative_to(root).as_posix()
            if report.name not in benchmark_index and relative not in benchmark_index:
                problems.append(f"benchmark index does not list: {relative}")

    changelog = _read_text(root / "CHANGELOG.md")
    if changelog and RELEASE_VERSION not in changelog:
        problems.append(f"CHANGELOG.md must contain version {RELEASE_VERSION}")
    citation = _read_text(root / "CITATION.cff")
    if citation and RELEASE_VERSION not in citation:
        problems.append(f"CITATION.cff must contain version {RELEASE_VERSION}")

    roadmap = _read_text(root / "docs" / "ROADMAP.md")
    if roadmap and not re.search(
        r"## 8\. Open-source release\s+\*\*Status: complete\.\*\*", roadmap
    ):
        problems.append("roadmap milestone 8 must be marked complete")

    return sorted(set(problems))
