# Open-source Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the completed AlphaZero Gomoku Lab milestones into a tested, bilingual, versioned `1.0.0` release candidate with container and GitHub Release automation.

**Architecture:** A standard-library release auditor is the executable contract connecting package metadata, documentation, model/benchmark evidence, and CI. Human-facing release files describe the project honestly, pull requests validate the Docker image, and tag-only automation builds Python artifacts plus multi-platform GHCR images before creating a GitHub Release.

**Tech Stack:** Python 3.10+, argparse, pathlib, tomllib/tomli, pytest, Ruff, setuptools/build, GitHub Actions, Docker Buildx, GHCR.

## Global Constraints

- The release version is exactly `1.0.0` and the publication tag is `v1.0.0`.
- Python 3.10 through 3.13 remain supported.
- No new runtime dependency is introduced for release auditing.
- Bundled legacy model provenance and playing strength must be described as unknown when no repository evidence exists.
- Pull requests build but never publish images or releases.
- Manual release-workflow dispatch validates artifacts but never publishes them.
- PyPI publication is outside this phase.
- The actual `v1.0.0` tag is created only after the final pull request merges to `main`.

---

## File structure

- `alphazero_gomoku/release.py`: repository audit and version/tag validation.
- `alphazero_gomoku/cli.py`: `gomoku release-check` command and canonical version import.
- `tests/test_release.py`: behavioral unit and repository integration tests.
- `README.md`, `README.zh-CN.md`: synchronized English and Chinese entry points.
- `models/README.md`, `benchmarks/README.md`: evidence-backed model and benchmark indexes.
- `CHANGELOG.md`, `CITATION.cff`, `SECURITY.md`, `SUPPORT.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`: release and community lifecycle.
- `.github/ISSUE_TEMPLATE/*`, `.github/pull_request_template.md`, `.github/release.yml`: contributor UX and generated release notes.
- `.github/workflows/container.yml`: non-publishing container validation.
- `.github/workflows/release.yml`: validated tag-driven artifact and GHCR publication.
- `docs/RELEASING.md`, `docs/ROADMAP.md`, `progress.md`: operator runbook and milestone status.

### Task 1: Executable release-readiness contract

**Files:**
- Create: `tests/test_release.py`
- Create: `alphazero_gomoku/release.py`
- Modify: `alphazero_gomoku/cli.py`

**Interfaces:**
- Produces: `audit_repository(root: Path) -> list[str]`.
- Produces: `validate_tag(version: str, tag: str) -> list[str]`.
- Produces: CLI command `gomoku release-check [--root PATH] [--tag TAG]`.
- Consumes: repository metadata and documentation only; it performs no writes.

- [x] **Step 1: Write failing audit tests**

Create tests that use a temporary repository fixture and independently assert:

```python
def test_audit_reports_missing_required_files(tmp_path: Path) -> None:
    problems = audit_repository(tmp_path)
    assert "missing required file: README.md" in problems


def test_audit_reports_version_mismatch(release_tree: Path) -> None:
    (release_tree / "pyproject.toml").write_text(
        '[project]\nversion = "0.9.0"\n', encoding="utf-8"
    )
    assert "pyproject version must be 1.0.0, found 0.9.0" in audit_repository(release_tree)


def test_audit_reports_unlisted_model(release_tree: Path) -> None:
    (release_tree / "models" / "extra.model").write_bytes(b"model")
    assert "model card does not list: models/extra.model" in audit_repository(release_tree)


def test_validate_tag_matches_version() -> None:
    assert validate_tag("1.0.0", "v1.0.0") == []
    assert validate_tag("1.0.0", "v1.0.1") == [
        "tag v1.0.1 does not match package version 1.0.0"
    ]
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_release.py -q`

Expected: collection fails because `alphazero_gomoku.release` does not exist.

- [x] **Step 3: Implement the minimum auditor**

Use `Path`, `re`, and `tomllib` with the Python 3.10 `tomli` fallback. Define
the exact required-file tuple from the design, read project version from
`pyproject.toml`, scan `models/*.model*`, scan committed JSON evidence under
`benchmarks/` and `reports/`, and return all violations in sorted order. Do not
invoke Git, access the network, or mutate the repository.

- [x] **Step 4: Verify GREEN and add CLI behavior tests**

Add tests calling `main(["release-check", "--root", str(path)])`. A valid fixture
must return `0` and print `Release readiness check passed for 1.0.0.`; an invalid
fixture must return `1` and print every violation to standard error.

Run: `python -m pytest tests/test_release.py -q`

Expected: all focused tests pass.

- [x] **Step 5: Refactor version ownership**

Import `alphazero_gomoku.__version__` in `cli.py` rather than maintaining an
independent literal. Preserve the public `VERSION` name for compatibility.

- [x] **Step 6: Run focused verification**

Run: `python -m pytest tests/test_release.py -q`

Expected: PASS with no warnings.

### Task 2: Versioned evidence and package metadata

**Files:**
- Modify: `pyproject.toml`
- Modify: `alphazero_gomoku/__init__.py`
- Create: `models/README.md`
- Modify: `benchmarks/README.md`
- Create: `CHANGELOG.md`
- Create: `CITATION.cff`

**Interfaces:**
- Consumes: `audit_repository` checks from Task 1.
- Produces: consistent `1.0.0` metadata and complete indexes for tracked evidence.

- [x] **Step 1: Record model integrity metadata**

Run:

```powershell
Get-FileHash -Algorithm SHA256 models\*.model* | Select-Object Path,Hash
```

Document the exact filenames and hashes. Identify `best_policy_6_6_4.*` as
6x6/4-in-a-row and `best_policy_8_8_5.*` as 8x8/5-in-a-row. State that these
legacy NumPy weights are intended for demonstrations and regression checks,
not as modern benchmark claims.

- [x] **Step 2: Update versions and lifecycle metadata**

Set both `pyproject.toml` and `alphazero_gomoku/__init__.py` to `1.0.0`. Add a
Keep a Changelog `1.0.0` section dated `2026-08-12`, and a valid `CITATION.cff`
with `cff-version: 1.2.0`, project title, version, release date, repository URL,
MIT license, and the repository owner as author.

- [x] **Step 3: Expand the benchmark evidence index**

List and explain:

- `benchmarks/smoke_cpu.json` — scalable self-play path smoke evidence;
- `benchmarks/gumbel_vs_puct_smoke.json` — equal-budget search-path evidence;
- `reports/arena_smoke.json` — arena statistics-path evidence.

Explicitly state that smoke evidence does not prove broad playing strength or
permit throughput comparisons across different hardware/workloads.

- [ ] **Step 4: Run the repository audit to expose remaining gaps**

Run: `python -m alphazero_gomoku release-check`

Expected: non-zero, with only not-yet-created documentation/workflow and roadmap
violations; model, benchmark, changelog, citation, and version violations are absent.

### Task 3: Bilingual project and community experience

**Files:**
- Modify: `README.md`
- Create: `README.zh-CN.md`
- Modify: `CONTRIBUTING.md`
- Create: `SECURITY.md`
- Create: `SUPPORT.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/pull_request_template.md`
- Create: `.github/release.yml`

**Interfaces:**
- Consumes: actual CLI commands, docs, reports, and optional dependencies.
- Produces: equivalent English/Chinese navigation and structured contribution paths.

- [x] **Step 1: Rewrite the English landing page as a release page**

Keep the existing capability detail but add a Chinese-language link, `1.0.0`
release badge, web/Docker quick start, architecture/research workflow, evidence
table, model-card link, limitations, release status, community links, and
citation. Every command must be copied from a working CLI parser path.

- [x] **Step 2: Create the Simplified Chinese landing page**

Mirror the same facts and navigation, link back to English at the top, and use
natural Chinese explanations rather than a sentence-by-sentence mechanical
translation. Preserve commands unchanged.

- [x] **Step 3: Add community policies and forms**

Use Contributor Covenant 2.1 text for `CODE_OF_CONDUCT.md`, private security
reporting through GitHub Security Advisories in `SECURITY.md`, GitHub Discussions
or Issues guidance in `SUPPORT.md`, required reproduction/environment fields in
the bug form, measurable acceptance criteria in the feature form, and test/docs/
benchmark checkboxes in the PR template.

- [x] **Step 4: Synchronize contribution commands**

Install `.[dev,train,web]`, run the full pytest suite, maintained-tree Ruff
command, `gomoku release-check`, and document benchmark/model metadata duties.

- [x] **Step 5: Run documentation cross-reference checks**

Run: `python -m pytest tests/test_release.py -q`

Expected: bilingual-link and community-file checks pass; workflow/roadmap gaps
may remain until Tasks 4 and 5.

### Task 4: Container and GitHub Release automation

**Files:**
- Create: `.github/workflows/container.yml`
- Create: `.github/workflows/release.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.dockerignore`
- Create: `docs/RELEASING.md`

**Interfaces:**
- Consumes: `gomoku release-check`, `pyproject.toml`, Dockerfile, and Git tag.
- Produces: PR image build validation and tag-only Python/GHCR/GitHub artifacts.

- [x] **Step 1: Add pull-request container validation**

Use `docker/setup-buildx-action` and `docker/build-push-action` with `push: false`,
GitHub Actions cache, and paths covering the Dockerfile, package, models, web
assets, and package metadata. Grant only `contents: read`.

- [x] **Step 2: Add the artifact validation job**

The release workflow triggers on `v*.*.*` tags and `workflow_dispatch`. Its
validation job checks version/tag equality for tag events, runs release audit,
pytest and Ruff, builds with `python -m build`, installs the wheel into a fresh
virtual environment, runs `doctor`, generates `SHA256SUMS`, and uploads `dist/`.

- [x] **Step 3: Add tag-only publication jobs**

For tag events only, authenticate to GHCR with `GITHUB_TOKEN`, generate OCI
metadata, push `linux/amd64,linux/arm64`, attach version and `latest` tags, then
create the GitHub Release from validated artifacts with generated notes. Use
job-level `packages: write` or `contents: write`; do not grant these permissions
to validation jobs.

- [x] **Step 4: Add release audit to normal CI**

Run `python -m alphazero_gomoku release-check` after tests so repository drift
fails pull requests before release day.

- [x] **Step 5: Document the operator sequence**

`docs/RELEASING.md` requires a clean `main`, green CI, local audit/build/wheel
smoke test, annotated `v1.0.0` tag, push, Actions monitoring, and post-release
GHCR/GitHub artifact verification. It must state that this phase prepares but
does not create the tag.

### Task 5: Milestone closure and complete verification

**Files:**
- Modify: `docs/ROADMAP.md`
- Modify: `progress.md`
- Modify: `docs/superpowers/plans/2026-08-12-open-source-release.md`

**Interfaces:**
- Consumes: all prior task outputs.
- Produces: a clean, verified release-candidate commit and PR to `main`.

- [x] **Step 1: Mark milestone 8 complete**

Add `**Status: complete.**` to roadmap phase 8, check the final README roadmap
item, and record release-audit, bilingual docs, model cards, CI/container, and
tag automation in `progress.md`.

- [x] **Step 2: Run the complete repository audit**

Run: `python -m alphazero_gomoku release-check`

Expected: `Release readiness check passed for 1.0.0.` and exit code `0`.

- [x] **Step 3: Run all Python checks**

Run:

```powershell
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/release.py alphazero_gomoku/policy_value_net_pytorch.py alphazero_gomoku/training alphazero_gomoku/evaluation alphazero_gomoku/gumbel alphazero_gomoku/web tests
python -m alphazero_gomoku doctor
```

Expected: all tests pass, Ruff exits zero, and doctor reports version `1.0.0`.

- [x] **Step 4: Build and smoke-test distribution artifacts**

Run `python -m build`. Create an isolated temporary virtual environment, install
the generated wheel with its dependencies, and run both `python -m
alphazero_gomoku doctor` and `python -m alphazero_gomoku release-check --root
<repository-root>`. Expected: both commands exit zero.

- [x] **Step 5: Validate declarative artifacts**

Parse every `.github/**/*.yml` and `CITATION.cff` with a YAML parser available in
the development environment, run `git diff --check`, and inspect `git status`
to ensure only intended phase files changed. If Docker is installed, run a local
image build; otherwise rely on the required GitHub container check and report
that limitation explicitly.

- [x] **Step 6: Commit, push, and open the final draft pull request**

Commit implementation as `Prepare 1.0.0 open-source release`, push
`codex/open-source-release`, and create a draft PR targeting `main`. The PR body
must summarize documentation, auditable release contract, container/release
automation, verification commands, the inclusion of phase 7, and the post-merge
`v1.0.0` action.
