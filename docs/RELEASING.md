# Release runbook

AlphaZero Gomoku Lab uses an explicit Git tag to publish Python distributions,
checksums, a multi-platform GHCR image, and generated GitHub release notes. The
workflow never publishes from a pull request or manual dispatch.

## 1. Prepare the release candidate

1. Merge the release pull request into `main`.
2. Pull the exact remote `main` and confirm the worktree is clean.
3. Confirm the version in `pyproject.toml`, `alphazero_gomoku/__init__.py`,
   `CHANGELOG.md`, and `CITATION.cff` is `1.0.0`.
4. Confirm every bundled model and committed JSON report appears in its index.
5. Review the English and Chinese landing pages together.

Run the local acceptance suite:

```bash
python -m pip install -e ".[dev,train,web]" build
python -m alphazero_gomoku release-check --tag v1.0.0
python -m pytest
ruff check alphazero_gomoku/cli.py alphazero_gomoku/release.py \
  alphazero_gomoku/policy_value_net_pytorch.py alphazero_gomoku/training \
  alphazero_gomoku/evaluation alphazero_gomoku/gumbel alphazero_gomoku/web tests
python -m build
```

Install the wheel in a new virtual environment and smoke-test its CLI. On
PowerShell:

```powershell
python -m venv .release-venv
.\.release-venv\Scripts\python -m pip install --upgrade pip
.\.release-venv\Scripts\python -m pip install (Get-ChildItem dist\*.whl)
.\.release-venv\Scripts\python -m alphazero_gomoku doctor
.\.release-venv\Scripts\python -m alphazero_gomoku release-check --root .
```

Remove the temporary environment after verification. Do not commit `dist/`,
`build/`, or the environment.

## 2. Verify GitHub before tagging

- The final `main` commit has green CI, PyTorch, and Container checks.
- The release workflow succeeds when manually dispatched. Manual dispatch runs
  validation only: it cannot push to GHCR or create a GitHub Release.
- The changelog contains no unreconciled release notes or unsupported claims.
- The Security tab permits private vulnerability reports.

## 3. Publish

Create and push an annotated tag from the verified `main` commit:

```bash
git switch main
git pull --ff-only origin main
git tag -a v1.0.0 -m "AlphaZero Gomoku Lab 1.0.0"
git push origin v1.0.0
```

The tag triggers `.github/workflows/release.yml`. Do not create the tag from a
feature branch and do not rerun publication against a different commit.

## 4. Verify published artifacts

After the workflow succeeds:

1. Download the wheel, source archive, and `SHA256SUMS` from the GitHub Release;
   verify both archive hashes.
2. Install the downloaded wheel in a clean environment and run `doctor`.
3. Pull `ghcr.io/louissma/alphazero-gomoku-lab:1.0.0` for an available platform,
   start it, and verify `http://127.0.0.1:8000/api/health`.
4. Confirm the GHCR package also exposes the `latest` tag and links back to the
   repository.
5. Confirm GitHub renders the citation metadata and generated release notes.

If validation fails before publication, fix the problem through a pull request
and tag only the corrected `main`. If a published artifact is compromised, use
the security process and publish a new patch version; never silently move an
existing release tag.
