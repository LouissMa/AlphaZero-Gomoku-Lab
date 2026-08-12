# Security policy

## Supported versions

Security fixes are applied to the latest release and the current `main` branch.
Historical research snapshots and archived framework backends are retained for
reference but are not maintained as deployment targets.

## Report a vulnerability privately

Do not open a public issue for a suspected vulnerability. Use GitHub's private
[security advisory form](https://github.com/LouissMa/AlphaZero-Gomoku-Lab/security/advisories/new)
and include:

- affected commit, release, command, API route, or container image;
- operating system, Python version, installation method, and relevant options;
- minimal reproduction steps and the observed security impact;
- whether the issue is already public or under active exploitation;
- any proposed mitigation, if known.

You should receive an acknowledgement within seven days. The maintainer will
validate the report, agree on a disclosure timeline, prepare a fix and tests,
and credit the reporter unless anonymity is requested. Please allow reasonable
time for a coordinated release before publishing details.

## Deployment notes

The web application is designed as a local research demo. The container runs as
an unprivileged user, but it does not provide authentication, authorization,
rate limiting, TLS termination, or multi-tenant isolation. Do not expose it
directly to an untrusted network without an appropriately configured reverse
proxy and operational controls. Treat checkpoints and pickle-compatible legacy
model files as trusted inputs; do not load artifacts from unknown sources.
