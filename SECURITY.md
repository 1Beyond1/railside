# Security & Data Compliance Policy

## Data Privacy & Non-Distribution Guarantee

`railside` is strictly a **software and algorithmic framework** for side-view line-scan metro train visual inspection:

- **No Raw Train Datasets Distributed**: This repository does not host or distribute any raw railway/metro line-scan datasets, vehicle serial identifiers, depot/operator configurations, or operational inspection records. Showcase screenshots under `docs/assets/` display anonymized underframe views scrubbed of any operator, line, or vehicle identifiers.
- **100% Synthetic Demo & Tests**: All automated tests (`源码/tests/`) and the built-in `railside demo` quickstart are generated programmatically from synthetic geometry and noise, requiring zero real imagery.
- **Local-First Runtime Isolation**: All runtime inputs, intermediate tiles, and outputs live under the gitignored `数据/` directory (or a custom directory via `RAILSIDE_DATA`), never written into version-controlled paths.
- **Read-Only Raw Data Protection**: `railside ingest` verifies integrity and leaves original raw frame directories unmodified.

## Reporting a Vulnerability

If you discover a security vulnerability (for example in the FastAPI local web server, file path validation, or MCP tool boundary):

1. Please **do not** open a public issue with exploit details.
2. Report it privately via GitHub Security Advisories (**Security → Advisories → Report a vulnerability**) or contact the maintainer through GitHub.
3. Include reproduction steps, affected command/route, and expected vs. actual behavior. Reports are reviewed promptly and patched in the next release.
