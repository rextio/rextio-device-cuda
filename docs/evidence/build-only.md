# Build-only evidence statement

The 0.1.0 candidate is limited to:

- Core Device Provider API 1 contract tests using injected, fixed reports;
- exact-schema/path-free probe parser tests;
- safe-loader compilation on hosted Linux, Windows, and macOS;
- Linux AArch64 cross-check;
- Rust RAII unit tests with fake CUDA symbols;
- package inspection proving the audited Rust runtime sources are present.

Mock reports and GitHub-hosted runners are not GPU evidence. The probe performs
inventory only. Every manifest capability remains `build-only`, and every
preflight/report has `support_claim=false`.

The manual-only runtime smoke is compiled but not executed in ordinary CI. Its
real-driver JSON also fixes `support_claim=false`, `kernel_executed=false`, and
`certification_ready=false`.

Manual Linux and Windows evidence scripts fail closed unless rustup selects
exactly rustc/cargo 1.93.1, and use `cargo +1.93.1` for both probe and
runtime-smoke builds.
