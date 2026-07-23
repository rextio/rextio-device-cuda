# Build-only evidence statement

The 0.1.0 candidate is limited to:

- Core Device Provider API 1 contract tests using injected, fixed reports;
- exact-schema/path-free probe parser tests;
- safe-loader compilation on hosted Linux, Windows, and macOS;
- Linux AArch64 cross-check;
- Rust RAII unit tests with fake CUDA symbols;
- package inspection proving the audited Rust runtime sources are present;
- contract tests for a separate Linux x86_64 host-extension-only
  libtorch/PyTorch 2.11.0 + tch 0.24.0 runtime-reuse request, its borrow-only
  framework resources, and fail-closed raw-only toolkit configuration.

Mock reports and GitHub-hosted runners are not GPU evidence. The probe performs
inventory only. Every manifest capability remains `build-only`, and every
preflight/report has `support_claim=false`.

Production Core provenance must receive configuration through request
`DeviceProviderOptions`; constructor config and injected runners/inspectors are
programmatic/manual-test boundaries. Ready observations bind only redacted
effective policy floors (minimum driver and, for raw lanes, minimum toolkit)
into Core's preflight record. They do not bind a configured probe executable's
content identity. Until source/content identity and execution evidence are
independently certified, a successful inventory probe is not certification
evidence and `support_claim` remains false.

The manual-only runtime smoke is compiled but not executed in ordinary CI. Its
real-driver JSON also fixes `support_claim=false`, `kernel_executed=false`, and
`certification_ready=false`.

The libtorch-reuse contract does not import torch, inspect the loaded
libtorch/ATen image, validate an ABI, or execute a GPU operation. Its preflight
pin observations mean "required and not yet verified", and its contribution
contains no raw CUDA runtime package, helper id, runtime-check id, provider
context, provider stream, or provider allocation.

Manual Linux and Windows evidence scripts fail closed unless rustup selects
exactly rustc/cargo 1.93.1, and use `cargo +1.93.1` for both probe and
runtime-smoke builds.
