# Changelog

## 0.1.0 — 2026-07-26

**Published non-certifying Alpha.** Package version `0.1.0` is tagged and
published to PyPI as the first-party NVIDIA CUDA Device Provider API 1 package
for Rextio. Publication does not promote any capability beyond build-only:
every public preflight/report retains `support_claim=false`, and this release
makes no formal CUDA execution, framework support, certification, or
performance claim.

- Add Device Provider API 1 manifest and exact entry-point identity.
- Declare build-only NVIDIA CUDA capabilities for Linux x86_64/AArch64 and
  Windows x86_64.
- Add a distinct build-only Linux x86_64 host-extension-only libtorch
  runtime-reuse capability for the frozen PyTorch/libtorch 2.11.0 + tch 0.24.0
  contract. It validates the typed `gpu:0` CUDA requirement and explicit
  provider SM, then contributes only framework
  tensor/allocator/current-stream borrow contracts.
- Keep the libtorch capability free of raw-runtime package references, helper
  ids, runtime-check ids, provider-owned contexts/allocations/streams, torch
  imports, one-image ABI claims, and real-GPU support claims.
- Add a distinct build-only Linux x86_64 host-extension-only
  `cuda-tensorflow-tfe-linux-x86_64` prerequisite for the frozen
  TensorFlow 2.21.0 + CPython 3.11 private eager ABI contract. It validates the
  exact typed `gpu:0` dense device requirement and explicit provider SM, then
  contributes only framework borrow/validate contracts.
- Keep the TensorFlow TFE capability free of raw CUDA resources, current-stream
  or event claims, native libraries, package references, helpers, runtime
  checks, TensorFlow imports, TFE/private-ABI image claims, real-GPU execution,
  support claims, and certification claims.
- Reject raw-driver-only `toolkit_root` inputs for libtorch reuse through both
  request options and constructor configuration.
- Invalidate a prior ready fingerprint before every repeated preflight and use
  a per-fingerprint generation token so only the newest concurrent successful
  preflight may restore contribution authority.
- Record redacted effective driver/toolkit policy floors in ready observations;
  keep constructor/injected inspection boundaries outside production Core
  provenance and make no probe-binary content-identity claim.
- Add explicit, fail-closed target/device/SM and driver preflight.
- Keep the public `CudaProviderConfig.minimum_driver_version` at or above the
  manifest's CUDA driver floor (`12000`), and apply that exact floor during
  preflight so configuration cannot weaken declared compatibility.
- Keep the public toolkit/runtime requirement at or above CUDA 12.0, reject
  mismatched toolkit/runtime report versions, and reject unsupported SM
  architectures even when `preflight()` is called outside Core resolution.
- Clarify that the host-owned shared loader must outlive injected runtime
  symbols and resources, remove the superseded probe-local loaders, pin manual
  validation to Rust 1.93.1, and reap a probe if stdout-reader startup fails.
- Port the reviewed path-safe CUDA Driver API inventory probe.
- Add optional explicitly rooted toolkit validation.
- Add a packaged no-dependency Rust crate for provider-owned raw context,
  dedicated-stream, and device-memory lifetime primitives.
- Keep framework tensors, allocators, current streams, and events outside the
  ownership surface.
- Add mocked/build-only CI and manual Linux/Windows NVIDIA inventory plus raw
  RAII lifetime-smoke scripts.
- Harden the Alpha review boundary with a provider-option allow-list, a
  64-KiB/timeout-bounded probe reader, dynamic-driver-only linking, explicit
  driver-image/API drop ordering, and current-context restoration tests.

The source repository and PyPI package are public. All capabilities remain
build-only and non-certifying; release publication is not a CUDA support claim.
