# Changelog

## 0.1.0 — Unreleased Alpha

- Add Device Provider API 1 manifest and exact entry-point identity.
- Declare build-only NVIDIA CUDA capabilities for Linux x86_64/AArch64 and
  Windows x86_64.
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

No tag, public repository, PyPI upload, or CUDA support claim exists yet.
