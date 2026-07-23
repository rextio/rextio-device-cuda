# Changelog

## 0.1.0 — Unreleased Alpha

- Add Device Provider API 1 manifest and exact entry-point identity.
- Declare build-only NVIDIA CUDA capabilities for Linux x86_64/AArch64 and
  Windows x86_64.
- Add explicit, fail-closed target/device/SM and driver preflight.
- Port the reviewed path-safe CUDA Driver API inventory probe.
- Add optional explicitly rooted toolkit validation.
- Add a packaged no-dependency Rust crate for provider-owned raw context,
  dedicated-stream, and device-memory lifetime primitives.
- Keep framework tensors, allocators, current streams, and events outside the
  ownership surface.
- Add mocked/build-only CI and manual Linux/Windows NVIDIA inventory plus raw
  RAII lifetime-smoke scripts.

No tag, public repository, PyPI upload, or CUDA support claim exists yet.
