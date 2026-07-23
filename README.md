# rextio-device-cuda

`rextio-device-cuda` is the incubating first-party NVIDIA CUDA device provider
for [Rextio](https://github.com/rextio/rextio). Version 0.1.0 implements a
bounded **Device Provider API 1 Alpha**. It is private, is not on PyPI, and
does not claim certified CUDA execution support.

This is a device/runtime integration layer, not a Python lowering plugin. It
does not claim AST nodes or understand NumPy, pandas, PyTorch, or TensorFlow
semantics.

## Current bounded surface

- NVIDIA CUDA only.
- Linux x86_64, Linux AArch64, and Windows x86_64 target declarations.
- Explicit target capability, device ordinal, and `sm_NN` selection.
- CUDA Driver API inventory through an explicitly configured probe executable.
- No `PATH`, registry, common-directory, or ambient-environment discovery.
- Optional validation of an explicitly configured CUDA toolkit root.
- Deterministic provider manifest, preflight observations, lock inputs, native
  link name, resource contracts, and packaged Rust runtime source.
- Provider-owned raw context, dedicated stream, and device-allocation RAII
  primitives. Children keep their context alive and are deliberately
  thread-affine (`!Send` / `!Sync`) in this Alpha.
- Framework tensor, allocator, current-stream, and event resources are
  framework-owned and may only be borrowed/validated by future domain adapters.

All preflight and report records serialize `support_claim: false`.

## Unsupported and deferred

- macOS, every 32-bit target, Windows ARM, and non-NVIDIA accelerators.
- CUDA kernel generation or performance claims.
- PyTorch/tch-rs and TensorFlow/TFE CUDA lowering.
- Adopting, replacing, allocating, or synchronizing framework-owned resources.
- Automatic provider selection or implicit hardware discovery.
- General Core materialization of the packaged Rust helper.
- Any claim based only on GitHub-hosted CI, a driver inventory, or a mock report.

The provider contribution contains a packaged-runtime reference, helper id, and
inventory check id. Current Core 0.1.6 integration deliberately rejects those
unmaterialized inputs before generated writes. Standalone provider resolution,
preflight, lock/report generation, Rust source packaging, and platform builds
can be tested now; E2/E3 domain integration must securely materialize and bind
the helper before an ordinary Rextio source build can use CUDA.

## Explicit selection

The future Core-side shape is explicit:

```toml
[target]
device_provider = "rextio-device-cuda"
device_capability = "cuda-linux-x86_64"

[target.device_options]
probe_executable = "/absolute/path/to/rextio-cuda-driver-probe"
device_ordinal = "0"
sm = "sm_80"
# toolkit_root = "/absolute/path/to/cuda"  # optional validation only
```

Raw option values remain in memory. Public locks and reports include option
names plus a SHA-256 digest, never the paths themselves. The target's domain
plugin must also emit the matching CUDA `DeviceRequirement`; configuration
alone does not turn CPU Python into GPU code.

The entry-point identity is exactly `rextio-device-cuda` in the
`rextio.device_providers` group.

## Probe trust boundary

The Rust inventory probe is ported from the reviewed Rextio Core implementation
at `ea568220467f469371452092493d00a2aa42d701`.

- Windows loads only `nvcuda.dll` from System32 with
  `LOAD_LIBRARY_SEARCH_SYSTEM32`.
- Linux tries an architecture-specific list of absolute `libcuda.so.1` paths,
  canonicalizes the selected image, and rejects paths outside reviewed system
  roots or mutable files/ancestors.
- The probe calls only `cuInit`, driver version, device enumeration, device
  name, and compute-capability entry points.

Absolute top-level selection does not neutralize `LD_PRELOAD`, transitive
`DT_NEEDED` resolution, a compromised same-UID process, or a hostile OS. Run
validation only on a trusted host/container filesystem.

## Development checks

With the unreleased Core 0.1.6 source on `PYTHONPATH`:

```bash
python -m pytest -q
python -m ruff check src tests
python -m mypy src
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

The packaged runtime source is independently compiled and tested, but Core does
not yet inject it into generated artifacts.

## Manual NVIDIA inventory

On a trusted NVIDIA Linux host:

```bash
./scripts/validate-linux-nvidia.sh sm_80 0 /usr/local/cuda
```

On Windows PowerShell:

```powershell
.\scripts\validate-windows-nvidia.ps1 -Sm sm_80 -DeviceOrdinal 0 `
  -ToolkitRoot "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
```

Both scripts build the exact local probe, run standalone provider resolution,
and execute a separate real-driver RAII lifetime smoke. Their JSON output
remains preflight/lifetime evidence only:
`support_claim=false`, `kernel_executed=false`, and
`certification_ready=false`. See [the support matrix](docs/support-matrix.md)
for the remaining real-device gate.

## License

MIT.
