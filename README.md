# rextio-device-cuda

`rextio-device-cuda` is the incubating first-party NVIDIA CUDA device provider
for [Rextio](https://github.com/rextio/rextio). Version 0.1.0 implements a
bounded **Device Provider API 1 Alpha**. The source repository is public, the
package is not on PyPI, and it does not claim certified CUDA execution support.

This is a device/runtime integration layer, not a Python lowering plugin. It
does not claim AST nodes or understand NumPy, pandas, PyTorch, or TensorFlow
semantics.

## Current bounded surface

- NVIDIA CUDA only.
- Linux x86_64, Linux AArch64, and Windows x86_64 target declarations.
- A separate Linux x86_64 host-extension-only
  `cuda-libtorch-linux-x86_64` build-only capability that accepts one `gpu:0`
  / `backend="cuda"` / `runtime="libtorch"` requirement and reuses the domain
  runtime. Its frozen contract requires libtorch/PyTorch 2.11.0, tch 0.24.0,
  inference + no-grad, `strided` layout, device memory, and an explicitly
  selected provider SM. An optional single SM in the domain requirement must
  match the provider selection.
- Explicit target capability, device ordinal, and `sm_NN` selection.
- A fixed minimum CUDA driver floor of `12000` (CUDA 12.0): explicit provider
  configuration may raise this requirement but cannot lower it beneath the
  manifest contract.
- A fixed raw-driver CUDA toolkit/runtime floor of 12.0: explicit provider
  configuration may raise it but cannot weaken the manifest, and
  toolkit/runtime reports must name one identical version at or above that
  floor. The libtorch-reuse lane does not inspect a standalone toolkit.
- CUDA Driver API inventory through an explicitly configured probe executable.
- No `PATH`, registry, common-directory, or ambient-environment discovery.
- Optional validation of an explicitly configured CUDA toolkit root for the
  raw-driver lanes.
- Deterministic provider manifest, preflight observations, lock inputs,
  resource contracts, and packaged Rust runtime source. A shared driver loader
  owned by the manual host resolves symbols from one reviewed CUDA Driver
  image and injects them into the runtime; generated artifacts do not add a
  direct `cuda` / `nvcuda` link directive. The host must retain that image
  handle until every injected `DriverApi` and child resource is dropped.
- Provider-owned raw context, dedicated stream, and device-allocation RAII
  primitives. Children keep their context alive and are deliberately
  thread-affine (`!Send` / `!Sync`) in this Alpha.
- Framework tensor, allocator, current-stream, and event resources are
  framework-owned. The libtorch-reuse capability contributes borrow/validate
  contracts only for the tensor, allocator, and current stream; it creates,
  replaces, allocates, and synchronizes none of them.

All preflight and report records serialize `support_claim: false`.

## Unsupported and deferred

- macOS, every 32-bit target, Windows ARM, and non-NVIDIA accelerators.
- CUDA kernel generation or performance claims.
- PyTorch/tch-rs and TensorFlow/TFE CUDA lowering.
- Importing `torch`, inspecting a loaded libtorch image, proving one-image ABI
  identity, or certifying any real GPU execution.
- Adopting, replacing, allocating, or synchronizing framework-owned resources.
- Automatic provider selection or implicit hardware discovery.
- General Core materialization of the packaged Rust helper.
- Any claim based only on GitHub-hosted CI, a driver inventory, or a mock report.

The raw-driver capabilities contribute a packaged-runtime reference, helper id,
and inventory check id. Current Core 0.1.6 integration deliberately rejects
those unmaterialized inputs before generated writes. The separate libtorch
capability contributes no raw runtime package, generated helper, runtime check,
raw context, raw stream, or raw allocation. Core can compose its borrow-only
record, but that does not make the CPU-only `rextio-torch` plugin CUDA-capable:
E2 still needs typed Torch CUDA lowering, one-libtorch-image/ABI proof, and
real-GPU certification.

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

Production Core resolution must supply the probe, device, SM, and raw-lane
toolkit inputs through the request's redacted `DeviceProviderOptions` so their
keys and digest enter provenance. Constructor `CudaProviderConfig` values and
injected probe/toolkit implementations are programmatic/manual-test boundaries,
not production Core provenance. Ready observations record the effective
minimum driver floor and, for raw lanes, the minimum toolkit floor; they contain
no paths.

The entry-point identity is exactly `rextio-device-cuda` in the
`rextio.device_providers` group.

For the build-only libtorch reuse lane, select
`cuda-libtorch-linux-x86_64` instead. The Torch domain plugin must emit the
following frozen facts; users must not forge them to bypass domain analysis:

```text
DeviceRequirement(
  logical_device="gpu:0",
  backend="cuda",
  runtime="libtorch",
  features=("inference", "no-grad"),
  layouts=("strided",),
  memory_spaces=("device",),
  architectures=(),              # optional one-SM cross-check
  reuse_domain_runtime=True,
)
RuntimeRequirement("libtorch", "2.11.0", ("cuda", "pytorch-wheel"))
RuntimeRequirement("tch", "0.24.0", ("cuda",))
```

`device_options.sm` remains mandatory and is checked against the driver probe.
The libtorch lane rejects `toolkit_root` whether supplied as a target option or
constructor configuration because standalone toolkit ownership belongs only to
the raw-driver lanes.
Preflight records the pins as required-but-unverified request facts. It does
not import PyTorch or assert that Python and generated Rust share one libtorch
image.

## Probe trust boundary

The Rust inventory probe is ported from the reviewed Rextio Core implementation
at `ea568220467f469371452092493d00a2aa42d701`.

`support_claim` remains false. Selecting or executing a configured probe does
not bind the probe binary's content identity, and therefore is not certification
evidence by itself.

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
cargo +1.93.1 fmt --all -- --check
cargo +1.93.1 test --workspace
cargo +1.93.1 clippy --workspace --all-targets -- -D warnings
```

The packaged runtime source is independently compiled and tested, but Core does
not yet inject it into generated artifacts. A future Core materializer must own
the shared loader, resolve the complete symbol table from one image, and prove
that the image handle outlives the runtime API, contexts, streams, and
allocations.

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
