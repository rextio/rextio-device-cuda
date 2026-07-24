# Support matrix

Status: **unreleased build-only Alpha**.

| Target | Declaration | Hosted CI | Real NVIDIA evidence | Certification |
|---|---|---:|---:|---|
| Linux x86_64 GNU | build-only | Python/Rust/mock | manual script available | not certified |
| Linux x86_64 GNU host extension, libtorch 2.11/tch 0.24 reuse | build-only | borrow-only contract | none | not certified |
| Linux x86_64 GNU host extension, TensorFlow TFE 2.21.0/CPython 3.11 reuse | build-only | borrow-only contract | none | not certified |
| Linux AArch64 GNU | build-only | cross-check only | manual script available | not certified |
| Windows x86_64 MSVC | build-only | Python/Rust/mock | manual script available | not certified |
| macOS arm64/x86_64 | unsupported | fail-closed contract | impossible (CUDA) | unsupported |
| 32-bit / Windows ARM | unsupported | none | none | unsupported |

The bounded architecture vocabulary is `sm_60`, `sm_61`, `sm_70`, `sm_72`,
`sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, and `sm_90`. A manifest entry is
declared compatibility metadata, not proof that an installed driver/toolkit can
execute it.

The libtorch row is a framework-runtime-reuse prerequisite, not PyTorch CUDA
support. It checks the frozen domain requirement and CUDA
driver/device/provider-SM inventory, then returns only framework-owned
tensor/allocator/current-stream borrow contracts. It does not load torch,
inspect libtorch/ATen images, validate the one-image ABI, execute an operation,
or provide real-device evidence. Rust-crate and executable artifacts are
outside this lane. A standalone `toolkit_root` is a raw-driver input and is
rejected for framework reuse.

The TensorFlow TFE row is likewise a provider prerequisite, not TensorFlow CUDA
support. It validates the exact `tensorflow-tfe` domain, TensorFlow 2.21.0 and
CPython 3.11 private eager ABI pins, driver/device/provider-SM inventory, then
returns only a framework borrow contract. It does not import TensorFlow,
resolve or validate TFE/private bridge symbols, inspect runtime images, execute
an operation, claim a current stream, or provide real-device evidence.

## Remaining certification gate

Before any target can be promoted beyond build-only:

1. run on physical NVIDIA hardware with the exact target, driver, device
   ordinal, SM, provider commit, Core commit, and Rust toolchain recorded;
2. verify the path-safe probe and negative driver/device/SM cases;
3. execute the packaged RAII runtime against the same driver, including
   context → allocation/stream → child release → context release ordering;
4. execute a deterministic native CUDA operation and verify its result;
5. repeat on the supported driver floor and current driver;
6. verify all public evidence remains path/secret-free and records
   `support_claim=false` until an owner-approved certification update;
7. add a secure Core helper-materialization contract and a domain adapter.

For the libtorch-reuse capability, promotion additionally requires the Torch E2
domain work: device-preserving lowering, one-libtorch-image/ABI proof, and a
real Linux x86_64 GPU vertical slice. Raw-driver E1 certification alone cannot
promote the framework-reuse row.

For the TensorFlow TFE capability, promotion additionally requires E3 typed
device-preserving lowering, exact TensorFlow-wheel/private-ABI image identity,
GPU:0 backing-device and same-context checks, and a real Linux x86_64 GPU
vertical slice. The evidence must retain `support_claim=false` and
`certification_ready=false` until a separate owner-approved promotion.

The current manual scripts cover steps 1–3: they run inventory and a separate
manual-only RAII smoke that resolves the runtime's nine symbols from the same
reviewed driver image, creates/detaches the context, creates/synchronizes and
destroys a dedicated stream, allocates/frees 4096 bytes, and closes the
context. They intentionally do not claim steps 4–7.

Record the immutable source and toolchain identity alongside each manual
evidence bundle:

```bash
git rev-parse HEAD
git -C ../rextio rev-parse HEAD
rustc +1.93.1 -Vv
cargo +1.93.1 -V
```

Also record the exact Rust target triple, NVIDIA driver version, selected
device ordinal, and selected `sm_NN`. Do not promote evidence that omits any of
these fields or that was produced from a dirty worktree.

The Linux and Windows validation scripts require and invoke the exact Rust
1.93.1 toolchain for both `rustc` and `cargo`; an unavailable or mismatched
toolchain fails before probe or runtime-smoke execution.
