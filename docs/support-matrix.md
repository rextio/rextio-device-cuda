# Support matrix

Status: **unreleased build-only Alpha**.

| Target | Declaration | Hosted CI | Real NVIDIA evidence | Certification |
|---|---|---:|---:|---|
| Linux x86_64 GNU | build-only | Python/Rust/mock | manual script available | not certified |
| Linux AArch64 GNU | build-only | cross-check only | manual script available | not certified |
| Windows x86_64 MSVC | build-only | Python/Rust/mock | manual script available | not certified |
| macOS arm64/x86_64 | unsupported | fail-closed contract | impossible (CUDA) | unsupported |
| 32-bit / Windows ARM | unsupported | none | none | unsupported |

The bounded architecture vocabulary is `sm_60`, `sm_61`, `sm_70`, `sm_72`,
`sm_75`, `sm_80`, `sm_86`, `sm_87`, `sm_89`, and `sm_90`. A manifest entry is
declared compatibility metadata, not proof that an installed driver/toolkit can
execute it.

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

The current manual scripts cover steps 1–3: they run inventory and a separate
manual-only RAII smoke that resolves the runtime's nine symbols from the same
reviewed driver image, creates/detaches the context, creates/synchronizes and
destroys a dedicated stream, allocates/frees 4096 bytes, and closes the
context. They intentionally do not claim steps 4–7.
