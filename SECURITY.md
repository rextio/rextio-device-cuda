# Security policy

## Reporting

Do not open a public issue for a suspected vulnerability. Contact
`rextio.co@gmail.com` with the affected commit, platform, and a minimal
reproducer that does not contain credentials or private machine paths.

## Alpha security boundary

- Production preflight requires an explicit absolute probe executable. It does
  not use `PATH`.
- Toolkit inspection occurs only under an explicit absolute root.
- Probe output is bounded, exact-schema, path-free, and never a support claim.
- Provider option values are redacted from public records and bound by digest.
- The Linux probe's absolute/canonical loader checks are provenance guards, not
  a sandbox. `LD_PRELOAD`, transitive dependencies, same-UID compromise, and a
  hostile kernel remain outside the boundary.
- The Windows probe selects only the System32 CUDA driver DLL.
- Rust resources are provider-owned, context-bound, and thread-affine.
- Framework resources are not adopted or destroyed by this package.

Do not run an untrusted probe executable or point toolkit validation at an
untrusted/mutable filesystem tree.

