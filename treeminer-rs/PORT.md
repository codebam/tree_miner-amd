# Rust port of TreeMiner — working rules

The C++/HIP miner lives at `../treeminer` in this same worktree. This workspace is a
rewrite of its **host code** in Rust. Read the C++ file named in your brief before porting;
behaviour must match unless the brief says otherwise, because the two are cross-checked
against each other (`tests/parity`).

## What stays C++

Only the Argon2 **device kernels** (`../treeminer/src/kernelrunner.cu`, the `__global__` /
`__device__` functions). They are compiled by `hipcc` from `crates/tm-gpu/kernel/` and
called through a tiny `extern "C"` launch shim. Everything else — buffers, streams, events,
batch sizing, the mining loop, journal, submitter, dashboard, TUI, CLI — is Rust.

Rewriting a hand-tuned wavefront-shuffle Argon2 kernel would risk exactly the
invalid-digest class of bug this project already fought (commit 12e241c), and it is the one
piece where a silent error costs real money — invalid digests are accepted locally and
rejected by the server after the work is spent.

**Phase 2 is complete: the kernels are Rust.** `crates/tm-kernel/` holds both Argon2 kernels
written in Rust and compiled for `amdgcn-amd-amdhsa`; they are the default
(`tm-gpu/rust-kernel`). The hipcc kernels are still built (`hip-kernel`) because they are the
differential oracle the kernel tests compare against, and the fallback if the tier-3 Rust GPU
toolchain breaks: `--no-default-features --features hip-kernel`.

Measured on an idle RX 7900 XTX, identical batch sizes: 5289 vs 3295 H/s at m=60000 and
7788 vs 5609 H/s at m=42069 — the Rust kernel keeps the per-thread scratch in registers
instead of round-tripping 1 KiB of LDS per Argon2 block, `4*m` times per hash.

The original staging note follows, kept because it records why the order was chosen.

**This was staged, not permanent.** The kernel was rewritten in Rust as a second phase,
once the host port is complete and confirmed working against the C++ miner. Sequencing it
that way means the kernel rewrite lands into a codebase where every other layer is already
trustworthy, so a digest mismatch can only have come from the kernel. Do not start it as
part of the host port; when it happens the fixtures in `fixtures/argon2_vectors.json` and
the GPU self-test are the acceptance gate.

## Rules

- `tm-core` is the contract. Do not redefine its types; import them.
- Match the C++ semantics, not its structure. Idiomatic Rust: `Result`, no panics on
  operator input, `&str` over `String` in arguments, no `unwrap()` outside tests.
- Every behavioural claim in your brief gets a test. Port the existing C++ test cases
  (`../treeminer/tests/unit/**`) where they exist — they encode hard-won bugs.
- No `unsafe` outside `tm-gpu`, and inside it keep unsafe blocks minimal and commented with
  the invariant that makes them sound.
- Comments explain *why*, at the level of the surrounding code. No narration of what the
  next line does.

## Building and testing

The box has no global Rust or ROCm; both come from nix. Always go through the wrapper:

```sh
./rs cargo test -p <your-crate>
./rs cargo clippy -p <your-crate> --all-targets
```

Set `CARGO_TARGET_DIR` to your own directory so parallel agents do not block on the target
lock, e.g. `export CARGO_TARGET_DIR=/tmp/tm-target-<crate>`.

GPU work: the machine has one AMD RX 7900 XTX (gfx1100, ROCm 7.2). `$ROCM_PATH` and
`$HIP_DEVICE_LIB_PATH` are set by `./rs`.

## Reference values

Argon2id, v=19, t=1, p=1, m=difficulty (KiB), 64-byte digest, salt = the 40 hex chars of
the miner address. Known-good vector (`../treeminer/src/hashapi/HashApiSelfTest.cpp`):

```
salt   e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc
key    52a13632690c0d5a7e528c91c8462f9d68d24975d4f80cc64d20504063f3590f
m      8
digest 2PKfnaEX2s+Yf/Drzi92D8HJ+B6K+FppyT7g5glp2knIMlFGWhnyOb9r1QIPf0GaVUEw8KumqQZ/pK2dkNTDxA
```

The C++ binary that produced it is at
`/tmp/claude-1000/-home-codebam-Documents-tree-miner-amd/083f64a4-9069-40f9-adf6-b87ab41c106e/scratchpad/nixbuild/bin/xenblocksMiner`
(`hash-one --salt <hex> --key <hex> --backend cpu|cuda --difficulty <n>`), useful for
generating more vectors.

## Rust GPU kernels (phase 2, staged)

The device kernels are being moved to Rust one at a time. `crates/tm-kernel/` holds them;
`crates/tm-gpu/kernel/argon2_kernel.hip` stays and remains the default. What is done:

| kernel | Rust | notes |
| --- | --- | --- |
| `argon2_first_blocks_kernel` | yes | stage 1 |
| `argon2_kernel_oneshot` | yes | stage 2 — the shuffle-heavy one |

### The feature flag

`tm-gpu/Cargo.toml` gains `rust-kernel`, **off by default**. With it on, `hip::launch_first_blocks`
and `hip::launch_oneshot` dispatch to `module::*` (the Rust kernels) instead of the hipcc ones;
nothing else about the batch changes, and `hip::launch_first_blocks_hip` / `hip::launch_oneshot_hip`
still reach the C++ kernels directly. The fallback is one flag away, deliberately, for as long as
a Rust kernel is unproven.

```sh
./rs cargo test -p tm-gpu --features rust-kernel   # Rust first blocks, on the real GPU
./rs cargo test --workspace                        # unchanged: HIP everywhere
```

### How it is built

`tm-kernel` is **not** a workspace member — its `Cargo.toml` carries an empty `[workspace]`
table so it is its own root. It could not be built for the host anyway (`#![no_std]`, a
GPU-only ABI), and keeping it out means `cargo test --workspace` never sees it. `tm-gpu`'s
`build.rs` drives it, only when `CARGO_FEATURE_RUST_KERNEL` is set, and points
`TM_RUST_KERNEL_ELF` at the resulting code object, which `src/module.rs` embeds with
`include_bytes!`.

`amdgcn-amd-amdhsa` is a **tier-3** target, so there is no prebuilt `core` and the build needs:

- `-Zbuild-std=core`, hence `RUSTC_BOOTSTRAP=1` — this box's rustc is nix's stable 1.97.1.
- The standard-library **source**. The nix rustc ships `lib/rustlib/rustc-src`, which contains
  only `proc_macro` — it is *not* the `rust-src` component and will silently look right.
  The real thing is `nixpkgs#rustPlatform.rustLibSrc`, whose output **is** the `library/`
  directory. `./rs` exports it as `TM_RUST_LIB_SRC`; `build.rs` falls back to running
  `nix build` itself, and to a rustup-style `rust-src` in the sysroot if there is one.
- A sysroot containing that source at `lib/rustlib/src/rust/library`. `build.rs` builds one in
  `OUT_DIR/gpu-sysroot` as a symlink farm over the real sysroot, and passes it through a
  generated `rustc` wrapper (`--sysroot` cannot go in `RUSTFLAGS`: cargo applies rustflags to
  the final crate only, and `core` has to be compiled against the same sysroot).
- ROCm's **real** `ld.lld`. rustc's `ld.lld` flavour passes `-flavor gnu`, which only the
  genuine lld driver accepts: nixpkgs wraps `ld.lld` in a shell script that rejects it
  (`unknown argument '-flavor'`), and nixpkgs' own lld 18 rejects the AMDGPU ABI version
  outright. `build.rs` therefore *probes* candidates with `-flavor gnu --version` rather than
  trusting a path — including the paths named inside the nix wrapper scripts. `TM_GPU_LLD`
  overrides.
- `-Ctarget-cpu=<arch>`, one architecture only: `hipModuleLoad` wants a single-arch code
  object, so the first entry of `offload_archs()` is used (gfx1100 here).

### Writing a kernel

- `#![no_std]`, `#![feature(abi_gpu_kernel, link_llvm_intrinsics)]`, a `#[panic_handler]`, and
  entry points declared `#[no_mangle] pub unsafe extern "gpu-kernel" fn`.
- **Kernel arguments go in one packed struct**, passed through
  `HIP_LAUNCH_PARAM_BUFFER_POINTER`. An array of pointers-to-arguments is HIP's *other*
  calling convention and faults the device, because a module-loaded code object has no
  argument metadata for the runtime to marshal against. The AMDGPU kernarg segment places
  each argument at its natural alignment, so order the parameters pointers → 64-bit →
  32-bit and there is no padding for the host's `#[repr(C)]` mirror to disagree about.
  Check the layout you assumed with
  `llvm-readelf --notes target/.../tm_kernel.elf` (`.args`, `.kernarg_segment_size`).
- There is **no `blockDim`**: workgroup size is only reachable through the HSA dispatch
  packet, so the first-blocks kernel takes `threads_per_block` as an explicit argument.
- **No shared memory.** rustc's amdgcn target cannot express `addrspace(3)` globals. The
  C++'s `extern __shared__ block_l` is not actually shared between threads —
  `block_l_store` writes slots `[i * 32 + thread]` and `block_l_load_xor` reads back the
  same thread's slots — it exists only to relieve NVIDIA register pressure. Stage 2 should
  use four `u64`s in registers, the shape of the C++'s own `move_block`/`xor_block`.
- Shuffles (stage 2): `llvm.amdgcn.ds.bpermute` with a **byte** index (`lane * 4`). Rebuild
  the absolute lane as `(lane_id() & !31) | (src & 31)` so it stays correct on wave64 parts;
  `lane_id` is `mbcnt_hi(!0, mbcnt_lo(!0, 0))`. This mirrors `TM_SHFL`/`TM_SHFL_XOR`, which
  pin every shuffle to `THREADS_PER_LANE` for the same reason.

### The acceptance gate

Both must pass on real hardware before a Rust kernel is believed:

1. `fixtures/argon2_vectors.json` (44 vectors) reproduces byte for byte with the feature on —
   `tests/gpu_vectors.rs`, which runs every vector through *both* first-block paths.
2. `kernel_differential` in `crates/tm-gpu/src/runner.rs`: for identical inputs the Rust and
   HIP kernels must produce identical **raw blocks** — the 2 KiB first-block pair, and the
   1 KiB final block of each job for the one-shot — not merely identical digests, because two
   compensating bugs can agree on a digest but not on a kilobyte of Argon2 output. It lives
   inside the crate because it needs both launch paths at once, which the public API
   deliberately does not expose.

Register pressure is the standing risk in the one-shot kernel: its inner loop runs `4 * m`
times per hash, so a single scratch spill would gut the hashrate. Check it after any edit with
`llvm-readelf --notes` on the code object — `vgpr_count`, `sgpr_spill_count`,
`private_segment_fixed_size`. As ported it is 67 VGPRs, 19 SGPRs, zero spills, zero scratch,
against the HIP kernel's 64 VGPRs / 20 SGPRs (`hipcc -Rpass-analysis=kernel-resource-usage`).
Keep the branchless `cmpeq_mask` selects in `BlockTh::get`/`set`: a `match` on a divergent
index becomes an indexed local array, which on AMDGPU means scratch.

### Fragile / version-pinned

- `abi_gpu_kernel` is an unstable feature reached through `RUSTC_BOOTSTRAP=1`. A rustc upgrade
  can rename or remove it; the store path in `./rs` pins what is known to work (1.97.1).
- `TM_RUST_LIB_SRC` must come from the **same** nixpkgs as the rustc, or `core` will not build.
- The kernarg layout is an ABI contract between `tm-kernel` and `tm-gpu/src/module.rs` that
  the compiler cannot check across the two builds. `debug_assert_eq!(size_of::<…>(), 72)`
  catches a size change; only the differential test catches a reordering.
- The generated code object is single-architecture. Cross-machine deployment needs either a
  per-arch build or a fat bundle; the HIP kernel is still fat.
