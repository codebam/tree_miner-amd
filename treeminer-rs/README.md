# treeminer-rs

The Rust TreeMiner: an XenBlocks Argon2 miner, host and device code both, with the
offline-resilient submission path this project exists for. `PORT.md` is the working
authority — read it before changing anything here.

## GPU support matrix

| vendor | build | status |
| --- | --- | --- |
| **AMD** — HIP / ROCm, `amdgcn-amd-amdhsa` | `cargo build` (default) | **Tested** on an RX 7900 XTX (gfx1100, ROCm 7.2) |
| **NVIDIA** — CUDA / PTX, `nvptx64-nvidia-cuda` | `cargo build --no-default-features --features nvidia` | **Compiles. Never executed.** |

The two are mutually exclusive — they link different device runtimes — and enabling both is
a compile error.

### AMD: what "tested" means

- All 44 vectors in `fixtures/argon2_vectors.json` reproduce byte for byte on the card, with
  GPU first blocks both on and off.
- `kernel_differential` (in `crates/tm-gpu/src/runner.rs`) compares the Rust kernels against
  the hipcc kernels on **raw Argon2 blocks**, not digests: the 2 KiB first-block pair and the
  1 KiB final block of every job must match exactly. Two compensating bugs can agree on a
  digest; they cannot agree on a kilobyte of Argon2 output.
- `tests/parity/run_parity.sh` cross-checks the whole miner against the C++ one.
- The one-shot kernel runs with zero register spills and no LDS, at roughly 1.5x the HIP
  kernel's hashrate.

### NVIDIA: what "never executed" means

Exactly what it says. **There is no NVIDIA GPU on the machine this was written on.** No
digest, no kernel launch, no `cuInit` has ever happened on this path. What has been checked
is only what can be checked without a device:

- the shared Argon2 kernel source compiles for `nvptx64-nvidia-cuda`;
- the emitted PTX contains both `argon2_first_blocks_kernel` and `argon2_kernel_oneshot`;
- the PTX is self-contained — no `.extern .func` the CUDA JIT would fail to resolve;
- every cross-lane shuffle is `shfl.sync.idx.b32` / `shfl.sync.bfly.b32` with a full-warp
  member mask and a width-32 clamp, which is what the C++ `__shfl_sync(..., 32)` means;
- the host's `cuLaunchKernel` argument list matches, parameter for parameter and width for
  width, the `.param` list the PTX declares;
- the CUDA driver-API host layer compiles.

None of that says the kernels compute Argon2 correctly on NVIDIA hardware. An invalid digest
is accepted locally and rejected by the server *after* the work is spent — this project has
already shipped that bug once (commit 12e241c) and it cost real submissions.

**If you have an NVIDIA card, you are the first.** On a rented cloud GPU, one command does
the whole thing — toolchain, build, and the gates in order, refusing to start on a card below
the sm_70 floor so an hour is not spent discovering that:

```sh
curl -fsSL https://raw.githubusercontent.com/codebam/tree_miner-amd/main/treeminer-rs/scripts/nvidia-smoke.sh | bash
```

Or by hand, from a checkout:

```sh
cargo test -p tm-gpu --no-default-features --features nvidia   # fixtures on the card
tests/parity/run_parity.sh                                     # against the C++ miner, if you have it built
```

Until those pass on real hardware, this is a compilation exercise. Please report the result:
the GPU model, its compute capability, and the throughput lines. That is what moves the row
above from "compiles" to "tested".

Known NVIDIA limitations before you start:

- **The PTX targets `sm_70`**, which is the lowest architecture this rustc will emit for
  `nvptx64`. Volta and newer only — Pascal and the GTX 10-series / P106 cards cannot run it.
- There is no hipcc-equivalent oracle for NVIDIA, so the raw-block differential that guards
  the AMD kernels has no counterpart. The fixtures are the whole safety net.
- `crates/tm-gpu/src/telemetry.rs` reads ROCm SMI. On an NVIDIA build it degrades to "no
  telemetry" rather than reading NVML; fan, temperature and power will be blank.

## Building

### With nix (reproducible, no toolchain setup)

From the **repository root**, everything — Rust, ROCm, the amdgcn kernel step, the vendored
crates — comes out of the flake. No network access is used during the build.

```sh
nix build                    # == nix build .#treeminer, the AMD miner
./result/bin/treeminer --help
nix run . -- --help          # meta.mainProgram is set

nix build .#treeminer-rocm   # the C++ miner this one replaces (bin/xenblocksMiner)
```

| attribute | what it is |
| --- | --- |
| `packages.treeminer` (= `packages.default`) | the Rust miner for AMD/ROCm, GPU kernels included. **This is the shipping package.** |
| `packages.treeminer-nvidia` | the same miner with `--no-default-features --features nvidia`. See below. |
| `packages.treeminer-rocm` | the old C++/HIP miner, unchanged |

**The GPU architecture is pinned to `gfx1100`** (RDNA3, RX 7900 — the card this is tested
on). It is a single architecture rather than a fat list on purpose: the Rust kernel is a
code object loaded with `hipModuleLoad`, which accepts exactly one architecture, and
`tm-gpu/build.rs` builds it for the *first* entry of the offload list. A fat list would
therefore fatten only the (fallback) hipcc kernels while silently leaving the Rust kernels —
the ones that actually run — targeted at whatever happened to be first. Override it:

```sh
nix build --impure --expr \
  '(builtins.getFlake (toString ./.)).packages.x86_64-linux.treeminer.override { gpuArch = "gfx1030"; }'
```

or, from a flake that consumes this one, `treeminer.override { gpuArch = "gfx942"; }`.

`doCheck` is off: every GPU test needs a real `/dev/kfd`, which the nix sandbox does not
have. Correctness of the nix-built binary is established by running the parity suite against
it afterwards:

```sh
nix build
nix develop --command bash -c \
  'cd treeminer-rs && bash tests/parity/run_parity.sh ../result/bin/treeminer <cpp-miner>'
# parity: 88 passed, 0 failed
```

#### What `packages.treeminer-nvidia` is and is not

- It **is** the same source built for CUDA, and a real derivation you can evaluate
  (`nix eval .#packages.x86_64-linux.treeminer-nvidia.drvPath` works for everyone).
- It has **never been built and never been executed.** Nothing in this repository has run on
  an NVIDIA card — see "NVIDIA: what 'never executed' means" above.
- It does **not** pull in nixpkgs' CUDA by default. That package is unfree and multiple
  gigabytes, and merely evaluating an attribute must not drag it in, so the flake only
  reaches for `cudaPackages.cuda_cudart` when `config.allowUnfree` is set. With the default
  configuration the build stops in `preBuild` with an explanation instead of downloading
  anything.
- To actually attempt it:
  `NIXPKGS_ALLOW_UNFREE=1 nix build --impure .#treeminer-nvidia`, or point it at a libcuda
  you already have with `.override { cudaLibDir = "/run/opengl-driver/lib"; }`.

### Without nix build (day-to-day development)

The box has no global Rust or ROCm; both come from nix. Always go through `./rs`, which
resolves the same store paths the flake uses and works outside `nix develop` too:

```sh
./rs cargo test --workspace
./rs cargo test -p tm-gpu                       # the real GPU
./rs cargo clippy --workspace --all-targets
```

Set `CARGO_TARGET_DIR` to your own directory so parallel work does not block on the target
lock.

Building the NVIDIA path needs a `libcuda` to link against; the CUDA toolkit's
`lib64/stubs/libcuda.so` is enough. `build.rs` fails with an explanation if it cannot find
one. To compile-check the NVIDIA path on a machine with no CUDA at all, set
`TM_CUDA_ALLOW_MISSING=1` — the library will build and nothing can be linked against it.

## Layout

| crate | what it is |
| --- | --- |
| `tm-core` | the shared contract: types, encoding, addresses, batch sizing, matching |
| `tm-argon2` | CPU Argon2 and the host-side helpers the GPU path calls into |
| `tm-kernel` | the Argon2 **device** kernels, in Rust. Not a workspace member; built for a GPU target by `tm-gpu`'s `build.rs`. `src/arch/` is the per-vendor floor |
| `tm-gpu` | device management, the batch pool, and the two driver layers (`hip.rs`, `cuda.rs`) behind one `driver` alias |
| `tm-journal` | the durable local queue of found hashes |
| `tm-submit` | submission, retry, circuit breaking, drain scheduling |
| `tm-dashboard` | the HTTP dashboard |
| `tm-tui` | the terminal console |
| `treeminer` | the CLI and the mining loop |
