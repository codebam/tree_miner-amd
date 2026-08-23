# treeminer-rs

The Rust TreeMiner: an XenBlocks Argon2 miner, host and device code both, with the
offline-resilient submission path this project exists for. `PORT.md` is the working
authority — read it before changing anything here.

## GPU support matrix

| vendor | build | status |
| --- | --- | --- |
| **AMD** — HIP / ROCm, `amdgcn-amd-amdhsa` | `cargo build` (default) | **Tested** on an RX 7900 XTX (gfx1100, ROCm 7.2) |
| **NVIDIA** — CUDA / PTX, `nvptx64-nvidia-cuda` | `cargo build --no-default-features --features nvidia` | **Tested** on an RTX 5070 Ti (sm_120, native PTX). Read the caveats below before trusting another architecture. |

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

### NVIDIA: what is and is not proven

The kernels have now executed on real NVIDIA hardware. On an RTX 5070 Ti (compute
capability 12.0, native `sm_120` PTX, driver-only container):

- all **44 fixture vectors reproduce byte for byte**, on both the CPU- and GPU-first-block
  paths, including the batched variant that catches pool-stride bugs a single-job test
  cannot;
- CPU and GPU digests agree through the shipped binary at m=8, 64, 1024 and 8192;
- an over-large pool is refused by the driver rather than silently falling back.

Plus what was already checkable without a device: the PTX is self-contained (no
`.extern .func` for the JIT to choke on), every cross-lane shuffle is
`shfl.sync.idx.b32` / `shfl.sync.bfly.b32` with a full-warp member mask and a width-32
clamp, and the host argument list matches the PTX `.param` list parameter for parameter.

**What that does not cover**, and matters if you are not on Blackwell:

- **One architecture.** sm_120 only. The shuffle translation is wave-width-correct by
  construction and the same source drives every target, but sm_70 through sm_90 have not
  been executed. Running `scripts/nvidia-smoke.sh` on another card is a 15-minute, ~$0.20
  contribution.
- **No differential oracle.** On AMD every kernel claim is checked against the hipcc kernel's
  raw block output. NVIDIA has no such counterpart, so the 44 vectors are the whole safety
  net — they compare final digests, not intermediate blocks.
- **No sustained run.** Minutes of hashing, not hours, and no block has ever been submitted
  from this path.

Measured on that card: 5130 H/s at m=60000, 8686 H/s at m=42069, 40944 H/s at m=8192 —
ahead of a 24 GB RX 7900 XTX at the lower two, behind at 60000 where 16 GB bounds the batch.
- **The sm_70 floor stands**: Pascal, the GTX 10-series and P106 cards cannot run this PTX at
  all. `TM_PTX_ARCH` emits natively for a newer card; the default sm_70 build is portable and
  JIT-compiled by the driver at load.

**If you have an NVIDIA card that is not Blackwell, you are the first for that
architecture.** On a rented cloud GPU, one command does the whole thing — toolchain, build,
and the gates in order, refusing to start on a card below the sm_70 floor so an hour is not
spent discovering that:

```sh
curl -fsSL https://raw.githubusercontent.com/codebam/tree_miner-amd/main/treeminer-rs/scripts/nvidia-smoke.sh | bash
```

Or by hand, from a checkout:

```sh
cargo test -p tm-gpu --no-default-features --features nvidia   # fixtures on the card
tests/parity/run_parity.sh                                     # against the C++ miner, if you have it built
```

Please report the result: the GPU model, its compute capability, and the throughput lines.
That is what extends the row above beyond the one architecture it currently covers.

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
