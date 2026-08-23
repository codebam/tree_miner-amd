# TreeMiner

An outage-proof GPU miner for [XenBlocks](https://xenblocks.io) (X1 Network), forked from
[woodysoil/XenblocksMiner](https://github.com/woodysoil/XenblocksMiner) (MIT).

**The problem it solves:** the XenBlocks central server goes down several times a day.
Existing miners keep found blocks in RAM and drop them after a few failed retries — every
outage permanently loses real finds. TreeMiner journals every find to durable storage
*before* the first network attempt and drains the journal with outage-aware retry.

## How it works

```
GPU find -> immutable PHC capture -> SQLite WAL journal (fsync'd)  ->  SubmissionManager
                                          |                              |  circuit breaker
                                     survives crashes,              adaptive drain, backoff,
                                     restarts, outages              /get_block confirmation
```

- **Journal-first invariant** — `append()` returns only after the find is crash-safe on
  disk (`treeminer-rs/crates/tm-journal/`, WAL + `synchronous=FULL`, one `BEGIN IMMEDIATE`
  transaction so the fsync has happened before the call returns).
- **9-state find lifecycle** — including `ParkedDifficulty` (401-difficulty finds
  auto-resubmit when difficulty allows) and `ParkedXuniWindow` (XUNI retry across
  future :55–:05 windows). Nothing is ever silently dropped.
- **Lying-200 detection** — the reference server can return 200 without storing the block;
  every accept is re-verified via `GET /get_block` before being counted.
- **Circuit breaker + adaptive drain** — no hammering a dead server; queued finds drain at
  a controlled, escalating rate on recovery.
- **Fixes inherited upstream bugs** — stale-difficulty silent find drops, weak 32-bit
  keygen seeding, VRAM-pool starvation spin on difficulty drops.

Validated end-to-end: the Rust workspace's own suites plus a chaos harness against a
`gpage.py`-faithful mock server with fault injection (60 s hard outage → 13/13 finds
recovered and acked), and live against the real server — 336 blocks acked and
`/get_block`-confirmed on one rig. The journal's schema and status strings are identical to
the C++ miner's, so both binaries read the same database and switching between them strands
nothing.

## Layout

| Path | Contents |
|---|---|
| `treeminer-rs/` | **The miner.** Rust, host and GPU kernels both. `treeminer-rs/PORT.md` is the working authority; `treeminer-rs/README.md` has the support matrix |
| `treeminer-rs/crates/tm-kernel/` | Argon2 device kernels in Rust, compiled for `amdgcn-amd-amdhsa` (and `nvptx64`, untested) |
| `treeminer-rs/crates/tm-journal/` | Durable SQLite find journal |
| `treeminer-rs/crates/tm-submit/` | Classifier, circuit breaker, drain scheduler, submission manager |
| `treeminer/` | The C++/HIP miner this was ported from. Still builds, still the differential oracle the Rust kernel tests compare against. `treeminer/CHANGES-FROM-UPSTREAM.md` is its divergence log |
| `docs/` | Research docs `01`–`08` plus `09-ops-stability.md` (this-box crash/reset record) |
| `docs/reviews/` | Verbatim review docs from other models (Kimmy, Grok, Sol) |
| `research/` | Experiment notes and validation records |

## Running it

```sh
nix run github:codebam/tree_miner-amd -- \
  --execute --minerAddr 0xYourAddress --totalDevFee 0 --display terminal
```

Run it from a directory you are happy writing into: the find journal
(`treeminer-journal.db`), `difficulty.cache` and `log/` are created in the working
directory. `--display logs` for a service, `--dashboard-bind 127.0.0.1` to keep the operator
console off the LAN.

## GPU support

| vendor | status |
|---|---|
| AMD (ROCm/HIP) | Tested on an RX 7900 XTX (gfx1100, ROCm 7.2) |
| NVIDIA (CUDA/PTX) | Tested on an RTX 5070 Ti (sm_120). `sm_70`+ only; other NVIDIA architectures are untested — see `treeminer-rs/README.md` |

The flake pins `gfx1100`, because a GPU module is loaded for exactly one architecture. On a
different card, override it — otherwise the miner fails to load its kernel at startup:

```sh
nix build --impure --expr \
  '(builtins.getFlake (toString ./.)).packages.x86_64-linux.treeminer.override { gpuArch = "gfx1030"; }'
```

Throughput on an idle RX 7900 XTX, against the C++ miner this was ported from, plus an
RTX 5070 Ti on the NVIDIA path for comparison:

| difficulty | C++/HIP (7900 XTX) | Rust (7900 XTX, 24 GB) | Rust (5070 Ti, 16 GB) |
|---|---|---|---|
| 60000 | 3215 H/s | **5289 H/s** | 5130 H/s |
| 42069 | 5529 H/s | **7891 H/s** | **8686 H/s** |
| 8192 | 19694 H/s | **39672 H/s** | **40944 H/s** |

The 5070 Ti leads at the difficulties where the batch fits comfortably and trails at 60000,
where its 16 GB bounds the batch the planner can choose — memory, not architecture.

The Argon2 kernel is the reason: it keeps the per-thread scratch in registers instead of
round-tripping 1 KiB of LDS per block, `4*m` times per hash.

## Building

```sh
nix build                  # the Rust miner -> ./result/bin/treeminer
nix build .#treeminer-rocm # the C++/HIP miner -> ./result/bin/xenblocksMiner
nix develop                # both toolchains; then: cd treeminer-rs && cargo test --workspace
```

Without nix, see `treeminer-rs/README.md` (Rust) and `treeminer/doc/BUILD_INSTRUCTIONS.md`
(C++, CMake + vcpkg, either GPU vendor).

## How it is checked

Every digest claim is backed by hardware, not inspection:

- **88/88 differential parity** — the same 44 Argon2 vectors through both this miner and the
  C++ one, on both backends, compared digest for digest
  (`treeminer-rs/tests/parity/run_parity.sh`).
- **Raw-block kernel differential** — the Rust and hipcc kernels must produce identical
  1 KiB output blocks, not merely matching digests, so compensating bugs cannot hide.
- **Negative controls** — the kernel gates have been shown to fail on a deliberately wrong
  shuffle mask and a dropped counter, then pass again on revert.
- 415 tests across the Rust workspace; the C++ tree keeps its 27 CTest suites.

## License

MIT (`LICENSE`, mirrored at `treeminer/LICENSE`), preserving woodysoil/XenblocksMiner
attribution — the Rust miner is a port of that C++ lineage and carries it forward. The reference
server repo (jacklevin74/xenminer) is unlicensed: it was read for protocol semantics only
and **no code from it is included**.
