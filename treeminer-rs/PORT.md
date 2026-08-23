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

**This is staged, not permanent.** The kernel is to be rewritten in Rust as a second phase,
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
