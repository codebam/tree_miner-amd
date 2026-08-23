#!/usr/bin/env bash
#
# First-run acceptance gate for the NVIDIA build, meant for a rented cloud GPU.
#
# The NVIDIA path is compile-verified only: no NVIDIA GPU has ever executed these kernels.
# This script is what turns that into evidence. It sets up a toolchain, builds, and then
# runs the gates in increasing order of cost — correctness first, speed last, because a
# hashrate from a kernel that computes the wrong digest is worse than no number at all.
#
#   curl -fsSL https://raw.githubusercontent.com/codebam/tree_miner-amd/main/treeminer-rs/scripts/nvidia-smoke.sh | bash
#
# or, from a checkout:  ./treeminer-rs/scripts/nvidia-smoke.sh
#
# Exit code is 0 only if every gate passed.
set -uo pipefail

# Every early exit below must carry a non-zero status: piped into bash, a zero exit
# from a failed preflight reads as success.

REPO_URL=${REPO_URL:-https://github.com/codebam/tree_miner-amd.git}
REPO_REF=${REPO_REF:-main}
WORKDIR=${WORKDIR:-$HOME/treeminer-nvidia}
SALT=${SALT:-e4bb184781bbc9c7004e8dafd4a9b49d203bc9bc}

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
info() { printf '  %s\n' "$*"; }
fail() { printf '\033[31mFAIL\033[0m  %s\n' "$*"; }
pass() { printf '\033[32mok\033[0m    %s\n' "$*"; }

RESULTS=()
record() { RESULTS+=("$1|$2"); }

# ---------------------------------------------------------------- preflight

bold "== Preflight"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    fail "nvidia-smi not found — this script is for a machine with an NVIDIA GPU."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
info "GPU: $GPU_NAME (compute capability $CAP)"

# rustc's nvptx64 target floor is sm_70. Anything older cannot run this PTX at all, and
# finding that out after paying for an hour of Pascal is exactly what this check prevents.
CAP_MAJOR=${CAP%%.*}
CAP_MINOR=${CAP##*.}
if [ "$((CAP_MAJOR * 10 + CAP_MINOR))" -lt 70 ]; then
    fail "compute capability $CAP is below the sm_70 floor of rustc's nvptx64 target."
    info "This card cannot run the Rust kernels. Rent Volta or newer:"
    info "  T4 (7.5), V100 (7.0), RTX 30xx (8.6), A100 (8.0), RTX 40xx (8.9)."
    exit 1
fi
pass "compute capability $CAP meets the sm_70 floor"

if ! ldconfig -p 2>/dev/null | grep -q "libcuda\.so"; then
    fail "libcuda.so not found. Use a CUDA driver image (nvidia/cuda:*-devel)."
    exit 1
fi
pass "libcuda.so present"

# ---------------------------------------------------------------- toolchain

bold "== Toolchain"

if command -v apt-get >/dev/null 2>&1 && ! command -v cc >/dev/null 2>&1; then
    info "installing build essentials"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq build-essential pkg-config curl git >/dev/null
fi

if ! command -v cargo >/dev/null 2>&1; then
    info "installing rustup"
    curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
fi
# shellcheck disable=SC1090,SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"

# The GPU kernels are built for a tier-2 target with -Z build-std, so the standard library
# *source* has to be installed. On a rustup toolchain this is one component; the nix box
# this was developed on has to assemble a sysroot by hand.
rustup component add rust-src >/dev/null 2>&1
rustup target add nvptx64-nvidia-cuda >/dev/null 2>&1
info "$(rustc --version)"
pass "rust toolchain ready (rust-src + nvptx64 target)"

# ---------------------------------------------------------------- source

bold "== Source"

if [ -f "treeminer-rs/Cargo.toml" ]; then
    SRC=$PWD
    info "using the checkout in $SRC"
elif [ -d "$WORKDIR/.git" ]; then
    SRC=$WORKDIR
    git -C "$SRC" fetch --depth 1 origin "$REPO_REF" -q && git -C "$SRC" checkout -q FETCH_HEAD
    info "updated $SRC to $REPO_REF"
else
    SRC=$WORKDIR
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SRC" -q
    info "cloned $REPO_REF into $SRC"
fi
cd "$SRC/treeminer-rs" || exit 1
pass "source at $(git -C "$SRC" rev-parse --short HEAD)"

# ---------------------------------------------------------------- build

bold "== Build (nvidia)"

NV=(--no-default-features --features nvidia)
if cargo build --release -p treeminer "${NV[@]}" 2>&1 | tail -20; then
    pass "treeminer built for NVIDIA"
    record ok "build"
else
    fail "build failed — everything below is untested"
    record fail "build"
    exit 1
fi
MINER=target/release/treeminer

# ---------------------------------------------------------------- gate 1

bold "== Gate 1: fixture vectors (the whole safety net on NVIDIA)"
info "There is no hipcc oracle on this vendor, so these 44 vectors are the only thing"
info "standing between a subtly wrong kernel and blocks the server rejects."

if cargo test -p tm-gpu "${NV[@]}" -- --nocapture 2>&1 | tail -30; then
    pass "tm-gpu suite passed"
    record ok "fixture vectors"
else
    fail "fixture vectors did NOT reproduce — the kernel is wrong. Stop here."
    record fail "fixture vectors"
fi

# ---------------------------------------------------------------- gate 2

bold "== Gate 2: CPU/GPU digest agreement through the shipped binary"

digest_of() {
    "$MINER" hash-one --salt "$SALT" --key "$2" --backend "$1" --difficulty "$3" --json 2>/dev/null |
        python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["hash"].rsplit("$",1)[-1] if d["ok"] else "ERROR:"+d["error"])'
}

mismatch=0
checked=0
for difficulty in 8 64 1024 8192; do
    key=$(printf '%064x' $((difficulty * 7919)))
    cpu=$(digest_of cpu "$key" "$difficulty")
    gpu=$(digest_of gpu "$key" "$difficulty")
    checked=$((checked + 1))
    if [ "$cpu" = "$gpu" ] && [ -n "$cpu" ] && [ "${cpu#ERROR}" = "$cpu" ]; then
        info "m=$difficulty  ${cpu:0:24}…  match"
    else
        fail "m=$difficulty  cpu=$cpu  gpu=$gpu"
        mismatch=$((mismatch + 1))
    fi
done
if [ "$mismatch" -eq 0 ]; then
    pass "$checked/$checked digests agree between CPU and GPU"
    record ok "cpu/gpu digests"
else
    fail "$mismatch of $checked digests disagree"
    record fail "cpu/gpu digests"
fi

# ---------------------------------------------------------------- gate 3

bold "== Gate 3: throughput (only meaningful if the gates above passed)"

for difficulty in 42069 8192; do
    line=$("$MINER" hash-benchmark --salt "$SALT" --backend gpu --seconds 8 \
        --auto-batch-size --difficulty "$difficulty" --no-xuni --json 2>/dev/null |
        python3 -c 'import json,sys
d=json.load(sys.stdin)
print(f"batch={d[\"batch_size\"]:6d}  {d[\"hashrate\"]:10.1f} H/s" if d["ok"] else "ERROR: "+d["error"])')
    info "m=$difficulty  $line"
done
record ok "throughput measured"

# ---------------------------------------------------------------- summary

bold "== Summary  ($GPU_NAME, compute $CAP)"
failed=0
for entry in "${RESULTS[@]}"; do
    status=${entry%%|*}
    name=${entry#*|}
    if [ "$status" = ok ]; then pass "$name"; else fail "$name"; failed=$((failed + 1)); fi
done

if [ "$failed" -eq 0 ]; then
    bold "NVIDIA path verified on real hardware."
    info "Report the GPU model, compute capability and the throughput lines above;"
    info "that is what promotes the support matrix from \"compiles\" to \"tested\"."
    exit 0
fi
bold "$failed gate(s) failed — the NVIDIA path is NOT verified."
info "Capture the full output above; the failing gate names the layer at fault."
exit 1
