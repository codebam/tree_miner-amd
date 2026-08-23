#!/usr/bin/env bash
# Differential test: the Rust miner against the C++ miner it replaces.
#
# Both binaries expose the same network-free hash API (`hash-one`, `hash-batch`), so the
# port can be checked digest-for-digest on real hardware without touching the network or
# submitting anything. Any divergence here is a port bug, full stop.
#
# Usage: run_parity.sh <rust-binary> <cpp-binary>
set -euo pipefail

RUST_BIN=${1:?usage: run_parity.sh <rust-binary> <cpp-binary>}
CPP_BIN=${2:?usage: run_parity.sh <rust-binary> <cpp-binary>}
FIXTURES="$(dirname "$0")/../../fixtures/argon2_vectors.json"

fail=0
pass=0

digest_of() { # binary backend salt key difficulty [extra flags...]
    local bin=$1 backend=$2 salt=$3 key=$4 difficulty=$5; shift 5
    "$bin" hash-one --salt "$salt" --key "$key" --backend "$backend" \
        --difficulty "$difficulty" --json "$@" |
        python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["hash"].rsplit("$",1)[-1] if d["ok"] else "ERROR:"+d["error"])'
}

while read -r salt key difficulty expected; do
    for backend in cpu cuda; do
        rust=$(digest_of "$RUST_BIN" "$backend" "$salt" "$key" "$difficulty")
        cpp=$(digest_of "$CPP_BIN" "$backend" "$salt" "$key" "$difficulty")
        if [[ "$rust" == "$expected" && "$cpp" == "$expected" ]]; then
            pass=$((pass + 1))
        else
            fail=$((fail + 1))
            echo "MISMATCH backend=$backend difficulty=$difficulty salt=$salt"
            echo "  expected $expected"
            echo "  rust     $rust"
            echo "  cpp      $cpp"
        fi
    done
done < <(python3 -c '
import json,sys
for v in json.load(open(sys.argv[1]))["vectors"]:
    print(v["salt_hex"], v["key"], v["difficulty"], v["digest_b64"])
' "$FIXTURES")

echo "parity: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
