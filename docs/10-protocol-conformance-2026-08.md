# XenBlocks protocol conformance & efficiency audit — 2026-08-24

Scope: verify TreeMiner follows the XenBlocks spec, mines correctly, and mines
efficiently. Sources: the reference server source, five other mining clients,
live probes of the production endpoints, and 19 hours of our own journal data.

Reference clones live in `repos/` (git-ignored): `xenminer` (jacklevin74 — the
server of record), `XenblocksMiner` (Woody), `xnminer` + `xnminer-linux`
(badnob/"Tony.x1", the only client maintained into 2026), `XENGPUMiner`,
`xenminer-go`, `xenblocks-webminer`, `xenblocks-lobby`, `x1-xenblocks`, `xgpu`.

Server of record: `repos/xenminer/gpage.py` — the Flask app serving `/verify`,
`/difficulty`, `/get_block` on xenblocks.io.

---

## 1. The spec, as verified

| Element | Value | Citation |
|---|---|---|
| Algorithm | Argon2**id**, v=19 (0x13) | `gpage.py:451` re-verifies; all clients agree |
| Time cost | t=1 | `miner.py:239` via `config.conf:2` |
| Parallelism | p=1 | `config.conf:3` |
| Memory cost | `m` = network difficulty, 1:1, KiB | `miner.py:133,250`; `gpage.py:404` |
| Digest | 64 bytes → 86 unpadded-base64 chars | `miner.py:239` |
| Salt | address `0x`-stripped, **hex-decoded to 20 raw bytes** | `miner.py:236-237` |
| Password | 64 lowercase hex chars, hashed as **ASCII** | `miner.py:149-155`; server allows ≤128 |
| Wire form | full PHC string, ≤150 chars | `gpage.py:376` |

Match rules, all tested against `hash_to_verify[-87:]` (which is exactly `$` plus
the 86-char digest — hence the magic number):

- **XEN11** — plain substring, position unconstrained (`gpage.py:421-425`)
- **XUNI** — regex `XUNI[0-9]` (`gpage.py:430`)
- **Superblock** — ≥50 uppercase chars in the final `$`-segment
  (`make_superblocks.py:41-46`, `utils/gen_balances.py:77-81`). The official
  docs' "65+ uppercase in 136 characters" is wrong and never fires.

Submission: `POST http://xenblocks.io/verify`, plain HTTP, no auth, no
signature, six fields: `hash_to_verify`, `key`, `account`, `attempts`,
`hashes_per_second`, `worker`.

### The three findings that matter most

**No timestamp validation exists.** The body has no timestamp field and the
server never asks for one. The only `datetime` call in the handler assigns to a
variable that is never read (`gpage.py:458`); its one consumer, the
`account_attempts` insert, is commented out at `:499`. The XEN11 branch carries
the comment `# no time restrictions for XEN11` (`gpage.py:481`). Storage times
come from SQLite's `CURRENT_TIMESTAMP` default.

**Reward is flat per accepted block and does not scale with difficulty.**
`utils/gen_balances.py:119-125` credits a fixed 10 for XEN11 and 1 for XUNI,
with no difficulty term and no time term. A superblock is credited 10 XNM **plus**
1 XBLK — the XBLK credit sits inside the XEN branch, it does not replace it.
Consequence: **higher difficulty is pure cost.** It lowers block rate for
identical per-block pay. Low difficulty is strictly the profitable regime.

**Difficulty drift is the only expiry mechanism, and it is reversible.**
`gpage.py:412` rejects only when `submitted_m < current` — a *strict* less-than,
so a block mined above current difficulty is accepted. The adjuster
(`manage_difficulty2.py:50-75`) runs every 300 s targeting 70 blocks/period:
**+1000 up, −2000 down**, floor 100, no ceiling. Because it falls twice as fast
as it rises, a parked block reliably becomes valid again.

### Protocol trivia worth knowing

- `worker` is **dead on the wire**. The server reads `data.get('worker_id')` — a
  different key — then discards it unless it is a string of ≤3 chars
  (`gpage.py:370-373`). No client sends `worker_id`. Per-worker attribution does
  not exist server-side. `attempts` and `hashes_per_second` are equally unused.
- `account` is **not bound to the salt**. Nothing checks that the base64 salt
  decodes to the `account` field.
- `account` is lowercased and capped at 43 chars server-side (`gpage.py:382-384`).
  Always compare accounts case-insensitively.
- Replay protection is a single `UNIQUE` on `blocks.key`, surfacing as HTTP 400
  `"Block already exists, continue"` — which every client correctly treats as
  success.
- The legacy salt `XEN10082022XEN` is still accepted, bypassing the address
  check entirely (`gpage.py:283,293-295`).
- No rate limiting of any kind. Difficulty is the only throughput control.
- `node_verify.py` is a non-persisting local test node (target `XEN1`,
  difficulty hardcoded 8, port 8888) — not production. Do not read it as spec.

---

## 2. Conformance verdict: we conform

Every rule checked against `crates/` on branch `rust`/`main` matches:

- Argon2 shape, salt derivation, ASCII password, unpadded base64, PHC assembly
- Matching against the bare 86-char digest ≡ the server's `[-87:]` slice
- XUNI regex, XUNI window `:55–:04`, superblock ≥50 uppercase
- Park/unpark boundary `m >= current`, matching the server's strict `<`

**Confirmed against the live server**, not merely by reading code:

```
GET /get_balance/0x655815CFaC22597C4339B76A8B7f8f3da6e648cD → {"balance":8320}
GET /get_super_blocks/…                                     → {"super_blocks":1}
```

Our blocks are in the ledger. And from 1937 journal rows: 1781 acked (91.9 %),
**610 of them confirmed more than 10 minutes after discovery, the slowest at
4.9 hours.** Delayed submission is not merely permitted by the source — it is
working in production for us today.

---

## 3. Defects found in our miner

Ranked by consequence.

1. **Two different XUNI windows in one binary.** The mining loop gates on the
   *local* wall clock (`mineunit.rs:111`); the submitter gates on
   server-clock-corrected UTC (`manager.rs:334`). Identical on whole-hour
   offsets, divergent on `:30`/`:45` zones (IST, NPT, ACST).
2. **`--testBlockPattern` mislabels every find.** `find.rs:145` hardcodes
   `find.digest.contains("XEN11")` to decide the kind, ignoring the configured
   pattern, so a custom pattern journals everything as XUNI and applies XUNI
   park semantics. Test-only flag, real bug.
3. **XUNI matches can never be flagged superblock** — `is_superblock: false` is
   hardcoded at `hash.rs:246`.
4. **The C++ `rocm-backend` branch lacks upstream `626f74f`** (warp-uniform
   address-word selection, sm75 registers 56→53). The Rust kernel already
   implements it — `tm-kernel/src/lib.rs:537`, used at `:854` — so only the
   stale branch is behind.
5. **`PowSubmitter.cpp:13` points at `xenminer.mooo.com`, which is NXDOMAIN.**
   Compiled but never called, in our fork *or* upstream Woody's — dead code, not
   a live bug. Superblocks are credited server-side by re-scanning the `blocks`
   table, never via `send_pow`; our 1 credited superblock proves it.

### Divergence we chose, and its risk

Upstream re-runs Argon2 on CPU for every find and drops it unless the GPU digest
is a substring (`main.cpp:377-381`). We assemble the PHC string from the GPU
digest instead. Ours is better in one respect — upstream re-hashes at *live*
`globalDifficulty`, so a find that sits across a difficulty change is silently
discarded. But given commit `12e241c` in our own history (an nvcc miscompile
emitting invalid digests), the startup self-test is now the only guard against a
bad toolchain. `xnminer` keeps both: GPU first blocks on, plus a CPU re-verify
before submit (`argon2_common.py:54-90`). That is the safer arrangement.

---

## 4. Efficiency

### The kernel is not the problem

At real difficulty the CUDA kernel is **92–98 % of wall time** (upstream
`goal.md:91-93`). Host-side finalize/base64/match is not the bottleneck, and
upstream's experiment ledger records **pinned host staging as tried and
rejected** — better transfer timings, worse throughput (42.5k → 35.7k H/s).
Also rejected there: `__launch_bounds__(THREADS_PER_LANE, 4)`, device-side final
hashing, multi-warp blocks. Do not re-litigate these without a materially
different hypothesis.

### The real opportunity: low difficulty is where the blocks are

From 19 h of our journal:

| mined m | wall time | finds | finds/min |
|---|---|---|---|
| 100 | 67 min | 596 | **8.9** |
| 1100 | 554 min | 1062 | 1.9 |

Since reward is flat per block (§1), **that 4.7× yield gap is a 4.7× revenue
gap**, and we spent only 6 % of the day in the profitable regime. Difficulty
collapses to 100 roughly hourly and recovers within minutes.

At m=100 the batch is **not** ceiling-capped — `recommended_batch_size` returns 0
(memory-limited) for any m >= 65, and the network difficulty floor is 100, so the
tuned low-`m` ceilings (2048/4096/3072) are unreachable in production. On a 24 GiB
card at m=100 the plan is already ~235k attempts filling the post-reserve budget.
**The card is full. More lanes cannot help, and a lane planner returns 1 at every
live difficulty.**

The loss is real but host-side. Measured from find timestamps alone (median gap
between consecutive finds at the same `m`, needing no dwell-time estimate):

| mined m | n | median gap | rate vs m=1100 | predicted at 1/m | captured |
|---|---|---|---|---|---|
| 100 | 490 | 3.0 s | 2.67x | 11.00x | **24 %** |
| 1100 | 984 | 8.0 s | 1.00x | 1.00x | 100 % |

We capture under a quarter of what low difficulty should pay. The cause is the
finalize loop in `tm-gpu/src/hash.rs:180-193`: after a full `stream.synchronize()`
it walks attempts one at a time on a single thread, doing a Blake2b 1024->64
compression, a base64 encode and a substring search each. At m=60000 that is noise
against a ~25 GB kernel; at m=100 it plausibly *is* the batch, and the GPU idles
through it.

Note this does not contradict upstream's "kernel is 92-98 % of wall time"
(`goal.md:91-93`) — those measurements were taken at m=4096 and above. The balance
inverts at low `m`, and low `m` is where the revenue is.

Fix: parallelize finalize/match across cores (`tm-argon2/src/host.rs:34-49` already
does this for CPU first blocks) and overlap it with the next batch's kernel.
Validate with a fixed-difficulty A/B on real hardware before trusting it.

### Outcome (measured, RX 7900 XTX, `--gpu-first-blocks`, quiet box, 3 runs each)

The finalize loop was parallelized across cores in commit `eba823a`:

| | before | after | speedup |
|---|---|---|---|
| m=100 | 632,672 H/s | **1,745,282 H/s** | **2.76x** |
| m=1100 | 225,116 H/s | **287,840 H/s** | **1.28x** |

Per attempt, finalize went 1.20 us -> 0.20 us on 16 threads. The m=100 profile
is now balanced — finalize 32.8 %, kernel 33.6 %, keygen 19.9 % — where it was
finalize 75.5 % / kernel 12.4 %. At m=1100 the kernel is 88.5 % of wall time, so
that difficulty is done without kernel work.

Two consequences worth recording. Throughput is now **sensitive to CPU
contention** in a way it was not before: benchmarking while other builds ran
produced a 1.6x spread between consecutive runs and one result *below* baseline.
And the next target at m=100 is **keygen** (19.9 %), which is still serial.

### Ranked recommendations

1. **Low-difficulty lane multiplication.** Fill the card when `m` is small.
   Largest expected gain by a wide margin.
2. **Gate XUNI matching in the kernel, not just at submit.** `xnminer` sets
   `allow_xuni = in_xuni_window()` per batch (`cuda_native.py:337`). We already
   decide `allow_xuni` per batch — verify it is off outside the window so we
   stop finding XUNI blocks that can never be sold.
3. **Mine and submit XUNI from `:55`, not `:56`.** The server allows minute 55
   (`gpage.py:40`); `xnminer` starts at `:56` and forfeits ~10 % of XUNI
   opportunity. Confirm we use the full window on both paths (see defect 1).
4. **Move the circuit-breaker probe off port 80.** Measured: **6 of 14** port-80
   requests timed out at 12 s, while ports 4445/4447 stayed healthy. Our breaker
   currently probes `/difficulty` on the least reliable route.
5. **Restore a CPU re-verify before submit.** Cheap insurance against the
   `12e241c` class of GPU miscompile; lets us keep GPU first blocks on.
6. **Adopt the rejection-string taxonomy** from `networking/submit_result.py`,
   including `"already exists"` ⇒ accepted on **400 and 409**, all four
   XUNI-window phrasings, and a *terminal* class for the six permanent failures
   (`"Invalid key format"`, `"Invalid salt format"`, `"Missing hash_to_verify,
   key, or account"`, `"Hash does not contain any of the valid targets"`,
   `"Length of hash_to_verify should not be greater than 150 characters."`,
   `"Hash verification failed."`). `xnminer` classifies these as rejects and
   then re-POSTs them every 30 s forever — a live defect there.
7. **Difficulty-transition quiesce** (`supervisor.py:274-289`): pause submits
   for a few seconds after a difficulty change and queue instead, avoiding a
   burst of racing 401s.
8. **Use `https://xenblocks.io/v1/leaderboard`** for self-confirmation. Returns
   per-account `blocks`, `superBlocks`, `xnm`/`xuni`/`xblk`, plus network
   totals and current difficulty, over HTTPS on a healthier route. Undocumented
   and unused by any client in `repos/`.
9. **On-chain balance oracle.** `wallet_balances.py:23-24,78,100` reads ERC-20
   `balanceOf` (`0x70a08231`) for XUNI `0x999999cf…00002` and XBLK
   `0x999999cf…00001`, 18 decimals. Ground truth for reward per block — and it
   would settle the halving question empirically. **Note the RPC it uses,
   `https://xenblocks.io:5556`, currently returns 502.**

### Explicitly not worth taking

- `strategies/fibonacci_strategy.py` is cargo cult. It SHA-256s a Fibonacci
  recurrence to produce the Argon2 password, destroying whatever structure the
  sequence had. Argon2id is a PRF in the password; non-random key selection is
  yield-neutral at best. It is also not the default and is dead on the GPU path.
- `worker`/`worker_id` — the server ignores both.
- Default-on telemetry to `woodyminer.com` carrying the wallet address.
- `xnminer`'s no-backoff flush loop, its non-WAL SQLite, and its missing SIGTERM
  handler.

---

## 5. Corrections to claims made during this audit

Recorded so they are not rediscovered as truth later.

- **"Server-side timestamp validation is real."** No. That inference came from
  Woody treating `"outside of time window"` as non-retryable
  (`main.cpp:415-419`), but that string is the XUNI-only rejection
  (`gpage.py:433,469`). XEN11 has no time restriction.
- **"Drop a held block permanently once difficulty exceeds its `m`."** No.
  Difficulty falls twice as fast as it rises; 71 of our late-accepted blocks
  were m=100 finds that waited for it to come back down. This advice would
  discard blocks we currently collect.
- **"Every XBLK find dies at the dead `xenminer.mooo.com` host."** No. That code
  path is never called, and superblocks are credited server-side without it.
- **"Difficulty adjusts in ±100 steps."** No — `+1000` / `−2000`
  (`manage_difficulty2.py:61-65`).
- **"At m=100 our batch is capped at 2048, so the card sits idle."** No. The
  low-`m` ceilings bind only at m <= 64 and the difficulty floor is 100, so the
  batch is memory-limited and the card is full. Multi-lane cannot help at any live
  difficulty. The low-difficulty loss is real (we capture 24 % of the 11x on
  offer) but its cause is the single-threaded host finalize loop, not VRAM.
- **"Our difficulty polling is too slow to catch low-difficulty windows."** No.
  `difficulty_seen` rows are written by the *submitter* on probe/401-hint
  (`manager.rs:563,689`), not by the poller, which runs every 10 s. The sparse
  rows measured probe cadence, not poll cadence.

## 6. Open questions

- Whether the live server still matches `gpage.py`. `xnminer` handles HTTP 409
  for duplicates, which the reference source never returns — suggesting prod has
  drifted, or defensive coding.
- Whether the yearly XNM halving is real on-chain. `rewards.py` back-fits genesis
  to 2023-09-13 from an observed halving; the reference server credits a flat 10
  with no halving logic anywhere.
- Whether a proxy in front of production adds a submission-age limit not present
  in the source. Our 4.9-hour delayed accept is strong evidence against, but it
  is one data point at one moment.
