# XenblocksMiner MQTT Protocol Specification

## Overview

This directory contains the formal MQTT message protocol schemas for the
XenblocksMiner hashpower marketplace. All messages are JSON-encoded and
published/subscribed via MQTT with QoS 1.

## Topic Structure

All topics follow the pattern:

```
xenminer/{worker_id}/{message_type}
```

Where `worker_id` is the machine-unique identifier (`machineId` global) and
`message_type` is one of the predefined suffixes.

### Worker -> Platform (published by worker)

| Topic Suffix | Description |
|---|---|
| `register` | Worker registration with GPU capabilities |
| `heartbeat` | Periodic stats (every 30 seconds) |
| `status` | State change notifications |
| `block` | Block-found report during a lease |

### Platform -> Worker (subscribed by worker)

| Topic Suffix | Description |
|---|---|
| `task` | Lease assignment commands (`register_ack`, `assign_task`, `release`) |
| `control` | Operational commands (`pause`, `resume`, `shutdown`) |

## Message Dispatch

Platform-to-worker messages arrive on two topics and are dispatched by
different fields:

- **`task` topic**: dispatched by the `command` field -- `register_ack`,
  `assign_task`, or `release`.
- **`control` topic**: dispatched by the `action` field -- `pause`, `resume`,
  or `shutdown`.

On the C++ side, if the `command` field does not match `register_ack`,
`assign_task`, or `release`, the message is forwarded to the control handler
which reads the `action` field instead.

## Schema Files

- `signing_vectors.json` - cross-language test vectors for the command
  signing envelope (see [Command Signing](#command-signing))
- `worker_to_platform.json` - JSON Schema for all worker-published messages
- `platform_to_worker.json` - JSON Schema for all platform-published messages
- `examples/` - Example payloads for each message type:
  - `register.json` - Worker registration
  - `heartbeat.json` - Periodic heartbeat
  - `status.json` - Status update with active lease
  - `status_offline.json` - Status update going offline
  - `block_found.json` - Block discovery report
  - `register_ack_accepted.json` - Registration accepted
  - `register_ack_rejected.json` - Registration rejected
  - `assign_task.json` - Lease assignment
  - `release.json` - Lease release
  - `control_pause.json` - Pause command
  - `control_resume.json` - Resume command
  - `control_shutdown.json` - Shutdown command

Note: The example files include `_description`, `_topic`, and `_source`
metadata fields for documentation purposes. These fields are not part of the
wire protocol and would not pass strict `additionalProperties: false` schema
validation.

## Worker States

The worker progresses through a 6-state machine, plus a virtual `offline`
state used in status messages when disconnecting:

```
IDLE -> AVAILABLE -> LEASED -> MINING -> COMPLETED -> AVAILABLE
                                    \-> ERROR -> IDLE -> AVAILABLE
```

| State | Description |
|---|---|
| `IDLE` | Not connected to platform |
| `AVAILABLE` | Registered and waiting for lease assignment |
| `LEASED` | Lease assigned, preparing to mine |
| `MINING` | Actively mining for a consumer |
| `COMPLETED` | Lease completed, transitioning back |
| `ERROR` | Error state, will attempt recovery |
| `offline` | Sent in status messages when the worker is shutting down (not a state machine state) |

## Command Signing

Every platform-to-worker command carries an HMAC-SHA256 envelope. The miner
verifies it before the message is interpreted at all; a command that fails
verification is dropped, and on a rig with no shared secret configured every
command that could move money -- `assign_task`, `set_config`, `shutdown` -- is
refused outright.

The canonical implementation is the Rust verifier in
`treeminer-rs/crates/tm-platform/src/envelope.rs`. Test vectors produced by it
live in [`signing_vectors.json`](signing_vectors.json) next to this file.

### The envelope

The signer adds one `auth` object to the command JSON. Nothing else changes:

```json
{
  "command": "assign_task",
  "lease_id": "lease-7",
  "consumer_address": "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
  "auth": {
    "worker_id":  "rig-01",
    "command_id": "cmd-0002",
    "issued_at":  1700000000,
    "expires_at": 1700000300,
    "nonce":      "3b1f2c0d4e5a6b7c8d9e0f1a2b3c4d5e",
    "sig":        "<64 lowercase hex characters>"
  }
}
```

| field | type | rule |
|---|---|---|
| `worker_id` | string | the TARGET worker's machine id. `[A-Za-z0-9._-]`, 1-128 chars. Must equal the worker's own id or the command is refused (this is what stops one rig's signed command being replayed onto another). |
| `command_id` | string | issuer-unique id. `[A-Za-z0-9._-]`, 1-128 chars. |
| `issued_at` | integer | Unix epoch seconds. JSON *number*, never a string; no floats. May be at most 30 s ahead of the worker's clock. |
| `expires_at` | integer | Unix epoch seconds. Must be `> issued_at`, and `expires_at - issued_at` must be <= 900 (15 minutes). |
| `nonce` | string | random hex, 16-128 characters (>= 64 bits of entropy). Must be unique per accepted command: the worker remembers accepted nonces until they expire and rejects a repeat. |
| `sig` | string | lowercase hex HMAC-SHA256, exactly 64 characters. Compared in constant time; upper-case hex also compares equal, but emit lowercase. |

The whole published message must be at most **65536 bytes**. The size is
checked on the raw bytes before any parsing.

### What is signed

```
signing_string = "TMv1" + LF
               + worker_id  + LF
               + command_id + LF
               + issued_at  + LF        (decimal, no padding, no "+")
               + expires_at + LF        (decimal)
               + nonce      + LF
               + canonical_body
```

`LF` is a single `\n` (0x0A). `TMv1` domain-separates this MAC from any other
use of the same secret. None of the first six fields may contain `\n` -- the
charsets above guarantee that -- so the encoding is unambiguous even though the
body, which comes last, may contain anything.

`canonical_body` is **the command message with the `auth` key removed**,
serialised as JSON with:

1. **Sorted keys, at every level of nesting.** Sort by Unicode code point,
   which for UTF-8 is the same as sorting the encoded bytes.
2. **No whitespace.** Separators are exactly `,` and `:` -- no space after
   either.
3. **No non-ASCII escaping.** Text is emitted as UTF-8. `café` is six bytes,
   not `café`. This is the single most common way a Python signer gets it
   wrong: `json.dumps` escapes non-ASCII *by default*.
4. **JSON's mandatory escapes only**: `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`,
   and any other control character below 0x20 as `\u00XX` with **lowercase**
   hex digits. `/` is not escaped. U+2028 and U+2029 are not escaped.
5. **Integers verbatim**, with no exponent and no trailing `.0`. Avoid floats
   entirely: no two languages agree on their shortest representation. The
   protocol has no float-valued command field.
6. An empty body is the two characters `{}`.

The MAC is then:

```
sig = lowercase_hex( HMAC-SHA256( key = utf8(shared_secret),
                                 msg = utf8(signing_string) ) )
```

The key is the raw UTF-8 bytes of the secret string -- not hex-decoded, not
base64-decoded, not NUL-terminated. The secret reaches the miner through the
`TREEMINER_PLATFORM_COMMAND_SECRET` environment variable and nothing else.

### Reference signer (Python)

```python
import hmac, hashlib, json

def canonical_body(message: dict) -> str:
    body = {k: v for k, v in message.items() if k != "auth"}
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)

def sign(message: dict, secret: str, worker_id: str, command_id: str,
         nonce: str, issued_at: int, expires_at: int) -> dict:
    signing_string = "TMv1\n{}\n{}\n{}\n{}\n{}\n{}".format(
        worker_id, command_id, issued_at, expires_at, nonce,
        canonical_body(message))
    sig = hmac.new(secret.encode("utf-8"),
                   signing_string.encode("utf-8"),
                   hashlib.sha256).hexdigest()
    return {**message, "auth": {
        "worker_id": worker_id, "command_id": command_id,
        "issued_at": issued_at, "expires_at": expires_at,
        "nonce": nonce, "sig": sig,
    }}
```

`ensure_ascii=False` and `separators=(",", ":")` are both load-bearing; so is
`sort_keys=True`. `json.dumps` defaults would produce a different string for
almost every real command.

### Verification order

The worker checks, and stops at the first failure:

1. schema and bounds of every `auth` field (`malformed auth envelope`)
2. `worker_id` equals its own (`wrong worker id`)
3. `issued_at <= now + 30` (`issued in the future`)
4. `issued_at < expires_at <= issued_at + 900` (`invalid lifetime`)
5. `now <= expires_at` (`expired`)
6. the signature (`bad signature`)
7. the nonce has not been seen before (`replayed nonce`)

The nonce is recorded only when everything else passed, so failed attempts
cannot fill the replay cache.

### The vector file

`signing_vectors.json` carries, for each vector: the `secret`, the
`expected_worker_id` the verifier is configured with, `verify_at` (the wall
clock to verify at), the envelope fields, the `body`, the exact
`canonical_body` and `signing_string` strings, the `signature`, and the
complete `message` as published. `must_verify` says whether the miner accepts
it; `expected_status` is the exact verdict, so a negative vector pins *why* it
is rejected rather than merely that it is.

Vectors cover a minimal command, `assign_task`, a nested `set_config`, a body
with non-ASCII text and an astral-plane emoji (signed with a non-ASCII secret),
an empty body, a body just under the size cap, a body edited after signing
(negative), and a valid envelope addressed to another rig (negative).

Start with the `self_check` object at the top of the file: it is a fixed
signing string and its signature, so you can prove your HMAC before debugging
your JSON serialisation.

Regenerate the file -- never edit it by hand -- with:

```sh
cd treeminer-rs && ./rs cargo run -p tm-platform --example gen_signing_vectors
```

## Key Constants

| Constant | Value | Description |
|---|---|---|
| `PLATFORM_PREFIX_LENGTH` | 16 | Required length of key prefix (hex chars) |
| `QOS` | 1 | MQTT Quality of Service level |
| `HEARTBEAT_INTERVAL_SEC` | 30 | Heartbeat publish interval |
| `WATCHDOG_INTERVAL_SEC` | 5 | Lease expiry check interval |
| `KEEPALIVE_INTERVAL_SEC` | 60 | MQTT keep-alive interval |
