"""
command_signing.py - HMAC-SHA256 signing for platform -> worker MQTT commands.

WHY: the MQTT broker is a shared rendezvous point. Anyone able to publish on
``xenminer/{worker_id}/task`` or ``.../control`` could otherwise redirect a rig's
payout address, change its difficulty, or shut it down. The miner verifies an
``auth`` envelope on every command it receives; this module is the platform side
of that contract.

The wire format and the exact bytes fed to HMAC are defined by the miner, which
is the authority:

  - Rust:  treeminer-rs/crates/tm-platform/src/envelope.rs
  - C++:   src/platform/CommandEnvelope.{h,cpp}
  - Tests: proto/signing_vectors.json (cross-language known-answer vectors)

Envelope::

    {
      "command": "assign_task", ...,
      "auth": {
        "worker_id":  "<target worker's machine id>",
        "command_id": "<issuer-unique id, 1..128 chars of [A-Za-z0-9._-]>",
        "issued_at":  1700000000,
        "expires_at": 1700000060,
        "nonce":      "<random hex, 16..128 chars>",
        "sig":        "<lowercase hex HMAC-SHA256, 64 chars>"
      }
    }

Signed string (newline delimited, no trailing newline)::

    "TMv1\\n" + worker_id + "\\n" + command_id + "\\n" + issued_at + "\\n"
              + expires_at + "\\n" + nonce + "\\n" + canonical_body(msg)

``canonical_body`` is the message with ``auth`` removed, serialised compactly
with sorted keys. The C++ uses ``nlohmann::json::dump()`` and Rust uses
``serde_json`` with a ``BTreeMap`` object type; both produce sorted keys, no
whitespace, and raw UTF-8 (no ``\\uXXXX`` escaping of non-ASCII). Python's
``json.dumps`` defaults do NOT match, so the separators / sort_keys /
ensure_ascii arguments below are load-bearing -- see ``canonical_body``.

THE SECRET comes from the environment variable
``TREEMINER_PLATFORM_COMMAND_SECRET`` and nowhere else: not a config file, not a
request parameter, not a CLI flag. It is never logged and never returned in a
response. If it is unset the server refuses to publish a command rather than
silently sending an unsigned one that a secret-configured miner would reject
(and that a secret-less miner would obey).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from typing import Any, Callable, Dict, Mapping, Optional

# --- Contract constants (mirror envelope.rs / CommandEnvelope.h) ------------

#: Domain separation tag + format version. Changing this invalidates every
#: signature, so it must move in lockstep with the miner.
SIGNING_VERSION = "TMv1"

#: The only place the shared secret may come from.
SECRET_ENV_VAR = "TREEMINER_PLATFORM_COMMAND_SECRET"

#: Hard cap on envelope lifetime, enforced by the miner (`MAX_LIFETIME_SEC`).
MAX_LIFETIME_SEC = 15 * 60

#: Default lifetime. Short, because a command is delivered over an already-open
#: MQTT session within milliseconds; the window only has to cover clock skew.
DEFAULT_LIFETIME_SEC = 60

#: Nonce is hex; the miner requires 16..128 hex chars. 32 chars = 128 bits.
NONCE_HEX_LEN = 32
MIN_NONCE_HEX_LEN = 16
MAX_NONCE_HEX_LEN = 128

#: `worker_id` / `command_id` length bound enforced by the miner.
MAX_ID_LEN = 128

#: Miner refuses payloads larger than this before parsing them.
MAX_PAYLOAD_BYTES = 64 * 1024

_ID_ALLOWED = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
_HEX_ALLOWED = set("0123456789abcdefABCDEF")


# --- Errors ----------------------------------------------------------------


class CommandSigningError(RuntimeError):
    """Base class for every refusal to produce a signed command."""


class CommandSecretMissing(CommandSigningError):
    """No shared secret configured, so no command may be published."""

    def __init__(self, message: Optional[str] = None):
        super().__init__(
            message
            or (
                "Refusing to publish an unsigned platform command: the shared "
                f"command secret is not configured. Set {SECRET_ENV_VAR} in the "
                "server's environment (never in a config file, request body, or "
                "command line) and restart. Miners configured with a secret "
                "reject unsigned commands; miners without one would obey them."
            )
        )


class CommandNotSignable(CommandSigningError):
    """The message or its addressing cannot be canonicalised / signed."""


# --- Validation helpers (mirror envelope.rs) -------------------------------


def is_safe_identifier(value: str, min_len: int = 1, max_len: int = MAX_ID_LEN) -> bool:
    """True iff ``value`` is ``[A-Za-z0-9._-]`` only and within the bounds.

    Conservative on purpose: identifiers travel into MQTT topics and log lines,
    so an embedded ``/`` or newline would be topic injection or log forging.
    """
    return (
        isinstance(value, str)
        and min_len <= len(value) <= max_len
        and all(c in _ID_ALLOWED for c in value)
    )


def is_hex_string(value: str, min_len: int, max_len: int) -> bool:
    return (
        isinstance(value, str)
        and min_len <= len(value) <= max_len
        and all(c in _HEX_ALLOWED for c in value)
    )


# --- Canonicalisation and MAC ----------------------------------------------


def _reject_uncanonicalisable(node: Any, path: str = "$") -> None:
    """Refuse values whose JSON text is not identical across languages.

    Floats are the trap: Rust's ``serde_json`` (ryu) and Python's ``repr`` agree
    on most values but not on all exponent forms, so a float in the body could
    produce a signature the miner recomputes differently. Since no command in
    ``proto/platform_to_worker.json`` carries a float, refusing one is strictly
    better than emitting a command the miner will silently reject.
    """
    if isinstance(node, bool) or node is None or isinstance(node, str):
        return
    if isinstance(node, int):
        return
    if isinstance(node, float):
        raise CommandNotSignable(
            f"Cannot sign a command containing a floating-point value at {path}: "
            "float JSON text is not byte-identical across the platform and miner "
            "implementations. Send an integer or a string instead."
        )
    if isinstance(node, Mapping):
        for key, value in node.items():
            if not isinstance(key, str):
                raise CommandNotSignable(
                    f"Cannot sign a command with a non-string object key at {path}"
                )
            _reject_uncanonicalisable(value, f"{path}.{key}")
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            _reject_uncanonicalisable(value, f"{path}[{index}]")
        return
    raise CommandNotSignable(
        f"Cannot sign a command containing a {type(node).__name__} at {path}"
    )


def canonical_body(message: Mapping[str, Any]) -> str:
    """The message minus ``auth``, serialised the way the miner re-serialises it.

    Every argument here matters:

    - ``sort_keys=True``   -- nlohmann::json and serde_json's ``Map`` are both
      ordered maps, so the miner always sees keys sorted.
    - ``separators=(",", ":")`` -- both dump compactly; Python's default adds a
      space after ``,`` and ``:``.
    - ``ensure_ascii=False`` -- both emit raw UTF-8; Python's default escapes
      non-ASCII as ``\\uXXXX``, which is a different byte string.
    """
    if not isinstance(message, Mapping):
        raise CommandNotSignable("A command must be a JSON object")
    body = {k: v for k, v in message.items() if k != "auth"}
    _reject_uncanonicalisable(body)
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def signing_string(
    worker_id: str,
    command_id: str,
    issued_at: int,
    expires_at: int,
    nonce: str,
    body: str,
) -> str:
    """The exact byte string the HMAC covers.

    None of the leading fields may contain ``\\n`` (the identifier and hex
    charsets guarantee that), so the newline framing is unambiguous even though
    the trailing body may contain escaped newlines.
    """
    return (
        f"{SIGNING_VERSION}\n{worker_id}\n{command_id}\n"
        f"{issued_at}\n{expires_at}\n{nonce}\n{body}"
    )


def hmac_sha256_hex(secret: str, data: str) -> str:
    """Lowercase-hex HMAC-SHA256 over the UTF-8 encoding of ``data``."""
    return hmac.new(
        secret.encode("utf-8"), data.encode("utf-8"), hashlib.sha256
    ).hexdigest()


# --- Secret handling -------------------------------------------------------


def load_command_secret(env: Optional[Mapping[str, str]] = None) -> str:
    """Read the shared secret from the environment.

    Raises :class:`CommandSecretMissing` when unset or blank. The value is
    returned to the caller and never logged, stored on an object, or echoed
    into an API response.
    """
    source = os.environ if env is None else env
    secret = (source.get(SECRET_ENV_VAR) or "").strip()
    if not secret:
        raise CommandSecretMissing()
    return secret


def command_secret_available(env: Optional[Mapping[str, str]] = None) -> bool:
    """True iff a non-blank secret is configured. Never reveals the value."""
    try:
        load_command_secret(env)
    except CommandSecretMissing:
        return False
    return True


# --- Signer ----------------------------------------------------------------


def _default_command_id(message: Mapping[str, Any]) -> str:
    """A per-command id derived from the command name plus a random suffix.

    The name makes an operator's log readable; the suffix makes the id unique
    per issuance, which is what the miner's replay bookkeeping keys off along
    with the nonce.
    """
    label = message.get("command") or message.get("action") or "cmd"
    if not isinstance(label, str):
        label = "cmd"
    cleaned = "".join(c if c in _ID_ALLOWED else "-" for c in label)[:32].strip("-")
    return f"{cleaned or 'cmd'}-{uuid.uuid4().hex}"[:MAX_ID_LEN]


class CommandSigner:
    """Produces the ``auth`` envelope for platform -> worker commands.

    The secret is *not* held on the instance: it is read from the environment on
    every signature, so rotating it needs no object surgery and so no long-lived
    object holds key material that could be pickled, repr'd, or logged.
    """

    def __init__(
        self,
        *,
        lifetime_sec: int = DEFAULT_LIFETIME_SEC,
        secret_loader: Callable[[], str] = load_command_secret,
        clock: Callable[[], float] = time.time,
        nonce_factory: Optional[Callable[[], str]] = None,
        command_id_factory: Optional[Callable[[Mapping[str, Any]], str]] = None,
    ):
        if not 1 <= lifetime_sec <= MAX_LIFETIME_SEC:
            raise ValueError(
                f"lifetime_sec must be in 1..{MAX_LIFETIME_SEC}, got {lifetime_sec}"
            )
        self.lifetime_sec = int(lifetime_sec)
        self._secret_loader = secret_loader
        self._clock = clock
        self._nonce_factory = nonce_factory or (
            lambda: secrets.token_hex(NONCE_HEX_LEN // 2)
        )
        self._command_id_factory = command_id_factory or _default_command_id
        # Guards against a nonce_factory that is not actually unique (a fixed
        # test double, or an exhausted entropy source). Bounded so it cannot
        # grow without limit on a long-running server.
        self._recent_nonces: "list[str]" = []
        self._recent_nonce_set: "set[str]" = set()
        self._recent_nonce_capacity = 4096

    def __repr__(self) -> str:  # pragma: no cover - trivial
        # Deliberately says nothing about the secret.
        return f"<CommandSigner lifetime={self.lifetime_sec}s>"

    # -- secret gate --------------------------------------------------------

    def secret_available(self) -> bool:
        try:
            self._secret_loader()
        except CommandSecretMissing:
            return False
        return True

    def require_secret(self) -> None:
        """Raise :class:`CommandSecretMissing` unless a secret is configured.

        Call this *before* mutating state (creating a lease, flipping a worker
        state) so a missing secret fails the whole operation instead of leaving
        the platform and the rig disagreeing about reality.
        """
        self._secret_loader()

    # -- signing ------------------------------------------------------------

    def _fresh_nonce(self) -> str:
        for _ in range(8):
            nonce = self._nonce_factory()
            if not is_hex_string(nonce, MIN_NONCE_HEX_LEN, MAX_NONCE_HEX_LEN):
                raise CommandNotSignable(
                    f"nonce must be {MIN_NONCE_HEX_LEN}..{MAX_NONCE_HEX_LEN} hex "
                    "characters"
                )
            if nonce not in self._recent_nonce_set:
                self._remember_nonce(nonce)
                return nonce
        raise CommandNotSignable(
            "Could not generate a unique nonce; the nonce source is repeating"
        )

    def _remember_nonce(self, nonce: str) -> None:
        self._recent_nonces.append(nonce)
        self._recent_nonce_set.add(nonce)
        while len(self._recent_nonces) > self._recent_nonce_capacity:
            self._recent_nonce_set.discard(self._recent_nonces.pop(0))

    def sign(
        self,
        message: Mapping[str, Any],
        worker_id: str,
        *,
        command_id: Optional[str] = None,
        lifetime_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a copy of ``message`` with a valid ``auth`` envelope attached.

        Raises :class:`CommandSecretMissing` when no secret is configured and
        :class:`CommandNotSignable` when the message or addressing is invalid.
        """
        secret = self._secret_loader()

        if not is_safe_identifier(worker_id, 1, MAX_ID_LEN):
            raise CommandNotSignable(
                "worker_id must be 1..%d characters of [A-Za-z0-9._-]" % MAX_ID_LEN
            )

        cid = command_id if command_id is not None else self._command_id_factory(message)
        if not is_safe_identifier(cid, 1, MAX_ID_LEN):
            raise CommandNotSignable(
                "command_id must be 1..%d characters of [A-Za-z0-9._-]" % MAX_ID_LEN
            )

        window = self.lifetime_sec if lifetime_sec is None else int(lifetime_sec)
        if not 1 <= window <= MAX_LIFETIME_SEC:
            raise CommandNotSignable(
                f"lifetime must be in 1..{MAX_LIFETIME_SEC} seconds, got {window}"
            )

        body = canonical_body(message)
        nonce = self._fresh_nonce()
        issued_at = int(self._clock())
        expires_at = issued_at + window

        sig = hmac_sha256_hex(
            secret,
            signing_string(worker_id, cid, issued_at, expires_at, nonce, body),
        )

        signed = {k: v for k, v in message.items() if k != "auth"}
        signed["auth"] = {
            "worker_id": worker_id,
            "command_id": cid,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "nonce": nonce,
            "sig": sig,
        }
        return signed


def sign_command(
    message: Mapping[str, Any],
    secret: str,
    worker_id: str,
    command_id: str,
    nonce: str,
    issued_at: int,
    expires_at: int,
) -> Dict[str, Any]:
    """Deterministic signer with every field supplied.

    Mirrors ``tm_platform::envelope::sign_command`` argument for argument; used
    by the cross-language vector tests and available for tooling that needs a
    reproducible envelope. Production code uses :class:`CommandSigner`, which
    sources the secret from the environment and the nonce from the OS.
    """
    body = canonical_body(message)
    sig = hmac_sha256_hex(
        secret,
        signing_string(worker_id, command_id, issued_at, expires_at, nonce, body),
    )
    signed = {k: v for k, v in message.items() if k != "auth"}
    signed["auth"] = {
        "worker_id": worker_id,
        "command_id": command_id,
        "issued_at": int(issued_at),
        "expires_at": int(expires_at),
        "nonce": nonce,
        "sig": sig,
    }
    return signed


def verify_signature(
    message: Mapping[str, Any], secret: str, expected_worker_id: str
) -> bool:
    """Signature-only check against the same canonicalisation, for tests.

    The miner is the real verifier and additionally enforces the time window and
    replay cache; this exists so the platform's own tests can prove a negative
    vector does not verify without reimplementing the whole verifier.
    """
    auth = message.get("auth") if isinstance(message, Mapping) else None
    if not isinstance(auth, Mapping):
        return False
    worker_id = auth.get("worker_id")
    command_id = auth.get("command_id")
    nonce = auth.get("nonce")
    sig = auth.get("sig")
    issued_at = auth.get("issued_at")
    expires_at = auth.get("expires_at")
    if not (
        isinstance(worker_id, str)
        and isinstance(command_id, str)
        and isinstance(nonce, str)
        and isinstance(sig, str)
        and isinstance(issued_at, int)
        and not isinstance(issued_at, bool)
        and isinstance(expires_at, int)
        and not isinstance(expires_at, bool)
    ):
        return False
    if worker_id != expected_worker_id:
        return False
    if not is_hex_string(sig, 64, 64):
        return False
    expected = hmac_sha256_hex(
        secret,
        signing_string(
            worker_id, command_id, issued_at, expires_at, nonce, canonical_body(message)
        ),
    )
    return hmac.compare_digest(sig.lower(), expected)
