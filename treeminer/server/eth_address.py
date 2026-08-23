"""
eth_address.py - EIP-55 checksummed Ethereum address validation.

WHY the platform must care: the miner refuses any address that is not full
EIP-55 checksummed (``src/EthereumAddressValidator.cpp``, and its Rust port
``tm_core::is_valid_ethereum_address``), because the address is compacted into
the Argon2 salt -- a typo would bind every block found to an address nobody
controls. The platform used to accept whatever a consumer typed, so a lowercase
address produced a lease that the miner then silently rejected: the platform
believed a rig was mining and billed for it, the rig never started.

Validation therefore happens at the API boundary, with the same rule the miner
applies, so a bad address is a 400 on the request that introduced it rather than
a lease nobody honours.

Keccak-256 comes from ``eth_utils`` when installed (it already is, via
``eth-account`` in server/requirements.in). The pure-Python fallback keeps this
module importable in a bare environment; it is covered by a known-answer test.
"""

from __future__ import annotations

import re
from typing import Optional

_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")

try:  # pragma: no cover - exercised by whichever branch the environment has
    from eth_utils import keccak as _eth_keccak
except ImportError:  # pragma: no cover
    _eth_keccak = None


# --- Pure-Python Keccak-256 fallback ---------------------------------------
#
# NOTE: hashlib.sha3_256 is NOT Keccak-256 -- SHA-3 uses a different padding
# byte (0x06 vs 0x01), so it produces different digests and cannot be
# substituted here.

_KECCAK_ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

_KECCAK_ROTATION_OFFSETS = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

_MASK64 = (1 << 64) - 1


def _rotl64(value: int, shift: int) -> int:
    shift %= 64
    return ((value << shift) | (value >> (64 - shift))) & _MASK64 if shift else value


def _keccak_f1600(state: list) -> None:
    for round_index in range(24):
        # Theta
        c = [
            state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4]
            for x in range(5)
        ]
        d = [c[(x - 1) % 5] ^ _rotl64(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x][y] ^= d[x]

        # Rho and Pi
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                b[y][(2 * x + 3 * y) % 5] = _rotl64(
                    state[x][y], _KECCAK_ROTATION_OFFSETS[x][y]
                )

        # Chi
        for x in range(5):
            for y in range(5):
                state[x][y] = b[x][y] ^ (
                    (~b[(x + 1) % 5][y] & _MASK64) & b[(x + 2) % 5][y]
                )

        # Iota
        state[0][0] ^= _KECCAK_ROUND_CONSTANTS[round_index]


def _keccak256_pure(data: bytes) -> bytes:
    rate = 136  # 1088 bits for Keccak-256
    state = [[0] * 5 for _ in range(5)]

    # Keccak (not SHA-3) padding: 0x01 ... 0x80
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % rate != 0:
        padded.append(0x00)
    padded[-1] ^= 0x80

    for offset in range(0, len(padded), rate):
        block = padded[offset : offset + rate]
        for i in range(rate // 8):
            lane = int.from_bytes(block[i * 8 : i * 8 + 8], "little")
            state[i % 5][i // 5] ^= lane
        _keccak_f1600(state)

    out = bytearray()
    while len(out) < 32:
        for i in range(rate // 8):
            if len(out) >= 32:
                break
            out += state[i % 5][i // 5].to_bytes(8, "little")
    return bytes(out[:32])


def keccak256(data: bytes) -> bytes:
    if _eth_keccak is not None:
        return _eth_keccak(data)
    return _keccak256_pure(data)


# --- EIP-55 ----------------------------------------------------------------


def to_checksum_address(address: str) -> Optional[str]:
    """EIP-55 checksum form of a ``0x``-prefixed address, or None if malformed."""
    if not isinstance(address, str) or not _ADDRESS_RE.match(address):
        return None
    body = address[2:].lower()
    digest = keccak256(body.encode("ascii")).hex().upper()
    out = ["0x"]
    for index, char in enumerate(body):
        out.append(char.upper() if digest[index] >= "8" else char)
    return "".join(out)


def is_valid_ethereum_address(address: str) -> bool:
    """True iff ``address`` is exactly its own EIP-55 checksum form.

    Mirrors ``tm_core::is_valid_ethereum_address`` / the C++
    ``EthereumAddressValidator``: an all-lowercase or all-uppercase address is
    rejected, not normalised, because the miner rejects it too.
    """
    checksummed = to_checksum_address(address)
    return checksummed is not None and checksummed == address


def describe_address_error(address: str, field: str = "consumer_address") -> str:
    """Operator-readable reason a given address is unusable."""
    if not isinstance(address, str) or not address:
        return f"{field} is required (EIP-55 checksummed 0x-prefixed address)"
    if not _ADDRESS_RE.match(address):
        return (
            f"{field} must be a 0x-prefixed 40-hex-character Ethereum address, "
            f"got {address!r}"
        )
    expected = to_checksum_address(address)
    return (
        f"{field} must use the EIP-55 checksum form the miner requires; "
        f"{address} should be {expected}"
    )
