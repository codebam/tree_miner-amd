"""
test_command_signing.py - HMAC command-signing contract and every publish path.

Three layers of assertion:

  1. The cross-language vectors in proto/signing_vectors.json. The Rust miner
     generates them (`cargo run -p tm-platform --example gen_signing_vectors`)
     and is the authority on canonicalisation; if this server's signer and that
     file ever disagree, the miner rejects every command the platform sends.
     This is the acceptance gate -- it fails loudly, it never skips.
  2. The signer's own policy: secret from the environment only, no unsigned
     fallback, unique nonces, bounded lifetime, no secret in logs or responses.
  3. Every publish path in the server, end to end through the real broker,
     matcher, and REST routers.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import aiosqlite
import pytest
import pytest_asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server import command_signing as cs
from server.broker import COMMAND_TOPIC_SUFFIXES, MQTTBroker
from server.command_signing import (
    CommandNotSignable,
    CommandSecretMissing,
    CommandSigner,
    canonical_body,
    hmac_sha256_hex,
    sign_command,
    signing_string,
    verify_signature,
)
from server.matcher import MatchingEngine
from server.storage import LeaseRepo, SCHEMA_SQL, SCHEMA_VERSION, WorkerRepo

VECTORS_PATH = PROJECT_ROOT / "proto" / "signing_vectors.json"

TEST_SECRET = "unit-test-command-secret"


# ═══════════════════════════════════════════════════════════════════════════
# 1. Cross-language vectors (the acceptance gate)
# ═══════════════════════════════════════════════════════════════════════════


def _load_vectors() -> dict:
    """Load the vector file, failing loudly if it is absent or unreadable.

    Deliberately not a skip: a silently skipped cross-language test is exactly
    how the platform and the miner drift apart on the wire format.
    """
    if not VECTORS_PATH.exists():
        pytest.fail(
            f"{VECTORS_PATH} is missing. It is generated from the authoritative "
            "Rust implementation with "
            "`cargo run -p tm-platform --example gen_signing_vectors`. Without it "
            "nothing proves this server signs commands the miner will accept."
        )
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))


VECTORS = _load_vectors()


def _vector_ids():
    return [v["name"] for v in VECTORS["vectors"]]


def test_vector_file_describes_the_algorithm_we_implement():
    assert VECTORS["algorithm"] == "HMAC-SHA256"
    assert VECTORS["domain"] == cs.SIGNING_VERSION
    assert VECTORS["signature_encoding"] == "lowercase hex, 64 characters"
    assert VECTORS["signing_string_format"] == (
        "TMv1\\n{worker_id}\\n{command_id}\\n{issued_at}\\n{expires_at}\\n"
        "{nonce}\\n{canonical_body}"
    )


def test_self_check_isolates_the_mac_from_canonicalisation():
    """Reproduce the raw MAC first: if this fails, JSON is not the problem."""
    check = VECTORS["self_check"]
    assert hmac_sha256_hex(check["secret"], check["signing_string"]) == (
        check["signature"]
    )


def test_hmac_known_answer_vector():
    """The classic KAT the C++ and Rust suites both pin."""
    assert hmac_sha256_hex("key", "The quick brown fox jumps over the lazy dog") == (
        "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
    )


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=_vector_ids())
def test_vector_canonical_body(vector):
    assert canonical_body(vector["body"]) == vector["canonical_body"]


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=_vector_ids())
def test_vector_signing_string(vector):
    built = signing_string(
        vector["worker_id"],
        vector["command_id"],
        vector["issued_at"],
        vector["expires_at"],
        vector["nonce"],
        canonical_body(vector["body"]),
    )
    assert built == vector["signing_string"]


@pytest.mark.parametrize("vector", VECTORS["vectors"], ids=_vector_ids())
def test_vector_signature_byte_for_byte(vector):
    """Recompute each vector's digest from its own signing string.

    Every vector except the tampered one pins the digest exactly. The
    `bad signature` vector is the exception by construction: its `signature`
    was produced over the ORIGINAL body and the body was then edited, so
    recomputing over the vector's (post-edit) signing string must differ --
    that difference is the whole point of the vector.
    """
    recomputed = hmac_sha256_hex(vector["secret"], vector["signing_string"])
    if vector["expected_status"] == "bad signature":
        assert recomputed != vector["signature"]
    else:
        assert recomputed == vector["signature"]


@pytest.mark.parametrize(
    "vector", [v for v in VECTORS["vectors"] if v["must_verify"]], ids=lambda v: v["name"]
)
def test_positive_vector_message_is_reproduced_exactly(vector):
    """Our signer rebuilds the whole envelope the miner was given, field for field."""
    produced = sign_command(
        vector["body"],
        vector["secret"],
        vector["worker_id"],
        vector["command_id"],
        vector["nonce"],
        vector["issued_at"],
        vector["expires_at"],
    )
    assert produced == vector["message"]
    assert produced["auth"]["sig"] == vector["signature"]
    assert verify_signature(produced, vector["secret"], vector["expected_worker_id"])


@pytest.mark.parametrize(
    "vector",
    [v for v in VECTORS["vectors"] if not v["must_verify"]],
    ids=lambda v: v["name"],
)
def test_negative_vector_does_not_verify(vector):
    assert not verify_signature(
        vector["message"], vector["secret"], vector["expected_worker_id"]
    )


def test_vector_file_carries_a_negative_vector():
    negatives = [v for v in VECTORS["vectors"] if not v["must_verify"]]
    assert negatives, "the vector file must exercise at least one rejection"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Canonicalisation rules in their own right
# ═══════════════════════════════════════════════════════════════════════════


def test_canonical_body_sorts_keys_and_drops_whitespace():
    body = canonical_body({"z": 1, "a": 2, "m": {"y": 1, "b": 2}})
    assert body == '{"a":2,"m":{"b":2,"y":1},"z":1}'
    # Python's json.dumps defaults would produce ", " and ": " separators.
    assert ", " not in body and '": ' not in body


def test_canonical_body_emits_raw_utf8_not_escapes():
    """ensure_ascii=False is load-bearing: serde_json/nlohmann emit raw UTF-8."""
    body = canonical_body({"reason": "café ☕"})
    assert body == '{"reason":"café ☕"}'
    assert "\\u" not in body


def test_canonical_body_excludes_the_auth_envelope():
    signed = {"command": "release", "auth": {"sig": "deadbeef"}}
    assert canonical_body(signed) == '{"command":"release"}'


def test_canonical_body_refuses_floats():
    """Float JSON text is not byte-identical across implementations."""
    with pytest.raises(CommandNotSignable, match="floating-point"):
        canonical_body({"command": "set_config", "difficulty": 1.5})


# ═══════════════════════════════════════════════════════════════════════════
# 3. Secret handling
# ═══════════════════════════════════════════════════════════════════════════


def test_secret_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, "  from-env  ")
    assert cs.load_command_secret() == "from-env"
    assert cs.command_secret_available()


def test_unset_secret_is_missing(monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    assert not cs.command_secret_available()
    with pytest.raises(CommandSecretMissing):
        cs.load_command_secret()


def test_blank_secret_counts_as_unset(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, "   ")
    assert not cs.command_secret_available()


def test_signer_refuses_to_sign_without_a_secret(monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    signer = CommandSigner()
    assert not signer.secret_available()
    with pytest.raises(CommandSecretMissing) as excinfo:
        signer.sign({"command": "release"}, "rig-01")
    # The refusal names the variable to set, and nothing else.
    assert cs.SECRET_ENV_VAR in str(excinfo.value)


def test_refusal_message_never_contains_the_secret(monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    try:
        cs.load_command_secret()
    except CommandSecretMissing as exc:
        assert "unit-test" not in str(exc)


def test_signer_repr_hides_key_material(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, "super-secret-value")
    signer = CommandSigner()
    signer.sign({"command": "release"}, "rig-01")
    assert "super-secret-value" not in repr(signer)
    assert "super-secret-value" not in str(vars(signer))


# ═══════════════════════════════════════════════════════════════════════════
# 4. Signer behaviour
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture()
def signer(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    return CommandSigner()


def test_signed_envelope_has_every_field_the_miner_checks(signer):
    before = int(time.time())
    signed = signer.sign({"command": "release", "lease_id": "L1"}, "rig-01")
    auth = signed["auth"]
    assert set(auth) == {
        "worker_id", "command_id", "issued_at", "expires_at", "nonce", "sig",
    }
    assert auth["worker_id"] == "rig-01"
    assert cs.is_safe_identifier(auth["command_id"], 1, cs.MAX_ID_LEN)
    assert cs.is_hex_string(auth["nonce"], cs.MIN_NONCE_HEX_LEN, cs.MAX_NONCE_HEX_LEN)
    assert cs.is_hex_string(auth["sig"], 64, 64)
    assert auth["sig"] == auth["sig"].lower()
    assert before <= auth["issued_at"] <= int(time.time())
    assert auth["expires_at"] - auth["issued_at"] == cs.DEFAULT_LIFETIME_SEC
    # The body survives untouched beside the envelope.
    assert signed["command"] == "release" and signed["lease_id"] == "L1"
    assert verify_signature(signed, TEST_SECRET, "rig-01")


def test_signature_covers_the_body(signer):
    signed = signer.sign({"command": "assign_task", "consumer_address": "0xA"}, "rig-01")
    signed["consumer_address"] = "0xEVIL"
    assert not verify_signature(signed, TEST_SECRET, "rig-01")


def test_signature_covers_the_envelope_fields(signer):
    signed = signer.sign({"command": "release"}, "rig-01")
    signed["auth"]["expires_at"] += 600
    assert not verify_signature(signed, TEST_SECRET, "rig-01")


def test_envelope_is_bound_to_one_worker(signer):
    signed = signer.sign({"command": "release"}, "rig-01")
    assert not verify_signature(signed, TEST_SECRET, "rig-02")


def test_wrong_secret_does_not_verify(signer):
    signed = signer.sign({"command": "release"}, "rig-01")
    assert not verify_signature(signed, "some-other-secret", "rig-01")


def test_lifetime_is_bounded(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    with pytest.raises(ValueError):
        CommandSigner(lifetime_sec=cs.MAX_LIFETIME_SEC + 1)
    signer = CommandSigner()
    with pytest.raises(CommandNotSignable):
        signer.sign({"command": "release"}, "rig-01",
                    lifetime_sec=cs.MAX_LIFETIME_SEC + 1)


def test_nonces_are_unique_across_publishes(signer):
    nonces = {
        signer.sign({"command": "release", "lease_id": str(i)}, "rig-01")["auth"]["nonce"]
        for i in range(500)
    }
    assert len(nonces) == 500


def test_command_ids_are_unique_across_publishes(signer):
    ids = {
        signer.sign({"command": "release"}, "rig-01")["auth"]["command_id"]
        for _ in range(200)
    }
    assert len(ids) == 200


def test_repeating_nonce_source_is_rejected(monkeypatch):
    """A broken nonce source must fail, not silently emit replayable commands."""
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    signer = CommandSigner(nonce_factory=lambda: "00112233445566778899aabbccddeeff")
    signer.sign({"command": "release"}, "rig-01")
    with pytest.raises(CommandNotSignable, match="repeating"):
        signer.sign({"command": "release"}, "rig-01")


@pytest.mark.parametrize(
    "worker_id",
    ["", "rig/01", "rig 01", "rig\n01", "#", "x" * (cs.MAX_ID_LEN + 1)],
)
def test_unsafe_worker_ids_are_refused(signer, worker_id):
    with pytest.raises(CommandNotSignable):
        signer.sign({"command": "release"}, worker_id)


def test_signing_is_idempotent_over_a_previously_signed_message(signer):
    """Re-signing replaces the old envelope instead of signing over it."""
    once = signer.sign({"command": "release"}, "rig-01")
    twice = signer.sign(once, "rig-01")
    assert canonical_body(twice) == '{"command":"release"}'
    assert verify_signature(twice, TEST_SECRET, "rig-01")


# ═══════════════════════════════════════════════════════════════════════════
# 5. Broker publish paths
# ═══════════════════════════════════════════════════════════════════════════


class RecordingBroker(MQTTBroker):
    """Real signing policy, no sockets: records what would go on the wire."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sent = []

    async def _publish_raw(self, topic, payload, qos=1):
        self.sent.append((topic, payload, qos))


@pytest.fixture()
def broker(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    return RecordingBroker()


def test_publish_task_signs(broker):
    asyncio.run(broker.publish_task("rig-01", {"command": "release", "lease_id": "L"}))
    topic, payload, _ = broker.sent[0]
    assert topic == "xenminer/rig-01/task"
    assert verify_signature(payload, TEST_SECRET, "rig-01")


def test_publish_control_signs(broker):
    asyncio.run(broker.publish_control("rig-01", {"action": "pause"}))
    topic, payload, _ = broker.sent[0]
    assert topic == "xenminer/rig-01/control"
    assert verify_signature(payload, TEST_SECRET, "rig-01")


def test_plain_publish_refuses_command_topics(broker):
    for suffix in COMMAND_TOPIC_SUFFIXES:
        with pytest.raises(CommandNotSignable):
            asyncio.run(
                broker.publish(f"xenminer/rig-01/{suffix}", {"command": "release"})
            )
    assert broker.sent == []


def test_plain_publish_still_works_for_non_command_topics(broker):
    asyncio.run(broker.publish("xenminer/rig-01/telemetry", {"hello": "world"}))
    assert broker.sent[0][0] == "xenminer/rig-01/telemetry"


def test_broker_refuses_to_publish_without_a_secret(monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    b = RecordingBroker()
    assert not b.command_secret_available()
    with pytest.raises(CommandSecretMissing):
        asyncio.run(b.publish_task("rig-01", {"command": "release"}))
    assert b.sent == [], "nothing may reach the wire unsigned"


def test_broker_rejects_an_oversized_command(broker):
    """The worker drops anything over the cap before parsing, so fail here."""
    huge = {"action": "set_config", "config": {"note": "x" * (cs.MAX_PAYLOAD_BYTES + 1)}}
    with pytest.raises(CommandNotSignable, match="cap"):
        asyncio.run(broker.publish_control("rig-01", huge))
    assert broker.sent == []


def test_broker_rejects_unknown_command_suffix(broker):
    with pytest.raises(CommandNotSignable):
        asyncio.run(broker.publish_command("rig-01", "heartbeat", {"a": 1}))


def test_broker_nonces_unique_across_all_publish_paths(broker):
    async def _run():
        for i in range(50):
            await broker.publish_task("rig-01", {"command": "release", "lease_id": str(i)})
            await broker.publish_control("rig-01", {"action": "pause"})

    asyncio.run(_run())
    nonces = [payload["auth"]["nonce"] for _, payload, _ in broker.sent]
    assert len(nonces) == 100 and len(set(nonces)) == 100


def test_secret_never_appears_in_broker_logs(broker, caplog):
    with caplog.at_level(logging.DEBUG):
        asyncio.run(broker.publish_task("rig-01", {"command": "release"}))
    assert TEST_SECRET not in caplog.text


# ═══════════════════════════════════════════════════════════════════════════
# 6. MatchingEngine publish paths + the state/address bug fixes
# ═══════════════════════════════════════════════════════════════════════════


class FakeAccounts:
    def __init__(self):
        self.created = []

    async def get_or_create_provider(self, worker_id, eth_address):
        self.created.append((worker_id, eth_address))
        return {"account_id": worker_id, "eth_address": eth_address}


CONSUMER_ADDRESS = "0x8ba1f109551bD432803012645Ac136ddd64DBA72"


@pytest_asyncio.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.executescript(SCHEMA_SQL)
    await conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, time.time()),
    )
    await conn.commit()
    yield conn
    await conn.close()


@pytest_asyncio.fixture
async def engine(db, monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    # leases.consumer_id is a foreign key into accounts.
    now = time.time()
    await db.execute(
        "INSERT INTO accounts (account_id, role, eth_address, balance, api_key,"
        " created_at, updated_at) VALUES (?, 'consumer', ?, 100.0, '', ?, ?)",
        ("consumer-1", CONSUMER_ADDRESS, now, now),
    )
    await db.commit()
    workers = WorkerRepo(db)
    leases = LeaseRepo(db)
    broker = RecordingBroker()
    return MatchingEngine(broker, FakeAccounts(), workers, leases)


async def _register(engine, worker_id="rig-01"):
    return await engine.register_worker({
        "worker_id": worker_id,
        "eth_address": "0x" + "ab" * 20,
        "gpu_count": 1,
        "total_memory_gb": 24,
        "gpus": [],
        "version": "2.0.0",
    })


@pytest.mark.asyncio
async def test_register_ack_is_signed(engine):
    assert await _register(engine)
    topic, payload, _ = engine.broker.sent[0]
    assert topic == "xenminer/rig-01/task"
    assert payload["command"] == "register_ack"
    assert verify_signature(payload, TEST_SECRET, "rig-01")


@pytest.mark.asyncio
async def test_register_ack_refused_without_secret(engine, monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    assert await _register(engine) is False
    assert engine.broker.sent == []


@pytest.mark.asyncio
async def test_assign_task_is_signed(engine):
    await _register(engine)
    engine.broker.sent.clear()
    lease = await engine.rent_hashpower("consumer-1", CONSUMER_ADDRESS, 60)
    assert lease is not None
    topic, payload, _ = engine.broker.sent[0]
    assert topic == "xenminer/rig-01/task"
    assert payload["command"] == "assign_task"
    assert payload["consumer_address"] == CONSUMER_ADDRESS
    assert verify_signature(payload, TEST_SECRET, "rig-01")


@pytest.mark.asyncio
async def test_release_is_signed(engine):
    await _register(engine)
    lease = await engine.rent_hashpower("consumer-1", CONSUMER_ADDRESS, 60)
    engine.broker.sent.clear()
    await engine.stop_lease(lease["lease_id"])
    topic, payload, _ = engine.broker.sent[0]
    assert payload["command"] == "release"
    assert verify_signature(payload, TEST_SECRET, "rig-01")


@pytest.mark.asyncio
async def test_expired_lease_release_is_signed(engine, db):
    await _register(engine)
    lease = await engine.rent_hashpower("consumer-1", CONSUMER_ADDRESS, 1)
    engine.broker.sent.clear()
    # Backdate so the watchdog sees it as expired.
    await db.execute(
        "UPDATE leases SET created_at = ? WHERE lease_id = ?",
        (time.time() - 3600, lease["lease_id"]),
    )
    await db.commit()
    expired = await engine.check_expired_leases()
    assert len(expired) == 1
    _, payload, _ = engine.broker.sent[0]
    assert payload["command"] == "release"
    assert verify_signature(payload, TEST_SECRET, "rig-01")


@pytest.mark.asyncio
async def test_rent_without_secret_creates_no_lease(engine, monkeypatch):
    """The secret check runs before any state is written."""
    await _register(engine)
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    with pytest.raises(CommandSecretMissing):
        await engine.rent_hashpower("consumer-1", CONSUMER_ADDRESS, 60)
    assert await engine.list_leases() == []
    worker = await engine._workers.get("rig-01")
    assert worker["state"] == "AVAILABLE"


@pytest.mark.asyncio
async def test_stop_without_secret_leaves_the_lease_active(engine, monkeypatch):
    await _register(engine)
    lease = await engine.rent_hashpower("consumer-1", CONSUMER_ADDRESS, 60)
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    with pytest.raises(CommandSecretMissing):
        await engine.stop_lease(lease["lease_id"])
    assert (await engine.get_lease(lease["lease_id"]))["state"] == "active"


# ── BUG FIX: consumer_address must be EIP-55, as the miner requires ────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "address",
    [
        "",
        "0x8ba1f109551bd432803012645ac136ddd64dba72",   # all lowercase
        "0x8BA1F109551BD432803012645AC136DDD64DBA72",   # all uppercase
        "8ba1f109551bD432803012645Ac136ddd64DBA72",     # no 0x
        "0x8ba1f109551bD432803012645Ac136ddd64DBA7",    # too short
        "0x8ba1f109551bD432803012645Ac136ddd64DBAZZ",   # not hex
    ],
)
async def test_rent_rejects_addresses_the_miner_would_refuse(engine, address):
    await _register(engine)
    with pytest.raises(ValueError):
        await engine.rent_hashpower("consumer-1", address, 60)
    assert await engine.list_leases() == [], "no lease may outlive a bad address"


@pytest.mark.asyncio
async def test_rent_accepts_the_checksummed_address(engine):
    await _register(engine)
    lease = await engine.rent_hashpower("consumer-1", CONSUMER_ADDRESS, 60)
    assert lease["consumer_address"] == CONSUMER_ADDRESS


# ── BUG FIX: the offline notice carries `status`, not `state` ─────────────


@pytest.mark.asyncio
async def test_offline_notice_status_field_is_recorded(engine):
    """MqttClient's LWT publishes {"worker_id", "status": "offline", ...}."""
    await _register(engine)
    await engine.update_worker_state(
        {"worker_id": "rig-01", "status": "offline", "timestamp": int(time.time())}
    )
    assert (await engine._workers.get("rig-01"))["state"] == "offline"


@pytest.mark.asyncio
async def test_ordinary_status_message_state_field_still_wins(engine):
    await _register(engine)
    await engine.update_worker_state(
        {"worker_id": "rig-01", "state": "MINING", "timestamp": int(time.time())}
    )
    assert (await engine._workers.get("rig-01"))["state"] == "MINING"


@pytest.mark.asyncio
async def test_status_without_a_state_does_not_blank_the_worker(engine):
    await _register(engine)
    await engine.update_worker_state({"worker_id": "rig-01", "timestamp": 1})
    assert (await engine._workers.get("rig-01"))["state"] == "AVAILABLE"
