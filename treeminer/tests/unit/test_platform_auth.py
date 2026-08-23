"""
test_platform_auth.py - Authentication hardening tests for the platform REST API.

These tests exercise the REAL FastAPI app (PlatformServer with real routers,
storage, and AuthService) — not hand-rolled fakes — and assert the systemic
rule: absence of credentials must never fall through an authorization check.

  - anonymous        -> 401 on every mutating / credential-sensitive route
  - invalid key      -> 401
  - wrong role       -> 403
  - correct role/key -> success (2xx, or a domain 404 past the auth gate)
  - registration cannot set a starting balance
  - default API bind is loopback
"""

import json
import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.auth import DEFAULT_ADMIN_KEY
from server.server import PlatformServer

ADMIN = {"X-API-Key": DEFAULT_ADMIN_KEY}

# The miner only accepts full EIP-55 checksummed addresses, so the platform
# rejects anything else at the /api/rent boundary.
CHECKSUMMED_ADDRESS = "0x8ba1f109551bD432803012645Ac136ddd64DBA72"
LOWERCASE_ADDRESS = CHECKSUMMED_ADDRESS.lower()


@pytest.fixture()
def client(tmp_path):
    """Real app + real services, initialized inside the TestClient event loop.

    Services are created in the startup hook (not directly in the fixture)
    because aiosqlite connections must live on the loop TestClient drives.
    """
    server = PlatformServer(enable_chain=False, db_path=str(tmp_path / "auth-test.db"))

    @server.app.on_event("startup")
    async def _startup():
        await server._init_services()

    @server.app.on_event("shutdown")
    async def _shutdown():
        await server.storage.close()

    with TestClient(server.app) as c:
        yield c


def register(client, account_id: str, role: str) -> dict:
    resp = client.post("/api/auth/register", json={"account_id": account_id, "role": role})
    assert resp.status_code == 200, resp.text
    return resp.json()


# ── Registration cannot mint funds ─────────────────────────────────────────

def test_register_ignores_client_supplied_balance(client):
    resp = client.post("/api/auth/register", json={
        "account_id": "rich-guy",
        "role": "consumer",
        "balance": 999999.0,  # must be ignored (field no longer in the model)
    })
    assert resp.status_code == 200, resp.text
    assert resp.json()["balance"] == 0.0

    # And the persisted account really has zero balance
    key = resp.json()["api_key"]
    bal = client.get("/api/accounts/rich-guy/balance", headers={"X-API-Key": key})
    assert bal.status_code == 200
    assert bal.json()["balance"] == 0.0


def test_legacy_login_route_gone(client):
    # The old /api/auth/login returned an account's API key for a bare
    # account ID (credential disclosure). It must not exist at all.
    resp = client.post("/api/auth/login", json={"account_id": "consumer-1"})
    assert resp.status_code in (404, 405)


# ── Anonymous / invalid credentials -> 401 ─────────────────────────────────

SENSITIVE_ROUTES = [
    ("GET", "/api/accounts/consumer-1/balance", None),
    ("POST", "/api/accounts/consumer-1/deposit", {"amount": 10.0}),
    ("POST", "/api/accounts/consumer-1/withdraw", {"amount": 10.0}),
    ("GET", "/api/accounts", None),
    ("GET", "/api/settlements", None),
    ("POST", "/api/workers/w1/control", {"action": "shutdown"}),
    ("POST", "/api/control/broadcast", {"action": "shutdown"}),
    ("PUT", "/api/workers/w1/pricing", {"price_per_min": 1.0}),
    ("PUT", "/api/provider/workers/w1/pricing", {"price_per_min": 1.0}),
    ("POST", "/api/rent", {"consumer_id": "consumer-1", "duration_sec": 60}),
    ("POST", "/api/stop", {"lease_id": "lease-x"}),
    ("POST", "/api/rental/start", {"consumer_id": "consumer-1", "duration_sec": 60}),
    ("POST", "/api/rental/stop", {"lease_id": "lease-x"}),
    ("GET", "/api/auth/me", None),
    ("GET", "/api/renter/stats", None),
]


@pytest.mark.parametrize("method,path,body", SENSITIVE_ROUTES,
                         ids=[f"{m} {p}" for m, p, _ in SENSITIVE_ROUTES])
def test_anonymous_gets_401(client, method, path, body):
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}: {resp.text}"


@pytest.mark.parametrize("method,path,body", SENSITIVE_ROUTES,
                         ids=[f"{m} {p}" for m, p, _ in SENSITIVE_ROUTES])
def test_invalid_api_key_gets_401(client, method, path, body):
    resp = client.request(method, path, json=body, headers={"X-API-Key": "not-a-real-key"})
    assert resp.status_code == 401, f"{method} {path} -> {resp.status_code}: {resp.text}"


# ── Wrong role -> 403 ──────────────────────────────────────────────────────

def test_consumer_cannot_use_admin_routes(client):
    key = {"X-API-Key": register(client, "c-role", "consumer")["api_key"]}
    assert client.get("/api/accounts", headers=key).status_code == 403
    assert client.get("/api/settlements", headers=key).status_code == 403
    assert client.post("/api/workers/w1/control",
                       json={"action": "shutdown"}, headers=key).status_code == 403
    assert client.post("/api/control/broadcast",
                       json={"action": "shutdown"}, headers=key).status_code == 403


def test_consumer_cannot_set_pricing(client):
    key = {"X-API-Key": register(client, "c-price", "consumer")["api_key"]}
    assert client.put("/api/workers/c-price/pricing",
                      json={"price_per_min": 1.0}, headers=key).status_code == 403
    assert client.put("/api/provider/workers/c-price/pricing",
                      json={"price_per_min": 1.0}, headers=key).status_code == 403


def test_provider_cannot_rent(client):
    key = {"X-API-Key": register(client, "p-rent", "provider")["api_key"]}
    resp = client.post("/api/rent", json={"duration_sec": 60}, headers=key)
    assert resp.status_code == 403


def test_provider_cannot_set_pricing_for_other_worker(client):
    key = {"X-API-Key": register(client, "p-own", "provider")["api_key"]}
    resp = client.put("/api/workers/someone-else/pricing",
                      json={"price_per_min": 1.0}, headers=key)
    assert resp.status_code == 403


def test_consumer_cannot_touch_other_accounts_funds(client):
    key = {"X-API-Key": register(client, "c-funds", "consumer")["api_key"]}
    assert client.get("/api/accounts/consumer-1/balance", headers=key).status_code == 403
    assert client.post("/api/accounts/consumer-1/deposit",
                       json={"amount": 10.0}, headers=key).status_code == 403
    assert client.post("/api/accounts/consumer-1/withdraw",
                       json={"amount": 10.0}, headers=key).status_code == 403


def test_consumer_cannot_rent_on_behalf_of_another(client):
    key = {"X-API-Key": register(client, "c-imp", "consumer")["api_key"]}
    resp = client.post("/api/rent",
                       json={"consumer_id": "consumer-1", "duration_sec": 60},
                       headers=key)
    assert resp.status_code == 403


# ── Correct credentials succeed ────────────────────────────────────────────

def test_admin_can_list_accounts_and_settlements(client):
    resp = client.get("/api/accounts", headers=ADMIN)
    assert resp.status_code == 200
    assert "consumer-1" in resp.json()  # default seeded account
    assert client.get("/api/settlements", headers=ADMIN).status_code == 200


def test_admin_can_send_control(client):
    resp = client.post("/api/workers/w1/control",
                       json={"action": "set_config", "config": {}}, headers=ADMIN)
    assert resp.status_code == 200
    assert resp.json()["status"] == "sent"
    resp = client.post("/api/control/broadcast",
                       json={"action": "set_config", "config": {}}, headers=ADMIN)
    assert resp.status_code == 200


def test_account_owner_can_manage_own_funds(client):
    acct = register(client, "c-self", "consumer")
    key = {"X-API-Key": acct["api_key"]}

    dep = client.post("/api/accounts/c-self/deposit", json={"amount": 25.0}, headers=key)
    assert dep.status_code == 200
    assert dep.json()["balance"] == 25.0

    bal = client.get("/api/accounts/c-self/balance", headers=key)
    assert bal.status_code == 200
    assert bal.json()["balance"] == 25.0

    wd = client.post("/api/accounts/c-self/withdraw", json={"amount": 10.0}, headers=key)
    assert wd.status_code == 200
    assert wd.json()["balance"] == 15.0


def test_authenticated_consumer_passes_rent_auth_gate(client):
    # With no workers registered the domain answer is 404 — the point is
    # that a valid consumer is not blocked by the auth layer (401/403).
    key = {"X-API-Key": register(client, "c-rent", "consumer")["api_key"]}
    resp = client.post(
        "/api/rent",
        json={"duration_sec": 60, "consumer_address": CHECKSUMMED_ADDRESS},
        headers=key,
    )
    assert resp.status_code == 404
    assert "No available workers" in resp.text


def test_provider_passes_pricing_auth_gate_for_own_worker(client):
    # Worker doesn't exist, so 404 — but not 401/403.
    key = {"X-API-Key": register(client, "p-mine", "provider")["api_key"]}
    resp = client.put("/api/workers/p-mine/pricing",
                      json={"price_per_min": 1.0}, headers=key)
    assert resp.status_code == 404


# ── Public read-only routes stay public (dashboard depends on them) ────────

def test_public_dashboard_reads_stay_open(client):
    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 200
    assert client.get("/api/workers").status_code == 200
    assert client.get("/api/marketplace").status_code == 200


# ── Network exposure default ───────────────────────────────────────────────

def test_default_api_bind_is_loopback(tmp_path):
    server = PlatformServer(enable_chain=False, db_path=str(tmp_path / "bind.db"))
    assert server.api_host == "127.0.0.1"


# ── Signed commands over the REST surface ──────────────────────────────────
#
# Every platform->worker command the API can trigger must leave the server with
# a valid `auth` envelope, and must not leave at all when no shared secret is
# configured. See server/command_signing.py for the contract.

import server.command_signing as cs  # noqa: E402
from server.command_signing import verify_signature  # noqa: E402
# Value the autouse `platform_command_secret` fixture in conftest.py installs.
TEST_COMMAND_SECRET = "unit-test-command-secret"


@pytest.fixture()
def recorded(client):
    """Capture what the server would put on the wire, past the signing gate."""
    srv = client.app.state.server
    sent = []
    original = srv.broker._publish_raw

    async def _capture(topic, payload, qos=1):
        sent.append((topic, payload, qos))
        return await original(topic, payload, qos)

    srv.broker._publish_raw = _capture
    yield sent
    srv.broker._publish_raw = original


def test_control_command_is_signed(client, recorded):
    resp = client.post("/api/workers/rig-01/control",
                       json={"action": "pause"}, headers=ADMIN)
    assert resp.status_code == 200
    topic, payload, _ = recorded[0]
    assert topic == "xenminer/rig-01/control"
    assert verify_signature(payload, TEST_COMMAND_SECRET, "rig-01")


def test_control_pause_carries_no_config_key(client, recorded):
    """proto/platform_to_worker.json types control as additionalProperties:false."""
    for action in ("pause", "resume", "shutdown"):
        recorded.clear()
        resp = client.post("/api/workers/rig-01/control",
                           json={"action": action}, headers=ADMIN)
        assert resp.status_code == 200, resp.text
        _, payload, _ = recorded[0]
        assert set(payload) == {"action", "auth"}, payload
        assert payload["action"] == action


def test_set_config_still_carries_its_config(client, recorded):
    resp = client.post("/api/workers/rig-01/control",
                       json={"action": "set_config", "config": {"difficulty": 8}},
                       headers=ADMIN)
    assert resp.status_code == 200
    _, payload, _ = recorded[0]
    assert payload["config"] == {"difficulty": 8}
    assert verify_signature(payload, TEST_COMMAND_SECRET, "rig-01")


def test_config_alongside_pause_is_rejected(client):
    resp = client.post("/api/workers/rig-01/control",
                       json={"action": "pause", "config": {"difficulty": 8}},
                       headers=ADMIN)
    assert resp.status_code == 400
    assert "set_config" in resp.text


def test_unknown_control_action_is_rejected(client):
    resp = client.post("/api/workers/rig-01/control",
                       json={"action": "self_destruct"}, headers=ADMIN)
    assert resp.status_code == 400


def test_control_refuses_to_publish_without_a_secret(client, recorded, monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    resp = client.post("/api/workers/rig-01/control",
                       json={"action": "pause"}, headers=ADMIN)
    assert resp.status_code == 503
    assert cs.SECRET_ENV_VAR in resp.text
    assert recorded == [], "an unsigned command must never reach the wire"


def test_broadcast_refuses_to_publish_without_a_secret(client, recorded, monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    resp = client.post("/api/control/broadcast",
                       json={"action": "pause"}, headers=ADMIN)
    assert resp.status_code == 503
    assert recorded == []


def test_secret_never_appears_in_a_response(client, recorded):
    for path, body in (
        ("/api/workers/rig-01/control", {"action": "pause"}),
        ("/api/control/broadcast", {"action": "pause"}),
    ):
        resp = client.post(path, json=body, headers=ADMIN)
        assert TEST_COMMAND_SECRET not in resp.text
    # Nor in the signed payload itself, only the digest derived from it.
    for _, payload, _ in recorded:
        assert TEST_COMMAND_SECRET not in json.dumps(payload)


def test_secret_never_appears_in_logs(client, recorded, caplog):
    with caplog.at_level(logging.DEBUG):
        client.post("/api/workers/rig-01/control",
                    json={"action": "set_config", "config": {"difficulty": 8}},
                    headers=ADMIN)
    assert TEST_COMMAND_SECRET not in caplog.text


def test_command_nonces_are_unique_across_requests(client, recorded):
    for _ in range(10):
        client.post("/api/workers/rig-01/control",
                    json={"action": "pause"}, headers=ADMIN)
    nonces = [p["auth"]["nonce"] for _, p, _ in recorded]
    assert len(nonces) == 10 and len(set(nonces)) == 10


# ── consumer_address must be EIP-55, the way the miner requires ────────────


def test_rent_rejects_a_lowercase_consumer_address(client):
    key = {"X-API-Key": register(client, "c-addr", "consumer")["api_key"]}
    resp = client.post("/api/rent",
                       json={"duration_sec": 60, "consumer_address": LOWERCASE_ADDRESS},
                       headers=key)
    assert resp.status_code == 400
    # The error names the form the miner wants, so the caller can fix it.
    assert CHECKSUMMED_ADDRESS in resp.text


def test_rent_rejects_a_missing_consumer_address(client):
    key = {"X-API-Key": register(client, "c-addr2", "consumer")["api_key"]}
    resp = client.post("/api/rent", json={"duration_sec": 60}, headers=key)
    assert resp.status_code == 400
    assert "consumer_address" in resp.text
