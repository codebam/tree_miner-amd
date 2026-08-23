"""The wallet console's worker-command endpoint speaks the miner's vocabulary.

WHY this exists: /api/wallet/workers/{id}/command used to publish
``{"command": "restart"|"stop"|"start"|"update_config", ...}``. The miner
dispatches control messages by ``action`` (pause / resume / shutdown /
set_config), so every one of those was a silent no-op on the rig -- accepted by
the API, signed, published, and dropped. These tests pin the translation, the
400 on the two console commands that have no counterpart at all, the 503 when no
command-signing secret is configured, and -- most importantly -- that whatever
this endpoint publishes validates against proto/platform_to_worker.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from server import command_signing as cs
from server.command_signing import verify_signature
from server.routers.wallet import router as wallet_router

from .test_proto_examples import _validate, load_schema

TEST_SECRET = "wallet-endpoint-test-secret"
WALLET = "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed"
WORKER = "rig-01"


# --- Minimal server double -------------------------------------------------


class FakeAuth:
    def decode_jwt(self, token):
        return {"sub": WALLET} if token == "good" else None


class FakeWorkers:
    def __init__(self):
        self.rows = {WORKER: {"worker_id": WORKER, "eth_address": WALLET}}
        self.state_updates = []

    async def get(self, worker_id):
        return self.rows.get(worker_id)

    async def update_state(self, worker_id, state):
        self.state_updates.append((worker_id, state))


class FakeStorage:
    def __init__(self):
        self.workers = FakeWorkers()


class FakeBroker:
    """Real signing policy, no network: publishes into a list."""

    def __init__(self):
        self.published = []
        self._signer = cs.CommandSigner()

    def require_command_secret(self):
        self._signer.require_secret()

    async def publish_control(self, worker_id, payload, qos=1):
        self.published.append((worker_id, self._signer.sign(payload, worker_id)))


class FakeServer:
    def __init__(self, broker=None):
        self.auth = FakeAuth()
        self.storage = FakeStorage()
        self.broker = FakeBroker() if broker is None else broker


@pytest.fixture()
def srv(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    return FakeServer()


@pytest.fixture()
def client(srv):
    app = FastAPI()
    app.include_router(wallet_router)
    app.state.server = srv
    return TestClient(app)


def send(client, command, params=None):
    body = {"command": command}
    if params is not None:
        body["params"] = params
    return client.post(
        f"/api/wallet/workers/{WORKER}/command",
        headers={"Authorization": "Bearer good"},
        json=body,
    )


def published_body(srv):
    """The last published message, minus its auth envelope."""
    _, message = srv.broker.published[-1]
    return {k: v for k, v in message.items() if k != "auth"}


# --- Vocabulary ------------------------------------------------------------


def test_stop_publishes_shutdown(client, srv):
    res = send(client, "stop")
    assert res.status_code == 200, res.text
    assert res.json()["action"] == "shutdown"
    assert published_body(srv) == {"action": "shutdown"}


def test_update_config_publishes_set_config(client, srv):
    res = send(
        client,
        "update_config",
        {"difficulty": 1749, "address": WALLET, "prefix": "c0ffee", "block_pattern": "XEN11"},
    )
    assert res.status_code == 200, res.text
    assert published_body(srv) == {
        "action": "set_config",
        "config": {
            "difficulty": 1749,
            "address": WALLET,
            "prefix": "c0ffee",
            "block_pattern": "XEN11",
        },
    }


def test_unlist_pauses_and_list_resumes(client, srv):
    assert send(client, "unlist").status_code == 200
    assert published_body(srv) == {"action": "pause"}
    assert srv.storage.workers.state_updates[-1] == (WORKER, "SELF_MINING")

    assert send(client, "list").status_code == 200
    assert published_body(srv) == {"action": "resume"}
    assert srv.storage.workers.state_updates[-1] == (WORKER, "AVAILABLE")


def test_listing_state_never_travels_inside_config(client, srv):
    """`state` is a platform concept; the miner's set_config has no such field."""
    send(client, "list")
    assert "config" not in published_body(srv)


# --- The dropped actions ---------------------------------------------------


@pytest.mark.parametrize("command", ["restart", "start"])
def test_restart_and_start_are_rejected(client, srv, command):
    res = send(client, command)
    assert res.status_code == 400
    detail = res.json()["detail"]
    assert command in detail
    for supported in ("stop", "list", "unlist", "update_config"):
        assert supported in detail
    assert srv.broker.published == [], "a rejected command must not reach the wire"


def test_unknown_command_is_rejected(client, srv):
    assert send(client, "sudo-rm-rf").status_code == 400
    assert send(client, None).status_code == 400
    assert srv.broker.published == []


def test_unknown_params_are_rejected(client, srv):
    res = send(client, "update_config", {"payout": WALLET})
    assert res.status_code == 400
    assert "payout" in res.json()["detail"]
    assert srv.broker.published == []


def test_update_config_with_nothing_to_change_is_rejected(client, srv):
    res = send(client, "update_config", {})
    assert res.status_code == 400
    assert srv.broker.published == []


def test_invalid_state_is_rejected(client, srv):
    res = send(client, "update_config", {"state": "TURBO"})
    assert res.status_code == 400
    assert "TURBO" in res.json()["detail"]
    assert srv.storage.workers.state_updates == []


def test_difficulty_must_be_an_integer(client, srv):
    """The signer refuses floats: their JSON text is not portable."""
    res = send(client, "update_config", {"difficulty": 1749.5})
    assert res.status_code == 400
    assert srv.broker.published == []


# --- Ownership and secrets -------------------------------------------------


def test_other_wallets_worker_is_forbidden(client, srv):
    srv.storage.workers.rows[WORKER]["eth_address"] = "0x" + "11" * 20
    assert send(client, "stop").status_code == 403


def test_missing_secret_is_a_503_and_changes_nothing(client, srv, monkeypatch):
    monkeypatch.delenv(cs.SECRET_ENV_VAR, raising=False)
    res = send(client, "list")
    assert res.status_code == 503
    assert cs.SECRET_ENV_VAR in res.json()["detail"]
    assert srv.broker.published == []
    assert srv.storage.workers.state_updates == [], (
        "the database must not record a listing change the rig never heard about"
    )


def test_no_broker_is_a_503(monkeypatch):
    monkeypatch.setenv(cs.SECRET_ENV_VAR, TEST_SECRET)
    srv = FakeServer(broker=None)
    srv.broker = None
    app = FastAPI()
    app.include_router(wallet_router)
    app.state.server = srv
    res = send(TestClient(app), "stop")
    assert res.status_code == 503


# --- The published message really is on-protocol ---------------------------


ALL_COMMANDS = [
    ("stop", None),
    ("list", None),
    ("unlist", None),
    ("update_config", {"difficulty": 1749}),
    ("update_config", {"address": WALLET}),
    ("update_config", {"prefix": ""}),
    ("update_config", {"block_pattern": "XEN11"}),
    ("update_config", {"difficulty": 42, "state": "AVAILABLE"}),
]


@pytest.mark.parametrize("command,params", ALL_COMMANDS)
def test_published_command_validates_against_the_schema(client, srv, command, params):
    assert send(client, command, params).status_code == 200
    _, message = srv.broker.published[-1]
    schema = load_schema("platform_to_worker.json")
    _validate(message, schema, schema, "$")


@pytest.mark.parametrize("command,params", ALL_COMMANDS)
def test_published_command_is_signed_for_this_worker(client, srv, command, params):
    assert send(client, command, params).status_code == 200
    _, message = srv.broker.published[-1]
    assert verify_signature(message, TEST_SECRET, WORKER)
