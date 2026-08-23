"""Wallet router — /api/wallet/* endpoints for history and achievements."""

import time
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query
from starlette.requests import Request

from server.command_signing import CommandSecretMissing, CommandSigningError
from server.deps import get_server

router = APIRouter()

OFFLINE_THRESHOLD = 90

# The miner dispatches control messages by `action`, not by `command`: see
# proto/platform_to_worker.json and tm_platform::proto::command_from_value.
# This endpoint used to publish {"command": "restart"|"stop"|"start"|
# "update_config"}, none of which the miner has ever recognised -- every one of
# them was a silent no-op on the rig. These are the mappings that have a real
# counterpart:
#
#   stop        -> {"action": "shutdown"}      (signed; kills mining)
#   unlist      -> {"action": "pause"}         (ends any lease, back to self-mining)
#   list        -> {"action": "resume"}        (re-register, become AVAILABLE)
#   update_config -> {"action": "set_config", "config": {...}}
#
# `restart` and `start` are gone: a shut-down miner is not connected to the
# broker, so "start" cannot be a broker command, and the miner has no restart
# action. They are rejected with a 400 naming what is supported rather than
# accepted and dropped.
WALLET_COMMANDS = {"stop", "list", "unlist", "update_config"}

#: The only keys `set_config` accepts, from tm_platform::proto::SetConfig.
#: `state` is a platform-side listing concept and is deliberately NOT one of
#: them -- sending it inside `config` would fail schema validation on the way
#: out and be ignored by the miner on the way in.
CONFIG_FIELDS = ("difficulty", "address", "prefix", "block_pattern")

VALID_WORKER_STATES = {"SELF_MINING", "AVAILABLE", "LEASED"}

#: Marketplace listing state -> the control action that puts the rig there.
STATE_ACTION = {"AVAILABLE": "resume", "SELF_MINING": "pause"}


def _control_payload(command: str, params: dict) -> dict:
    """Translate a console command into a schema-valid control message.

    Raises HTTPException(400) for anything the miner could not act on, so an
    operator gets told instead of watching a command vanish.
    """
    if command not in WALLET_COMMANDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported command {command!r}; supported commands are "
                + ", ".join(sorted(WALLET_COMMANDS))
            ),
        )

    unknown = set(params) - set(CONFIG_FIELDS) - {"state"}
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported params {sorted(unknown)}; a worker command accepts "
                "state and " + ", ".join(CONFIG_FIELDS)
            ),
        )

    if command == "stop":
        if params:
            raise HTTPException(
                status_code=400, detail="stop takes no params"
            )
        return {"action": "shutdown"}

    config = {k: params[k] for k in CONFIG_FIELDS if k in params}
    if config:
        # difficulty must be a JSON integer: the signer refuses floats outright
        # (their JSON text is not byte-identical across implementations) and the
        # miner's parser does not coerce strings.
        if "difficulty" in config and (
            isinstance(config["difficulty"], bool)
            or not isinstance(config["difficulty"], int)
        ):
            raise HTTPException(
                status_code=400, detail="difficulty must be an integer"
            )
        for key in ("address", "prefix", "block_pattern"):
            if key in config and not isinstance(config[key], str):
                raise HTTPException(
                    status_code=400, detail=f"{key} must be a string"
                )
        return {"action": "set_config", "config": config}

    # No config fields: this is purely a listing change, which on the rig means
    # pause (stop offering hashpower, self-mine) or resume (register and become
    # available).
    state = params.get("state")
    if state not in STATE_ACTION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"{command} needs either a config field ("
                + ", ".join(CONFIG_FIELDS)
                + ") or state in " + ", ".join(sorted(STATE_ACTION))
            ),
        )
    return {"action": STATE_ACTION[state]}


async def _get_wallet_address(srv, authorization: str) -> str:
    """Extract wallet address from JWT."""
    if not authorization.startswith("Bearer ") or not srv.auth:
        raise HTTPException(status_code=401, detail="Wallet authentication required")
    claims = srv.auth.decode_jwt(authorization[7:])
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return claims["sub"]


@router.get("/api/wallet/history")
async def wallet_history(
    request: Request,
    period: str = Query(default="30d", pattern="^(24h|7d|30d)$"),
    authorization: str = Header(default=""),
):
    """Get historical snapshots for the connected wallet (always hourly)."""
    srv = get_server(request)
    address = await _get_wallet_address(srv, authorization)

    hours_map = {"24h": 24, "7d": 168, "30d": 720}
    hours = hours_map.get(period, 720)

    snapshots = await srv.storage.wallet_snapshots.query(
        eth_address=address,
        hours=hours,
        interval_type="hourly",
    )

    return {
        "address": address,
        "period": period,
        "count": len(snapshots),
        "data": snapshots,
    }


@router.get("/api/wallet/achievements")
async def wallet_achievements(
    request: Request,
    authorization: str = Header(default=""),
):
    """Get achievement stats for the connected wallet."""
    srv = get_server(request)
    address = await _get_wallet_address(srv, authorization)

    # Get achievements from snapshots
    achievements = await srv.storage.wallet_snapshots.get_achievements(address)

    # Also get current live stats
    now = time.time()
    workers = await srv.storage.workers.list_all()
    my_workers = [w for w in workers if w.get("eth_address", "").lower() == address.lower()]

    current_hashrate = 0.0
    online_count = 0
    for w in my_workers:
        if (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD:
            online_count += 1
            current_hashrate += w.get("hashrate", 0)

    # Get total blocks from workers (more accurate than snapshots)
    total_blocks = sum(w.get("self_blocks_found", 0) for w in my_workers)

    # Calculate mining time
    total_uptime_sec = sum(w.get("total_online_sec", 0) for w in my_workers)
    mining_days = total_uptime_sec / 86400

    return {
        "address": address,
        "current_hashrate": round(current_hashrate, 2),
        "online_workers": online_count,
        "total_workers": len(my_workers),
        "total_blocks": total_blocks,
        "peak_hashrate": achievements.get("peak_hashrate", 0),
        "total_earnings": achievements.get("total_earnings", 0),
        "mining_days": round(mining_days, 1),
        "first_seen": achievements.get("first_seen"),
    }


@router.get("/api/wallet/stats")
async def wallet_stats(
    request: Request,
    authorization: str = Header(default=""),
):
    """Get current stats for the connected wallet (real-time)."""
    srv = get_server(request)
    address = await _get_wallet_address(srv, authorization)
    now = time.time()

    workers = await srv.storage.workers.list_all()
    my_workers = [w for w in workers if w.get("eth_address", "").lower() == address.lower()]

    online_count = 0
    total_hashrate = 0.0
    total_gpus = 0
    total_blocks = 0

    for w in my_workers:
        is_online = (now - w["last_heartbeat"]) < OFFLINE_THRESHOLD
        if is_online:
            online_count += 1
            total_hashrate += w.get("hashrate", 0)
        total_gpus += w.get("gpu_count", 0)
        total_blocks += w.get("self_blocks_found", 0)

    # Get earnings from settlements
    settlements = await srv.storage.settlements.list_all()
    worker_ids = {w["worker_id"] for w in my_workers}
    total_earnings = sum(
        s.get("provider_payout", 0) for s in settlements if s.get("worker_id") in worker_ids
    )

    return {
        "address": address,
        "online_workers": online_count,
        "total_workers": len(my_workers),
        "total_hashrate": round(total_hashrate, 2),
        "total_gpus": total_gpus,
        "total_blocks": total_blocks,
        "total_earnings": round(total_earnings, 4),
    }


@router.post("/api/wallet/workers/{worker_id}/command")
async def send_worker_command(
    request: Request,
    worker_id: str,
    authorization: str = Header(default=""),
):
    """Send a remote control command to a worker."""
    srv = get_server(request)
    address = await _get_wallet_address(srv, authorization)

    # Verify worker belongs to this wallet
    worker = await srv.storage.workers.get(worker_id)
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")
    if worker.get("eth_address", "").lower() != address.lower():
        raise HTTPException(status_code=403, detail="Worker does not belong to this wallet")

    # Get command from request body
    body = await request.json()
    command = body.get("command")
    params = dict(body.get("params") or {})

    # Resolve the list/unlist aliases to the state they mean, then let
    # _control_payload turn the whole thing into a control message.
    if command == "list":
        params["state"] = "AVAILABLE"
    elif command == "unlist":
        params["state"] = "SELF_MINING"

    new_state = params.get("state")
    if new_state is not None and new_state not in VALID_WORKER_STATES:
        raise HTTPException(status_code=400, detail=f"Invalid state: {new_state}")

    payload = _control_payload(command, params)

    if not srv.broker:
        raise HTTPException(status_code=503, detail="Control service unavailable")

    # Check the secret BEFORE touching the database: a missing secret must fail
    # the whole operation rather than leave the platform believing it changed a
    # rig that never heard about it.
    try:
        srv.broker.require_command_secret()
    except CommandSecretMissing as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if new_state:
        await srv.storage.workers.update_state(worker_id, new_state)

    try:
        await srv.broker.publish_control(worker_id, payload)
    except CommandSigningError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {
        "status": "sent",
        "worker_id": worker_id,
        "command": command,
        "action": payload["action"],
    }


@router.get("/api/wallet/share")
async def wallet_share_data(
    request: Request,
    authorization: str = Header(default=""),
):
    """Get data for generating a shareable achievement card."""
    srv = get_server(request)
    address = await _get_wallet_address(srv, authorization)

    achievements = await srv.storage.wallet_snapshots.get_achievements(address)

    # Get current stats
    workers = await srv.storage.workers.list_all()
    my_workers = [w for w in workers if w.get("eth_address", "").lower() == address.lower()]
    total_blocks = sum(w.get("self_blocks_found", 0) for w in my_workers)
    total_uptime_sec = sum(w.get("total_online_sec", 0) for w in my_workers)

    return {
        "address": address[:6] + "..." + address[-4:] if len(address) > 12 else address,
        "full_address": address,
        "total_blocks": total_blocks,
        "peak_hashrate": achievements.get("peak_hashrate", 0),
        "total_earnings": achievements.get("total_earnings", 0),
        "mining_hours": round(total_uptime_sec / 3600, 1),
        "worker_count": len(my_workers),
        "generated_at": time.time(),
    }
