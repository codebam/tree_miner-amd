"""Admin router — /api/accounts, /api/settlements, /api/status, /api/workers/{id}/control, etc."""

from fastapi import APIRouter, Depends, HTTPException
from starlette.requests import Request

from server.command_signing import CommandSecretMissing, CommandSigningError
from server.deps import get_server, require_auth
from server.models import ControlRequest

router = APIRouter()


# proto/platform_to_worker.json types the control message as `{"action": ...}`
# with additionalProperties: false, and PlatformManager::handleControl only
# reads `config` for set_config. Sending `config` alongside pause/resume made
# every control message fail strict schema validation, so build the payload the
# schema actually allows.
CONFIG_BEARING_ACTIONS = {"set_config"}
CONTROL_ACTIONS = {"pause", "resume", "shutdown"} | CONFIG_BEARING_ACTIONS


def _control_payload(req: ControlRequest) -> dict:
    """Build a schema-valid control payload, or 400 on an unknown action."""
    if req.action not in CONTROL_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown control action {req.action!r}; expected one of "
                + ", ".join(sorted(CONTROL_ACTIONS))
            ),
        )
    payload = {"action": req.action}
    if req.action in CONFIG_BEARING_ACTIONS:
        payload["config"] = req.config
    elif req.config:
        raise HTTPException(
            status_code=400,
            detail=f"config is only accepted with action=set_config, not {req.action!r}",
        )
    return payload


def _command_unavailable(exc: CommandSigningError) -> HTTPException:
    """503 carrying the refusal reason. The secret itself is never in `exc`."""
    return HTTPException(status_code=503, detail=str(exc))


# The root banner and /api/status expose only aggregate counters and are
# deliberately public (the dashboard polls them anonymously).

@router.get("/")
async def root(request: Request):
    srv = get_server(request)
    return {
        "service": "XenMiner Mock Platform",
        "mqtt_port": srv.mqtt_port,
        "api_port": srv.api_port,
        "connected_workers": len(srv.broker.connected_client_ids),
        "uptime": "running",
    }


@router.get("/api/status")
async def server_status(request: Request):
    srv = get_server(request)
    return {
        "mqtt_clients": srv.broker.connected_client_ids,
        "workers": await srv.storage.workers.count(),
        "active_leases": await srv.storage.leases.count(state="active"),
        "total_blocks": await srv.storage.blocks.count(),
        "self_mined_blocks": await srv.storage.blocks.count_self_mined(),
        "total_settlements": await srv.storage.settlements.count(),
    }


@router.get("/api/accounts")
async def list_accounts(
    request: Request,
    caller: dict = Depends(require_auth("admin")),
):
    srv = get_server(request)
    accounts = await srv.accounts.list_accounts()
    sanitized = {}
    for k, v in accounts.items():
        sanitized[k] = {
            "account_id": v["account_id"],
            "role": v["role"],
            "eth_address": v["eth_address"],
            "balance": v["balance"],
        }
    return sanitized


@router.get("/api/settlements")
async def list_settlements(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    caller: dict = Depends(require_auth("admin")),
):
    srv = get_server(request)
    items = await srv.settlement.list_settlements(limit=limit, offset=offset)
    total = await srv.storage.settlements.count()
    return {"items": items, "total": total, "limit": limit, "offset": offset}


# Worker control publishes MQTT commands (config changes, shutdown) to live
# miners — operator-only. Wallet owners have their own scoped command route
# at /api/wallet/workers/{id}/command.

@router.post("/api/workers/{worker_id}/control")
async def send_control(
    request: Request,
    worker_id: str,
    req: ControlRequest,
    caller: dict = Depends(require_auth("admin")),
):
    srv = get_server(request)
    payload = _control_payload(req)
    try:
        await srv.broker.publish_control(worker_id, payload)
    except CommandSigningError as exc:
        raise _command_unavailable(exc)
    return {"status": "sent", "worker_id": worker_id, "action": req.action}


@router.post("/api/control/broadcast")
async def broadcast_control(
    request: Request,
    req: ControlRequest,
    caller: dict = Depends(require_auth("admin")),
):
    srv = get_server(request)
    payload = _control_payload(req)
    # Refuse the whole broadcast up front rather than signing for the first N
    # workers and failing partway through.
    try:
        srv.broker.require_command_secret()
    except CommandSecretMissing as exc:
        raise _command_unavailable(exc)
    workers = await srv.matcher.get_available_workers()
    sent_to = []
    for w in workers:
        wid = w["worker_id"]
        try:
            await srv.broker.publish_control(wid, payload)
        except CommandSigningError as exc:
            raise _command_unavailable(exc)
        sent_to.append(wid)
    return {"status": "sent", "workers": sent_to, "action": req.action}
