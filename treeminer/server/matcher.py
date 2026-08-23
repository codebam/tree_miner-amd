"""
matcher.py - Order matching engine.

Matches consumer rent requests to available workers and manages active leases.
Backed by StorageManager's WorkerRepo and LeaseRepo.
"""

import asyncio
import logging
import secrets
import time
import uuid
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

from server.command_signing import CommandSecretMissing
from server.eth_address import describe_address_error, is_valid_ethereum_address

if TYPE_CHECKING:
    from server.broker import MQTTBroker
    from server.account import AccountService
    from server.storage import WorkerRepo, LeaseRepo

logger = logging.getLogger("matcher")

PLATFORM_PREFIX_LENGTH = 16


class LeaseState(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MatchingEngine:
    """Matches consumer rent requests to available workers."""

    def __init__(
        self,
        broker: "MQTTBroker",
        accounts: "AccountService",
        worker_repo: "WorkerRepo",
        lease_repo: "LeaseRepo",
    ):
        self.broker = broker
        self.accounts = accounts
        self._workers = worker_repo
        self._leases = lease_repo
        self._lock = asyncio.Lock()
        # Ephemeral runtime config reported by workers (not persisted to DB)
        self._worker_config: dict[str, dict] = {}

    # -------------------------------------------------------------------
    # Worker management
    # -------------------------------------------------------------------

    async def register_worker(self, msg: dict) -> bool:
        """Register a new worker and send an acknowledgement via MQTT."""
        worker_id = msg["worker_id"]
        await self._workers.upsert(
            worker_id=worker_id,
            eth_address=msg.get("eth_address", ""),
            gpu_count=msg.get("gpu_count", 0),
            total_memory_gb=msg.get("total_memory_gb", 0),
            gpus=msg.get("gpus", []),
            version=msg.get("version", "unknown"),
            state="AVAILABLE",
        )
        # Ensure provider account exists
        await self.accounts.get_or_create_provider(worker_id, msg.get("eth_address", ""))

        # Send register_ack. A worker that never gets one stays IDLE, but that
        # is strictly better than publishing an unsigned command: refuse loudly
        # so the operator sees the missing secret instead of a silent takeover
        # surface.
        try:
            await self.broker.publish_task(
                worker_id,
                {"command": "register_ack", "accepted": True},
            )
        except CommandSecretMissing as exc:
            logger.error("Cannot acknowledge registration of %s: %s", worker_id, exc)
            return False
        logger.info("Worker %s registered (%d GPUs, %dGB)",
                     worker_id, msg.get("gpu_count", 0), msg.get("total_memory_gb", 0))
        return True

    async def update_heartbeat(self, msg: dict):
        """Update worker heartbeat and lease hashrate statistics."""
        worker_id = msg.get("worker_id", "")
        hashrate = msg.get("hashrate", 0.0)
        active_gpus = msg.get("active_gpus", 0)
        await self._workers.update_heartbeat(worker_id, hashrate, active_gpus)
        # Store ephemeral config fields reported in heartbeat
        self._worker_config[worker_id] = {
            "current_address": msg.get("address", ""),
            "current_prefix": msg.get("prefix", ""),
            "current_block_pattern": msg.get("block_pattern", ""),
        }
        # Update lease hashrate stats
        lease = await self._leases.get_active_lease_for_worker(worker_id)
        if lease and lease["state"] == "active":
            await self._leases.update_hashrate_stats(lease["lease_id"], hashrate)

    async def update_worker_state(self, msg: dict):
        """Update a worker's state from an incoming status message.

        Two shapes arrive on the status topic. Ordinary transitions carry
        `state` (proto/worker_to_platform.json `status`). The Last Will and the
        clean-disconnect notice carry `status` instead -- MqttClient.cpp builds
        `{"worker_id", "status": "offline", "timestamp"}` and the Rust port
        keeps that shape verbatim. Reading only `state` recorded a disconnecting
        worker with an empty state, so it never showed as offline anywhere.
        Accept both, and never overwrite a known state with an empty one.
        """
        worker_id = msg.get("worker_id", "")
        state = msg.get("state") or msg.get("status") or ""
        if not worker_id:
            logger.debug("Ignoring status message without a worker_id")
            return
        if not state:
            logger.debug("Ignoring status message for %s with no state", worker_id)
            return
        await self._workers.update_state(worker_id, state)
        logger.debug("Worker %s state -> %s", worker_id, state)

    async def get_available_workers(self) -> List[dict]:
        """Return all registered workers with their current status."""
        workers = await self._workers.list_all()
        return [
            {
                "worker_id": w["worker_id"],
                "eth_address": w["eth_address"],
                "gpu_count": w["gpu_count"],
                "total_memory_gb": w["total_memory_gb"],
                "gpus": w["gpus"],
                "state": w["state"],
                "hashrate": w["hashrate"],
                "active_gpus": w["active_gpus"],
                "last_heartbeat": w["last_heartbeat"],
                "price_per_min": w.get("price_per_min", 0.60),
                "min_duration_sec": w.get("min_duration_sec", 60),
                "max_duration_sec": w.get("max_duration_sec", 86400),
                "self_blocks_found": w.get("self_blocks_found", 0),
                "current_address": self._worker_config.get(w["worker_id"], {}).get("current_address", ""),
                "current_prefix": self._worker_config.get(w["worker_id"], {}).get("current_prefix", ""),
                "current_block_pattern": self._worker_config.get(w["worker_id"], {}).get("current_block_pattern", ""),
            }
            for w in workers
        ]

    # -------------------------------------------------------------------
    # Lease management
    # -------------------------------------------------------------------

    async def rent_hashpower(
        self,
        consumer_id: str,
        consumer_address: str,
        duration_sec: int = 3600,
        worker_id: Optional[str] = None,
    ) -> Optional[dict]:
        """Create a lease: match consumer to an available worker.

        Raises ValueError for a consumer_address the miner would refuse, and
        CommandSecretMissing when no command-signing secret is configured. Both
        checks run BEFORE any state is written: a lease the worker will never
        accept is worse than no lease, because the platform would bill for it.
        """
        if not is_valid_ethereum_address(consumer_address):
            raise ValueError(describe_address_error(consumer_address))
        # Fail before mutating: the assign_task publish below must be able to
        # sign, or the worker sits LEASED with no task.
        self.broker.require_command_secret()

        async with self._lock:
            # Find an available worker
            target = None
            if worker_id:
                w = await self._workers.get(worker_id)
                active_lease = await self._leases.get_active_lease_for_worker(worker_id)
                if w and w["state"] == "AVAILABLE" and active_lease is None:
                    target = w
            else:
                # Find any available worker without active lease
                all_workers = await self._workers.list_all()
                for w in all_workers:
                    if w["state"] == "AVAILABLE":
                        active_lease = await self._leases.get_active_lease_for_worker(w["worker_id"])
                        if active_lease is None:
                            target = w
                            break

            if target is None:
                logger.warning("No available workers for rent request from %s", consumer_id)
                return None

            # Generate prefix (16 hex chars)
            prefix = secrets.token_hex(PLATFORM_PREFIX_LENGTH // 2)
            lease_id = f"lease-{uuid.uuid4()}"

            # Use worker's pricing (convert price_per_min to price_per_sec)
            price_per_min = target.get("price_per_min", 0.60)
            price_per_sec = price_per_min / 60.0

            lease = await self._leases.create(
                lease_id=lease_id,
                worker_id=target["worker_id"],
                consumer_id=consumer_id,
                consumer_address=consumer_address,
                prefix=prefix,
                duration_sec=duration_sec,
                price_per_sec=price_per_sec,
            )
            await self._workers.update_state(target["worker_id"], "LEASED")

        # Send assign_task to worker (signed; see server/command_signing.py)
        await self.broker.publish_task(
            target["worker_id"],
            {
                "command": "assign_task",
                "lease_id": lease_id,
                "consumer_id": consumer_id,
                "consumer_address": consumer_address,
                "prefix": prefix,
                "duration_sec": duration_sec,
            },
        )
        logger.info("Lease %s created: worker=%s consumer=%s duration=%ds prefix=%s",
                     lease_id, target["worker_id"], consumer_id, duration_sec, prefix)
        return lease

    async def stop_lease(self, lease_id: str) -> Optional[dict]:
        """Stop a lease early by sending release to the worker."""
        # Same ordering rule as rent_hashpower: refuse before mutating.
        self.broker.require_command_secret()
        async with self._lock:
            lease = await self._leases.get(lease_id)
            if lease is None or lease["state"] != "active":
                return None
            await self._leases.update_state(lease_id, "completed", ended_at=time.time())
            await self._workers.update_state(lease["worker_id"], "AVAILABLE")

        await self.broker.publish_task(
            lease["worker_id"], {"command": "release", "lease_id": lease_id}
        )
        logger.info("Lease %s stopped (worker=%s)", lease_id, lease["worker_id"])
        # Return updated lease
        return await self._leases.get(lease_id)

    async def check_expired_leases(self) -> List[dict]:
        """Check for and complete expired leases. Returns list of newly expired."""
        expired_leases = await self._leases.find_expired()
        completed = []
        for lease in expired_leases:
            await self._leases.update_state(lease["lease_id"], "completed", ended_at=time.time())
            await self._workers.update_state(lease["worker_id"], "AVAILABLE")
            # The watchdog runs on a timer with no caller to report to, so a
            # missing secret is logged per lease rather than raised. The lease
            # is still closed and settled: expiry is time-driven and the worker
            # enforces it locally too, so the release is a courtesy nudge.
            try:
                await self.broker.publish_task(
                    lease["worker_id"],
                    {"command": "release", "lease_id": lease["lease_id"]},
                )
            except CommandSecretMissing as exc:
                logger.error(
                    "Lease %s expired but the release command could not be signed: %s",
                    lease["lease_id"], exc,
                )
            # Re-fetch with updated state
            updated = await self._leases.get(lease["lease_id"])
            completed.append(updated)
            logger.info("Lease %s expired (worker=%s, blocks=%d)",
                         lease["lease_id"], lease["worker_id"], lease["blocks_found"])
        return completed

    async def get_lease(self, lease_id: str) -> Optional[dict]:
        """Retrieve a lease by ID."""
        return await self._leases.get(lease_id)

    async def get_active_lease_for_worker(self, worker_id: str) -> Optional[dict]:
        """Return the active lease for a worker, if any."""
        return await self._leases.get_active_lease_for_worker(worker_id)

    async def list_leases(self, state: Optional[str] = None, limit: Optional[int] = None, offset: int = 0) -> List[dict]:
        """List leases, optionally filtered by state."""
        leases = await self._leases.list_all(state=state, limit=limit, offset=offset)
        return [
            {
                "lease_id": l["lease_id"],
                "worker_id": l["worker_id"],
                "consumer_id": l["consumer_id"],
                "consumer_address": l["consumer_address"],
                "prefix": l["prefix"],
                "duration_sec": l["duration_sec"],
                "state": l["state"],
                "created_at": l["created_at"],
                "ended_at": l["ended_at"],
                "blocks_found": l["blocks_found"],
                "avg_hashrate": l["avg_hashrate"],
                "elapsed_sec": l["elapsed_sec"],
            }
            for l in leases
        ]
