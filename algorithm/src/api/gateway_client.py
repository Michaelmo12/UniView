"""
Gateway HTTP Client
"""
import logging

import httpx

logger = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


async def init_client(base_url: str) -> None:
    global _client
    _client = httpx.AsyncClient(base_url=base_url, timeout=5.0)
    logger.info("GatewayClient initialized: %s", base_url)


async def close_client() -> None:
    global _client
    if _client:
        await _client.aclose()
        _client = None


async def post_payload(payload: dict) -> None:
    """POST a StreamPayload dict to gateway /api/internal/push. Fire-and-forget."""
    if _client is None:
        logger.warning("GatewayClient not initialized — skipping POST")
        return
    try:
        await _client.post("/api/internal/push", json=payload)
    except Exception as exc:
        logger.warning("Gateway POST failed: %s", exc)
