"""
SSE Broadcaster

Fan-out broadcaster for Server-Sent Events.
Each connected SSE client gets its own asyncio.Queue (maxsize=10).
push_event() puts the serialized payload into every client queue.
subscribe() is an async generator — one instance per client connection.

Wire format (SSE spec):
  "data: {json_string}\\n\\n"    — data event (browser fires 'message')
  ": keepalive\\n\\n"            — comment, keeps connection alive through proxies
"""
import asyncio
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class SSEBroadcaster:
    def __init__(self) -> None:
        # list of mailboxes 
        self._queues: list[asyncio.Queue] = []

    async def push_event(self, payload: dict) -> None:
        """Fan out payload to all connected SSE clients. Skip slow clients (queue full)."""
        # Serialize the dict to a JSON string once — same string goes to every client
        data_str = json.dumps(payload)
        # for each mailbox
        for q in list(self._queues):
            try:
                # put_nowait = drop the message into the client's mailbox instantly.
                # If the queue is full (client is too slow to consume), raises QueueFull
                # instead of waiting — so the algorithm pipeline is never blocked.
                q.put_nowait(data_str)
            except asyncio.QueueFull:
                pass  # Drop frame for slow client — don't block pipeline

    async def subscribe(self) -> AsyncIterator[str]:
        """Async generator yielding SSE-formatted strings. One per client connection."""
        # create a dedicated mailbox for this client\
        q: asyncio.Queue = asyncio.Queue(maxsize=10)
        # register this client so push_event() will deliver to it
        self._queues.append(q)
        logger.info("SSE client subscribed. Total: %d", len(self._queues))
        try:
            # loop forever — one iteration per frame event or keepalive
            while True:
                try:
                    # wait up to 25s for the next message from push_event()
                    data_str = await asyncio.wait_for(q.get(), timeout=25.0)
                    # SSE wire format: "data: {json}\n\n"
                    yield f"data: {data_str}\n\n"
                except asyncio.TimeoutError:
                    # no frame in 25s — send heartbeat so proxies don't close the connection
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            # browser disconnected — exit cleanly
            pass
        finally:
            # remove this client's queue so push_event() stops delivering to it
            if q in self._queues:
                self._queues.remove(q)
            logger.info("SSE client unsubscribed. Total: %d", len(self._queues))


# Module-level singleton imported by routes.py
broadcaster = SSEBroadcaster()
