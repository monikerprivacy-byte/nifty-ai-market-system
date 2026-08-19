"""WebSocket Connection Manager — pushes live events to dashboard clients."""

import asyncio, json, logging
from datetime import datetime
from fastapi import WebSocket

logger = logging.getLogger("ws_manager")

class ConnectionManager:
    def __init__(self):
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        logger.info(f"WS client connected ({len(self._clients)} total)")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._clients.discard(ws)
        logger.info(f"WS client disconnected ({len(self._clients)} remaining)")

    async def broadcast(self, event_type: str, data: dict):
        """Send event to all connected clients."""
        payload = json.dumps({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        }, default=str)
        async with self._lock:
            dead = set()
            for ws in self._clients:
                try:
                    await ws.send_text(payload)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

    @property
    def client_count(self):
        return len(self._clients)


# Singleton
_manager = None

def get_ws_manager():
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
