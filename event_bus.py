"""Event Bus — pub/sub for module communication. Modules never call each other directly."""

import asyncio, logging, uuid
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("event_bus")

class Event:
    def __init__(self, topic, data=None, source=""):
        self.id = uuid.uuid4().hex[:12]
        self.topic = topic
        self.data = data or {}
        self.source = source
        self.timestamp = datetime.now()
        self._consumed = False

    def __repr__(self):
        return f"Event({self.topic}, source={self.source})"

class EventBus:
    def __init__(self):
        self._subscribers = defaultdict(list)
        self._history = []
        self._max_history = 1000
        self._running = True

    def subscribe(self, topic, callback, name=""):
        """Subscribe to a topic. callback receives (event)"""
        sub_id = uuid.uuid4().hex[:8]
        self._subscribers[topic].append({
            "id": sub_id,
            "name": name or f"sub_{sub_id}",
            "callback": callback,
        })
        logger.debug(f"Subscribed {name} to '{topic}'")
        return sub_id

    def unsubscribe(self, topic, sub_id):
        self._subscribers[topic] = [
            s for s in self._subscribers.get(topic, []) if s["id"] != sub_id
        ]

    async def publish(self, topic, data=None, source=""):
        """Publish event — all subscribers get called asynchronously"""
        event = Event(topic, data, source)
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history.pop(0)

        subs = self._subscribers.get(topic, [])
        if not subs:
            subs = self._subscribers.get("*", [])  # wildcard catch-all

        tasks = []
        for sub in subs:
            tasks.append(self._safe_call(sub, event))
        if tasks:
            await asyncio.gather(*tasks)

    async def _safe_call(self, sub, event):
        try:
            if asyncio.iscoroutinefunction(sub["callback"]):
                await sub["callback"](event)
            else:
                sub["callback"](event)
        except Exception as e:
            logger.error(f"Event handler '{sub['name']}' failed: {e}")

    def history(self, topic=None, limit=20):
        events = self._history
        if topic:
            events = [e for e in events if e.topic == topic]
        return events[-limit:]

    def subscriber_count(self, topic=None):
        if topic:
            return len(self._subscribers.get(topic, []))
        return sum(len(s) for s in self._subscribers.values())

    def stop(self):
        self._running = False

    def on(self, topic, callback, name=""):
        return self.subscribe(topic, callback, name)

    def emit(self, topic, data=None, source=""):
        """Fire-and-forget alias for publish — schedules on event loop."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(self.publish(topic, data, source))
            else:
                loop.run_until_complete(self.publish(topic, data, source))
        except:
            pass

# Global singleton
_bus = None

def get_bus():
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus

def get_event_bus():
    """Alias for backward compatibility."""
    return get_bus()
