"""
ROX Proven Edge Engine v3.0 - Event Bus
======================================
Event-driven architecture with pub/sub pattern.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Awaitable
from collections import defaultdict
import json


class EventType(Enum):
    """Types of market events"""
    # Price events
    TICK_UPDATE = "TICK_UPDATE"
    QUOTE_UPDATE = "QUOTE_UPDATE"
    OHLCV_UPDATE = "OHLCV_UPDATE"
    
    # Volume events
    VOLUME_SPIKE = "VOLUME_SPIKE"
    LARGE_TRADE = "LARGE_TRADE"
    
    # Flow events
    FII_FLOW_UPDATE = "FII_FLOW_UPDATE"
    DII_FLOW_UPDATE = "DII_FLOW_UPDATE"
    BLOCK_DEAL = "BLOCK_DEAL"
    
    # Derivatives events
    OPTIONS_OI_CHANGE = "OPTIONS_OI_CHANGE"
    PCR_CHANGE = "PCR_CHANGE"
    IV_SPIKE = "IV_SPIKE"
    
    # Sentiment events
    NEWS_EVENT = "NEWS_EVENT"
    SOCIAL_SENTIMENT = "SOCIAL_SENTIMENT"
    ANALYST_RATING = "ANALYST_RATING"
    
    # System events
    MARKET_OPEN = "MARKET_OPEN"
    MARKET_CLOSE = "MARKET_CLOSE"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    
    # Agent events
    AGENT_SIGNAL = "AGENT_SIGNAL"
    CONSENSUS_UPDATE = "CONSENSUS_UPDATE"
    TRADE_SIGNAL = "TRADE_SIGNAL"
    
    # Risk events
    RISK_ALERT = "RISK_ALERT"
    POSITION_UPDATE = "POSITION_UPDATE"
    DRAWDOWN_ALERT = "DRAWDOWN_ALERT"


@dataclass
class Event:
    """Event data structure"""
    event_type: EventType
    timestamp: datetime
    source: str
    data: Dict[str, Any]
    priority: int = 0  # Higher = more urgent
    correlation_id: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "priority": self.priority,
            "correlation_id": self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Event":
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            data=data["data"],
            priority=data.get("priority", 0),
            correlation_id=data.get("correlation_id", "")
        )
    
    def __lt__(self, other: "Event") -> bool:
        """Compare by priority for queue ordering"""
        return self.priority < other.priority


class EventHandler(ABC):
    """Abstract event handler"""
    
    @abstractmethod
    async def handle(self, event: Event) -> bool:
        """Handle event, return success status"""
        pass


@dataclass
class Subscription:
    """Subscription details"""
    subscriber_id: str
    event_types: Set[EventType]
    callback: Callable[[Event], Awaitable[None]]
    filter_func: Optional[Callable[[Event], bool]] = None
    active: bool = True


class EventBus:
    """
    Central event bus for pub/sub messaging.
    
    Features:
    - Async event processing
    - Event filtering
    - Priority queuing
    - Subscriber management
    - Event history
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("EventBus")
        
        # Subscribers indexed by event type
        self._subscribers: Dict[EventType, List[Subscription]] = defaultdict(list)
        self._all_subscribers: List[Subscription] = []
        
        # Event queues
        self._event_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._history: List[Event] = []
        self._max_history = config.get("max_history", 10000)
        
        # State
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        self._subscriber_counter = 0
    
    async def start(self):
        """Start event processing"""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_events())
        self.logger.info("Event bus started")
    
    async def stop(self):
        """Stop event processing"""
        self._running = False
        
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Event bus stopped")
    
    def subscribe(self, event_types: List[EventType], 
                  callback: Callable[[Event], Awaitable[None]],
                  filter_func: Callable[[Event], bool] = None) -> str:
        """
        Subscribe to events.
        
        Args:
            event_types: List of event types to subscribe to
            callback: Async function to call when event received
            filter_func: Optional filter function
            
        Returns:
            Subscriber ID for unsubscribing
        """
        self._subscriber_counter += 1
        subscriber_id = f"sub_{self._subscriber_counter}"
        
        subscription = Subscription(
            subscriber_id=subscriber_id,
            event_types=set(event_types),
            callback=callback,
            filter_func=filter_func
        )
        
        # Add to type-specific subscribers
        for event_type in event_types:
            self._subscribers[event_type].append(subscription)
        
        self.logger.info(f"Subscriber {subscriber_id} registered for {[e.value for e in event_types]}")
        return subscriber_id
    
    def subscribe_all(self, callback: Callable[[Event], Awaitable[None]]) -> str:
        """Subscribe to all events"""
        self._subscriber_counter += 1
        subscriber_id = f"sub_all_{self._subscriber_counter}"
        
        subscription = Subscription(
            subscriber_id=subscriber_id,
            event_types=set(),  # Empty means all
            callback=callback
        )
        
        self._all_subscribers.append(subscription)
        return subscriber_id
    
    def unsubscribe(self, subscriber_id: str) -> bool:
        """Unsubscribe from events"""
        # Remove from type-specific subscribers
        for event_type in list(self._subscribers.keys()):
            self._subscribers[event_type] = [
                s for s in self._subscribers[event_type]
                if s.subscriber_id != subscriber_id
            ]
        
        # Remove from all subscribers
        self._all_subscribers = [
            s for s in self._all_subscribers
            if s.subscriber_id != subscriber_id
        ]
        
        self.logger.info(f"Unsubscribed {subscriber_id}")
        return True
    
    async def publish(self, event: Event):
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        if not self._running:
            self.logger.warning("Event bus not running, event dropped")
            return
        
        # Add to queue for processing
        await self._event_queue.put((event.priority, event))
    
    async def publish_batch(self, events: List[Event]):
        """Publish multiple events"""
        for event in events:
            await self.publish(event)
    
    async def _process_events(self):
        """Process events from queue"""
        while self._running:
            try:
                # Get event from queue
                priority, event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                # Store in history
                self._history.append(event)
                if len(self._history) > self._max_history:
                    self._history.pop(0)
                
                # Dispatch to subscribers
                await self._dispatch_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Event processing error: {e}")
    
    async def _dispatch_event(self, event: Event):
        """Dispatch event to relevant subscribers"""
        # Get type-specific subscribers
        subscribers = list(self._subscribers.get(event.event_type, []))
        
        # Add all-event subscribers
        subscribers.extend(self._all_subscribers)
        
        # Dispatch to each subscriber
        tasks = []
        for subscription in subscribers:
            if not subscription.active:
                continue
            
            # Apply filter if present
            if subscription.filter_func and not subscription.filter_func(event):
                continue
            
            tasks.append(self._call_subscriber(subscription, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _call_subscriber(self, subscription: Subscription, event: Event):
        """Call subscriber callback with error handling"""
        try:
            await subscription.callback(event)
        except Exception as e:
            self.logger.error(f"Subscriber {subscription.subscriber_id} error: {e}")
    
    def get_history(self, event_type: EventType = None, 
                    limit: int = 100) -> List[Event]:
        """Get event history"""
        history = self._history
        if event_type:
            history = [e for e in history if e.event_type == event_type]
        return history[-limit:]
    
    def get_stats(self) -> Dict:
        """Get event bus statistics"""
        return {
            "queue_size": self._event_queue.qsize(),
            "history_size": len(self._history),
            "subscriber_count": sum(len(s) for s in self._subscribers.values()),
            "running": self._running
        }


class KafkaEventBus(EventBus):
    """
    Kafka-backed event bus for distributed systems.
    
    Extends basic EventBus with Kafka integration for:
    - Cross-service communication
    - Event persistence
    - Horizontal scaling
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.bootstrap_servers = config.get("kafka_servers", "localhost:9092")
        self._producer = None
        self._consumer = None
        self._topics = {
            EventType.TICK_UPDATE: "ticks",
            EventType.QUOTE_UPDATE: "quotes",
            EventType.TRADE_SIGNAL: "signals",
            EventType.RISK_ALERT: "alerts"
        }
    
    async def start(self):
        """Start with Kafka connection"""
        await super().start()
        
        try:
            from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
            
            self._producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers
            )
            await self._producer.start()
            
            self.logger.info("Kafka event bus started")
            
        except ImportError:
            self.logger.warning("aiokafka not installed, using in-memory event bus")
        except Exception as e:
            self.logger.error(f"Kafka connection error: {e}")
    
    async def stop(self):
        """Stop Kafka connections"""
        if self._producer:
            await self._producer.stop()
        
        await super().stop()
    
    async def publish(self, event: Event):
        """Publish to both local and Kafka"""
        # Local publishing
        await super().publish(event)
        
        # Kafka publishing
        if self._producer:
            topic = self._topics.get(event.event_type, "events")
            try:
                await self._producer.send_and_wait(
                    topic,
                    json.dumps(event.to_dict()).encode(),
                    key=event.source.encode()
                )
            except Exception as e:
                self.logger.error(f"Kafka publish error: {e}")


# Event factory functions
def create_tick_event(symbol: str, price: float, volume: int,
                      bid: float = 0, ask: float = 0, source: str = "data_feed") -> Event:
    """Create a tick update event"""
    return Event(
        event_type=EventType.TICK_UPDATE,
        timestamp=datetime.now(),
        source=source,
        data={
            "symbol": symbol,
            "price": price,
            "volume": volume,
            "bid": bid,
            "ask": ask
        },
        priority=1
    )


def create_volume_spike_event(symbol: str, volume: int, 
                               avg_volume: int, source: str = "volume_monitor") -> Event:
    """Create a volume spike event"""
    spike_ratio = volume / avg_volume if avg_volume > 0 else 0
    
    return Event(
        event_type=EventType.VOLUME_SPIKE,
        timestamp=datetime.now(),
        source=source,
        data={
            "symbol": symbol,
            "volume": volume,
            "avg_volume": avg_volume,
            "spike_ratio": spike_ratio
        },
        priority=5
    )


def create_agent_signal_event(agent: str, symbol: str, direction: str,
                               conviction: float, reason: str) -> Event:
    """Create an agent signal event"""
    return Event(
        event_type=EventType.AGENT_SIGNAL,
        timestamp=datetime.now(),
        source=agent,
        data={
            "agent": agent,
            "symbol": symbol,
            "direction": direction,
            "conviction": conviction,
            "reason": reason
        },
        priority=3
    )


def create_risk_alert_event(alert_type: str, message: str, 
                             severity: str, data: Dict = None) -> Event:
    """Create a risk alert event"""
    priority_map = {"LOW": 1, "MEDIUM": 5, "HIGH": 8, "CRITICAL": 10}
    
    return Event(
        event_type=EventType.RISK_ALERT,
        timestamp=datetime.now(),
        source="prudence",
        data={
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            **(data or {})
        },
        priority=priority_map.get(severity, 5)
    )


def create_trade_signal_event(symbol: str, direction: str, entry: float,
                               stop_loss: float, target: float,
                               conviction: int, agents: List[str]) -> Event:
    """Create a trade signal event"""
    return Event(
        event_type=EventType.TRADE_SIGNAL,
        timestamp=datetime.now(),
        source="coordinator",
        data={
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "stop_loss": stop_loss,
            "target": target,
            "conviction": conviction,
            "recommending_agents": agents
        },
        priority=7
    )
