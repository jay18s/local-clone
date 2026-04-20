"""
ROX Proven Edge Engine v3.0 - WebSocket Handler
==============================================
WebSocket connection management for real-time data.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Set
from enum import Enum
try:
    import websockets
    from websockets.client import WebSocketClientProtocol
except ImportError:
    websockets = None  # Optional: install with 'pip install websockets'
    WebSocketClientProtocol = None  # type: ignore


class ConnectionState(Enum):
    """WebSocket connection states"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


@dataclass
class WebSocketConfig:
    """WebSocket configuration"""
    url: str
    reconnect_interval: float = 5.0
    max_reconnect_attempts: int = 10
    ping_interval: float = 30.0
    ping_timeout: float = 10.0
    message_timeout: float = 60.0
    headers: Dict[str, str] = field(default_factory=dict)
    subscriptions: List[str] = field(default_factory=list)


class WebSocketManager:
    """
    Manages WebSocket connections for real-time data.
    
    Features:
    - Automatic reconnection
    - Heartbeat/ping-pong
    - Message queuing
    - Subscription management
    - Connection pooling
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger("WebSocketManager")
        
        # Connection management
        self._connections: Dict[str, WebSocketClientProtocol] = {}
        self._states: Dict[str, ConnectionState] = {}
        self._configs: Dict[str, WebSocketConfig] = {}
        
        # Callbacks
        self._message_callbacks: List[Callable] = []
        self._connection_callbacks: List[Callable] = []
        self._error_callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._reconnect_counts: Dict[str, int] = {}
    
    async def start(self):
        """Start WebSocket manager"""
        self._running = True
        self.logger.info("WebSocket manager started")
    
    async def stop(self):
        """Stop all connections"""
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        # Close connections
        for name, ws in self._connections.items():
            if ws and not ws.closed:
                await ws.close()
        
        self._connections.clear()
        self._states.clear()
        self.logger.info("WebSocket manager stopped")
    
    async def add_connection(self, name: str, config: WebSocketConfig) -> bool:
        """Add a new WebSocket connection"""
        self._configs[name] = config
        self._states[name] = ConnectionState.DISCONNECTED
        self._reconnect_counts[name] = 0
        
        # Start connection task
        task = asyncio.create_task(self._maintain_connection(name))
        self._tasks.append(task)
        
        return True
    
    async def _maintain_connection(self, name: str):
        """Maintain a WebSocket connection with auto-reconnect"""
        config = self._configs.get(name)
        if not config:
            return
        
        while self._running:
            try:
                self._states[name] = ConnectionState.CONNECTING
                
                # Connect
                async with websockets.connect(
                    config.url,
                    extra_headers=config.headers,
                    ping_interval=config.ping_interval,
                    ping_timeout=config.ping_timeout
                ) as ws:
                    self._connections[name] = ws
                    self._states[name] = ConnectionState.CONNECTED
                    self._reconnect_counts[name] = 0
                    
                    self.logger.info(f"WebSocket {name} connected")
                    await self._notify_connection(name, True)
                    
                    # Send subscriptions
                    if config.subscriptions:
                        await self._subscribe(name, config.subscriptions)
                    
                    # Receive messages
                    await self._receive_messages(name, ws)
                    
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning(f"WebSocket {name} connection closed")
                self._states[name] = ConnectionState.RECONNECTING
                
            except Exception as e:
                self.logger.error(f"WebSocket {name} error: {e}")
                self._states[name] = ConnectionState.ERROR
                await self._notify_error(name, str(e))
            
            # Reconnect logic
            if self._running:
                self._reconnect_counts[name] += 1
                
                if self._reconnect_counts[name] > config.max_reconnect_attempts:
                    self.logger.error(f"WebSocket {name} max reconnect attempts reached")
                    self._states[name] = ConnectionState.ERROR
                    break
                
                await asyncio.sleep(config.reconnect_interval)
    
    async def _receive_messages(self, name: str, ws: WebSocketClientProtocol):
        """Receive and process messages"""
        config = self._configs.get(name)
        
        while self._running and not ws.closed:
            try:
                message = await asyncio.wait_for(
                    ws.recv(),
                    timeout=config.message_timeout if config else 60.0
                )
                
                # Parse message
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    data = {"raw": message}
                
                # Add metadata
                data["_connection"] = name
                data["_timestamp"] = datetime.now().isoformat()
                
                # Notify callbacks
                await self._notify_message(name, data)
                
            except asyncio.TimeoutError:
                self.logger.debug(f"WebSocket {name} message timeout")
                continue
                
            except websockets.exceptions.ConnectionClosed:
                raise
                
            except Exception as e:
                self.logger.error(f"WebSocket {name} message error: {e}")
    
    async def _subscribe(self, name: str, subscriptions: List[str]):
        """Send subscription messages"""
        ws = self._connections.get(name)
        if not ws or ws.closed:
            return
        
        message = json.dumps({
            "type": "subscribe",
            "symbols": subscriptions
        })
        
        await ws.send(message)
        self.logger.info(f"WebSocket {name} subscribed to {len(subscriptions)} symbols")
    
    async def subscribe(self, name: str, symbols: List[str]) -> bool:
        """Subscribe to additional symbols"""
        config = self._configs.get(name)
        if config:
            config.subscriptions.extend(symbols)
        
        ws = self._connections.get(name)
        if ws and not ws.closed:
            await self._subscribe(name, symbols)
            return True
        
        return False
    
    async def send(self, name: str, message: Dict) -> bool:
        """Send message to WebSocket"""
        ws = self._connections.get(name)
        if not ws or ws.closed:
            return False
        
        try:
            await ws.send(json.dumps(message))
            return True
        except Exception as e:
            self.logger.error(f"WebSocket {name} send error: {e}")
            return False
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connections"""
        for name in self._connections:
            await self.send(name, message)
    
    def register_message_callback(self, callback: Callable):
        """Register callback for messages"""
        self._message_callbacks.append(callback)
    
    def register_connection_callback(self, callback: Callable):
        """Register callback for connection events"""
        self._connection_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable):
        """Register callback for errors"""
        self._error_callbacks.append(callback)
    
    async def _notify_message(self, name: str, data: Dict):
        """Notify message callbacks"""
        for callback in self._message_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(name, data)
                else:
                    callback(name, data)
            except Exception as e:
                self.logger.error(f"Message callback error: {e}")
    
    async def _notify_connection(self, name: str, connected: bool):
        """Notify connection callbacks"""
        for callback in self._connection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(name, connected)
                else:
                    callback(name, connected)
            except Exception as e:
                self.logger.error(f"Connection callback error: {e}")
    
    async def _notify_error(self, name: str, error: str):
        """Notify error callbacks"""
        for callback in self._error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(name, error)
                else:
                    callback(name, error)
            except Exception as e:
                self.logger.error(f"Error callback error: {e}")
    
    def get_state(self, name: str) -> ConnectionState:
        """Get connection state"""
        return self._states.get(name, ConnectionState.DISCONNECTED)
    
    def get_all_states(self) -> Dict[str, str]:
        """Get all connection states"""
        return {name: state.value for name, state in self._states.items()}
    
    def is_connected(self, name: str) -> bool:
        """Check if connection is active"""
        return self._states.get(name) == ConnectionState.CONNECTED


class FyersWebSocketManager(WebSocketManager):
    """Specialized WebSocket manager for FYERS API"""
    
    FYERS_WS_URL = "wss://api.fyers.in/socket/v2"
    
    async def add_fyers_connection(self, access_token: str, 
                                    symbols: List[str]) -> bool:
        """Add FYERS WebSocket connection"""
        config = WebSocketConfig(
            url=self.FYERS_WS_URL,
            headers={
                "Authorization": access_token
            },
            subscriptions=symbols
        )
        
        return await self.add_connection("fyers", config)
    
    async def subscribe_symbols(self, symbols: List[str]) -> bool:
        """Subscribe to symbols on FYERS"""
        return await self.subscribe("fyers", symbols)


class ZerodhaWebSocketManager(WebSocketManager):
    """Specialized WebSocket manager for Zerodha Kite"""
    
    KITE_WS_URL = "wss://ws.kite.trade"
    
    async def add_kite_connection(self, api_key: str, access_token: str,
                                   instrument_tokens: List[int]) -> bool:
        """Add Kite WebSocket connection"""
        config = WebSocketConfig(
            url=self.KITE_WS_URL,
            headers={
                "X-Kite-Version": "3",
                "Authorization": f"token {api_key}:{access_token}"
            },
            subscriptions=[str(t) for t in instrument_tokens]
        )
        
        return await self.add_connection("kite", config)
