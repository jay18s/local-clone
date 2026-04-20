"""
ROX Proven Edge Engine v3.0 - Alert Manager
==========================================
Centralized alert management with multi-channel delivery.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
import json


class AlertPriority(Enum):
    """Alert priority levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AlertChannelType(Enum):
    """Alert channel types"""
    EMAIL = "EMAIL"
    SMS = "SMS"
    TELEGRAM = "TELEGRAM"
    WHATSAPP = "WHATSAPP"
    WEBHOOK = "WEBHOOK"
    IN_APP = "IN_APP"
    LOG = "LOG"


@dataclass
class AlertConfig:
    """Alert configuration"""
    channel_type: AlertChannelType
    enabled: bool = True
    min_priority: AlertPriority = AlertPriority.MEDIUM
    
    # Channel-specific config
    recipients: List[str] = field(default_factory=list)
    webhook_url: str = ""
    api_key: str = ""
    template: str = ""
    
    # Rate limiting
    max_alerts_per_hour: int = 20
    cooldown_minutes: int = 5


@dataclass
class AlertMessage:
    """Alert message"""
    alert_id: str
    priority: AlertPriority
    title: str
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def format(self, template: str = None) -> str:
        """Format alert message"""
        if template:
            return template.format(
                title=self.title,
                message=self.message,
                symbol=self.symbol or "N/A",
                timestamp=self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                priority=self.priority.value
            )
        
        return f"[{self.priority.value}] {self.title}\n{self.message}"


class AlertChannel(ABC):
    """Abstract alert channel"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.logger = logging.getLogger(f"AlertChannel.{config.channel_type.value}")
        
        # Rate limiting
        self._recent_alerts: List[datetime] = []
    
    @abstractmethod
    async def send(self, alert: AlertMessage) -> bool:
        """Send alert through this channel"""
        pass
    
    def can_send(self, alert: AlertMessage) -> bool:
        """Check if can send (rate limiting)"""
        if not self.config.enabled:
            return False
        
        # Check priority
        priority_order = [AlertPriority.CRITICAL, AlertPriority.HIGH, 
                        AlertPriority.MEDIUM, AlertPriority.LOW, AlertPriority.INFO]
        
        if priority_order.index(alert.priority) > priority_order.index(self.config.min_priority):
            return False
        
        # Check rate limit
        self._cleanup_recent()
        
        if len(self._recent_alerts) >= self.config.max_alerts_per_hour:
            return False
        
        return True
    
    def record_sent(self):
        """Record that alert was sent"""
        self._recent_alerts.append(datetime.now())
    
    def _cleanup_recent(self):
        """Clean up old entries"""
        cutoff = datetime.now() - timedelta(hours=1)
        self._recent_alerts = [t for t in self._recent_alerts if t > cutoff]


class LogChannel(AlertChannel):
    """Log-based alert channel"""
    
    async def send(self, alert: AlertMessage) -> bool:
        """Log the alert"""
        if not self.can_send(alert):
            return False
        
        if alert.priority == AlertPriority.CRITICAL:
            self.logger.critical(alert.format())
        elif alert.priority == AlertPriority.HIGH:
            self.logger.error(alert.format())
        elif alert.priority == AlertPriority.MEDIUM:
            self.logger.warning(alert.format())
        else:
            self.logger.info(alert.format())
        
        self.record_sent()
        return True


class EmailChannel(AlertChannel):
    """Email alert channel"""
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send email alert"""
        if not self.can_send(alert):
            return False
        
        try:
            # Placeholder - would use smtplib or email service
            self.logger.info(f"EMAIL to {self.config.recipients}: {alert.title}")
            self.record_sent()
            return True
        except Exception as e:
            self.logger.error(f"Email send error: {e}")
            return False


class TelegramChannel(AlertChannel):
    """Telegram alert channel"""
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send Telegram alert"""
        if not self.can_send(alert):
            return False
        
        try:
            import aiohttp
            
            bot_token = self.config.api_key
            chat_ids = self.config.recipients
            
            for chat_id in chat_ids:
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                
                data = {
                    "chat_id": chat_id,
                    "text": alert.format(),
                    "parse_mode": "HTML"
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data) as resp:
                        if resp.status != 200:
                            self.logger.error(f"Telegram API error: {resp.status}")
                            return False
            
            self.record_sent()
            return True
            
        except ImportError:
            self.logger.warning("aiohttp not installed for Telegram")
            return False
        except Exception as e:
            self.logger.error(f"Telegram send error: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Webhook alert channel"""
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send webhook alert"""
        if not self.can_send(alert):
            return False
        
        try:
            import aiohttp
            
            payload = {
                "alert_id": alert.alert_id,
                "priority": alert.priority.value,
                "title": alert.title,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "symbol": alert.symbol,
                "metadata": alert.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as resp:
                    if resp.status not in [200, 201, 202]:
                        self.logger.error(f"Webhook error: {resp.status}")
                        return False
            
            self.record_sent()
            return True
            
        except Exception as e:
            self.logger.error(f"Webhook send error: {e}")
            return False


class AlertManager:
    """
    Centralized alert management.
    
    Features:
    - Multi-channel delivery
    - Priority-based routing
    - Rate limiting
    - Alert grouping
    - Historical tracking
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("AlertManager")
        
        # Channels
        self.channels: Dict[str, AlertChannel] = {}
        
        # Alert tracking
        self.alert_counter = 0
        self.alert_history: List[AlertMessage] = []
        
        # Initialize default channel
        self._init_default_channels()
    
    def _init_default_channels(self):
        """Initialize default channels"""
        # Log channel (always enabled)
        self.add_channel("log", LogChannel(AlertConfig(
            channel_type=AlertChannelType.LOG,
            enabled=True,
            min_priority=AlertPriority.INFO
        )))
    
    def add_channel(self, name: str, channel: AlertChannel):
        """Add alert channel"""
        self.channels[name] = channel
        self.logger.info(f"Added alert channel: {name}")
    
    def configure_telegram(self, bot_token: str, chat_ids: List[str],
                          min_priority: AlertPriority = AlertPriority.HIGH):
        """Configure Telegram channel"""
        config = AlertConfig(
            channel_type=AlertChannelType.TELEGRAM,
            enabled=True,
            min_priority=min_priority,
            recipients=chat_ids,
            api_key=bot_token
        )
        
        self.add_channel("telegram", TelegramChannel(config))
    
    def configure_webhook(self, url: str, 
                         min_priority: AlertPriority = AlertPriority.MEDIUM):
        """Configure webhook channel"""
        config = AlertConfig(
            channel_type=AlertChannelType.WEBHOOK,
            enabled=True,
            min_priority=min_priority,
            webhook_url=url
        )
        
        self.add_channel("webhook", WebhookChannel(config))
    
    async def send_alert(self, priority: AlertPriority, title: str,
                        message: str, symbol: str = None,
                        metadata: Dict = None) -> AlertMessage:
        """Send alert through all channels"""
        self.alert_counter += 1
        alert_id = f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self.alert_counter}"
        
        alert = AlertMessage(
            alert_id=alert_id,
            priority=priority,
            title=title,
            message=message,
            timestamp=datetime.now(),
            symbol=symbol,
            metadata=metadata or {}
        )
        
        # Store in history
        self.alert_history.append(alert)
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Send through all channels
        tasks = []
        for name, channel in self.channels.items():
            tasks.append(self._send_to_channel(name, channel, alert))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        return alert
    
    async def _send_to_channel(self, name: str, channel: AlertChannel,
                               alert: AlertMessage):
        """Send to single channel with error handling"""
        try:
            await channel.send(alert)
        except Exception as e:
            self.logger.error(f"Channel {name} error: {e}")
    
    async def send_critical(self, title: str, message: str, 
                           symbol: str = None) -> AlertMessage:
        """Send critical alert"""
        return await self.send_alert(
            priority=AlertPriority.CRITICAL,
            title=title,
            message=message,
            symbol=symbol
        )
    
    async def send_high(self, title: str, message: str,
                       symbol: str = None) -> AlertMessage:
        """Send high priority alert"""
        return await self.send_alert(
            priority=AlertPriority.HIGH,
            title=title,
            message=message,
            symbol=symbol
        )
    
    async def send_medium(self, title: str, message: str,
                         symbol: str = None) -> AlertMessage:
        """Send medium priority alert"""
        return await self.send_alert(
            priority=AlertPriority.MEDIUM,
            title=title,
            message=message,
            symbol=symbol
        )
    
    async def send_trade_signal(self, symbol: str, direction: str,
                               entry: float, stop: float, target: float,
                               conviction: int) -> AlertMessage:
        """Send trade signal alert"""
        return await self.send_alert(
            priority=AlertPriority.HIGH,
            title=f"Trade Signal: {symbol} {direction}",
            message=f"Entry: ₹{entry:.2f}\nSL: ₹{stop:.2f}\nTarget: ₹{target:.2f}\nConviction: {conviction}%",
            symbol=symbol,
            metadata={"entry": entry, "stop": stop, "target": target}
        )
    
    async def send_risk_alert(self, alert_type: str, message: str,
                             current_value: float, threshold: float) -> AlertMessage:
        """Send risk alert"""
        return await self.send_alert(
            priority=AlertPriority.CRITICAL,
            title=f"Risk Alert: {alert_type}",
            message=f"{message}\nCurrent: {current_value:.2%}\nThreshold: {threshold:.2%}"
        )
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts"""
        return [
            {
                "alert_id": a.alert_id,
                "priority": a.priority.value,
                "title": a.title,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self.alert_history[-limit:]
        ]


from datetime import timedelta
