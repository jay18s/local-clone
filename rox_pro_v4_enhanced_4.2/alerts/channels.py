"""
ROX Proven Edge Engine v3.0 - Alert Channels
===========================================
Implementation of various alert delivery channels.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio
import logging

from .alert_manager import AlertChannel, AlertConfig, AlertMessage


class EmailChannel(AlertChannel):
    """Email alert channel"""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.smtp_server = config.metadata.get("smtp_server", "smtp.gmail.com")
        self.smtp_port = config.metadata.get("smtp_port", 587)
        self.sender_email = config.metadata.get("sender_email", "")
        self.sender_password = config.metadata.get("sender_password", "")
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send email alert"""
        if not self.can_send(alert):
            return False
        
        try:
            # Placeholder - in production would use aiosmtplib
            self.logger.info(f"EMAIL: {alert.title} to {self.config.recipients}")
            self.record_sent()
            return True
        except Exception as e:
            self.logger.error(f"Email error: {e}")
            return False


class SMSChannel(AlertChannel):
    """SMS alert channel"""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.provider = config.metadata.get("provider", "twilio")
        self.account_sid = config.metadata.get("account_sid", "")
        self.auth_token = config.metadata.get("auth_token", "")
        self.from_number = config.metadata.get("from_number", "")
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send SMS alert"""
        if not self.can_send(alert):
            return False
        
        try:
            # Placeholder - in production would use Twilio API
            self.logger.info(f"SMS: {alert.title[:50]} to {self.config.recipients}")
            self.record_sent()
            return True
        except Exception as e:
            self.logger.error(f"SMS error: {e}")
            return False


class TelegramChannel(AlertChannel):
    """Telegram alert channel"""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.bot_token = config.api_key
        self.chat_ids = config.recipients
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send Telegram message"""
        if not self.can_send(alert):
            return False
        
        try:
            import aiohttp
            
            # Format message with HTML
            emoji_map = {
                "CRITICAL": "🚨",
                "HIGH": "⚠️",
                "MEDIUM": "📢",
                "LOW": "ℹ️",
                "INFO": "📌"
            }
            
            emoji = emoji_map.get(alert.priority.value, "📢")
            
            text = f"""
{emoji} <b>{alert.title}</b>

{alert.message}

📊 Symbol: {alert.symbol or 'N/A'}
⏰ Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Priority: {alert.priority.value}
"""
            
            for chat_id in self.chat_ids:
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json={
                        "chat_id": chat_id,
                        "text": text,
                        "parse_mode": "HTML"
                    }) as resp:
                        if resp.status != 200:
                            return False
            
            self.record_sent()
            return True
            
        except Exception as e:
            self.logger.error(f"Telegram error: {e}")
            return False


class WebhookChannel(AlertChannel):
    """Webhook alert channel"""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.url = config.webhook_url
        self.headers = config.metadata.get("headers", {})
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send webhook POST"""
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
                    self.url,
                    json=payload,
                    headers={**self.headers, "Content-Type": "application/json"}
                ) as resp:
                    if resp.status not in [200, 201, 202, 204]:
                        self.logger.error(f"Webhook returned {resp.status}")
                        return False
            
            self.record_sent()
            return True
            
        except Exception as e:
            self.logger.error(f"Webhook error: {e}")
            return False


class WhatsAppChannel(AlertChannel):
    """WhatsApp alert channel via Twilio"""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.account_sid = config.metadata.get("account_sid", "")
        self.auth_token = config.metadata.get("auth_token", "")
        self.from_number = config.metadata.get("from_number", "")
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send WhatsApp message"""
        if not self.can_send(alert):
            return False
        
        try:
            # Placeholder - would use Twilio WhatsApp API
            self.logger.info(f"WhatsApp: {alert.title} to {self.config.recipients}")
            self.record_sent()
            return True
        except Exception as e:
            self.logger.error(f"WhatsApp error: {e}")
            return False


class SlackChannel(AlertChannel):
    """Slack alert channel"""
    
    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self.webhook_url = config.webhook_url
    
    async def send(self, alert: AlertMessage) -> bool:
        """Send Slack message"""
        if not self.can_send(alert):
            return False
        
        try:
            import aiohttp
            
            color_map = {
                "CRITICAL": "#FF0000",
                "HIGH": "#FF6600",
                "MEDIUM": "#FFCC00",
                "LOW": "#3399FF",
                "INFO": "#999999"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.priority.value, "#999999"),
                    "title": alert.title,
                    "text": alert.message,
                    "fields": [
                        {"title": "Symbol", "value": alert.symbol or "N/A", "short": True},
                        {"title": "Priority", "value": alert.priority.value, "short": True}
                    ],
                    "footer": f"ROX Edge Engine • {alert.timestamp.strftime('%H:%M:%S')}"
                }]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    if resp.status != 200:
                        return False
            
            self.record_sent()
            return True
            
        except Exception as e:
            self.logger.error(f"Slack error: {e}")
            return False
