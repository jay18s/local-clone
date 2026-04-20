"""
ROX Proven Edge Engine v3.0 - Risk Monitor
=========================================
Continuous risk monitoring and alerting.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from enum import Enum


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Types of risk alerts"""
    POSITION_LIMIT = "POSITION_LIMIT"
    SECTOR_LIMIT = "SECTOR_LIMIT"
    PORTFOLIO_HEAT = "PORTFOLIO_HEAT"
    DRAWDOWN = "DRAWDOWN"
    DAILY_LOSS = "DAILY_LOSS"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    CORRELATION = "CORRELATION"
    NEWS_EVENT = "NEWS_EVENT"
    STALE_DATA = "STALE_DATA"
    SYSTEM_ERROR = "SYSTEM_ERROR"


@dataclass
class Alert:
    """Risk alert"""
    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    current_value: float = 0.0
    threshold: float = 0.0
    recommended_action: str = ""
    acknowledged: bool = False
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "recommended_action": self.recommended_action,
            "acknowledged": self.acknowledged
        }


class RiskMonitor:
    """
    Continuous risk monitoring system.
    
    Features:
    - Real-time monitoring
    - Multi-channel alerting
    - Alert prioritization
    - Historical tracking
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger("RiskMonitor")
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_active_alerts = config.get("max_active_alerts", 100)
        
        # Monitoring thresholds
        self.thresholds = {
            "position_size": config.get("max_position_size", 0.15),
            "sector_exposure": config.get("max_sector_exposure", 0.25),
            "portfolio_heat": config.get("max_portfolio_heat", 0.08),
            "drawdown_warning": config.get("drawdown_warning", 0.10),
            "drawdown_critical": config.get("drawdown_critical", 0.15),
            "daily_loss_warning": config.get("daily_loss_warning", -0.03),
            "daily_loss_critical": config.get("daily_loss_critical", -0.05),
            "vix_high": config.get("vix_high", 20),
            "vix_extreme": config.get("vix_extreme", 28),
            "spread_wide": config.get("spread_wide", 0.01),
            "spread_extreme": config.get("spread_extreme", 0.02),
            "correlation_high": config.get("correlation_high", 0.7)
        }
        
        # Callbacks
        self._alert_callbacks: List[Callable] = []
        
        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Alert counter
        self._alert_counter = 0
    
    def start(self):
        """Start risk monitor"""
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self.logger.info("Risk monitor started")
    
    def stop(self):
        """Stop risk monitor"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        self.logger.info("Risk monitor stopped")
    
    async def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
    
    def check_position_limits(self, positions: Dict[str, Dict],
                              portfolio_value: float) -> List[Alert]:
        """Check position size limits"""
        alerts = []
        
        for symbol, pos in positions.items():
            position_value = pos.get("value", 0)
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            
            if position_pct > self.thresholds["position_size"]:
                alert = self._create_alert(
                    alert_type=AlertType.POSITION_LIMIT,
                    severity=AlertSeverity.HIGH,
                    title=f"Position Size Exceeded: {symbol}",
                    message=f"Position in {symbol} is {position_pct:.1%} of portfolio, exceeds {self.thresholds['position_size']:.1%} limit",
                    symbol=symbol,
                    current_value=position_pct,
                    threshold=self.thresholds["position_size"],
                    recommended_action=f"Reduce {symbol} position to within limits"
                )
                alerts.append(alert)
        
        return alerts
    
    def check_sector_exposure(self, sector_positions: Dict[str, float],
                             portfolio_value: float) -> List[Alert]:
        """Check sector concentration"""
        alerts = []
        
        for sector, value in sector_positions.items():
            exposure = value / portfolio_value if portfolio_value > 0 else 0
            
            if exposure > self.thresholds["sector_exposure"]:
                alert = self._create_alert(
                    alert_type=AlertType.SECTOR_LIMIT,
                    severity=AlertSeverity.MEDIUM,
                    title=f"Sector Concentration: {sector}",
                    message=f"Exposure to {sector} is {exposure:.1%}, exceeds {self.thresholds['sector_exposure']:.1%} limit",
                    current_value=exposure,
                    threshold=self.thresholds["sector_exposure"],
                    recommended_action=f"Reduce positions in {sector}"
                )
                alerts.append(alert)
        
        return alerts
    
    def check_portfolio_heat(self, heat: float) -> Optional[Alert]:
        """Check portfolio heat"""
        if heat > self.thresholds["portfolio_heat"]:
            return self._create_alert(
                alert_type=AlertType.PORTFOLIO_HEAT,
                severity=AlertSeverity.HIGH,
                title="Portfolio Heat Critical",
                message=f"Portfolio heat is {heat:.1%}, exceeds {self.thresholds['portfolio_heat']:.1%} limit",
                current_value=heat,
                threshold=self.thresholds["portfolio_heat"],
                recommended_action="No new positions until heat reduces below 6%"
            )
        return None
    
    def check_drawdown(self, drawdown: float) -> Optional[Alert]:
        """Check drawdown level"""
        if abs(drawdown) > self.thresholds["drawdown_critical"]:
            return self._create_alert(
                alert_type=AlertType.DRAWDOWN,
                severity=AlertSeverity.CRITICAL,
                title="Critical Drawdown",
                message=f"Portfolio drawdown is {drawdown:.1%}",
                current_value=drawdown,
                threshold=self.thresholds["drawdown_critical"],
                recommended_action="Exit 50% of positions, preserve capital"
            )
        elif abs(drawdown) > self.thresholds["drawdown_warning"]:
            return self._create_alert(
                alert_type=AlertType.DRAWDOWN,
                severity=AlertSeverity.HIGH,
                title="Drawdown Warning",
                message=f"Portfolio drawdown is {drawdown:.1%}",
                current_value=drawdown,
                threshold=self.thresholds["drawdown_warning"],
                recommended_action="Reduce position sizes, review stops"
            )
        return None
    
    def check_daily_pnl(self, daily_pnl_pct: float) -> Optional[Alert]:
        """Check daily P&L"""
        if daily_pnl_pct < self.thresholds["daily_loss_critical"]:
            return self._create_alert(
                alert_type=AlertType.DAILY_LOSS,
                severity=AlertSeverity.CRITICAL,
                title="Critical Daily Loss",
                message=f"Daily loss is {daily_pnl_pct:.1%}",
                current_value=daily_pnl_pct,
                threshold=self.thresholds["daily_loss_critical"],
                recommended_action="Stop trading for the day"
            )
        elif daily_pnl_pct < self.thresholds["daily_loss_warning"]:
            return self._create_alert(
                alert_type=AlertType.DAILY_LOSS,
                severity=AlertSeverity.HIGH,
                title="Daily Loss Warning",
                message=f"Daily loss is {daily_pnl_pct:.1%}",
                current_value=daily_pnl_pct,
                threshold=self.thresholds["daily_loss_warning"],
                recommended_action="Reduce position sizes, tighten stops"
            )
        return None
    
    def check_volatility(self, vix: float) -> Optional[Alert]:
        """Check VIX level"""
        if vix > self.thresholds["vix_extreme"]:
            return self._create_alert(
                alert_type=AlertType.VOLATILITY,
                severity=AlertSeverity.CRITICAL,
                title="Extreme Volatility",
                message=f"India VIX at {vix:.1f}, indicates panic conditions",
                current_value=vix,
                threshold=self.thresholds["vix_extreme"],
                recommended_action="Exit all discretionary positions"
            )
        elif vix > self.thresholds["vix_high"]:
            return self._create_alert(
                alert_type=AlertType.VOLATILITY,
                severity=AlertSeverity.HIGH,
                title="High Volatility",
                message=f"India VIX at {vix:.1f}",
                current_value=vix,
                threshold=self.thresholds["vix_high"],
                recommended_action="Reduce position sizes by 50%"
            )
        return None
    
    def check_liquidity(self, symbol: str, spread_pct: float) -> Optional[Alert]:
        """Check liquidity/spread"""
        if spread_pct > self.thresholds["spread_extreme"]:
            return self._create_alert(
                alert_type=AlertType.LIQUIDITY,
                severity=AlertSeverity.HIGH,
                title=f"Liquidity Alert: {symbol}",
                message=f"Bid-ask spread for {symbol} is {spread_pct:.2%}",
                symbol=symbol,
                current_value=spread_pct,
                threshold=self.thresholds["spread_extreme"],
                recommended_action="Use limit orders only, avoid market orders"
            )
        return None
    
    def raise_custom_alert(self, alert_type: AlertType, severity: AlertSeverity,
                          title: str, message: str, recommended_action: str = "",
                          symbol: str = None, **kwargs) -> Alert:
        """Raise a custom alert"""
        return self._create_alert(
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            symbol=symbol,
            recommended_action=recommended_action,
            **kwargs
        )
    
    def _create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     title: str, message: str, **kwargs) -> Alert:
        """Create and store an alert"""
        self._alert_counter += 1
        alert_id = f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alert_counter}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            **kwargs
        )
        
        # Store
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Limit history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
        
        # Notify callbacks
        self._notify_alert(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            return True
        return False
    
    def dismiss_alert(self, alert_id: str) -> bool:
        """Dismiss an alert"""
        if alert_id in self.active_alerts:
            del self.active_alerts[alert_id]
            return True
        return False
    
    def clear_all_alerts(self):
        """Clear all active alerts"""
        self.active_alerts.clear()
    
    def get_active_alerts(self, severity: AlertSeverity = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity"""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_critical_alerts(self) -> List[Alert]:
        """Get all critical alerts"""
        return [a for a in self.active_alerts.values() 
                if a.severity == AlertSeverity.CRITICAL]
    
    def has_critical_alerts(self) -> bool:
        """Check if there are any critical alerts"""
        return any(a.severity == AlertSeverity.CRITICAL 
                  for a in self.active_alerts.values())
    
    def register_callback(self, callback: Callable):
        """Register alert callback"""
        self._alert_callbacks.append(callback)
    
    def _notify_alert(self, alert: Alert):
        """Notify all callbacks"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")
    
    def get_summary(self) -> Dict:
        """Get alert summary"""
        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([
                a for a in self.active_alerts.values() if a.severity == severity
            ])
        
        return {
            "total_active": len(self.active_alerts),
            "by_severity": severity_counts,
            "has_critical": self.has_critical_alerts(),
            "recent_count": len([a for a in self.alert_history 
                                if a.timestamp > datetime.now() - timedelta(hours=1)])
        }
