"""
ROX Proven Edge Engine v3.0 - Logging Module
===========================================
Comprehensive logging with Windows compatibility.
"""

import io
import os
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from typing import Optional
from functools import wraps
import traceback


def _utf8_console_stream() -> io.TextIOWrapper:
    """
    Return a UTF-8-encoded stream wrapping stdout.

    On Windows the default console code-page is cp1252, which cannot
    encode Indian-Rupee (₹), arrows (→), warning signs (⚠), or any
    other Unicode character outside the Latin-1 range.  Every ROX log
    line that contains those symbols raises UnicodeEncodeError and gets
    truncated.

    Strategy (in priority order):
      1. If stdout already reports UTF-8 encoding, use it as-is — no
         wrapping needed and we avoid the double-wrap pitfall.
      2. If stdout has a raw binary buffer (the normal case on Windows),
         wrap that buffer in a TextIOWrapper with UTF-8 + 'replace' as
         the error handler so that, in the absolute worst case (a legacy
         terminal that truly can't display a glyph), the character is
         replaced by '?' rather than raising an exception.
      3. Fallback: return stdout unchanged (Linux/macOS where this is
         almost never needed anyway).
    """
    current_encoding = getattr(sys.stdout, "encoding", "") or ""
    if current_encoding.lower().replace("-", "") == "utf8":
        return sys.stdout  # already fine

    if hasattr(sys.stdout, "buffer"):
        return io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",   # '?' instead of a crash
            line_buffering=True,
        )

    return sys.stdout  # best-effort fallback


def setup_logging(config=None) -> logging.Logger:
    """
    Set up comprehensive logging for the application.

    Args:
        config: LoggingConfig instance with logging settings

    Returns:
        Configured root logger
    """
    if config is None:
        from .config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG.logging

    # Create logs directory
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).parent
    else:
        base_dir = Path(__file__).parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt=config.log_format,
        datefmt=config.date_format,
    )

    # ------------------------------------------------------------------
    # Console handler — UTF-8 safe
    # ------------------------------------------------------------------
    if config.log_to_console:
        console_handler = logging.StreamHandler(_utf8_console_stream())
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # File handler with rotation — already UTF-8, unchanged
    # ------------------------------------------------------------------
    if config.log_to_file:
        log_file = log_dir / f"{config.log_file_prefix}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=config.max_log_size_mb * 1024 * 1024,
            backupCount=config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Error file handler — already UTF-8, unchanged
    # ------------------------------------------------------------------
    error_log_file = log_dir / f"{config.log_file_prefix}_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=config.max_log_size_mb * 1024 * 1024,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module."""
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capability to any class."""

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger


def log_execution(func):
    """Decorator to log function execution with timing."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = func.__qualname__

        logger.debug(f"Executing: {func_name}")
        start_time = datetime.now()

        try:
            result = func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Completed: {func_name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed: {func_name} after {elapsed:.3f}s - {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    return wrapper


def log_async_execution(func):
    """Decorator to log async function execution with timing."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        func_name = func.__qualname__

        logger.debug(f"Executing async: {func_name}")
        start_time = datetime.now()

        try:
            result = await func(*args, **kwargs)
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.debug(f"Completed async: {func_name} in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed async: {func_name} after {elapsed:.3f}s - {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    return wrapper


class TradeLoggerAdapter(logging.LoggerAdapter):
    """Custom logger adapter for trade-related logging."""

    def process(self, msg, kwargs):
        trade_id = self.extra.get("trade_id", "N/A")
        stock = self.extra.get("stock", "N/A")
        return f"[TRADE:{trade_id}|{stock}] {msg}", kwargs


class AuditLogger:
    """Audit logger for compliance and tracking."""

    def __init__(self, log_dir: Path = None):
        if log_dir is None:
            if getattr(sys, "frozen", False):
                base_dir = Path(sys.executable).parent
            else:
                base_dir = Path(__file__).parent.parent
            log_dir = base_dir / "logs"

        log_dir.mkdir(parents=True, exist_ok=True)

        self.audit_file = log_dir / "audit.log"
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.FileHandler(self.audit_file, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            self.logger.addHandler(handler)

    def log_action(self, action: str, details: dict):
        """Log an audit action."""
        import json
        self.logger.info(f"{action} | {json.dumps(details)}")

    def log_trade(self, action: str, trade_details: dict):
        """Log trade-related action."""
        self.log_action(f"TRADE_{action}", trade_details)

    def log_agent_decision(self, agent: str, decision: str, details: dict):
        """Log agent decision."""
        self.log_action(f"AGENT_{agent}", {"decision": decision, **details})


# ---------------------------------------------------------------------------
# Module-level initialisation
# ---------------------------------------------------------------------------
_initialized = False


def init_logging():
    """Initialize logging (called once on startup)."""
    global _initialized
    if not _initialized:
        setup_logging()
        _initialized = True
