"""ROX Engine v5.0 — Logging Setup"""
import os
import logging
import sys


def setup_logging(level: str = "INFO", log_file: str = "logs/rox_engine.log", max_files: int = 7):
    """Configure root logger with console and file handlers."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    log_format = "[%(asctime)s] %(levelname)-7s %(name)-30s %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='a', encoding='utf-8'),
    ]
    
    for handler in handlers:
        handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        datefmt=date_format,
        handlers=handlers,
    )
    
    # Set component log levels
    logging.getLogger("llm.client").setLevel(logging.INFO)
    logging.getLogger("reasoning").setLevel(logging.DEBUG)
    logging.getLogger("reasoning.debate").setLevel(logging.INFO)
    logging.getLogger("reasoning.calibrator").setLevel(logging.INFO)
    logging.getLogger("reasoning.rule_validator").setLevel(logging.DEBUG)
    logging.getLogger("reasoning.pattern_memory").setLevel(logging.INFO)
    logging.getLogger("reasoning.adaptive").setLevel(logging.INFO)
    logging.getLogger("agents").setLevel(logging.INFO)
    
    return logging.getLogger("rox.engine")
