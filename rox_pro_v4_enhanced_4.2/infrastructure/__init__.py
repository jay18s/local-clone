"""
ROX Proven Edge Engine v4.0 Unified - Infrastructure Package
=============================================================
Real-time data infrastructure, caching, event-driven architecture,
F&O instrument management, margin calculation, and MWPL monitoring.

v3.2 components: DataFeedManager, WebSocketManager, EventBus, Cache,
                 DataNormalizer, GreeksCalculator, OptionChainStream,
                 PhysicalSettlementManager, HistoricalDataManager.

v4.0 additions: FnoInstrumentManager, MarginCalculator, MWPLMonitor,
                FNO StrategyBuilders (IronCondor, Calendar, Bull, Bear, Collar).
"""

def __getattr__(name):
    """Lazy import — load heavy/optional-dep modules only when needed."""
    # v3.2 real-time data (optional: requires websockets/aiohttp)
    if name in ("DataFeedManager", "TickData", "MarketDataSource"):
        from .data_feed import DataFeedManager, TickData, MarketDataSource
        globals().update({"DataFeedManager": DataFeedManager,
                          "TickData": TickData, "MarketDataSource": MarketDataSource})
        return globals()[name]
    if name in ("RedisCacheManager", "CircularBuffer"):
        from .cache import RedisCacheManager, CircularBuffer
        globals().update({"RedisCacheManager": RedisCacheManager,
                          "CircularBuffer": CircularBuffer})
        return globals()[name]
    if name in ("EventBus", "Event", "EventType"):
        from .event_bus import EventBus, Event, EventType
        globals().update({"EventBus": EventBus, "Event": Event, "EventType": EventType})
        return globals()[name]
    if name == "WebSocketManager":
        from .websocket_handler import WebSocketManager
        globals()["WebSocketManager"] = WebSocketManager
        return WebSocketManager
    if name in ("DataNormalizer", "NormalizedTick"):
        from .data_normalizer import DataNormalizer, NormalizedTick
        globals().update({"DataNormalizer": DataNormalizer,
                          "NormalizedTick": NormalizedTick})
        return globals()[name]
    # v3.2 F&O data
    if name == "HistoricalDataManager":
        from .historical_data_manager import HistoricalDataManager
        globals()["HistoricalDataManager"] = HistoricalDataManager
        return HistoricalDataManager
    if name in ("GreeksCalculator", "Greeks", "GreeksResult", "OptionsLeg", "PortfolioGreeks"):
        from .greeks_calculator import (GreeksCalculator, Greeks,
                                        OptionsLeg, PortfolioGreeks)
        globals().update({"GreeksCalculator": GreeksCalculator,
                          "Greeks": Greeks, "GreeksResult": Greeks,
                          "OptionsLeg": OptionsLeg,
                          "PortfolioGreeks": PortfolioGreeks})
        return globals()[name]
    if name in ("OptionChainStream", "OptionChainSnapshot"):
        from .option_chain_stream import OptionChainStream, OptionChainSnapshot
        globals().update({"OptionChainStream": OptionChainStream,
                          "OptionChainSnapshot": OptionChainSnapshot})
        return globals()[name]
    if name == "PhysicalSettlementManager":
        from .physical_settlement_manager import PhysicalSettlementManager
        globals()["PhysicalSettlementManager"] = PhysicalSettlementManager
        return PhysicalSettlementManager
    # v4.0 additions
    if name in ("FnoInstrumentManager", "FnoContract", "OptionChain",
                "StrikeInfo", "InstrumentType", "get_instrument_manager"):
        from .fno_instrument_manager import (FnoInstrumentManager, FnoContract,
                                             OptionChain, StrikeInfo, InstrumentType,
                                             get_instrument_manager)
        for k, v in [("FnoInstrumentManager", FnoInstrumentManager),
                     ("FnoContract", FnoContract), ("OptionChain", OptionChain),
                     ("StrikeInfo", StrikeInfo), ("InstrumentType", InstrumentType),
                     ("get_instrument_manager", get_instrument_manager)]:
            globals()[k] = v
        return globals()[name]
    if name in ("MarginCalculator", "MarginResult", "PortfolioMargin",
                "PositionType", "calculate_order_margin"):
        from .margin_calculator import (MarginCalculator, MarginResult,
                                        PortfolioMargin, PositionType,
                                        calculate_order_margin)
        for k, v in [("MarginCalculator", MarginCalculator),
                     ("MarginResult", MarginResult),
                     ("PortfolioMargin", PortfolioMargin),
                     ("PositionType", PositionType),
                     ("calculate_order_margin", calculate_order_margin)]:
            globals()[k] = v
        return globals()[name]
    if name in ("MWPLMonitor", "MWPLData", "MWPLAlertLevel",
                "ClientPositionLimit", "get_mwpl_monitor"):
        from .mwpl_monitor import (MWPLMonitor, MWPLData, MWPLAlertLevel,
                                   ClientPositionLimit, get_mwpl_monitor)
        for k, v in [("MWPLMonitor", MWPLMonitor), ("MWPLData", MWPLData),
                     ("MWPLAlertLevel", MWPLAlertLevel),
                     ("ClientPositionLimit", ClientPositionLimit),
                     ("get_mwpl_monitor", get_mwpl_monitor)]:
            globals()[k] = v
        return globals()[name]
    if name in ("StrategyFactory", "IronCondorBuilder", "CalendarSpreadBuilder",
                "BullSpreadBuilder", "BearSpreadBuilder", "CollarBuilder",
                "StrategyType", "MarketBias", "StrategyLeg", "StrategyResult"):
        from .fno_strategy_builders import (
            StrategyFactory, IronCondorBuilder, CalendarSpreadBuilder,
            BullSpreadBuilder, BearSpreadBuilder, CollarBuilder,
            StrategyType, MarketBias, StrategyLeg, StrategyResult,
        )
        for attr in ("StrategyFactory","IronCondorBuilder","CalendarSpreadBuilder",
                     "BullSpreadBuilder","BearSpreadBuilder","CollarBuilder",
                     "StrategyType","MarketBias","StrategyLeg","StrategyResult"):
            globals()[attr] = locals()[attr]
        return globals()[name]
    raise AttributeError(f"module 'infrastructure' has no attribute {name!r}")

__all__ = [
    # v3.2
    "DataFeedManager","TickData","MarketDataSource",
    "RedisCacheManager","CircularBuffer",
    "EventBus","Event","EventType",
    "WebSocketManager",
    "DataNormalizer","NormalizedTick",
    "HistoricalDataManager",
    "GreeksCalculator","GreeksResult","OptionsLeg","PortfolioGreeks",
    "OptionChainStream","OptionChainSnapshot",
    "PhysicalSettlementManager",
    # v4.0
    "FnoInstrumentManager","FnoContract","OptionChain","StrikeInfo",
    "InstrumentType","get_instrument_manager",
    "MarginCalculator","MarginResult","PortfolioMargin",
    "PositionType","calculate_order_margin",
    "MWPLMonitor","MWPLData","MWPLAlertLevel","ClientPositionLimit","get_mwpl_monitor",
    "StrategyFactory","IronCondorBuilder","CalendarSpreadBuilder",
    "BullSpreadBuilder","BearSpreadBuilder","CollarBuilder",
    "StrategyType","MarketBias","StrategyLeg","StrategyResult",
]
