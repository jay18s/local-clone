"""
ROX Proven Edge Engine — AI Brain (Multi-LLM)
=============================================
Meta-intelligence layer on top of the 7-agent system.
Supports Claude (Anthropic), GPT (OpenAI), Gemini (Google), Groq.

Configure via .env — no code changes needed to switch providers:

    BRAIN_LLM_PROVIDER=anthropic          # anthropic | openai | gemini | groq
    BRAIN_MODEL=claude-sonnet-4-6         # model name for that provider
    ANTHROPIC_API_KEY=sk-ant-...
    OPENAI_API_KEY=sk-...
    GOOGLE_API_KEY=AIza...
    GROQ_API_KEY=gsk_...

What the Brain does:
  1. SYNTHESIZE   — reads full consensus + setups, explains WHY in plain English
  2. CROSS-EXAMINE — challenges each setup against news / macro context
  3. RE-RANK      — adjusts conviction scores accounting for correlation & regime
  4. LEARN        — post-mortem after trades close, persists in brain_memory.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rox.brain")

# --------------------------------------------------------------------------- #
#  Model defaults per provider                                                #
# --------------------------------------------------------------------------- #

PROVIDER_DEFAULTS: Dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "gemini":    "gemini-2.5-pro",
    "groq":      "llama-3.3-70b-versatile",
}

# --------------------------------------------------------------------------- #
#  Dataclasses                                                                #
# --------------------------------------------------------------------------- #

@dataclass
class AdjustedSetup:
    symbol:           str
    original_rank:    int
    brain_rank:       int
    direction:        str
    conviction:       int
    brain_conviction: int
    brain_note:       str
    proceed:          bool


@dataclass
class BrainOutput:
    timestamp:       str
    provider:        str
    model:           str
    narrative:       str
    market_stance:   str   # BULLISH | NEUTRAL | BEARISH | CAUTIOUS
    risk_score:      int   # 1-10
    top_setup:       str
    adjusted_setups: List[AdjustedSetup]
    cautions:        List[str]
    options_note:    str
    learning_note:   str
    latency_sec:     float = 0.0
    tokens_used:     int   = 0
    raw_response:    str   = ""


@dataclass
class TradeResult:
    symbol:     str
    direction:  str
    entry:      float
    exit:       float
    pnl:        float
    held_days:  int
    notes:      str = ""


# --------------------------------------------------------------------------- #
#  Prompts                                                                    #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are ROX Brain, the meta-intelligence layer of a
professional algorithmic trading system called ROX Proven Edge Engine.

Your job is to synthesize the outputs of 8 specialised sub-agents
(Orion, Vesper, Kairo, Sentinel, Nexus, Prudence, Catalyst, Optimus)
and produce a final, holistic assessment for the Indian equity market.

You think like a seasoned senior portfolio manager at a prop desk.
You are direct, risk-conscious, concise, and never verbose.
You always respond with VALID JSON only — no markdown fences, no preamble.

Response schema (strict):
{
  "narrative":       "<2-3 paragraph synthesis in plain English>",
  "market_stance":   "BULLISH|NEUTRAL|BEARISH|CAUTIOUS",
  "risk_score":      <integer 1-10>,
  "top_setup":       "<SYMBOL>",
  "adjusted_setups": [
    {
      "symbol":           "<SYMBOL>",
      "original_rank":    <int>,
      "brain_rank":       <int>,
      "brain_conviction": <int 0-100>,
      "brain_note":       "<one sentence reason>",
      "proceed":          true|false
    }
  ],
  "cautions":      ["<string>", ...],
  "options_note":  "<one sentence on the options signal>",
  "learning_note": "<empty string unless trade_results were provided>"
}"""


def _build_user_prompt(
    consensus:      Dict[str, Any],
    setups:         List[Dict[str, Any]],
    market_context: Dict[str, Any],
    options_signal: Optional[Dict[str, Any]] = None,
    trade_results:  Optional[List[TradeResult]] = None,
    memory:         Optional[str] = None,
) -> str:
    parts = []
    if memory:
        parts.append(f"=== BRAIN MEMORY (last 5 sessions) ===\n{memory}")
    parts.append(f"=== DATE / TIME ===\n{datetime.now().strftime('%Y-%m-%d %H:%M IST')}")
    parts.append("=== MULTI-AGENT CONSENSUS ===\n" + json.dumps(consensus, indent=2))
    parts.append("=== TOP SWING SETUPS ===\n"      + json.dumps(setups,    indent=2))
    if options_signal:
        parts.append("=== OPTIONS SIGNAL ===\n"    + json.dumps(options_signal, indent=2))
    if market_context:
        parts.append("=== MARKET CONTEXT ===\n"    + json.dumps(market_context, indent=2))
    if trade_results:
        parts.append("=== RECENTLY CLOSED TRADES (post-mortem) ===\n" + json.dumps(
            [{"symbol": r.symbol, "direction": r.direction, "entry": r.entry,
              "exit": r.exit, "pnl": r.pnl, "held_days": r.held_days, "notes": r.notes}
             for r in trade_results], indent=2))
    parts.append("\nSynthesize ALL of the above and return your JSON response.")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
#  Memory                                                                     #
# --------------------------------------------------------------------------- #

MEMORY_PATH = Path(__file__).parent.parent / "data" / "brain_memory.jsonl"


def _load_memory(last_n: int = 5) -> str:
    if not MEMORY_PATH.exists():
        return ""
    lines = [l for l in MEMORY_PATH.read_text().strip().split("\n") if l.strip()]
    entries = []
    for line in lines[-last_n:]:
        try:
            o = json.loads(line)
            entries.append(
                f"[{o['date']}] provider={o.get('provider','?')} "
                f"stance={o['market_stance']} risk={o['risk_score']} "
                f"top={o['top_setup']} cautions={o.get('cautions', [])}"
            )
        except Exception:
            pass
    return "\n".join(entries)


def _save_memory(output: BrainOutput):
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "date":          output.timestamp[:10],
        "provider":      output.provider,
        "model":         output.model,
        "market_stance": output.market_stance,
        "risk_score":    output.risk_score,
        "top_setup":     output.top_setup,
        "cautions":      output.cautions,
        "narrative":     output.narrative[:300],
    }
    with open(MEMORY_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
#  LLM Provider backends (all synchronous — matches ROX threading model)     #
# --------------------------------------------------------------------------- #

class _AnthropicBackend:
    """Claude via Anthropic SDK  →  pip install anthropic"""

    def __init__(self, api_key: str, model: str, max_tokens: int):
        self.api_key    = api_key
        self.model      = model
        self.max_tokens = max_tokens

    def call(self, system: str, user: str):
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Run: pip install anthropic")
        client  = anthropic.Anthropic(api_key=self.api_key)
        msg     = client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        tokens = msg.usage.input_tokens + msg.usage.output_tokens
        return msg.content[0].text, tokens


class _OpenAIBackend:
    """GPT via OpenAI SDK  →  pip install openai"""

    def __init__(self, api_key: str, model: str, max_tokens: int):
        self.api_key    = api_key
        self.model      = model
        self.max_tokens = max_tokens

    def call(self, system: str, user: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Run: pip install openai")
        client = OpenAI(api_key=self.api_key)
        resp   = client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        return resp.choices[0].message.content, tokens


class _GeminiBackend:
    """
    Gemini via Google GenAI SDK.

    gemini-2.5-pro requires the NEW sdk:  pip install google-genai
    Older models (gemini-1.5-pro) used:   pip install google-generativeai

    This backend tries the new SDK first, then falls back to the old one so
    existing installations are not broken.
    """

    def __init__(self, api_key: str, model: str, max_tokens: int):
        self.api_key    = api_key
        self.model      = model
        self.max_tokens = max_tokens

    def call(self, system: str, user: str):
        # ── Try new google-genai SDK first (required for gemini-2.5-pro) ──
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model   = self.model,
                contents= user,
                config  = types.GenerateContentConfig(
                    system_instruction = system,
                    max_output_tokens  = self.max_tokens,
                    response_mime_type = "application/json",
                ),
            )
            tokens = (response.usage_metadata.total_token_count
                      if hasattr(response, "usage_metadata") else 0)
            text = response.text
            if text is None:
                # Gemini returned no text — decode finish_reason for clarity.
                finish_reason_map = {
                    1: "STOP (normal)", 2: "MAX_TOKENS — increase BRAIN_MAX_TOKENS in .env",
                    3: "SAFETY block", 4: "RECITATION", 5: "OTHER",
                }
                finish = None
                try:
                    fr = response.candidates[0].finish_reason
                    finish = finish_reason_map.get(int(fr), str(fr))
                except Exception:
                    pass
                raise RuntimeError(
                    f"Gemini returned None text (finish_reason={finish}). "
                    "Check safety settings or increase BRAIN_MAX_TOKENS."
                )
            return text, tokens

        except ImportError:
            pass  # fall through to old SDK

        # ── Fallback: old google-generativeai SDK (gemini-1.5-pro etc.) ──
        try:
            import google.generativeai as genai_old
        except ImportError:
            raise RuntimeError(
                "No Gemini SDK found. Install with:\n"
                "  pip install google-genai          # for gemini-2.5-pro (recommended)\n"
                "  pip install google-generativeai   # for gemini-1.5-pro (legacy)"
            )

        genai_old.configure(api_key=self.api_key)
        mdl  = genai_old.GenerativeModel(
            model_name         = self.model,
            system_instruction = system,
            generation_config  = genai_old.GenerationConfig(
                max_output_tokens  = self.max_tokens,
                response_mime_type = "application/json",
            ),
        )
        resp   = mdl.generate_content(user)
        tokens = (resp.usage_metadata.total_token_count
                  if hasattr(resp, "usage_metadata") else 0)
        text = resp.text
        if text is None:
            finish_reason_map = {
                1: "STOP (normal)", 2: "MAX_TOKENS — increase BRAIN_MAX_TOKENS in .env",
                3: "SAFETY block", 4: "RECITATION", 5: "OTHER",
            }
            finish = None
            try:
                fr = resp.candidates[0].finish_reason
                finish = finish_reason_map.get(int(fr), str(fr))
            except Exception:
                pass
            raise RuntimeError(
                f"Gemini (legacy SDK) returned None text (finish_reason={finish}). "
                "Check safety settings or increase BRAIN_MAX_TOKENS."
            )
        return text, tokens


class _GroqBackend:
    """Fast inference via Groq SDK  →  pip install groq"""

    def __init__(self, api_key: str, model: str, max_tokens: int):
        self.api_key    = api_key
        self.model      = model
        self.max_tokens = max_tokens

    def call(self, system: str, user: str):
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("Run: pip install groq")
        client = Groq(api_key=self.api_key)
        resp   = client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            response_format={"type": "json_object"},
        )
        tokens = resp.usage.total_tokens if resp.usage else 0
        return resp.choices[0].message.content, tokens


_BACKENDS = {
    "anthropic": _AnthropicBackend,
    "openai":    _OpenAIBackend,
    "gemini":    _GeminiBackend,
    "groq":      _GroqBackend,
}


# --------------------------------------------------------------------------- #
#  Main class                                                                 #
# --------------------------------------------------------------------------- #

class AIBrain:
    """
    Multi-LLM meta-intelligence for ROX.
    All config via .env — see BRAIN_LLM_PROVIDER, BRAIN_MODEL, *_API_KEY.
    """

    def __init__(self, use_memory: bool = True):
        self.use_memory = use_memory
        self._backend   = None
        self._provider  = "offline"
        self._model     = "none"
        self._enabled   = os.environ.get("BRAIN_ENABLED", "true").lower() != "false"
        self._setup_backend()

    def _setup_backend(self):
        if not self._enabled:
            logger.info("AIBrain disabled (BRAIN_ENABLED=false)")
            return

        provider = os.environ.get("BRAIN_LLM_PROVIDER", "anthropic").lower().strip()
        if provider not in _BACKENDS:
            logger.warning(f"AIBrain: unknown provider '{provider}'. "
                           f"Valid options: {list(_BACKENDS.keys())}. Brain disabled.")
            return

        key_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai":    "OPENAI_API_KEY",
            "gemini":    "GOOGLE_API_KEY",
            "groq":      "GROQ_API_KEY",
        }
        env_key = key_map[provider]
        api_key = os.environ.get(env_key, "").strip()
        if not api_key:
            logger.warning(f"AIBrain: {env_key} not set — brain disabled")
            return

        model      = os.environ.get("BRAIN_MODEL", "").strip() or PROVIDER_DEFAULTS[provider]
        max_tokens = int(os.environ.get("BRAIN_MAX_TOKENS", "8192"))

        try:
            self._backend  = _BACKENDS[provider](api_key, model, max_tokens)
            self._provider = provider
            self._model    = model
            logger.info(f"AIBrain ready — provider={provider} | model={model}")
        except Exception as e:
            logger.error(f"AIBrain init failed: {e}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def synthesize_sync(
        self,
        consensus:      Dict[str, Any],
        setups:         List[Dict[str, Any]],
        market_context: Dict[str, Any],
        options_signal: Optional[Dict[str, Any]] = None,
        trade_results:  Optional[List[TradeResult]] = None,
    ) -> BrainOutput:
        """
        Main entry point — synchronous, safe to call from any context.
        """
        if not self._backend:
            return self._fallback(setups)

        memory   = _load_memory() if self.use_memory else None
        user_msg = _build_user_prompt(
            consensus, setups, market_context,
            options_signal, trade_results, memory
        )

        t0 = time.monotonic()
        try:
            raw, tokens = self._backend.call(SYSTEM_PROMPT, user_msg)
        except Exception as e:
            logger.error(f"AIBrain LLM call failed ({self._provider}): {e}")
            return self._fallback(setups)

        latency = time.monotonic() - t0
        logger.info(f"AIBrain: {self._provider}/{self._model} | "
                    f"{latency:.1f}s | {tokens} tokens")

        output = self._parse(raw, setups, latency, tokens)
        _save_memory(output)
        return output

    def post_mortem_sync(self, trade_results: List[TradeResult]) -> str:
        """Standalone post-market post-mortem. Returns plain text."""
        if not self._backend:
            return "AIBrain not configured."
        prompt = (
            "These trades just closed. Write a concise post-mortem: "
            "what worked, what failed, one improvement rule for next time.\n\n"
            + json.dumps([
                {"symbol": r.symbol, "pnl": r.pnl,
                 "held_days": r.held_days, "notes": r.notes}
                for r in trade_results
            ], indent=2)
        )
        try:
            raw, _ = self._backend.call(
                "You are a trading coach. Be blunt, concise, actionable.", prompt
            )
            return raw
        except Exception as e:
            return f"Post-mortem failed: {e}"

    @property
    def is_ready(self) -> bool:
        return self._backend is not None

    @property
    def info(self) -> str:
        return f"{self._provider}/{self._model}"

    # ------------------------------------------------------------------ #
    #  Internals                                                          #
    # ------------------------------------------------------------------ #

    def _parse(self, raw: str, setups: List[Dict],
               latency: float, tokens: int) -> BrainOutput:
        now = datetime.now().isoformat()
        if not raw:
            logger.error("Brain received empty/None response from LLM — falling back")
            return self._fallback(setups, raw or "", latency, tokens)
        try:
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.error(f"Brain JSON parse error: {e} | raw[:300]={raw[:300]}")
            return self._fallback(setups, raw, latency, tokens)

        adjusted = []
        for i, s in enumerate(data.get("adjusted_setups", [])):
            orig = next(
                (x.get("conviction", 65) for x in setups
                 if x.get("stock") == s.get("symbol")), 65
            )
            adjusted.append(AdjustedSetup(
                symbol           = s.get("symbol", ""),
                original_rank    = s.get("original_rank", i + 1),
                brain_rank       = s.get("brain_rank",    i + 1),
                direction        = s.get("direction", "LONG"),
                conviction       = orig,
                brain_conviction = s.get("brain_conviction", orig),
                brain_note       = s.get("brain_note", ""),
                proceed          = s.get("proceed", True),
            ))

        return BrainOutput(
            timestamp       = now,
            provider        = self._provider,
            model           = self._model,
            narrative       = data.get("narrative", ""),
            market_stance   = data.get("market_stance", "NEUTRAL"),
            risk_score      = int(data.get("risk_score", 5)),
            top_setup       = data.get("top_setup", ""),
            adjusted_setups = adjusted,
            cautions        = data.get("cautions", []),
            options_note    = data.get("options_note", ""),
            learning_note   = data.get("learning_note", ""),
            latency_sec     = round(latency, 2),
            tokens_used     = tokens,
            raw_response    = raw,
        )

    def _fallback(self, setups: List[Dict], raw: str = "",
                  latency: float = 0, tokens: int = 0) -> BrainOutput:
        top = setups[0].get("stock", "") if setups else ""
        adjusted = [
            AdjustedSetup(
                symbol=s.get("stock", ""), original_rank=i + 1, brain_rank=i + 1,
                direction=s.get("direction", "LONG"),
                conviction=s.get("conviction", 65),
                brain_conviction=s.get("conviction", 65),
                brain_note="Brain unavailable — using raw agent output.",
                proceed=True,
            )
            for i, s in enumerate(setups)
        ]
        return BrainOutput(
            timestamp="", provider="offline", model="none",
            narrative="AI Brain offline. Using raw agent consensus.",
            market_stance="NEUTRAL", risk_score=5, top_setup=top,
            adjusted_setups=adjusted,
            cautions=["AIBrain offline — check .env LLM config"],
            options_note="", learning_note="",
            latency_sec=latency, tokens_used=tokens, raw_response=raw,
        )


# --------------------------------------------------------------------------- #
#  Terminal pretty-printer                                                    #
# --------------------------------------------------------------------------- #

def print_brain_output(output: BrainOutput):
    sep  = "=" * 65
    sep2 = "-" * 65
    ts   = output.timestamp[:16] if output.timestamp else "offline"
    print(f"\n{sep}")
    print(f"AI BRAIN  |  {ts}  |  {output.provider.upper()} / {output.model}")
    print(sep)
    print(f"\nMARKET STANCE : {output.market_stance}  |  "
          f"RISK SCORE : {output.risk_score}/10  |  "
          f"{output.latency_sec}s | {output.tokens_used} tokens\n")
    print(output.narrative)

    if output.cautions:
        print("\n⚠️  CAUTIONS")
        for c in output.cautions:
            print(f"   • {c}")

    print(f"\n{sep}")
    print("BRAIN-ADJUSTED SETUPS  (re-ranked)")
    print(sep2)
    for s in sorted(output.adjusted_setups, key=lambda x: x.brain_rank):
        flag  = "✅" if s.proceed else "🚫"
        delta = s.brain_conviction - s.conviction
        d_str = f"(+{delta})" if delta > 0 else f"({delta})" if delta < 0 else "(=)"
        print(f"\n  #{s.brain_rank}  {flag}  {s.symbol}  [{s.direction}]  "
              f"Conviction: {s.conviction} → {s.brain_conviction} {d_str}")
        print(f"       {s.brain_note}")

    print(f"\n  ⭐ TOP PICK : {output.top_setup}")
    if output.options_note:
        print(f"  📊 OPTIONS  : {output.options_note}")
    if output.learning_note:
        print(f"  📚 LEARNING : {output.learning_note}")
    print(f"\n{sep}\n")
