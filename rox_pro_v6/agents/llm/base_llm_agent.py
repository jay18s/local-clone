"""
Base LLM Agent - Foundation for all LLM-powered trading intelligence
=====================================================================

Provides common infrastructure for LLM integration:
- OpenRouter API client management
- Prompt construction and caching
- Response parsing and validation
- Graceful degradation with fallbacks
- Cost tracking and rate limiting
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from enum import Enum
import threading
import asyncio

import httpx

# ── OpenRouter config from env ───────────────────────────────────────────────
OPENROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API", "")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")

# ── OPENROUTER_MODELS — centralized model routing ───────────────────────────
OPENROUTER_MODELS = {
    "planner": os.getenv("ROX_PLANNER_MODEL", OPENROUTER_DEFAULT_MODEL),
    "debate": os.getenv("ROX_DEBATE_MODEL", OPENROUTER_DEFAULT_MODEL),
    "analysis": os.getenv("ROX_ANALYSIS_MODEL", OPENROUTER_DEFAULT_MODEL),
    "swarm": os.getenv("ROX_SWARM_MODEL", OPENROUTER_DEFAULT_MODEL),
    "fast": os.getenv("ROX_FAST_MODEL", OPENROUTER_DEFAULT_MODEL),
    "smart": os.getenv("ROX_SMART_MODEL", OPENROUTER_DEFAULT_MODEL),
    "news": os.getenv("ROX_NEWS_MODEL", OPENROUTER_DEFAULT_MODEL),
}


@dataclass
class LLMConfig:
    """Configuration for LLM-powered agents."""
    enabled: bool = True
    api_key: str = ""  # OpenRouter API key
    model_name: str = OPENROUTER_DEFAULT_MODEL  # Default model
    fallback_model: str = OPENROUTER_DEFAULT_MODEL
    max_retries: int = 3
    timeout_seconds: int = 30
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_enabled: bool = True
    temperature: float = 0.3  # Low temperature for analytical tasks
    max_output_tokens: int = 8192
    rate_limit_per_minute: int = 15
    log_prompts: bool = True
    log_responses: bool = True
    fallback_on_error: bool = True  # Graceful degradation

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables."""
        api_key = os.getenv("OPEN_ROUTER_API", "") or os.getenv("OPENROUTER_API_KEY", "")
        model_name = os.getenv("LLM_MODEL", "") or \
                     os.getenv("OPEN_ROUTER_MODEL", OPENROUTER_DEFAULT_MODEL)

        return cls(
            enabled=os.getenv("LLM_ENABLED", "true").lower() == "true",
            api_key=api_key,
            model_name=model_name,
            cache_ttl_seconds=int(os.getenv("LLM_CACHE_TTL", "300")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        )


@dataclass
class LLMResponse:
    """Standardized LLM response container."""
    content: str
    parsed_json: Optional[Dict] = None
    raw_response: Any = None
    model_used: str = ""
    tokens_used: int = 0
    latency_ms: int = 0
    cached: bool = False
    source: str = "LLM"  # "LLM" or "FALLBACK"
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry with TTL."""
    response: LLMResponse
    created_at: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)


def _openrouter_headers() -> dict:
    """Build standard OpenRouter request headers."""
    return {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPEN_ROUTER_HTTP_REFERER", "https://rox-engine.local"),
        "X-Title": os.getenv("OPEN_ROUTER_X_TITLE", "ROX Trading Engine"),
    }


def _call_openrouter_sync(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_instruction: Optional[str] = None,
    expect_json: bool = False,
) -> dict:
    """
    Synchronous OpenRouter call via httpx.
    Returns the parsed JSON response dict.
    Raises on non-200.
    """
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if expect_json:
        payload["response_format"] = {"type": "json_object"}
        # FIX-JSON-PROVIDER: Tell OpenRouter to only route to providers that natively
        # support ALL requested parameters, including json_object response_format.
        # Prevents cascading 400/405 failures when free-tier routing picks gemma,
        # stepfun, or other models that reject json_object mode.
        payload["provider"] = {"require_parameters": True}

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=_openrouter_headers(),
            json=payload,
        )

    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:500]}")

    return resp.json()


class BaseLLMAgent:
    """
    Base class for all LLM-powered agents.

    Provides:
    - OpenRouter API client
    - Prompt caching with TTL
    - JSON response parsing with validation
    - Fallback handling
    - Rate limiting
    - Cost tracking
    """

    def __init__(self, config: LLMConfig, logger_name: str = "BaseLLMAgent"):
        self.config = config
        self.logger = logging.getLogger(logger_name)
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0

    def _generate_cache_key(self, prompt: str) -> str:
        """Generate a cache key from prompt."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _get_cached_response(self, prompt: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired."""
        if not self.config.cache_enabled:
            return None

        cache_key = self._generate_cache_key(prompt)

        with self._cache_lock:
            entry = self._cache.get(cache_key)
            if entry and not entry.is_expired():
                self.logger.debug(f"Cache hit for prompt hash: {cache_key[:8]}...")
                return LLMResponse(
                    content=entry.response.content,
                    parsed_json=entry.response.parsed_json,
                    model_used=entry.response.model_used,
                    cached=True,
                    source="LLM"
                )
        return None

    def _cache_response(self, prompt: str, response: LLMResponse):
        """Cache a response."""
        if not self.config.cache_enabled:
            return

        cache_key = self._generate_cache_key(prompt)

        with self._cache_lock:
            self._cache[cache_key] = CacheEntry(
                response=response,
                created_at=datetime.now(),
                ttl_seconds=self.config.cache_ttl_seconds
            )

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        expect_json: bool = True,
        fallback_handler: Optional[Callable[[], LLMResponse]] = None
    ) -> LLMResponse:
        """
        Generate a response from the LLM via OpenRouter.

        Args:
            prompt: The prompt to send
            system_instruction: Optional system instruction
            temperature: Override default temperature
            max_tokens: Override default max tokens
            expect_json: Whether to parse response as JSON
            fallback_handler: Function to call if LLM fails

        Returns:
            LLMResponse with content and metadata
        """
        # Check cache first
        cached = self._get_cached_response(prompt)
        if cached:
            return cached

        # Check API key
        if not OPENROUTER_API_KEY and not self.config.api_key:
            self.logger.warning("No OpenRouter API key configured (set OPEN_ROUTER_API)")
            if fallback_handler and self.config.fallback_on_error:
                return fallback_handler()
            return self._create_fallback_response()

        # Check rate limit
        if not self._rate_limiter.allow_request():
            self.logger.warning("Rate limit exceeded, using fallback")
            if fallback_handler and self.config.fallback_on_error:
                return fallback_handler()
            return self._create_fallback_response()

        actual_model = self.config.model_name
        _temp = temperature if temperature is not None else self.config.temperature
        _max_tokens = max_tokens or self.config.max_output_tokens

        if self.config.log_prompts:
            self.logger.debug(f"LLM Prompt: {prompt[:500]}...")

        start_time = time.time()

        try:
            api_key = self.config.api_key or OPENROUTER_API_KEY
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            payload: Dict[str, Any] = {
                "model": actual_model,
                "messages": messages,
                "temperature": _temp,
                "max_tokens": _max_tokens,
            }
            if expect_json:
                payload["response_format"] = {"type": "json_object"}
                # FIX-JSON-PROVIDER: Restrict OpenRouter routing to models that support
                # json_object mode.  Without this, free-tier routing can select gemma /
                # stepfun variants that return 400/405, taking down all 7 LLM modules in
                # the same cycle.  require_parameters=True is a no-op for named models
                # (e.g. deepseek-v3.2) — it only matters when the router has discretion.
                payload["provider"] = {"require_parameters": True}

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPEN_ROUTER_HTTP_REFERER", "https://rox-engine.local"),
                "X-Title": os.getenv("OPEN_ROUTER_X_TITLE", "ROX Trading Engine"),
            }

            with httpx.Client(timeout=60.0) as client:
                resp = client.post(
                    f"{OPENROUTER_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                )

            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            choice = data.get("choices", [{}])[0]
            content = choice.get("message", {}).get("content", "")
            usage = data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            latency_ms = int((time.time() - start_time) * 1000)

            parsed_json = None
            if expect_json:
                parsed_json = self._parse_json_response(content)

            self._total_requests += 1
            self._total_tokens += tokens_used

            llm_response = LLMResponse(
                content=content,
                parsed_json=parsed_json,
                model_used=actual_model,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                source="LLM"
            )

            self._cache_response(prompt, llm_response)

            if self.config.log_responses:
                self.logger.debug(f"LLM Response: {content[:500]}...")

            return llm_response

        except Exception as e:
            self._total_errors += 1
            self.logger.error(f"LLM generation failed: {e}")

            if fallback_handler and self.config.fallback_on_error:
                return fallback_handler()

            return LLMResponse(
                content="",
                error=str(e),
                source="FALLBACK"
            )

    def _safe_extract_text(self, response: Any) -> str:
        """
        Safely extract text from a response.
        Retained for backward compat; OpenRouter returns text directly.
        """
        if isinstance(response, str):
            return response
        if hasattr(response, 'text'):
            return response.text or ""
        return str(response)

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        if not content:
            return None

        text = content.strip()

        # Try direct JSON parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Collapse bare newlines inside strings
        text_normalised = re.sub(r'[\r\n]+', ' ', text)
        try:
            return json.loads(text_normalised)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        for match in re.findall(json_block_pattern, text):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # Scan for first complete top-level JSON object
        depth = 0
        start_idx = None
        in_string = False
        escape_next = False
        for i, ch in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start_idx is not None:
                    candidate = text[start_idx:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        start_idx = None

        self.logger.warning(f"Failed to parse JSON from response: {content[:200]}...")

        # Last-chance: attempt to repair truncated JSON
        try:
            first_brace = text.find('{')
            if first_brace != -1:
                partial = text[first_brace:]
                depth_brace = 0
                depth_bracket = 0
                in_str = False
                esc = False
                for ch in partial:
                    if esc:
                        esc = False
                        continue
                    if ch == '\\' and in_str:
                        esc = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == '{':
                        depth_brace += 1
                    elif ch == '}':
                        depth_brace -= 1
                    elif ch == '[':
                        depth_bracket += 1
                    elif ch == ']':
                        depth_bracket -= 1
                repair = partial
                if in_str:
                    repair += '"'
                repair += ']' * max(0, depth_bracket)
                repair += '}' * max(0, depth_brace)
                return json.loads(repair)
        except Exception:
            pass

        return None

    def _create_fallback_response(self) -> LLMResponse:
        """Create a fallback response when LLM is unavailable."""
        return LLMResponse(
            content="",
            parsed_json=None,
            source="FALLBACK",
            error="LLM unavailable"
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "total_errors": self._total_errors,
            "cache_size": len(self._cache),
            "model": self.config.model_name,
            "provider": "openrouter",
        }

    def clear_cache(self):
        """Clear the response cache."""
        with self._cache_lock:
            self._cache.clear()
        self.logger.info("LLM response cache cleared")


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self._requests: List[float] = []
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        """Check if a request is allowed under rate limiting."""
        with self._lock:
            now = time.time()
            self._requests = [t for t in self._requests if now - t < 60]

            if len(self._requests) >= self.requests_per_minute:
                return False

            self._requests.append(now)
            return True


T = TypeVar('T')


class PromptBuilder(Generic[T]):
    """
    Helper class for building structured prompts with consistent formatting.
    """

    @staticmethod
    def format_market_data(data: Dict[str, Any]) -> str:
        """Format market data for LLM consumption."""
        lines = []
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'price' in key.lower() or 'level' in key.lower():
                    lines.append(f"- {key}: {value:,.2f}")
                elif 'pct' in key.lower() or 'rate' in key.lower():
                    lines.append(f"- {key}: {value:.2f}%")
                else:
                    lines.append(f"- {key}: {value}")
            elif isinstance(value, dict):
                lines.append(f"- {key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {', '.join(str(v) for v in value[:5])}")
            else:
                lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def format_agent_reports(reports: Dict[str, Any]) -> str:
        """Format agent reports for LLM consumption."""
        lines = []
        for agent_name, report in reports.items():
            if hasattr(report, 'verdict'):
                v = report.verdict
                lines.append(f"- {agent_name}: {v.direction.value} (conviction: {v.conviction}, weight: {v.weight:.2f})")
                if hasattr(v, 'reason') and v.reason:
                    lines.append(f"  Reason: {v.reason[:100]}")
            elif isinstance(report, dict):
                direction = report.get('direction', 'N/A')
                conviction = report.get('conviction', 0)
                lines.append(f"- {agent_name}: {direction} (conviction: {conviction})")
        return "\n".join(lines)
