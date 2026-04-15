"""
Base LLM Agent - Foundation for all LLM-powered trading intelligence
=====================================================================

Provides common infrastructure for LLM integration:
- Gemini API client management (supports both old and new SDK)
- Prompt construction and caching
- Response parsing and validation
- Graceful degradation with fallbacks
- Cost tracking and rate limiting
"""

from __future__ import annotations

import json
import logging
import os
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from enum import Enum
import threading

# Gemini imports - supports both old and new SDK (same pattern as news_core.py)
try:
    from google import genai
    from google.genai import types
    GEMINI_SDK = "new"
except ImportError:
    try:
        import google.generativeai as genai
        GEMINI_SDK = "old"
    except ImportError:
        GEMINI_SDK = "none"
        genai = None
        types = None


@dataclass
class LLMConfig:
    """Configuration for LLM-powered agents."""
    enabled: bool = True
    api_key: str = ""  # Gemini API key
    model_name: str = "gemini-3.1-pro-preview"  # Default model
    fallback_model: str = "gemini-3-flash"
    max_retries: int = 3
    timeout_seconds: int = 30
    cache_ttl_seconds: int = 300  # 5 minutes
    cache_enabled: bool = True
    temperature: float = 0.3  # Low temperature for analytical tasks
    max_output_tokens: int = 8192  # Raised from 2048 — prevents JSON truncation in verbose responses
    rate_limit_per_minute: int = 15
    log_prompts: bool = True
    log_responses: bool = True
    fallback_on_error: bool = True  # Graceful degradation

    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Load configuration from environment variables."""
        # Support multiple env var names for flexibility
        api_key = os.getenv("GEMINI_API_KEY", "") or \
                  os.getenv("GOOGLE_API_KEY", "") or \
                  os.getenv("BRAIN_API_KEY", "")
        
        # Support multiple model name env vars
        model_name = os.getenv("LLM_MODEL", "") or \
                     os.getenv("BRAIN_MODEL", "") or \
                     "gemini-3.1-pro-preview"
        
        # Check provider if specified
        provider = os.getenv("BRAIN_LLM_PROVIDER", "").lower()
        if provider and provider != "gemini":
            return cls(enabled=False, api_key="")
        
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


class BaseLLMAgent:
    """
    Base class for all LLM-powered agents.

    Provides:
    - Gemini API client initialization (supports both old and new SDK)
    - Prompt caching with TTL
    - JSON response parsing with validation
    - Fallback handling
    - Rate limiting
    - Cost tracking
    """

    def __init__(self, config: LLMConfig, logger_name: str = "BaseLLMAgent"):
        self.config = config
        self.logger = logging.getLogger(logger_name)
        self._client = None  # For new SDK: genai.Client
        self._model = None  # For old SDK: GenerativeModel
        self._cache: Dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._rate_limiter = RateLimiter(config.rate_limit_per_minute)
        self._total_requests = 0
        self._total_tokens = 0
        self._total_errors = 0
        self._initialized = False

    def _initialize_client(self) -> bool:
        """Initialize Gemini client lazily (supports both SDKs)."""
        if self._initialized:
            return self._client is not None or self._model is not None

        self._initialized = True

        if GEMINI_SDK == "none":
            self.logger.warning("Google Generative AI library not available")
            return False

        if not self.config.api_key:
            # Support multiple env var names for API key
            self.config.api_key = os.getenv("GEMINI_API_KEY", "") or \
                                  os.getenv("GOOGLE_API_KEY", "") or \
                                  os.getenv("BRAIN_API_KEY", "")

        if not self.config.api_key:
            self.logger.warning("No Gemini API key configured (set GEMINI_API_KEY or GOOGLE_API_KEY)")
            return False

        try:
            if GEMINI_SDK == "new":
                # New SDK (google-genai) — uses Client
                self._client = genai.Client(api_key=self.config.api_key)
                self.logger.info(f"Gemini client initialized (new SDK) with model: {self.config.model_name}")
            else:
                # Old SDK (google-generativeai) — uses configure + GenerativeModel
                genai.configure(api_key=self.config.api_key)
                self._model = genai.GenerativeModel(self.config.model_name)
                self.logger.info(f"Gemini client initialized (old SDK) with model: {self.config.model_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            return False

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
        Generate a response from the LLM.

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

        # Initialize client if needed
        if not self._initialize_client():
            self.logger.warning("LLM client not available, using fallback")
            if fallback_handler and self.config.fallback_on_error:
                return fallback_handler()
            return self._create_fallback_response()

        # Check rate limit
        if not self._rate_limiter.allow_request():
            self.logger.warning("Rate limit exceeded, using fallback")
            if fallback_handler and self.config.fallback_on_error:
                return fallback_handler()
            return self._create_fallback_response()

        # Log prompt if configured
        if self.config.log_prompts:
            self.logger.debug(f"LLM Prompt: {prompt[:500]}...")

        start_time = time.time()

        try:
            # Generate response based on SDK type
            if GEMINI_SDK == "new":
                response = self._generate_new_sdk(prompt, system_instruction, temperature, max_tokens, expect_json)
            else:
                response = self._generate_old_sdk(prompt, system_instruction, temperature, max_tokens, expect_json)

            latency_ms = int((time.time() - start_time) * 1000)

            # Extract content safely — handles thinking models and blocked/empty responses
            content = self._safe_extract_text(response)

            # Parse JSON if expected
            parsed_json = None
            if expect_json:
                parsed_json = self._parse_json_response(content)

            # Get token usage if available
            tokens_used = 0
            if hasattr(response, 'usage_metadata'):
                tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)

            llm_response = LLMResponse(
                content=content,
                parsed_json=parsed_json,
                raw_response=response,
                model_used=self.config.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                cached=False,
                source="LLM"
            )

            # Update stats
            self._total_requests += 1
            self._total_tokens += tokens_used

            # Cache response
            self._cache_response(prompt, llm_response)

            # Log response if configured
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

    def _generate_new_sdk(self, prompt: str, system_instruction: Optional[str],
                          temperature: Optional[float], max_tokens: Optional[int],
                          expect_json: bool = True):
        """Generate using new SDK (google-genai)."""
        config_params = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_output_tokens,
        }

        # Only request JSON mime type when the caller expects JSON output.
        # NOTE: Do NOT set response_mime_type for thinking models (gemini-2.5-pro)
        # as it can conflict with the reasoning/thinking tokens in the response.
        if expect_json and "2.5" not in self.config.model_name:
            config_params["response_mime_type"] = "application/json"

        gen_config = types.GenerateContentConfig(**config_params)

        return self._client.models.generate_content(
            model=self.config.model_name,
            contents=prompt,
            config=gen_config
        )

    def _generate_old_sdk(self, prompt: str, system_instruction: Optional[str],
                          temperature: Optional[float], max_tokens: Optional[int],
                          expect_json: bool = True):
        """Generate using old SDK (google-generativeai)."""
        gen_config = {
            "temperature": temperature or self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_output_tokens,
        }

        # Create model with system instruction if provided
        model = self._model
        if system_instruction:
            model = genai.GenerativeModel(
                self.config.model_name,
                system_instruction=system_instruction
            )

        return model.generate_content(prompt, generation_config=gen_config)

    def _safe_extract_text(self, response: Any) -> str:
        """
        Safely extract text from a Gemini SDK response.

        Handles edge cases:
        - response.text raises TypeError when candidates is None (safety block / quota error)
        - Thinking models (gemini-2.5-pro) which include thought parts alongside text parts
        - Old SDK vs new SDK response shapes
        """
        # 1. Try the simple .text property first
        try:
            text = response.text
            if text is not None:
                return text
        except Exception:
            pass  # Fall through to manual extraction

        # 2. Try walking candidates → content → parts manually
        try:
            candidates = getattr(response, 'candidates', None)
            if candidates:
                parts = candidates[0].content.parts
                # Concatenate all non-thought text parts
                texts = []
                for part in parts:
                    # In thinking models, thought parts have thought=True; skip them
                    if getattr(part, 'thought', False):
                        continue
                    if hasattr(part, 'text') and part.text:
                        texts.append(part.text)
                if texts:
                    return "\n".join(texts)
        except Exception:
            pass

        # 3. Last resort: stringify the response (useful for debugging)
        self.logger.warning("Could not extract text from response; falling back to str(response)")
        return str(response)

    def _parse_json_response(self, content: str) -> Optional[Dict]:
        """Parse JSON from LLM response, handling markdown code blocks and thinking-model output."""
        if not content:
            return None

        # Strip leading/trailing whitespace
        text = content.strip()

        # Try direct JSON parse first (fastest path, works when mime_type=application/json)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block (```json ... ``` or ``` ... ```)
        import re
        json_block_pattern = r'```(?:json)?\s*([\s\S]*?)```'
        for match in re.findall(json_block_pattern, text):
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue

        # For thinking models: scan for the FIRST complete top-level JSON object.
        # We walk character by character to find a balanced { ... } block.
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
                        # Reset and keep looking
                        start_idx = None

        self.logger.warning(f"Failed to parse JSON from response: {content[:200]}...")

        # Last-chance: attempt to repair truncated JSON by closing open braces/brackets.
        # This handles cases where max_output_tokens cuts the response mid-JSON.
        try:
            first_brace = text.find('{')
            if first_brace != -1:
                partial = text[first_brace:]
                # Count unclosed braces and brackets
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
                # Close any open strings/brackets/braces to produce valid JSON
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
            "sdk": GEMINI_SDK,
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
            # Remove requests older than 1 minute
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
