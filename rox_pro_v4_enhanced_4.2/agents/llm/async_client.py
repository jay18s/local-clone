"""
ROX PROVEN EDGE ENGINE v6.0 — OpenRouter Async LLM Client
Supports retry, rate limiting, token tracking, and structured JSON output.
Migrated from Gemini to OpenRouter on 2026-04-17.
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import httpx
import logging

logger = logging.getLogger(__name__)

# ── Environment config ───────────────────────────────────────────────────────
OPENROUTER_BASE_URL = os.getenv("OPEN_ROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_API_KEY = os.getenv("OPEN_ROUTER_API", "")
OPENROUTER_DEFAULT_MODEL = os.getenv("OPEN_ROUTER_MODEL", "openrouter/free")


class LLMError(Exception):
    """Custom exception for LLM API errors."""
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AsyncLLMResponse:
    """v5 standardized async response"""
    text: str
    model: str
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    tokens_prompt: int = 0
    tokens_completion: int = 0
    json_data: Optional[dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Keep LLMResponse as alias for backward compatibility with main_v5_pipeline.py
LLMResponse = AsyncLLMResponse


class OpenRouterClient:
    """
    Async OpenRouter API client with retry logic, rate limiting,
    and automatic JSON extraction. Drop-in replacement for GeminiClient.
    """

    def __init__(self, api_key: str = None, config=None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = OPENROUTER_BASE_URL.rstrip("/")
        self.default_model = OPENROUTER_DEFAULT_MODEL
        self._config = config
        self._semaphore = asyncio.Semaphore(getattr(config, 'max_concurrent', 3) if config else 3)
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
        self._last_call_time = 0.0
        self._cache: dict[str, tuple[float, AsyncLLMResponse]] = {}
        self._cache_ttl = getattr(config, 'cache_ttl_seconds', 300) if config else 300
        self._cache_hits = 0
        self._cache_misses = 0

    def _cache_key(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        raw = f"{model}:{temperature}:{max_tokens}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[AsyncLLMResponse]:
        if key in self._cache:
            timestamp, response = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return response
            del self._cache[key]
        return None

    def _set_cached(self, key: str, response: AsyncLLMResponse):
        if len(self._cache) > 100:
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time(), response)

    @staticmethod
    def _is_timeout_error(error_str: str) -> bool:
        lower = error_str.lower()
        return "timeout" in lower or "timed out" in lower or "readtimeout" in lower

    def _get_headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("OPEN_ROUTER_HTTP_REFERER", "https://rox-engine.local"),
            "X-Title": os.getenv("OPEN_ROUTER_X_TITLE", "ROX Trading Engine"),
        }

    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        system_instruction: str = "",
        expect_json: bool = False,
        response_format: str = "text",
    ) -> AsyncLLMResponse:
        """
        Generate a response from OpenRouter.

        Args:
            prompt: The prompt to send
            model: Model override (default from env)
            temperature: Sampling temperature
            max_tokens: Max output tokens
            system_instruction: System prompt
            expect_json: Whether to parse response as JSON (sets response_format too)
            response_format: "text" or "json"

        Returns:
            AsyncLLMResponse with text and metadata
        """
        # Map expect_json to response_format
        if expect_json and response_format == "text":
            response_format = "json"

        actual_model = model or self.default_model
        cache_key = self._cache_key(prompt, actual_model, temperature, max_tokens)
        cached = self._get_cached(cache_key)
        if cached:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        async with self._semaphore:
            min_interval = 1.0 / (getattr(self._config, 'requests_per_minute', 15) if self._config else 15)
            elapsed = time.time() - self._last_call_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)

            start = time.time()
            max_retries = getattr(self._config, 'retry_max', 3) if self._config else 3

            for attempt in range(1, max_retries + 1):
                try:
                    # Build messages
                    messages = []
                    if system_instruction:
                        messages.append({"role": "system", "content": system_instruction})
                    messages.append({"role": "user", "content": prompt})

                    # Build payload
                    payload: Dict[str, Any] = {
                        "model": actual_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }

                    if response_format == "json":
                        payload["response_format"] = {"type": "json_object"}

                    async with httpx.AsyncClient(timeout=60.0) as client:
                        resp = await client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self._get_headers(),
                            json=payload,
                        )

                    if resp.status_code != 200:
                        error_body = resp.text[:500]
                        raise LLMError(
                            f"OpenRouter API error {resp.status_code}: {error_body}",
                            status_code=resp.status_code,
                        )

                    data = resp.json()

                    # Extract response
                    choice = data.get("choices", [{}])[0]
                    text = choice.get("message", {}).get("content", "")
                    finish_reason = choice.get("finish_reason", "")

                    # Token usage
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                    latency_ms = int((time.time() - start) * 1000)
                    self._call_count += 1
                    self._total_tokens += prompt_tokens + completion_tokens
                    self._total_latency_ms += latency_ms
                    self._last_call_time = time.time()

                    json_data = self._extract_json(text) if (expect_json or response_format == "json") else None

                    result = AsyncLLMResponse(
                        text=text,
                        json_data=json_data,
                        model=actual_model,
                        tokens_prompt=prompt_tokens,
                        tokens_completion=completion_tokens,
                        latency_ms=latency_ms,
                        success=True,
                    )
                    self._set_cached(cache_key, result)
                    return result

                except LLMError:
                    raise
                except Exception as e:
                    error_str = str(e)
                    # Retry on timeout once
                    if self._is_timeout_error(error_str) and attempt < max_retries:
                        logger.warning(f"[OpenRouter] Timeout on attempt {attempt}, retrying...")
                        await asyncio.sleep(2.0 * attempt)
                        continue
                    if attempt < max_retries:
                        delay = 2.0 * attempt
                        await asyncio.sleep(delay)
                    else:
                        return AsyncLLMResponse(
                            text="",
                            model=actual_model,
                            latency_ms=int((time.time() - start) * 1000),
                            success=False,
                            error=error_str,
                        )

    async def generate_parallel(self, calls: list[dict]) -> list[AsyncLLMResponse]:
        """
        Run multiple generate calls in parallel.
        Maintains GeminiClient.generate_parallel() interface.

        Each dict in calls should have: prompt, model, temperature, etc.
        Maps 'expect_json' from calls; if dict has 'system_instruction', passes it.
        """
        tasks = [self.generate(**c) for c in calls]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except Exception:
                pass
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        try:
            return json.loads(text.strip())
        except Exception:
            return None

    def get_stats(self) -> dict:
        avg = self._total_latency_ms / self._call_count if self._call_count else 0
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": round(avg, 1),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }


import re

# ── Module-level singleton ───────────────────────────────────────────────────
llm_client = OpenRouterClient()

# Backward-compatible alias used by debate_engine.py and other modules
GeminiClient = OpenRouterClient
