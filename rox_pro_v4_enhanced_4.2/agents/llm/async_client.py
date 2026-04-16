"""
ROX PROVEN EDGE ENGINE v5.0 — Async Gemini LLM Client
Supports retry, rate limiting, token tracking, and structured JSON output.
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from google import genai
from google.genai import types
import logging

logger = logging.getLogger(__name__)

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

class GeminiClient:
    """
    Async Gemini API client with retry logic, rate limiting,
    and automatic JSON extraction. Uses google-genai SDK.
    """

    def __init__(self, api_key: str, config=None):
        self.client = genai.Client(api_key=api_key or os.getenv('GEMINI_API_KEY'))
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
    def _is_rate_limit_error(error_str: str) -> bool:
        lower = error_str.lower()
        return "429" in lower or "resource_exhausted" in lower or "rate limit" in lower

    async def generate(
        self,
        prompt: str,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.3,
        max_tokens: int = 4096,
        system_instruction: str = "",
        expect_json: bool = False,
    ) -> AsyncLLMResponse:
        cache_key = self._cache_key(prompt, model, temperature, max_tokens)
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
                    # Build config for new SDK
                    gen_config = types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                        system_instruction=system_instruction if system_instruction else None,
                    )

                    # New SDK is sync — run in thread
                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=model,
                        contents=prompt,
                        config=gen_config
                    )

                    text = response.text or ""
                    latency_ms = int((time.time() - start) * 1000)

                    # Token counts from usage_metadata if available
                    usage = getattr(response, 'usage_metadata', None)
                    prompt_tokens = getattr(usage, 'prompt_token_count', 0) if usage else int(len(prompt.split()) * 1.3)
                    completion_tokens = getattr(usage, 'candidates_token_count', 0) if usage else int(len(text.split()) * 1.3)

                    self._call_count += 1
                    self._total_tokens += prompt_tokens + completion_tokens
                    self._total_latency_ms += latency_ms
                    self._last_call_time = time.time()

                    json_data = self._extract_json(text) if expect_json else None

                    result = AsyncLLMResponse(
                        text=text,
                        json_data=json_data,
                        model=model,
                        tokens_prompt=prompt_tokens,
                        tokens_completion=completion_tokens,
                        latency_ms=latency_ms,
                        success=True,
                    )
                    self._set_cached(cache_key, result)
                    return result

                except Exception as e:
                    error_str = str(e)
                    if attempt < max_retries:
                        delay = 5.0 if self._is_rate_limit_error(error_str) else 2.0 * attempt
                        await asyncio.sleep(delay)
                    else:
                        return AsyncLLMResponse(
                            text="",
                            model=model,
                            latency_ms=int((time.time() - start) * 1000),
                            success=False,
                            error=error_str,
                        )

    async def generate_parallel(self, calls: list[dict]) -> list[AsyncLLMResponse]:
        tasks = [self.generate(**c) for c in calls]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        import re
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try: return json.loads(json_match.group(1))
            except: pass
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try: return json.loads(json_match.group(0))
            except: pass
        try: return json.loads(text.strip())
        except: return None

    def get_stats(self) -> dict:
        avg = self._total_latency_ms / self._call_count if self._call_count else 0
        return {
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "avg_latency_ms": round(avg, 1),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }