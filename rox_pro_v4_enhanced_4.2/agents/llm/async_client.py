"""
ROX PROVEN EDGE ENGINE v5.0 — Async Gemini LLM Client
Supports retry, rate limiting, token tracking, and structured JSON output.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("llm.client")

try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logger.warning("google-generativeai not installed. pip install google-generativeai")


@dataclass
class AsyncLLMResponse:
    """Standardized response from any LLM call."""
    text: str
    json_data: Optional[dict] = None
    model: str = ""
    tokens_prompt: int = 0
    tokens_completion: int = 0
    latency_ms: int = 0
    success: bool = True
    error: Optional[str] = None
    
    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion


class GeminiClient:
    """
    Async Gemini API client with retry logic, rate limiting,
    and automatic JSON extraction.
    """
    
    def __init__(self, api_key: str, config=None):
        if genai is None:
            raise ImportError("Install google-generativeai: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self._semaphore = asyncio.Semaphore(config.max_concurrent if config else 3)
        self._config = config
        self._call_count = 0
        self._total_tokens = 0
        self._total_latency_ms = 0
        self._last_call_time = 0.0
        self._cache: dict[str, tuple[float, AsyncLLMResponse]] = {}  # {hash: (timestamp, response)}
        self._cache_ttl = getattr(config, 'cache_ttl_seconds', 300) if config else 300
        self._cache_hits = 0
        self._cache_misses = 0
        
    def _cache_key(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate a cache key from call parameters."""
        raw = f"{model}:{temperature}:{max_tokens}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()
    
    def _get_cached(self, key: str) -> Optional[AsyncLLMResponse]:
        """Check cache for a valid (non-expired) response."""
        if key in self._cache:
            timestamp, response = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return response
            else:
                del self._cache[key]
        return None
    
    def _set_cached(self, key: str, response: AsyncLLMResponse):
        """Store response in cache."""
        # Limit cache size to prevent memory bloat
        if len(self._cache) > 100:
            # Remove oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (time.time(), response)
    
    @staticmethod
    def _is_rate_limit_error(error_str: str) -> bool:
        """Detect 429 / rate-limit errors from the API."""
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
        """
        Generate a completion from the Gemini API.
        
        Args:
            prompt: The user prompt.
            model: Gemini model identifier.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum output tokens.
            system_instruction: Optional system prompt.
            expect_json: If True, attempt to parse response as JSON.
        
        Returns:
            AsyncLLMResponse with text, optional JSON, and metadata.
        """
        # --- Cache check (outside semaphore for fast path) ---
        cache_key = self._cache_key(prompt, model, temperature, max_tokens)
        cached = self._get_cached(cache_key)
        if cached is not None:
            self._cache_hits += 1
            logger.info(f"LLM cache HIT | model={model} | cache_size={len(self._cache)}")
            return cached
        self._cache_misses += 1
        
        async with self._semaphore:
            # Rate limiting: minimum interval between calls
            min_interval = 1.0 / (self._config.requests_per_minute if self._config else 15)
            elapsed = time.time() - self._last_call_time
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            
            start_ms = time.monotonic() * 1000
            
            # Retry loop — exponential backoff: 2s, 4s, 8s (general) / 5s, 10s, 20s (429)
            max_retries = self._config.retry_max if self._config else 3
            general_backoffs = [2.0, 4.0, 8.0]
            rate_limit_backoffs = [5.0, 10.0, 20.0]
            
            for attempt in range(1, max_retries + 1):
                try:
                    gen_model = genai.GenerativeModel(
                        model_name=model,
                        system_instruction=system_instruction if system_instruction else None,
                    )
                    
                    generation_config = genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    
                    response = await asyncio.to_thread(
                        gen_model.generate_content,
                        prompt,
                        generation_config=generation_config,
                    )
                    
                    text = response.text
                    latency_ms = int(time.monotonic() * 1000 - start_ms)
                    
                    # Token estimation (rough)
                    prompt_tokens = len(prompt.split()) * 1.3
                    completion_tokens = len(text.split()) * 1.3
                    
                    self._call_count += 1
                    self._total_tokens += int(prompt_tokens + completion_tokens)
                    self._total_latency_ms += latency_ms
                    self._last_call_time = time.time()
                    
                    # JSON extraction
                    json_data = None
                    if expect_json:
                        json_data = self._extract_json(text)
                    
                    logger.info(
                        f"LLM response OK | model={model} | "
                        f"latency={latency_ms}ms | tokens~{int(prompt_tokens + completion_tokens)} | "
                        f"attempt={attempt}"
                    )
                    
                    result = AsyncLLMResponse(
                        text=text,
                        json_data=json_data,
                        model=model,
                        tokens_prompt=int(prompt_tokens),
                        tokens_completion=int(completion_tokens),
                        latency_ms=latency_ms,
                        success=True,
                    )
                    
                    # Store successful response in cache
                    self._set_cached(cache_key, result)
                    return result
                    
                except Exception as e:
                    latency_ms = int(time.monotonic() * 1000 - start_ms)
                    error_str = str(e)
                    
                    if attempt < max_retries:
                        # Choose backoff schedule based on error type
                        backoffs = rate_limit_backoffs if self._is_rate_limit_error(error_str) else general_backoffs
                        delay = backoffs[attempt - 1]
                        label = "rate-limit" if self._is_rate_limit_error(error_str) else "general"
                        
                        logger.warning(
                            f"LLM call failed ({label}, attempt {attempt}/{max_retries}): {error_str} | "
                            f"retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"LLM call FAILED after {max_retries} attempts: {error_str}"
                        )
                        return AsyncLLMResponse(
                            text="",
                            model=model,
                            latency_ms=latency_ms,
                            success=False,
                            error=error_str,
                        )
            
            # Should not reach here
            return AsyncLLMResponse(text="", model=model, success=False, error="Unexpected error")
    
    async def generate_parallel(
        self,
        calls: list[dict],
    ) -> list[AsyncLLMResponse]:
        """
        Execute multiple LLM calls in parallel (respecting concurrency limit).
        
        Args:
            calls: List of dicts with keys: prompt, model, temperature, 
                   system_instruction, expect_json.
        
        Returns:
            List of AsyncLLMResponse objects in the same order as input calls.
        """
        tasks = [
            self.generate(
                prompt=c.get("prompt", ""),
                model=c.get("model", "gemini-2.0-flash"),
                temperature=c.get("temperature", 0.3),
                max_tokens=c.get("max_tokens", 4096),
                system_instruction=c.get("system_instruction", ""),
                expect_json=c.get("expect_json", False),
            )
            for c in calls
        ]
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """
        Extract JSON from LLM response text.
        Handles markdown code blocks and raw JSON.
        """
        import re
        
        # Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try parsing entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        logger.warning("Failed to extract JSON from LLM response")
        return None
    
    def get_stats(self) -> dict:
        """Return API usage statistics including cache metrics."""
        avg_latency = self._total_latency_ms / self._call_count if self._call_count > 0 else 0
        return {
            "total_calls": self._call_count,
            "total_tokens_estimate": self._total_tokens,
            "total_latency_ms": self._total_latency_ms,
            "avg_latency_ms": round(avg_latency, 1),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
            "cache_ttl_seconds": self._cache_ttl,
        }
