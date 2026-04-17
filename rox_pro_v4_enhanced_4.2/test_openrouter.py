"""
Quick smoke test for OpenRouter client migration.
Run: python test_openrouter.py
Requires: OPEN_ROUTER_API env var set.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.llm.async_client import OpenRouterClient, llm_client, AsyncLLMResponse


async def test_basic_generate():
    """Test basic generate() call."""
    print("[TEST] Basic generate()...")
    client = OpenRouterClient()
    resp = await client.generate(
        prompt="Say hello in exactly 3 words.",
        temperature=0.1,
        max_tokens=50,
    )
    assert isinstance(resp, AsyncLLMResponse), f"Expected AsyncLLMResponse, got {type(resp)}"
    assert resp.text, "Empty response text"
    assert resp.success, f"Generation failed: {resp.error}"
    print(f"  ✅ Response: '{resp.text.strip()}' (model={resp.model}, {resp.latency_ms}ms)")


async def test_json_mode():
    """Test JSON response_format."""
    print("[TEST] JSON mode...")
    client = OpenRouterClient()
    resp = await client.generate(
        prompt='Return {"answer": 42} as JSON.',
        response_format="json",
        temperature=0.0,
        max_tokens=100,
    )
    assert resp.success, f"JSON generation failed: {resp.error}"
    assert resp.json_data is not None, f"No JSON parsed from: {resp.text[:200]}"
    assert resp.json_data.get("answer") == 42, f"Unexpected JSON: {resp.json_data}"
    print(f"  ✅ JSON parsed: {resp.json_data}")


async def test_parallel():
    """Test generate_parallel()."""
    print("[TEST] Parallel generate()...")
    client = OpenRouterClient()
    calls = [
        {"prompt": "Reply with exactly: ALOHA", "temperature": 0.0, "max_tokens": 20},
        {"prompt": "Reply with exactly: BONJOUR", "temperature": 0.0, "max_tokens": 20},
    ]
    results = await client.generate_parallel(calls)
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    for i, r in enumerate(results):
        assert r.success, f"Parallel call {i} failed: {r.error}"
    print(f"  ✅ Parallel results: '{results[0].text.strip()}' | '{results[1].text.strip()}'")


async def test_singleton():
    """Test the module-level llm_client singleton."""
    print("[TEST] Singleton llm_client...")
    resp = await llm_client.generate(
        prompt="Reply with OK.",
        temperature=0.0,
        max_tokens=10,
    )
    assert resp.success, f"Singleton call failed: {resp.error}"
    print(f"  ✅ Singleton response: '{resp.text.strip()}'")


async def test_stats():
    """Test get_stats()."""
    print("[TEST] Stats...")
    stats = llm_client.get_stats()
    assert "total_calls" in stats
    assert "total_tokens" in stats
    print(f"  ✅ Stats: {stats}")


async def main():
    print("=" * 60)
    print("ROX OpenRouter Migration — Smoke Test")
    print("=" * 60)

    api_key = os.getenv("OPEN_ROUTER_API", "")
    if not api_key:
        print("⚠️  OPEN_ROUTER_API not set — tests will fail on API calls")
        print("   Set it and re-run: OPEN_ROUTER_API=sk-or-v1-... python test_openrouter.py")
        sys.exit(1)

    print(f"  API key: {api_key[:12]}...")
    print(f"  Base URL: {os.getenv('OPEN_ROUTER_BASE_URL', 'https://openrouter.ai/api/v1')}")
    print(f"  Model: {os.getenv('OPEN_ROUTER_MODEL', 'openrouter/free')}")
    print()

    try:
        await test_basic_generate()
        await test_json_mode()
        await test_parallel()
        await test_singleton()
        await test_stats()
        print()
        print("✅ ALL TESTS PASSED")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
