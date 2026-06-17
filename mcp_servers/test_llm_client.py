"""
Smoke-test the LLM client abstraction layer.

Verifies create_client dispatch, tool-definition conversion, and that
the OpenAI path imports cleanly.  Anthropic/Gemini paths are tested
only if their SDKs are installed (they're optional dependencies).

Run: python -m mcp_servers.test_llm_client
"""
from __future__ import annotations

import json
import sys

from pydantic import BaseModel, Field

from mcp_servers.llm_client import (
    LLMClient,
    LLMResponse,
    ToolCall,
    create_client,
    _openai_tools_to_anthropic,
    _openai_tools_to_gemini,
    OpenAIClient,
    AnthropicClient,
    GeminiClient,
)

# ---------------------------------------------------------------------------
# Sample tool definitions (OpenAI format)
# ---------------------------------------------------------------------------
SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_pdk",
            "description": "Search the PDK catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
]


class TestModel(BaseModel):
    """A simple Pydantic model for structured output testing."""
    name: str = Field(..., description="A name")
    count: int = Field(..., description="A count")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_create_client_dispatch():
    """Verify create_client routes to the correct implementation class."""
    import os
    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    if has_openai_key:
        openai_models = ["gpt-5.4", "gpt-4o", "gpt-4o-mini", "o3-mini", "o1"]
        for m in openai_models:
            c = create_client(m)
            assert isinstance(c, OpenAIClient), f"{m} -> {type(c)}"
            assert c.model == m
        print("  [PASS] OpenAI dispatch")
    else:
        print("  [SKIP] OpenAI dispatch (no API key)")

    try:
        import anthropic  # noqa: F401
        claude_models = [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-sonnet-4-20250514",
            "claude-3.5-sonnet-20241022",
        ]
        for m in claude_models:
            c = create_client(m)
            assert isinstance(c, AnthropicClient), f"{m} -> {type(c)}"
            assert c.model == m
        print("  [PASS] Anthropic dispatch")
    except ImportError:
        print("  [SKIP] Anthropic SDK not installed")

    try:
        import google.generativeai  # noqa: F401
        gemini_models = [
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ]
        for m in gemini_models:
            c = create_client(m)
            assert isinstance(c, GeminiClient), f"{m} -> {type(c)}"
            assert c.model == m
        print("  [PASS] Gemini dispatch")
    except ImportError:
        print("  [SKIP] Gemini SDK not installed")

    try:
        create_client("unknown-model-abc")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [PASS] Unknown model raises ValueError")


def test_tool_converter_anthropic():
    """Verify OpenAI -> Anthropic tool format conversion."""
    converted = _openai_tools_to_anthropic(SAMPLE_TOOLS)
    assert len(converted) == 1
    t = converted[0]
    assert t["name"] == "search_pdk"
    assert "description" in t
    assert "input_schema" in t
    assert "query" in t["input_schema"]["properties"]
    print("  [PASS] Anthropic tool conversion")


def test_tool_converter_gemini():
    """Verify OpenAI -> Gemini tool format conversion."""
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        print("  [SKIP] Gemini SDK not installed")
        return

    converted = _openai_tools_to_gemini(SAMPLE_TOOLS)
    assert len(converted) == 1
    tool = converted[0]
    assert hasattr(tool, "function_declarations")
    decls = tool.function_declarations
    assert len(decls) == 1
    assert decls[0].name == "search_pdk"
    print("  [PASS] Gemini tool conversion")


def test_tool_call_dataclass():
    """Verify ToolCall and LLMResponse dataclasses."""
    tc = ToolCall(id="tc_1", name="search_pdk", arguments='{"query": "MZI"}')
    assert tc.id == "tc_1"
    assert tc.name == "search_pdk"
    assert json.loads(tc.arguments) == {"query": "MZI"}

    resp = LLMResponse(
        content="test",
        tool_calls=[tc],
        stop_reason="tool_calls",
        raw=None,
    )
    assert len(resp.tool_calls) == 1
    assert resp.stop_reason == "tool_calls"
    print("  [PASS] ToolCall / LLMResponse dataclasses")


def test_openai_tool_result_message():
    """Verify OpenAI tool result message format."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  [SKIP] OpenAI tool_result_message (no API key)")
        return
    client = create_client("gpt-4o")
    tc = ToolCall(id="call_abc123", name="search_pdk", arguments='{"query": "MZI"}')
    msg = client.tool_result_message(tc, '{"results": []}')
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_abc123"
    assert msg["content"] == '{"results": []}'
    print("  [PASS] OpenAI tool_result_message format")


def test_anthropic_tool_result_message():
    """Verify Anthropic tool result message format."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        print("  [SKIP] Anthropic SDK not installed")
        return

    client = create_client("claude-sonnet-4-20250514")
    tc = ToolCall(id="toolu_abc123", name="search_pdk", arguments='{"query": "MZI"}')
    msg = client.tool_result_message(tc, '{"results": []}')
    assert msg["role"] == "user"
    assert isinstance(msg["content"], list)
    block = msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_abc123"
    print("  [PASS] Anthropic tool_result_message format")


def test_assistant_message_openai():
    """Verify OpenAI assistant_message returns the raw object."""
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("  [SKIP] OpenAI assistant_message (no API key)")
        return
    client = create_client("gpt-4o")
    resp = LLMResponse(content="Hello", tool_calls=[], stop_reason="stop", raw="sentinel")
    msg = client.assistant_message(resp)
    assert msg == "sentinel"
    print("  [PASS] OpenAI assistant_message returns raw")


def main():
    print("=" * 60)
    print("LLM Client Abstraction Layer — Smoke Tests")
    print("=" * 60)

    tests = [
        test_create_client_dispatch,
        test_tool_converter_anthropic,
        test_tool_converter_gemini,
        test_tool_call_dataclass,
        test_openai_tool_result_message,
        test_anthropic_tool_result_message,
        test_assistant_message_openai,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {test_fn.__name__}: {exc}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
