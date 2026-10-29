"""Text typed in interrupt mode while context compaction runs is held, not refused.

``redirect()`` used to answer False when no model request was in flight, so a correction typed
during compaction was refused ("correction rejected — input retained") and never resent. It now
degrades to ``steer()`` while a compaction is active; the pre-API drain then delivers it as a
user row at the first model request after the compaction.
"""
from __future__ import annotations

from unittest.mock import patch

import agent.conversation_compression as cc
from agent.prompt_builder import STEER_DISPLAY_KIND
from agent.turn_iteration_prep import prepare_iteration
from run_agent import AIAgent


def _agent() -> AIAgent:
    with (
        patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
    ):
        return AIAgent(
            api_key="test-key", base_url="https://openrouter.ai/api/v1", model="test/model",
            quiet_mode=True, skip_context_files=True, skip_memory=True,
        )


def _tool_turn(call_id: str) -> list[dict]:
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": call_id, "content": f"output {call_id}"},
    ]


def test_redirect_during_compaction_reaches_the_next_model_request(monkeypatch):
    agent = _agent()
    history = [{"role": "user", "content": "build it"}, *_tool_turn("a"), *_tool_turn("b")]
    compacted = [{"role": "user", "content": "[CONTEXT SUMMARY] earlier work"}, *_tool_turn("b")]
    replies = {}

    def compress_context(agent_, messages, system_message, **_kwargs):
        # The user types while the summary is being produced.
        replies["during"] = agent_.redirect("also run the linter")
        return compacted, "system"

    monkeypatch.setattr(cc, "compress_context", compress_context)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda *a, **k: (30.0, 60.0))
    messages, _ = agent._compress_context(history, "system", approx_tokens=1000, trigger="post_tool")

    assert replies["during"] is True, "a correction typed during compaction must be held, not refused"
    assert agent._interrupt_requested is False, "compaction must not be interrupted"
    assert agent._pending_redirect is None
    assert messages is compacted

    prep = prepare_iteration(agent, messages=messages, api_call_count=2)
    roles = [m["role"] for m in prep.messages]
    assert roles[-3:] == ["assistant", "tool", "user"], roles
    assert prep.messages[-1]["display_kind"] == STEER_DISPLAY_KIND
    assert "also run the linter" in prep.messages[-1]["content"]
    assert agent._drain_pending_steer() is None
    print("PROVED: redirect during compaction is held as a steer and lands after the newest tool row")


def test_redirect_outside_compaction_and_model_request_is_still_refused():
    agent = _agent()
    assert agent.redirect("too late") is False
    assert agent._drain_pending_steer() is None
    print("PROVED: with no compaction and no model request, redirect keeps the refusal contract")


def test_stop_pending_during_compaction_refuses_the_redirect():
    agent = _agent()
    agent._active_compression_commit_fence = cc.CompressionCommitFence()
    agent._interrupt_requested = True
    assert agent.redirect("after stop") is False
    assert agent._drain_pending_steer() is None
    print("PROVED: a pending /stop wins over a correction typed during compaction")
