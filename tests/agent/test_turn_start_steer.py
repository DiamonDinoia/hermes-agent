"""A steer queued during a turn-start (pre_api) compaction reaches the first request after the prompt.

After a pre_api compaction the tail is the turn's prompt. The pre-API steer drain looked only for
the newest tool row: with none it requeued the steer, which then missed the request; with an older
one it put the steer in front of the turn's prompt.
"""
from __future__ import annotations

from unittest.mock import patch

import agent.conversation_compression as cc
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


def _steer_during_turn_start_compaction(monkeypatch, history, compacted):
    agent = _agent()
    replies = {}

    def compress_context(agent_, messages, system_message, **_kwargs):
        replies["during"] = agent_.steer("also run the linter")
        return compacted, "system"

    monkeypatch.setattr(cc, "compress_context", compress_context)
    monkeypatch.setattr(cc, "resolve_context_compression_timeouts", lambda *a, **k: (30.0, 60.0))
    messages, _ = agent._compress_context(history, "system", approx_tokens=1000, trigger="pre_api")
    assert replies["during"] is True
    prep = prepare_iteration(agent, messages=messages, api_call_count=1, current_turn_user_idx=len(messages) - 1)
    assert agent._drain_pending_steer() is None, "the steer must reach this request, not wait for a tool batch"
    return prep.messages


def test_steer_during_turn_start_compaction_without_tool_rows(monkeypatch):
    prompt = {"role": "user", "content": "build it"}
    out = _steer_during_turn_start_compaction(
        monkeypatch, [{"role": "user", "content": "old"}, {"role": "assistant", "content": "ok"}, prompt],
        [{"role": "user", "content": "[CONTEXT SUMMARY] earlier work"}, {"role": "assistant", "content": "ok"},
         prompt],
    )
    assert out[-1]["role"] == "user" and out[-2]["role"] == "assistant", [m["role"] for m in out]
    assert out[-1]["content"].index("build it") < out[-1]["content"].index("also run the linter")


def test_steer_during_turn_start_compaction_lands_after_the_prompt(monkeypatch):
    prompt = {"role": "user", "content": "build it"}
    old_tools = _tool_turn("a")
    out = _steer_during_turn_start_compaction(
        monkeypatch, [{"role": "user", "content": "old"}, *old_tools, {"role": "assistant", "content": "done"}, prompt],
        [{"role": "user", "content": "[CONTEXT SUMMARY] earlier work"}, *old_tools,
         {"role": "assistant", "content": "done"}, prompt],
    )
    text = "\n".join(m["content"] for m in out if isinstance(m.get("content"), str))
    assert text.index("build it") < text.index("also run the linter"), "the steer must follow the turn's prompt"
    assert out[-1]["role"] == "user" and "also run the linter" in out[-1]["content"]
