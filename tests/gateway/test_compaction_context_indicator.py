"""The context indicator around a compaction: every producer the TUI status bar reads.

Recorded sequence (agent.log, 2026-10-29: ``messages=364->119 rough_tokens=~65,583 awaiting_real_usage=true``):
real usage before, the summary call running, the commit, the first reply after it. After the commit the
bar shows the compacted size, marked estimated, until that reply reports real usage.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from gateway.config import Platform

WINDOW, BEFORE, ROUGH, AFTER = 262_144, 150_000, 65_583, 61_000


def _agent():
    from agent.context_compressor import ContextCompressor
    comp = ContextCompressor(model="fixture", config_context_length=WINDOW, quiet_mode=True)
    return SimpleNamespace(model="fixture", provider="custom", session_id="sid", context_compressor=comp,
                           _session_messages=[], _turn_base_usage_anchor=None, get_rate_limit_state=lambda: None)


def _commit(comp):
    """What ``agent/conversation_compression.py::_finish_compaction_boundary`` sets on a commit."""
    comp.last_compression_rough_tokens = ROUGH
    comp.last_prompt_tokens = -1
    comp.last_completion_tokens = 0
    comp.awaiting_real_usage_after_compression = True


def _shown(usage):
    return usage.get("context_used"), usage.get("context_estimated")


def _views(agent):
    """Embedded TUI (``_get_usage``: ticker, session.info, message.complete) and canonical TUI
    (``session.usage`` frame's nested ``usage``)."""
    from gateway.session_usage import tui_usage, usage_stats
    from tui_gateway.server import _get_usage
    return {"embedded": _shown(_get_usage(agent)), "canonical": _shown(tui_usage(usage_stats(agent)))}


def test_recorded_compaction_sequence_shows_the_estimate_until_real_usage():
    agent = _agent()
    comp = agent.context_compressor
    comp.update_from_response({"prompt_tokens": BEFORE, "completion_tokens": 500})
    steps = {"before": _views(agent)}
    steps["compacting"] = _views(agent)  # the summary call does not touch the compressor
    _commit(comp)
    steps["committed"] = _views(agent)
    comp.update_from_response({"prompt_tokens": AFTER, "completion_tokens": 300})
    steps["first_reply"] = _views(agent)
    expected = {"before": (BEFORE, False), "compacting": (BEFORE, False),
                "committed": (ROUGH, True), "first_reply": (AFTER, False)}
    for step, want in expected.items():
        assert steps[step] == {"embedded": want, "canonical": want}, (step, steps[step])
    print("proved: embedded and canonical usage show", expected)


def test_classic_cli_bar_shows_the_estimate_not_zero_after_a_commit():
    from datetime import datetime
    from cli import HermesCLI
    agent = _agent()
    _commit(agent.context_compressor)
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model, cli_obj.agent, cli_obj.conversation_history = "fixture", agent, []
    cli_obj.session_start = datetime.now()
    snapshot = cli_obj._get_status_bar_snapshot()
    assert (snapshot["context_tokens"], snapshot["context_estimated"]) == (ROUGH, True), snapshot


def test_a_sentinel_without_an_estimate_shows_no_reading():
    """The classic CLI's gateway mirror parks at -1 with no rough figure: no context fields, never -1 or 0."""
    from tui_gateway.server import _get_usage
    agent = _agent()
    agent.context_compressor.last_prompt_tokens = -1
    assert "context_used" not in _get_usage(agent)


def _turn(published, agent):
    from gateway.run_turn_runner import TurnRunner
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL, chat_id="chat"),
                                _run_still_current=lambda: True, agent_holder=[agent],
                                mute_notification_reply=True, _hooks_ref=SimpleNamespace(loaded_hooks=[]))
    turn._runner = SimpleNamespace()
    turn._approval_owner = (SimpleNamespace(sessions={"s": SimpleNamespace()}), "s", 1)
    turn._publish_execution = lambda kind, payload: published.append((kind, payload)) or True
    return turn


@pytest.mark.parametrize("event_type", ["compacted", "lifecycle"])
def test_canonical_viewer_gets_the_estimate_when_the_commit_lands(event_type):
    """The canonical gateway publishes usage only after API calls; the commit must publish too,
    or the bar keeps the pre-compaction reading unmarked until the next reply."""
    agent = _agent()
    agent.context_compressor.update_from_response({"prompt_tokens": BEFORE, "completion_tokens": 500})
    _commit(agent.context_compressor)
    published = []
    _turn(published, agent)._status_callback_sync(event_type, "✓ Context compression complete")
    frames = [payload for kind, payload in published if kind == "session.usage"]
    if event_type != "compacted":
        assert frames == []
        return
    assert len(frames) == 1
    assert _shown(frames[0]["usage"]) == (ROUGH, True), frames[0]["usage"]

