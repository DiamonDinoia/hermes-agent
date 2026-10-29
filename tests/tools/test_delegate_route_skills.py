"""A delegation route's ``skills`` load into the child prompt like skills.auto_load."""

from unittest.mock import MagicMock, patch

from tests.tools.test_delegate import _make_mock_parent
from tools.delegate_tool import _build_child_agent
from tools.delegate_tool_routes import _clean_route


def _child_prompt(tmp_path, monkeypatch, routing_cfg):
    home = tmp_path / ".hermes"
    skill = home / "skills" / "probe-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: probe-skill\ndescription: probe\n---\nPROBE-SKILL-BODY\n")
    (home / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    with patch("run_agent.AIAgent", return_value=MagicMock()) as agent_cls:
        _build_child_agent(task_index=0, goal="g", context=None, toolsets=None, model=None, max_iterations=5,
                           parent_agent=_make_mock_parent(), task_count=1, routing_cfg=routing_cfg)
    return agent_cls.call_args.kwargs["ephemeral_system_prompt"]


def test_route_keeps_skills_key():
    assert _clean_route("p", {"skills": ["probe-skill"], "unknown": 1}) == {"skills": ["probe-skill"]}


def test_route_skills_reach_the_child_prompt(tmp_path, monkeypatch):
    prompt = _child_prompt(tmp_path, monkeypatch, {"skills": ["probe-skill", "missing-skill"]})
    assert "PROBE-SKILL-BODY" in prompt


def test_route_without_skills_loads_none(tmp_path, monkeypatch):
    assert "PROBE-SKILL-BODY" not in _child_prompt(tmp_path, monkeypatch, {"model": "m"})
