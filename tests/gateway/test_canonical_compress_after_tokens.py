"""A committed canonical ``/compress`` reports the compacted request size the TUI status bar shows
(marked estimated) until the next reply: the messages plus the system prompt and tool schemas."""
from types import SimpleNamespace

import pytest

import hermes_state_runtime as rt
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session


@pytest.mark.asyncio
async def test_committed_compress_summary_carries_the_request_estimate(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from agent.context_compressor import ContextCompressor
    from agent.model_metadata import estimate_request_tokens_rough

    summary_row = {'role': 'user', 'content': 'SUMMARY ' * 50, '_compressed_summary': True}
    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    monkeypatch.setattr(ContextCompressor, '__init__', lambda self, *a, **k: None)
    monkeypatch.setattr(ContextCompressor, 'compress', lambda self, messages, **kw: [dict(summary_row)])
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='cold', source='tui', cwd=str(tmp_path),
                                                              model='frozen', toolsets=[]))
    for i in range(6):
        db.append_message(ref.session_id, 'user' if i % 2 == 0 else 'assistant', f'turn {i} ' * 200)
    prompt = 'You are Hermes. ' * 400
    tools = [{'type': 'function', 'function': {'name': 'read_file', 'description': 'Read a file. ' * 50,
                                               'parameters': {'type': 'object', 'properties': {}}}}]
    db.update_system_prompt(ref.session_id, prompt)
    db.update_session_tool_names(ref.session_id, {'version': 'v', 'tools': tools})
    expected = estimate_request_tokens_rough([summary_row], system_prompt=prompt, tools=tools)
    messages_only = estimate_request_tokens_rough([summary_row])
    before = db.get_session(ref.session_id)
    try:
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'commit', 'expected_revision': before['runtime_revision'],
            'expected_generation': before['runtime_generation'], 'operation': 'compress', 'payload': {}}})
        result = reply.get('result') or {}
        assert result.get('execution_generation') == before['runtime_generation'] + 1, reply
        assert expected > messages_only  # the check separates the two bases
        assert result['summary']['after_tokens'] == expected, result['summary']
        assert f'~{expected:,} tokens' in result['summary']['token_line']
    finally:
        await owner.close()


def test_manual_compress_summary_carries_the_after_estimate():
    from agent.manual_compression_feedback import summarize_manual_compression
    summary = summarize_manual_compression([{"role": "user"}] * 4, [{"role": "user"}], 9_000, 3_000)
    assert summary["after_tokens"] == 3_000
