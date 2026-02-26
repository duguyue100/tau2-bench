import pytest
from fastapi.testclient import TestClient

from tau2.api_service import interactive_chat_service as service


@pytest.fixture(autouse=True)
def reset_session_manager_state():
    service.session_manager._sessions.clear()
    service.session_manager._relay_sessions.clear()
    service.session_manager.update_config(service.InteractiveChatServiceConfig())


class FakeAgentGymEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_action = None

    def reset(self):
        return "user: hello from user", {"task": {"id": self.kwargs["task_id"]}}

    def step(self, action: str):
        self.last_action = action
        return f"user: got {action}", 0.25, False, False, {}


class FakeUserGymEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_action = None

    def reset(self):
        return "assistant: hello from agent", {"task": {"id": self.kwargs["task_id"]}}

    def step(self, action: str):
        self.last_action = action
        return f"assistant: got {action}", 0.5, False, False, {}


def test_agent_session_and_chat(monkeypatch):
    monkeypatch.setattr(service, "AgentGymEnv", FakeAgentGymEnv)
    client = TestClient(service.app)

    create_response = client.post(
        "/v1/agent/sessions",
        json={"domain": "mock", "task_id": "create_task_1"},
    )
    assert create_response.status_code == 200
    create_data = create_response.json()
    assert create_data["initial_message"]["role"] == "user"

    session_id = create_data["session_id"]
    session = service.session_manager._sessions[session_id]
    env_kwargs = getattr(session.env, "kwargs", {})
    assert env_kwargs["all_messages_as_observation"] is True

    completion_response = client.post(
        "/v1/agent/chat/completions",
        json={
            "session_id": session_id,
            "messages": [{"role": "assistant", "content": "Hello there"}],
        },
    )
    assert completion_response.status_code == 200
    completion_data = completion_response.json()
    assert completion_data["choices"][0]["message"]["role"] == "user"
    assert completion_data["choices"][0]["message"]["content"] == "got Hello there"


def test_user_session_and_tool_call_chat(monkeypatch):
    monkeypatch.setattr(service, "UserGymEnv", FakeUserGymEnv)
    client = TestClient(service.app)

    create_response = client.post(
        "/v1/user/sessions",
        json={"domain": "mock", "task_id": "create_task_1"},
    )
    assert create_response.status_code == 200
    create_data = create_response.json()
    session_id = create_data["session_id"]
    session = service.session_manager._sessions[session_id]
    env_kwargs = getattr(session.env, "kwargs", {})
    assert env_kwargs["all_messages_as_observation"] is True

    completion_response = client.post(
        "/v1/user/chat/completions",
        json={
            "session_id": session_id,
            "messages": [
                {
                    "role": "user",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "check_status",
                                "arguments": '{"order_id": "123"}',
                            },
                        }
                    ],
                }
            ],
        },
    )
    assert completion_response.status_code == 200
    completion_data = completion_response.json()
    assert completion_data["choices"][0]["message"]["role"] == "assistant"
    assert "check_status" in completion_data["observation"]


def test_session_can_disable_full_observation(monkeypatch):
    monkeypatch.setattr(service, "AgentGymEnv", FakeAgentGymEnv)
    client = TestClient(service.app)

    create_response = client.post(
        "/v1/agent/sessions",
        json={
            "domain": "mock",
            "task_id": "create_task_1",
            "full_observation": False,
        },
    )
    assert create_response.status_code == 200
    create_data = create_response.json()
    assert create_data["full_observation"] is False

    session_id = create_data["session_id"]
    session = service.session_manager._sessions[session_id]
    env_kwargs = getattr(session.env, "kwargs", {})
    assert env_kwargs["all_messages_as_observation"] is False


def test_delete_unknown_session():
    client = TestClient(service.app)
    response = client.delete("/v1/sessions/does-not-exist")
    assert response.status_code == 404


def test_config_defaults_can_define_domain_and_models(monkeypatch):
    monkeypatch.setattr(service, "AgentGymEnv", FakeAgentGymEnv)
    service.session_manager.update_config(
        service.InteractiveChatServiceConfig(
            agent_session_defaults=service.SessionDefaults(
                domain="mock",
                task_id="create_task_1",
                user_llm="config-user-model",
                user_llm_args={"temperature": 0.2},
                full_observation=False,
            )
        )
    )
    client = TestClient(service.app)

    create_response = client.post("/v1/agent/sessions", json={})
    assert create_response.status_code == 200
    create_data = create_response.json()
    assert create_data["domain"] == "mock"
    assert create_data["task_id"] == "create_task_1"
    assert create_data["full_observation"] is False

    session_id = create_data["session_id"]
    session = service.session_manager._sessions[session_id]
    env_kwargs = getattr(session.env, "kwargs", {})
    assert env_kwargs["user_llm"] == "config-user-model"
    assert env_kwargs["user_llm_args"] == {"temperature": 0.2}
    assert env_kwargs["all_messages_as_observation"] is False


def test_load_service_config_from_toml(tmp_path):
    config_path = tmp_path / "interactive_chat.toml"
    config_path.write_text(
        """
[agent_session_defaults]
domain = "mock"
task_id = "create_task_1"

[user_session_defaults]
domain = "mock"
agent_llm = "gpt-4.1"

[relay_session_defaults]
domain = "mock"
task_id = "create_task_1"
""".strip(),
        encoding="utf-8",
    )

    config = service.load_service_config(str(config_path))
    assert config.agent_session_defaults.domain == "mock"
    assert config.agent_session_defaults.task_id == "create_task_1"
    assert config.user_session_defaults.agent_llm == "gpt-4.1"
    assert config.relay_session_defaults.task_id == "create_task_1"


class FakeRelaySessionEngine:
    def __init__(
        self, domain: str, task_id: str, max_steps: int, full_observation: bool
    ):
        self.domain = domain
        self.task_id = task_id
        self.max_steps = max_steps
        self.full_observation = full_observation
        self.expected_turn = "user"

    def start(self):
        return "assistant: hello", "user"

    def get_bootstrap_info(self):
        return {
            "domain": self.domain,
            "task_id": self.task_id,
            "policy": "fake-policy",
            "user_scenario": "fake-user-scenario",
            "agent_tools": [],
            "user_tools": [],
        }

    def step(self, turn: str, action: str):
        if turn != self.expected_turn:
            raise RuntimeError(f"It is not the {turn}'s turn")
        if turn == "user":
            self.expected_turn = "agent"
            return f"assistant: got {action}", False, 0.0, "agent"
        self.expected_turn = "user"
        return f"user: got {action}", False, 0.0, "user"


def test_relay_session_ping_pong(monkeypatch):
    monkeypatch.setattr(service, "RelaySessionEngine", FakeRelaySessionEngine)
    client = TestClient(service.app)

    create_response = client.post(
        "/v1/relay/sessions",
        json={"domain": "mock", "task_id": "create_task_1"},
    )
    assert create_response.status_code == 200
    create_data = create_response.json()
    assert create_data["next_turn"] == "user"

    session_id = create_data["session_id"]
    user_step_response = client.post(
        "/v1/relay/user/chat/completions",
        json={
            "session_id": session_id,
            "messages": [{"role": "user", "content": "hello agent"}],
        },
    )
    assert user_step_response.status_code == 200
    user_step_data = user_step_response.json()
    assert user_step_data["next_turn"] == "agent"
    assert user_step_data["choices"][0]["message"]["role"] == "assistant"

    agent_step_response = client.post(
        "/v1/relay/agent/chat/completions",
        json={
            "session_id": session_id,
            "messages": [{"role": "assistant", "content": "hello user"}],
        },
    )
    assert agent_step_response.status_code == 200
    agent_step_data = agent_step_response.json()
    assert agent_step_data["next_turn"] == "user"
    assert agent_step_data["choices"][0]["message"]["role"] == "user"


def test_relay_wrong_turn_returns_conflict(monkeypatch):
    monkeypatch.setattr(service, "RelaySessionEngine", FakeRelaySessionEngine)
    client = TestClient(service.app)

    create_response = client.post(
        "/v1/relay/sessions",
        json={"domain": "mock", "task_id": "create_task_1"},
    )
    session_id = create_response.json()["session_id"]

    response = client.post(
        "/v1/relay/agent/chat/completions",
        json={
            "session_id": session_id,
            "messages": [{"role": "assistant", "content": "out of turn"}],
        },
    )
    assert response.status_code == 409


def test_relay_uses_relay_defaults(monkeypatch):
    monkeypatch.setattr(service, "RelaySessionEngine", FakeRelaySessionEngine)
    service.session_manager.update_config(
        service.InteractiveChatServiceConfig(
            relay_session_defaults=service.SessionDefaults(
                domain="mock",
                task_id="create_task_1",
                max_steps=77,
                full_observation=False,
            )
        )
    )
    client = TestClient(service.app)

    create_response = client.post("/v1/relay/sessions", json={})
    assert create_response.status_code == 200
    create_data = create_response.json()
    assert create_data["domain"] == "mock"
    assert create_data["task_id"] == "create_task_1"
    assert create_data["full_observation"] is False

    relay_session = service.session_manager._relay_sessions[create_data["session_id"]]
    assert relay_session.engine.max_steps == 77
