from fastapi.testclient import TestClient

from tau2.api_service import interactive_chat_service as service


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


def test_delete_unknown_session():
    client = TestClient(service.app)
    response = client.delete("/v1/sessions/does-not-exist")
    assert response.status_code == 404
