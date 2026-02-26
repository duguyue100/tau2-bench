import json
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tau2.gym.gym_agent import AgentGymEnv, UserGymEnv
from tau2.run import get_tasks

ChatRole = Literal["system", "user", "assistant", "tool"]
SessionMode = Literal["agent", "user"]


class OpenAIFunctionCall(BaseModel):
    name: str
    arguments: str | dict[str, Any] = Field(default_factory=dict)


class OpenAIToolCall(BaseModel):
    id: str = ""
    type: Literal["function"] = "function"
    function: OpenAIFunctionCall


class OpenAIMessage(BaseModel):
    role: ChatRole
    content: Optional[str] = None
    tool_calls: Optional[list[OpenAIToolCall]] = None


class UsageStats(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop"


class SessionCreateRequest(BaseModel):
    domain: str
    task_id: Optional[str] = None
    task_split_name: str = "base"
    max_steps: int = 100
    agent_llm: Optional[str] = None
    agent_llm_args: Optional[dict[str, Any]] = None
    user_llm: Optional[str] = None
    user_llm_args: Optional[dict[str, Any]] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    mode: SessionMode
    domain: str
    task_id: str
    initial_observation: str
    initial_message: OpenAIMessage
    info: dict[str, Any]


class ChatCompletionRequest(BaseModel):
    session_id: str
    model: Optional[str] = None
    messages: list[OpenAIMessage]
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageStats
    session_id: str
    terminated: bool
    reward: float
    observation: str


@dataclass
class SessionState:
    mode: SessionMode
    domain: str
    task_id: str
    env: AgentGymEnv | UserGymEnv
    terminated: bool = False
    last_observation: str = ""


def _extract_last_turn(observation: str, default_role: str) -> OpenAIMessage:
    lines = [line.strip() for line in observation.splitlines() if line.strip()]
    if not lines:
        return OpenAIMessage(role=default_role, content="")
    last_line = lines[-1]
    if ":" not in last_line:
        return OpenAIMessage(role=default_role, content=last_line)
    role, content = last_line.split(":", 1)
    role = role.strip().lower()
    if role not in {"system", "user", "assistant", "tool"}:
        role = default_role
    return OpenAIMessage(role=role, content=content.strip())


def _select_task_id(domain: str, task_split_name: str, task_id: Optional[str]) -> str:
    if task_id is not None:
        return task_id
    tasks = get_tasks(
        task_set_name=domain,
        task_split_name=task_split_name,
        num_tasks=1,
    )
    if not tasks:
        raise ValueError(f"No tasks available for domain '{domain}'")
    return tasks[0].id


def _tool_call_to_action(tool_call: OpenAIToolCall, requestor: str) -> str:
    arguments = tool_call.function.arguments
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid function arguments JSON: {exc}") from exc
    action = {
        "id": tool_call.id,
        "name": tool_call.function.name,
        "arguments": arguments,
        "requestor": requestor,
    }
    return json.dumps(action)


def _extract_action(request: ChatCompletionRequest, expected_role: str) -> str:
    if request.stream:
        raise ValueError("Streaming is not supported")
    if not request.messages:
        raise ValueError("At least one message is required")
    last_message = request.messages[-1]
    if last_message.role != expected_role:
        raise ValueError(
            f"Last message role must be '{expected_role}', got '{last_message.role}'"
        )
    if last_message.tool_calls:
        if len(last_message.tool_calls) != 1:
            raise ValueError("Only one tool call per step is supported")
        return _tool_call_to_action(last_message.tool_calls[0], requestor=expected_role)
    content = (last_message.content or "").strip()
    if not content:
        raise ValueError("Last message content cannot be empty")
    return content


class SessionManager:
    def __init__(self):
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        mode: SessionMode,
        request: SessionCreateRequest,
    ) -> SessionCreateResponse:
        task_id = _select_task_id(
            domain=request.domain,
            task_split_name=request.task_split_name,
            task_id=request.task_id,
        )
        if mode == "agent":
            env = AgentGymEnv(
                domain=request.domain,
                task_id=task_id,
                max_steps=request.max_steps,
                user_llm=request.user_llm,
                user_llm_args=request.user_llm_args,
            )
            default_role = "user"
        else:
            env = UserGymEnv(
                domain=request.domain,
                task_id=task_id,
                max_steps=request.max_steps,
                agent_llm=request.agent_llm,
                agent_llm_args=request.agent_llm_args,
            )
            default_role = "assistant"

        observation, info = env.reset()
        session_id = f"session-{uuid.uuid4().hex}"
        with self._lock:
            self._sessions[session_id] = SessionState(
                mode=mode,
                domain=request.domain,
                task_id=task_id,
                env=env,
                last_observation=observation,
            )

        return SessionCreateResponse(
            session_id=session_id,
            mode=mode,
            domain=request.domain,
            task_id=task_id,
            initial_observation=observation,
            initial_message=_extract_last_turn(observation, default_role=default_role),
            info=info,
        )

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def step(
        self,
        mode: SessionMode,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        with self._lock:
            session = self._sessions.get(request.session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {request.session_id}")
        if session.mode != mode:
            raise ValueError(
                f"Session mode mismatch. Session is '{session.mode}', endpoint is '{mode}'"
            )
        if session.terminated:
            raise RuntimeError("Session is already terminated")

        expected_role = "assistant" if mode == "agent" else "user"
        default_reply_role = "user" if mode == "agent" else "assistant"
        action = _extract_action(request=request, expected_role=expected_role)

        observation, reward, terminated, _, _ = session.env.step(action)
        session.terminated = terminated
        session.last_observation = observation

        choice = ChatCompletionChoice(
            message=_extract_last_turn(observation, default_role=default_reply_role),
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model or f"tau2-{mode}-endpoint",
            choices=[choice],
            usage=UsageStats(),
            session_id=request.session_id,
            terminated=terminated,
            reward=reward,
            observation=observation,
        )


app = FastAPI(title="Tau2 Interactive Chat API")
session_manager = SessionManager()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/agent/sessions", response_model=SessionCreateResponse)
def create_agent_session(request: SessionCreateRequest) -> SessionCreateResponse:
    try:
        return session_manager.create_session(mode="agent", request=request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/user/sessions", response_model=SessionCreateResponse)
def create_user_session(request: SessionCreateRequest) -> SessionCreateResponse:
    try:
        return session_manager.create_session(mode="user", request=request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/agent/chat/completions", response_model=ChatCompletionResponse)
def agent_chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        return session_manager.step(mode="agent", request=request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/user/chat/completions", response_model=ChatCompletionResponse)
def user_chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        return session_manager.step(mode="user", request=request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/v1/sessions/{session_id}")
def delete_session(session_id: str) -> dict[str, bool]:
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return {"deleted": True}


def main(host: str = "127.0.0.1", port: int = 8005):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
