import json
import threading
import time
import uuid
from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal, Optional, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.gym.gym_agent import AgentGymEnv, GymAgent, GymUser, UserGymEnv
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.registry import registry
from tau2.run import get_tasks
from tau2.utils.tools import parse_action_string, to_functional_format
from tau2.utils.io_utils import load_file

ChatRole = Literal["system", "user", "assistant", "tool"]
SessionMode = Literal["agent", "user"]
RelayTurn = Literal["agent", "user"]


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
    domain: Optional[str] = None
    task_id: Optional[str] = None
    task_split_name: Optional[str] = None
    max_steps: Optional[int] = None
    agent_llm: Optional[str] = None
    agent_llm_args: Optional[dict[str, Any]] = None
    user_llm: Optional[str] = None
    user_llm_args: Optional[dict[str, Any]] = None
    full_observation: Optional[bool] = None


class SessionCreateResponse(BaseModel):
    session_id: str
    mode: SessionMode
    domain: str
    task_id: str
    initial_observation: str
    initial_message: OpenAIMessage
    info: dict[str, Any]
    full_observation: bool
    next_turn: Optional[RelayTurn] = None


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
    next_turn: Optional[RelayTurn] = None


@dataclass
class SessionState:
    mode: SessionMode
    domain: str
    task_id: str
    env: AgentGymEnv | UserGymEnv
    full_observation: bool = True
    terminated: bool = False
    last_observation: str = ""


@dataclass
class RelaySessionState:
    domain: str
    task_id: str
    engine: "RelaySessionEngine"
    full_observation: bool = True


class RelaySessionEngine:
    def __init__(
        self, domain: str, task_id: str, max_steps: int, full_observation: bool
    ):
        self.domain = domain
        self.task_id = task_id
        self.max_steps = max_steps
        self.full_observation = full_observation
        self._lock = Lock()
        self._simulation_done = threading.Event()
        self._orchestrator_thread: Optional[threading.Thread] = None
        self._simulation_run: Optional[Any] = None

        self._environment = registry.get_env_constructor(domain)()
        self._task = self._get_task(domain=domain, task_id=task_id)
        self._agent_tools = self._environment.get_tools()
        self._agent = GymAgent(
            tools=self._agent_tools,
            domain_policy=self._environment.get_policy(),
        )
        try:
            user_tools = self._environment.get_user_tools()
        except ValueError:
            user_tools = None
        self._user_tools = user_tools or []
        self._user = GymUser(
            tools=user_tools,
            instructions=self._task.user_scenario,
        )
        self._orchestrator = Orchestrator(
            domain=domain,
            agent=self._agent,
            user=self._user,
            environment=self._environment,
            task=self._task,
            max_steps=max_steps,
            solo_mode=False,
        )

    @staticmethod
    def _get_task(domain: str, task_id: str) -> Any:
        tasks = registry.get_tasks_loader(domain)(None)
        for task in tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"No task found with id {task_id} for domain {domain}")

    def get_bootstrap_info(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "task_id": self.task_id,
            "policy": self._environment.get_policy(),
            "user_scenario": str(self._task.user_scenario),
            "agent_tools": [tool.openai_schema for tool in self._agent_tools],
            "user_tools": [tool.openai_schema for tool in self._user_tools],
        }

    def start(self) -> tuple[str, RelayTurn]:
        with self._lock:
            self._simulation_run = None
            self._simulation_done.clear()
            self._orchestrator_thread = threading.Thread(
                target=self._run_orchestrator,
                daemon=True,
            )
            assert self._orchestrator_thread is not None
            self._orchestrator_thread.start()

        self._wait_for_external_turn()
        next_turn = self._get_next_turn()
        if next_turn is None:
            return "", "user"
        messages = self._current_messages(next_turn)
        return self._format_observation(messages), next_turn

    def _run_orchestrator(self) -> None:
        simulation_run = None
        try:
            simulation_run = self._orchestrator.run()
        finally:
            self._simulation_run = simulation_run
            self._simulation_done.set()

    def _wait_for_external_turn(self) -> None:
        while not self._simulation_done.is_set():
            if self._user.is_user_turn or self._agent.is_agent_turn:
                return
            self._simulation_done.wait(timeout=0.01)

    def _get_next_turn(self) -> Optional[RelayTurn]:
        if self._simulation_done.is_set():
            return None
        if self._user.is_user_turn:
            return "user"
        if self._agent.is_agent_turn:
            return "agent"
        return None

    def _format_observation(self, messages: list[Any]) -> str:
        if not messages:
            return ""
        turns: list[str] = []
        for message in messages:
            if getattr(message, "tool_calls", None):
                tool_calls = ", ".join(
                    [
                        to_functional_format(tool_call)
                        for tool_call in message.tool_calls
                    ]
                )
                turns.append(f"{message.role}: {tool_calls}")
            else:
                turns.append(f"{message.role}: {message.content}")
        if self.full_observation:
            return "\n".join(turns)
        return turns[-1]

    def _current_messages(self, turn: RelayTurn) -> list[Any]:
        if turn == "user":
            return self._user.observation.copy()
        return self._agent.observation.copy()

    def step(
        self, turn: RelayTurn, action: str
    ) -> tuple[str, bool, float, Optional[RelayTurn]]:
        if self._simulation_done.is_set():
            return "", True, self._get_reward(), None

        if turn == "user":
            if not self._user.is_user_turn:
                raise RuntimeError("It is not the user's turn")
            action_msg = cast(Any, parse_action_string(action, requestor="user"))
            self._user.set_action(action_msg)
        else:
            if not self._agent.is_agent_turn:
                raise RuntimeError("It is not the agent's turn")
            action_msg = cast(Any, parse_action_string(action, requestor="assistant"))
            self._agent.set_action(action_msg)

        self._wait_for_external_turn()
        next_turn = self._get_next_turn()
        terminated = self._simulation_done.is_set()
        reward = self._get_reward()
        if next_turn is None:
            if turn == "user":
                messages = self._agent.observation.copy()
            else:
                messages = self._user.observation.copy()
        else:
            messages = self._current_messages(next_turn)
        return self._format_observation(messages), terminated, reward, next_turn

    def _get_reward(self) -> float:
        if self._simulation_run is None:
            return 0.0
        evaluation_result = evaluate_simulation(
            simulation=self._simulation_run,
            task=self._task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=False,
            domain=self.domain,
        )
        return evaluation_result.reward


class SessionDefaults(BaseModel):
    domain: Optional[str] = None
    task_id: Optional[str] = None
    task_split_name: str = "base"
    max_steps: int = 100
    agent_llm: Optional[str] = None
    agent_llm_args: Optional[dict[str, Any]] = None
    user_llm: Optional[str] = None
    user_llm_args: Optional[dict[str, Any]] = None
    full_observation: bool = True


class InteractiveChatServiceConfig(BaseModel):
    agent_session_defaults: SessionDefaults = Field(default_factory=SessionDefaults)
    user_session_defaults: SessionDefaults = Field(default_factory=SessionDefaults)
    relay_session_defaults: SessionDefaults = Field(default_factory=SessionDefaults)


def _extract_last_turn(observation: str, default_role: ChatRole) -> OpenAIMessage:
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
    return OpenAIMessage(role=cast(ChatRole, role), content=content.strip())


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
    def __init__(self, config: Optional[InteractiveChatServiceConfig] = None):
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}
        self._relay_sessions: dict[str, RelaySessionState] = {}
        self._config = config or InteractiveChatServiceConfig()

    def update_config(self, config: InteractiveChatServiceConfig):
        with self._lock:
            self._config = config

    def _resolve_request(
        self,
        mode: SessionMode | Literal["relay"],
        request: SessionCreateRequest,
    ) -> SessionCreateRequest:
        if mode == "agent":
            defaults = self._config.agent_session_defaults
        elif mode == "user":
            defaults = self._config.user_session_defaults
        else:
            defaults = self._config.relay_session_defaults
        default_values = defaults.model_dump()
        request_values = request.model_dump(exclude_none=True)
        merged = {**default_values, **request_values}
        resolved_defaults = SessionDefaults.model_validate(merged)
        if not resolved_defaults.domain:
            raise ValueError(
                "'domain' is required either in request body or config defaults"
            )
        return SessionCreateRequest.model_validate(resolved_defaults.model_dump())

    def create_session(
        self,
        mode: SessionMode,
        request: SessionCreateRequest,
    ) -> SessionCreateResponse:
        request = self._resolve_request(mode=mode, request=request)
        assert request.domain is not None
        assert request.task_split_name is not None
        assert request.max_steps is not None
        assert request.full_observation is not None
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
                all_messages_as_observation=request.full_observation,
            )
            default_role: ChatRole = "user"
        else:
            env = UserGymEnv(
                domain=request.domain,
                task_id=task_id,
                max_steps=request.max_steps,
                agent_llm=request.agent_llm,
                agent_llm_args=request.agent_llm_args,
                all_messages_as_observation=request.full_observation,
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
                full_observation=request.full_observation,
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
            full_observation=request.full_observation,
        )

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            deleted = self._sessions.pop(session_id, None) is not None
            deleted_relay = self._relay_sessions.pop(session_id, None) is not None
            return deleted or deleted_relay

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
        default_reply_role: ChatRole = "user" if mode == "agent" else "assistant"
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

    def create_relay_session(
        self, request: SessionCreateRequest
    ) -> SessionCreateResponse:
        request = self._resolve_request(mode="relay", request=request)
        assert request.domain is not None
        assert request.task_split_name is not None
        assert request.max_steps is not None
        assert request.full_observation is not None

        task_id = _select_task_id(
            domain=request.domain,
            task_split_name=request.task_split_name,
            task_id=request.task_id,
        )

        engine = RelaySessionEngine(
            domain=request.domain,
            task_id=task_id,
            max_steps=request.max_steps,
            full_observation=request.full_observation,
        )
        initial_observation, next_turn = engine.start()
        session_id = f"relay-{uuid.uuid4().hex}"
        with self._lock:
            self._relay_sessions[session_id] = RelaySessionState(
                domain=request.domain,
                task_id=task_id,
                engine=engine,
                full_observation=request.full_observation,
            )

        return SessionCreateResponse(
            session_id=session_id,
            mode="user",
            domain=request.domain,
            task_id=task_id,
            initial_observation=initial_observation,
            initial_message=_extract_last_turn(
                initial_observation, default_role="assistant"
            ),
            info={"relay": True, **engine.get_bootstrap_info()},
            full_observation=request.full_observation,
            next_turn=next_turn,
        )

    def step_relay(
        self,
        turn: RelayTurn,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        with self._lock:
            relay_session = self._relay_sessions.get(request.session_id)
        if relay_session is None:
            raise KeyError(f"Unknown relay session_id: {request.session_id}")

        expected_role = "assistant" if turn == "agent" else "user"
        action = _extract_action(request=request, expected_role=expected_role)
        observation, terminated, reward, next_turn = relay_session.engine.step(
            turn=turn,
            action=action,
        )
        default_reply_role: ChatRole = "user" if turn == "agent" else "assistant"
        choice = ChatCompletionChoice(
            message=_extract_last_turn(observation, default_role=default_reply_role),
        )
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=request.model or f"tau2-relay-{turn}",
            choices=[choice],
            usage=UsageStats(),
            session_id=request.session_id,
            terminated=terminated,
            reward=reward,
            observation=observation,
            next_turn=next_turn,
        )


def load_service_config(path: str) -> InteractiveChatServiceConfig:
    config_data = load_file(path)
    if not isinstance(config_data, dict):
        raise ValueError("Interactive chat config file must contain a dictionary")
    return InteractiveChatServiceConfig.model_validate(config_data)


app = FastAPI(title="Tau2 Interactive Chat API")
session_manager = SessionManager()


@app.get("/v1/config")
def get_config() -> InteractiveChatServiceConfig:
    return session_manager._config


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


@app.post("/v1/relay/sessions", response_model=SessionCreateResponse)
def create_relay_session(request: SessionCreateRequest) -> SessionCreateResponse:
    try:
        return session_manager.create_relay_session(request=request)
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


@app.post("/v1/relay/user/chat/completions", response_model=ChatCompletionResponse)
def relay_user_chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    try:
        return session_manager.step_relay(turn="user", request=request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/v1/relay/agent/chat/completions", response_model=ChatCompletionResponse)
def relay_agent_chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    try:
        return session_manager.step_relay(turn="agent", request=request)
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


def main(
    host: str = "127.0.0.1",
    port: int = 8005,
    config_path: Optional[str] = None,
):
    if config_path is not None:
        session_manager.update_config(load_service_config(config_path))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
