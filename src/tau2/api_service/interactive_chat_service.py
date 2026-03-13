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
from tau2.run import get_tasks, run_task
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


class SimulationRunRequest(BaseModel):
    domain: str
    task_id: Optional[str] = None
    task_split_name: str = "base"
    agent: str = "llm_agent"
    user: str = "user_simulator"
    agent_llm: Optional[str] = None
    agent_llm_args: Optional[dict[str, Any]] = None
    user_llm: Optional[str] = None
    user_llm_args: Optional[dict[str, Any]] = None
    max_steps: int = 100
    max_errors: int = 10
    evaluation_type: Literal["all", "env", "communicate", "action"] = "all"
    seed: Optional[int] = None
    enforce_communication_protocol: bool = False


class SimulationRunResponse(BaseModel):
    domain: str
    task_id: str
    terminated: bool
    termination_reason: Optional[str]
    reward: float
    transcript: list[OpenAIMessage]
    simulation: dict[str, Any]


class DuoSessionCreateRequest(BaseModel):
    domain: str
    task_id: Optional[str] = None
    task_split_name: str = "base"
    agent_llm: Optional[str] = None
    agent_llm_args: Optional[dict[str, Any]] = None
    user_llm: Optional[str] = None
    user_llm_args: Optional[dict[str, Any]] = None
    max_steps: int = 100
    full_observation: bool = True


class DuoSessionCreateResponse(BaseModel):
    session_id: str
    domain: str
    task_id: str
    initial_message: OpenAIMessage
    next_turn: Literal["agent", "user"]


class DuoStepRequest(BaseModel):
    session_id: str
    role: Literal["agent", "user"]
    message: str


class DuoStepResponse(BaseModel):
    session_id: str
    terminated: bool
    reward: float
    agent_message: Optional[str]
    user_message: Optional[str]
    observation: str


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
class DuoSessionState:
    domain: str
    task_id: str
    agent: GymAgent
    user: GymUser
    environment: Any
    task: Any
    orchestrator: Orchestrator
    max_steps: int
    full_observation: bool = True
    step_count: int = 0
    terminated: bool = False
    last_agent_message: str = ""
    last_user_message: str = ""


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


def _message_to_openai(message: Any) -> OpenAIMessage:
    role = getattr(message, "role", "assistant")
    content = getattr(message, "content", None)
    tool_calls = getattr(message, "tool_calls", None)
    openai_tool_calls = None
    if tool_calls:
        openai_tool_calls = []
        for tool_call in tool_calls:
            openai_tool_calls.append(
                OpenAIToolCall(
                    id=getattr(tool_call, "id", ""),
                    function=OpenAIFunctionCall(
                        name=getattr(tool_call, "name", ""),
                        arguments=getattr(tool_call, "arguments", {}),
                    ),
                )
            )
    if role not in {"system", "user", "assistant", "tool"}:
        role = "assistant"
    return OpenAIMessage(
        role=cast(ChatRole, role),
        content=content,
        tool_calls=openai_tool_calls,
    )


class SessionManager:
    def __init__(self, config: Optional[InteractiveChatServiceConfig] = None):
        self._lock = Lock()
        self._sessions: dict[str, SessionState] = {}
        self._duo_sessions: dict[str, DuoSessionState] = {}
        self._config = config or InteractiveChatServiceConfig()

    def update_config(self, config: InteractiveChatServiceConfig):
        with self._lock:
            self._config = config

    def _resolve_request(
        self,
        mode: SessionMode,
        request: SessionCreateRequest,
    ) -> SessionCreateRequest:
        if mode == "agent":
            defaults = self._config.agent_session_defaults
        else:
            defaults = self._config.user_session_defaults
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
            deleted_duo = self._duo_sessions.pop(session_id, None) is not None
            return deleted or deleted_duo

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

    def run_simulation(self, request: SimulationRunRequest) -> SimulationRunResponse:
        task_id = _select_task_id(
            domain=request.domain,
            task_split_name=request.task_split_name,
            task_id=request.task_id,
        )
        tasks = get_tasks(
            task_set_name=request.domain,
            task_split_name=request.task_split_name,
            task_ids=[task_id],
        )
        if not tasks:
            raise ValueError(
                f"No task found with id {task_id} for domain {request.domain}"
            )
        task = tasks[0]

        simulation = run_task(
            domain=request.domain,
            task=task,
            agent=request.agent,
            user=request.user,
            llm_agent=request.agent_llm,
            llm_args_agent=request.agent_llm_args,
            llm_user=request.user_llm,
            llm_args_user=request.user_llm_args,
            max_steps=request.max_steps,
            max_errors=request.max_errors,
            evaluation_type=EvaluationType(request.evaluation_type),
            seed=request.seed,
            enforce_communication_protocol=request.enforce_communication_protocol,
        )
        transcript = [_message_to_openai(message) for message in simulation.messages]
        termination_reason = (
            simulation.termination_reason.value
            if simulation.termination_reason is not None
            else None
        )
        reward = simulation.reward_info.reward if simulation.reward_info else 0.0
        return SimulationRunResponse(
            domain=request.domain,
            task_id=task_id,
            terminated=simulation.termination_reason is not None,
            termination_reason=termination_reason,
            reward=reward,
            transcript=transcript,
            simulation=simulation.model_dump(mode="json"),
        )

    def create_duo_session(
        self, request: DuoSessionCreateRequest
    ) -> DuoSessionCreateResponse:
        task_id = _select_task_id(
            domain=request.domain,
            task_split_name=request.task_split_name,
            task_id=request.task_id,
        )
        task = _get_task(domain=request.domain, task_id=task_id)
        environment = registry.get_env_constructor(request.domain)()
        agent_tools = environment.get_tools()
        agent = GymAgent(
            tools=agent_tools,
            domain_policy=environment.get_policy(),
            llm=request.agent_llm,
            llm_args=request.agent_llm_args,
        )
        try:
            user_tools = environment.get_user_tools()
        except ValueError:
            user_tools = None
        user = GymUser(
            tools=user_tools,
            instructions=task.user_scenario,
            llm=request.user_llm,
            llm_args=request.user_llm_args,
        )
        orchestrator = Orchestrator(
            domain=request.domain,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=request.max_steps,
            solo_mode=False,
        )
        session_id = f"duo-{uuid.uuid4().hex}"
        duo_state = DuoSessionState(
            domain=request.domain,
            task_id=task_id,
            agent=agent,
            user=user,
            environment=environment,
            task=task,
            orchestrator=orchestrator,
            max_steps=request.max_steps,
            full_observation=request.full_observation,
        )
        with self._lock:
            self._duo_sessions[session_id] = duo_state

        agent_msg, user_msg = _generate_first_turn(duo_state)
        if user_msg:
            return DuoSessionCreateResponse(
                session_id=session_id,
                domain=request.domain,
                task_id=task_id,
                initial_message=OpenAIMessage(role="user", content=user_msg),
                next_turn="user",
            )
        return DuoSessionCreateResponse(
            session_id=session_id,
            domain=request.domain,
            task_id=task_id,
            initial_message=OpenAIMessage(role="assistant", content=agent_msg),
            next_turn="agent",
        )

    def step_duo(self, request: DuoStepRequest) -> DuoStepResponse:
        with self._lock:
            duo = self._duo_sessions.get(request.session_id)
        if duo is None:
            raise KeyError(f"Unknown duo session_id: {request.session_id}")
        if duo.terminated:
            raise RuntimeError("Duo session is already terminated")

        duo.step_count += 1
        if request.role == "user":
            action_msg = UserMessage(role="user", content=request.message)
            duo.user.set_action(action_msg)
        else:
            action_msg = AssistantMessage(role="assistant", content=request.message)
            duo.agent.set_action(action_msg)

        agent_msg, user_msg, done, reward = _generate_next_turn(duo)
        duo.terminated = done
        duo.last_agent_message = agent_msg or ""
        duo.last_user_message = user_msg or ""

        observation_parts = []
        if duo.last_agent_message:
            observation_parts.append(f"assistant: {duo.last_agent_message}")
        if duo.last_user_message:
            observation_parts.append(f"user: {duo.last_user_message}")
        observation = "\n".join(observation_parts) if observation_parts else ""

        return DuoStepResponse(
            session_id=request.session_id,
            terminated=duo.terminated,
            reward=reward,
            agent_message=agent_msg,
            user_message=user_msg,
            observation=observation,
        )


def _get_task(domain: str, task_id: str) -> Any:
    tasks = registry.get_tasks_loader(domain)(None)
    for task in tasks:
        if task.id == task_id:
            return task
    raise ValueError(f"No task found with id {task_id} for domain {domain}")


def _generate_first_turn(duo: DuoSessionState) -> tuple[Optional[str], Optional[str]]:
    duo.orchestrator.run()
    if duo.user.observation:
        last_msg = duo.user.observation[-1]
        return None, getattr(last_msg, "content", None)
    if duo.agent.observation:
        last_msg = duo.agent.observation[-1]
        return getattr(last_msg, "content", None), None
    return None, None


def _generate_next_turn(
    duo: DuoSessionState,
) -> tuple[Optional[str], Optional[str], bool, float]:
    if duo.user.observation:
        last_msg = duo.user.observation[-1]
        user_content = getattr(last_msg, "content", None)
    else:
        user_content = None

    if duo.agent.observation:
        last_msg = duo.agent.observation[-1]
        agent_content = getattr(last_msg, "content", None)
    else:
        agent_content = None

    done = duo.step_count >= duo.max_steps
    reward = 0.0
    return agent_content, user_content, done, reward


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
