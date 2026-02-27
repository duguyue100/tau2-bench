"""
Chat server for step-by-step simulation of tau2 conversations.

Provides OpenAI-compatible chat completion endpoints plus session management
and tool execution:

  POST /v1/session
      Create a session for a specific task. Returns a session_id. The server
      initialises the domain environment (with the task's initial_state) and
      keeps it alive for the lifetime of the session. Required when the agent
      will make tool calls.

  GET /v1/session/{session_id}
      Check whether a session is still alive. Returns 200 with metadata if it
      exists, 404 if it has been evicted or never created.

  DELETE /v1/session/{session_id}
      Destroy a session and free its resources.

  POST /v1/tool/execute
      Execute one or more tool calls against a session's environment and
      return the results as role="tool" messages. Call this whenever the
      agent endpoint returns finish_reason=="tool_calls".

  POST /v1/agent/chat/completions
      Plays the customer-service agent.  Pass the full conversation so far
      (system + user/assistant/tool messages) and receive the agent's next
      response, which may contain tool calls.

  POST /v1/agent/turn
      High-level agent turn: runs the full LLM → tool-execute → LLM → …
      loop internally until the agent produces a final text message (or a
      stop signal).  Returns the final assistant message together with all
      intermediate tool calls and tool results that occurred during the turn,
      so the caller can append everything to their history.

  POST /v1/user/chat/completions
      Plays the user simulator.  Pass the full conversation so far plus
      either a task_id or a raw scenario string; receive the user's next
      response.

Typical simple flow using /v1/agent/turn
-----------------------------------------
1. Start the server: tau2 chat-server --domain airline --agent-llm gpt-4.1 ...
2. POST /v1/session  {"task_id": "3"}
   -> {"session_id": "..."}
3. POST /v1/agent/turn  {"session_id": "...", "messages": []}
   -> Agent sends greeting.
4. POST /v1/user/chat/completions   {"session_id": "...", "task_id": "3",
                                     "messages": [agent greeting]}
   -> User replies.
5. POST /v1/agent/turn  {"session_id": "...", "messages": [...]}
   -> Agent completes its full turn (including any tool calls) and returns
      the final message plus all intermediate steps.
6. Repeat from step 4 until is_stop==True or max turns reached.
7. DELETE /v1/session/{session_id}

Session lifecycle
-----------------
Sessions are stored in memory for the lifetime of the server process.
They are evicted automatically after ``session_ttl`` seconds of inactivity
(configurable via ``--session-ttl``; set to 0 to disable). You can also
delete a session explicitly with DELETE /v1/session/{session_id}.

Stop signals
------------
- Agent stop: response content contains "###STOP###" or "###TRANSFER###"
- User stop:  response content contains "###STOP###", "###TRANSFER###", or
              "###OUT-OF-SCOPE###"
"""

import asyncio
import time
import uuid
from copy import deepcopy
from typing import Any, Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from tau2.config import (
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_ARGS_AGENT,
    DEFAULT_LLM_ARGS_USER,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
)
from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.environment import Environment
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

# ---------------------------------------------------------------------------
# Server configuration – set once at startup, shared across all requests
# ---------------------------------------------------------------------------

CHAT_SERVER_PORT = 8002


class ChatServerConfig(BaseModel):
    """All settings that are fixed when the chat server starts."""

    domain: str = Field(description="The tau2 domain to simulate.")
    agent_llm: str = Field(
        default=DEFAULT_LLM_AGENT,
        description="LLM model for the agent.",
    )
    agent_llm_args: dict = Field(
        default_factory=lambda: deepcopy(DEFAULT_LLM_ARGS_AGENT),
        description="Extra kwargs forwarded to the agent LLM (e.g. temperature, seed).",
    )
    user_llm: str = Field(
        default=DEFAULT_LLM_USER,
        description="LLM model for the user simulator.",
    )
    user_llm_args: dict = Field(
        default_factory=lambda: deepcopy(DEFAULT_LLM_ARGS_USER),
        description="Extra kwargs forwarded to the user LLM.",
    )
    session_ttl: int = Field(
        default=3600,
        description=(
            "Seconds of inactivity after which a session is automatically evicted. "
            "Set to 0 to disable auto-eviction."
        ),
    )


# ---------------------------------------------------------------------------
# Request / response models (OpenAI-compatible)
# ---------------------------------------------------------------------------


class ChatMessageInput(BaseModel):
    """A single message in an input conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message author."
    )
    content: Optional[str] = Field(
        default=None, description="Text content of the message."
    )
    # For assistant messages that contain tool calls
    tool_calls: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Tool calls made by the assistant. Each entry follows the OpenAI schema: "
            "{'id': str, 'type': 'function', 'function': {'name': str, 'arguments': str}}."
        ),
    )
    # For tool messages (results of tool calls)
    tool_call_id: Optional[str] = Field(
        default=None,
        description="ID of the tool call this message is responding to (role='tool' only).",
    )
    # Requestor hint – needed to distinguish agent vs user tool calls when
    # replaying histories through the user simulator endpoint.
    requestor: Optional[Literal["user", "assistant"]] = Field(
        default=None,
        description=(
            "Who made the original tool call: 'user' or 'assistant'. "
            "Only relevant for role='tool' messages."
        ),
    )
    name: Optional[str] = Field(
        default=None,
        description="Tool name (used in some tool message formats).",
    )


class AgentChatRequest(BaseModel):
    """Request body for the agent chat-completion endpoint."""

    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Session ID returned by POST /v1/session. "
            "When provided, the agent uses the session's live environment for "
            "correct tool schemas. Required if you intend to execute tool calls."
        ),
    )
    messages: list[ChatMessageInput] = Field(
        default_factory=list,
        description=(
            "The full conversation history so far. "
            "If empty the agent returns its initial greeting without an LLM call. "
            "Messages follow the OpenAI chat format."
        ),
    )


class UserChatRequest(BaseModel):
    """Request body for the user-simulator chat-completion endpoint.

    Supply exactly one of `task_id` or `scenario`:

    - `task_id`: the server looks up the task in the configured domain and
      uses ``str(task.user_scenario)`` (persona + instructions) as the
      scenario.  This is the recommended way to simulate a specific task.
    - `scenario`: a raw scenario string passed directly to the user simulator.
      Useful for ad-hoc testing when you don't have a task_id.
    """

    session_id: Optional[str] = Field(
        default=None,
        description="Session ID returned by POST /v1/session (optional).",
    )
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "ID of the task to simulate. When provided, the server resolves "
            "the user scenario from the domain's task set. "
            "Mutually exclusive with `scenario`."
        ),
    )
    scenario: Optional[str] = Field(
        default=None,
        description=(
            "Raw user scenario / instructions string passed directly to the "
            "user simulator. Use this for ad-hoc testing. "
            "Mutually exclusive with `task_id`."
        ),
    )
    messages: list[ChatMessageInput] = Field(
        description=(
            "The full conversation history so far. "
            "The last message must be an assistant message."
        )
    )


# Session management and tool-execution models


class CreateSessionRequest(BaseModel):
    """Request body for POST /v1/session."""

    task_id: str = Field(
        description=(
            "The task ID to initialise the session for. "
            "The server loads the task from the configured domain, creates a "
            "fresh environment, and seeds it with the task's initial_state."
        )
    )


class SessionInfo(BaseModel):
    """Response from POST /v1/session."""

    session_id: str = Field(description="Opaque session identifier.")
    task_id: str = Field(description="Task ID this session was created for.")
    domain: str = Field(description="Domain the session is running in.")


class SessionStatus(BaseModel):
    """Response from GET /v1/session/{session_id}."""

    session_id: str = Field(description="Opaque session identifier.")
    task_id: str = Field(description="Task ID this session was created for.")
    domain: str = Field(description="Domain the session is running in.")
    last_accessed: float = Field(
        description="Unix timestamp of the last activity on this session."
    )
    alive: bool = Field(
        default=True, description="Always True when the session exists."
    )


class ToolCallInput(BaseModel):
    """An OpenAI-style tool call to execute."""

    id: str = Field(description="Tool call ID (from the agent's response).")
    type: Literal["function"] = "function"
    function: dict = Field(
        description="{'name': str, 'arguments': str (JSON-encoded)}."
    )


class ExecuteToolsRequest(BaseModel):
    """Request body for POST /v1/tool/execute."""

    session_id: str = Field(description="Session ID returned by POST /v1/session.")
    tool_calls: list[ToolCallInput] = Field(
        description="Tool calls from the agent's last response."
    )


class ToolResultMessage(BaseModel):
    """A single tool result message (role='tool')."""

    role: Literal["tool"] = "tool"
    tool_call_id: str = Field(description="Mirrors the tool call's id.")
    content: str = Field(description="JSON-encoded tool output, or an error string.")
    requestor: Literal["user", "assistant"] = "assistant"
    error: bool = Field(default=False, description="True if the tool raised an error.")


class ExecuteToolsResponse(BaseModel):
    """Response from POST /v1/tool/execute."""

    results: list[ToolResultMessage]


# OpenAI-compatible response models


class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON-encoded string


class ToolCallOutput(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChoiceMessage(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCallOutput]] = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: str


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    # tau2-specific: True when the response contains a stop signal
    is_stop: bool = False


class AgentTurnStep(BaseModel):
    """One tool-call/result pair that occurred during an agent turn."""

    tool_calls: list[ToolCallOutput] = Field(
        description="Tool calls the agent made in this step."
    )
    tool_results: list[ToolResultMessage] = Field(
        description="Results returned by the environment for those tool calls."
    )


class AgentTurnResponse(BaseModel):
    """Response from POST /v1/agent/turn.

    Contains the final assistant message (after all tool loops have completed)
    plus the full list of intermediate steps so the caller can reconstruct the
    complete message history if needed.
    """

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: Literal["agent.turn"] = "agent.turn"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    # All tool-call/result rounds that happened before the final message
    steps: list[AgentTurnStep] = Field(
        default_factory=list,
        description="Intermediate tool-call / tool-result rounds (may be empty).",
    )
    # The final text message from the agent
    final_message: ChoiceMessage = Field(
        description="The agent's final text message after all tool loops."
    )
    finish_reason: str = Field(
        default="stop",
        description="Always 'stop' – tool loops are resolved internally.",
    )
    usage: UsageInfo = Field(default_factory=UsageInfo)
    is_stop: bool = False


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _parse_tool_call_dict(tc: dict) -> ToolCall:
    """Convert an OpenAI-style tool-call dict to a tau2 ToolCall."""
    import json as _json

    func = tc.get("function", {})
    raw_args = func.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            args = _json.loads(raw_args)
        except _json.JSONDecodeError:
            args = {}
    else:
        args = raw_args
    return ToolCall(
        id=tc.get("id", ""),
        name=func.get("name", ""),
        arguments=args,
        requestor="assistant",
    )


def _input_messages_to_tau2(
    messages: list[ChatMessageInput],
    default_tool_requestor: Literal["user", "assistant"] = "assistant",
) -> list[Any]:
    """
    Convert OpenAI-style ChatMessageInput list to tau2 message objects.

    Tool messages are annotated with a requestor field. When the caller
    omits the requestor, *default_tool_requestor* is used.
    """
    tau2_messages = []
    for msg in messages:
        if msg.role == "system":
            tau2_messages.append(SystemMessage(role="system", content=msg.content))
        elif msg.role == "assistant":
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [_parse_tool_call_dict(tc) for tc in msg.tool_calls]
            tau2_messages.append(
                AssistantMessage(
                    role="assistant",
                    content=msg.content,
                    tool_calls=tool_calls,
                )
            )
        elif msg.role == "user":
            tau2_messages.append(UserMessage(role="user", content=msg.content))
        elif msg.role == "tool":
            requestor = msg.requestor or default_tool_requestor
            tau2_messages.append(
                ToolMessage(
                    id=msg.tool_call_id or "",
                    role="tool",
                    content=msg.content,
                    requestor=requestor,
                )
            )
        # Skip unknown roles silently
    return tau2_messages


def _assistant_message_to_response(
    assistant_msg: AssistantMessage,
    model: str,
    is_stop: bool,
) -> ChatCompletionResponse:
    """Convert a tau2 AssistantMessage to an OpenAI-compatible response."""
    import json as _json

    tool_calls_out: Optional[list[ToolCallOutput]] = None
    finish_reason = "stop"

    if assistant_msg.tool_calls:
        finish_reason = "tool_calls"
        tool_calls_out = []
        for tc in assistant_msg.tool_calls:
            tool_calls_out.append(
                ToolCallOutput(
                    id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                    function=FunctionCall(
                        name=tc.name,
                        arguments=_json.dumps(tc.arguments),
                    ),
                )
            )

    usage = UsageInfo()
    if assistant_msg.usage:
        usage = UsageInfo(
            prompt_tokens=assistant_msg.usage.get("prompt_tokens", 0),
            completion_tokens=assistant_msg.usage.get("completion_tokens", 0),
            total_tokens=assistant_msg.usage.get("total_tokens", 0),
        )

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChoiceMessage(
                    role="assistant",
                    content=assistant_msg.content,
                    tool_calls=tool_calls_out,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        is_stop=is_stop,
    )


def _user_message_to_response(
    user_msg: UserMessage,
    model: str,
    is_stop: bool,
) -> ChatCompletionResponse:
    """Convert a tau2 UserMessage to an OpenAI-compatible response."""
    import json as _json

    tool_calls_out: Optional[list[ToolCallOutput]] = None
    finish_reason = "stop"

    if user_msg.tool_calls:
        finish_reason = "tool_calls"
        tool_calls_out = []
        for tc in user_msg.tool_calls:
            tool_calls_out.append(
                ToolCallOutput(
                    id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                    function=FunctionCall(
                        name=tc.name,
                        arguments=_json.dumps(tc.arguments),
                    ),
                )
            )

    usage = UsageInfo()
    if user_msg.usage:
        usage = UsageInfo(
            prompt_tokens=user_msg.usage.get("prompt_tokens", 0),
            completion_tokens=user_msg.usage.get("completion_tokens", 0),
            total_tokens=user_msg.usage.get("total_tokens", 0),
        )

    return ChatCompletionResponse(
        model=model,
        choices=[
            Choice(
                message=ChoiceMessage(
                    role="user",
                    content=user_msg.content,
                    tool_calls=tool_calls_out,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        is_stop=is_stop,
    )


# ---------------------------------------------------------------------------
# Core generation helpers
# ---------------------------------------------------------------------------


def _resolve_scenario(config: ChatServerConfig, request: "UserChatRequest") -> str:
    """Return the scenario string from a UserChatRequest.

    Exactly one of ``request.task_id`` or ``request.scenario`` must be set.
    When ``task_id`` is given the task is loaded from the configured domain
    and ``str(task.user_scenario)`` is returned.
    """
    if request.task_id is not None and request.scenario is not None:
        raise ValueError("Provide either 'task_id' or 'scenario', not both.")
    if request.task_id is None and request.scenario is None:
        raise ValueError("One of 'task_id' or 'scenario' must be provided.")

    if request.scenario is not None:
        return request.scenario

    # Resolve from task_id
    from tau2.registry import registry

    task_loader = registry.get_tasks_loader(config.domain)
    tasks = task_loader(task_split_name=None)
    matching = [t for t in tasks if t.id == request.task_id]
    if not matching:
        raise ValueError(
            f"Task '{request.task_id}' not found in domain '{config.domain}'."
        )
    return str(matching[0].user_scenario)


def _load_task(domain: str, task_id: str):
    """Load a single task by ID from the given domain."""
    task_loader = registry.get_tasks_loader(domain)
    tasks = task_loader(task_split_name=None)
    matching = [t for t in tasks if t.id == task_id]
    if not matching:
        raise ValueError(f"Task '{task_id}' not found in domain '{domain}'.")
    return matching[0]


def _run_agent(
    config: ChatServerConfig,
    messages: list[ChatMessageInput],
    environment: Optional[Environment] = None,
) -> AssistantMessage:
    """
    Instantiate a fresh LLMAgent for the configured domain, replay *messages*
    as its history, and return the agent's next AssistantMessage.

    If *environment* is provided (from a session) it is used directly;
    otherwise a throw-away environment is constructed for the request.

    Empty history special case
    --------------------------
    When *messages* is empty we return the hard-coded default greeting
    ("Hi! How can I help you today?") without making an LLM call, mirroring
    the orchestrator's behaviour.
    """
    from tau2.agent.base import is_valid_agent_history_message
    from tau2.agent.llm_agent import LLMAgent
    from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE

    # Empty history → default greeting, no LLM call
    if not messages:
        return deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)

    own_env = False
    if environment is None:
        env_constructor = registry.get_env_constructor(config.domain)
        environment = env_constructor()
        own_env = True

    agent = LLMAgent(
        tools=environment.get_tools(),
        domain_policy=environment.get_policy(),
        llm=config.agent_llm,
        llm_args=deepcopy(config.agent_llm_args),
    )

    tau2_msgs = _input_messages_to_tau2(messages, default_tool_requestor="assistant")
    agent_history = [m for m in tau2_msgs if is_valid_agent_history_message(m)]

    if not agent_history:
        raise ValueError(
            "No valid agent-history messages found in the provided messages. "
            "Pass an empty messages list to get the initial greeting."
        )

    last_valid = agent_history[-1]

    if isinstance(last_valid, AssistantMessage):
        # History ends with an assistant message – unusual; prompt with an empty
        # user turn to elicit a new agent response.
        state = agent.get_init_state(message_history=agent_history)
        dummy_user_msg = UserMessage(role="user", content="")
        assistant_msg, _ = agent.generate_next_message(dummy_user_msg, state)
    else:
        # Normal case: history ends with a user or tool message.
        state = agent.get_init_state(message_history=agent_history[:-1])
        assistant_msg, _ = agent.generate_next_message(last_valid, state)

    if own_env:
        environment.sync_tools()
    return assistant_msg


def _run_user(
    config: ChatServerConfig,
    scenario: str,
    messages: list[ChatMessageInput],
    environment: Optional[Environment] = None,
) -> UserMessage:
    """
    Instantiate a fresh UserSimulator for the configured domain / scenario,
    replay *messages* as its history, and return the user's next UserMessage.

    If *environment* is provided (from a session) its user_tools are used.
    """
    from tau2.user.base import is_valid_user_history_message

    if environment is None:
        env_constructor = registry.get_env_constructor(config.domain)
        environment = env_constructor()

    try:
        user_tools = environment.get_user_tools()
    except Exception:
        user_tools = None

    user_sim = UserSimulator(
        tools=user_tools,
        instructions=scenario,
        llm=config.user_llm,
        llm_args=deepcopy(config.user_llm_args),
    )

    tau2_msgs = _input_messages_to_tau2(messages, default_tool_requestor="user")
    user_history = [m for m in tau2_msgs if is_valid_user_history_message(m)]

    if not user_history:
        raise ValueError(
            "The message history must contain at least one assistant message "
            "for the user simulator to respond to."
        )

    state = user_sim.get_init_state(message_history=user_history[:-1])
    last_valid = user_history[-1]
    user_msg, _ = user_sim.generate_next_message(last_valid, state)
    return user_msg


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _evict_once(sessions: dict[str, dict], session_ttl: int) -> list[str]:
    """Evict sessions idle for longer than *session_ttl* seconds.

    No-op when *session_ttl* is 0 (disabled).  Returns the list of evicted
    session IDs — useful for testing.
    """
    if session_ttl <= 0:
        return []
    now = time.time()
    stale = [
        sid
        for sid, data in list(sessions.items())
        if now - data["last_accessed"] > session_ttl
    ]
    for sid in stale:
        sessions.pop(sid, None)
    return stale


def create_app(config: ChatServerConfig) -> FastAPI:
    """
    Create and return a FastAPI application configured for *config*.

    The config is captured in the closure of each route handler so that
    individual requests carry no configuration fields.
    """
    app = FastAPI(
        title="Tau2 Chat Server",
        description=(
            "Step-by-step simulation of tau2 conversations via OpenAI-compatible endpoints.\n\n"
            f"Domain: **{config.domain}** | "
            f"Agent LLM: **{config.agent_llm}** | "
            f"User LLM: **{config.user_llm}**"
        ),
        version="0.1.0",
    )

    @app.get("/health")
    def get_health() -> dict[str, str]:
        return {"app_health": "OK"}

    @app.get("/config")
    def get_config() -> ChatServerConfig:
        """Return the active server configuration."""
        return config

    # -----------------------------------------------------------------------
    # Session store: session_id -> {"environment": Environment, "task_id": str,
    #                                "last_accessed": float}
    # Sessions are evicted after config.session_ttl seconds of inactivity
    # (when session_ttl > 0).
    # The store is also exposed on app.state.sessions for testability.
    # -----------------------------------------------------------------------
    sessions: dict[str, dict] = {}
    app.state.sessions = sessions  # expose for tests / introspection

    async def _evict_stale_sessions() -> None:
        """Background task: remove sessions idle for longer than session_ttl.

        The sweep interval is capped at 60s but reduced to half the TTL when
        the TTL is short, so short-lived sessions are evicted promptly.
        """
        sweep_interval = (
            min(60, max(1, config.session_ttl // 2)) if config.session_ttl > 0 else 60
        )
        while True:
            await asyncio.sleep(sweep_interval)
            _evict_once(sessions, config.session_ttl)

    @app.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_evict_stale_sessions())

    @app.post("/v1/session", status_code=201)
    def create_session(request: CreateSessionRequest) -> SessionInfo:
        """
        Create a session for a specific task.

        The server loads the task, constructs a fresh environment, and seeds
        it with the task's `initial_state`.  Returns a `session_id` that must
        be passed to subsequent agent/user/tool requests so they share the
        same live environment.
        """
        try:
            task = _load_task(config.domain, request.task_id)
            env_constructor = registry.get_env_constructor(config.domain)
            environment: Environment = env_constructor()

            initial_state = task.initial_state
            environment.set_state(
                initialization_data=(
                    initial_state.initialization_data if initial_state else None
                ),
                initialization_actions=(
                    initial_state.initialization_actions if initial_state else None
                ),
                message_history=(
                    (initial_state.message_history or []) if initial_state else []
                ),
            )

            session_id = uuid.uuid4().hex
            sessions[session_id] = {
                "environment": environment,
                "task_id": request.task_id,
                "last_accessed": time.time(),
            }
            return SessionInfo(
                session_id=session_id,
                task_id=request.task_id,
                domain=config.domain,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/session/{session_id}")
    def get_session(session_id: str) -> SessionStatus:
        """
        Check whether a session is still alive.

        Returns 200 with session metadata if the session exists, or 404 if it
        has been evicted (TTL expired) or was never created.  Useful for
        detecting TTL-based eviction without waiting for a tool/agent request
        to fail.
        """
        if session_id not in sessions:
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found."
            )
        data = sessions[session_id]
        return SessionStatus(
            session_id=session_id,
            task_id=data["task_id"],
            domain=config.domain,
            last_accessed=data["last_accessed"],
            alive=True,
        )

    @app.delete("/v1/session/{session_id}", status_code=204)
    def delete_session(session_id: str) -> None:
        """Destroy a session and free its environment."""
        if session_id not in sessions:
            raise HTTPException(
                status_code=404, detail=f"Session '{session_id}' not found."
            )
        del sessions[session_id]

    @app.post("/v1/tool/execute")
    def execute_tools(request: ExecuteToolsRequest) -> ExecuteToolsResponse:
        """
        Execute tool calls against a session's live environment.

        Call this whenever the agent endpoint returns `finish_reason=="tool_calls"`.
        Append the returned `role="tool"` messages to your history before
        calling the agent endpoint again.
        """
        if request.session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found.",
            )
        try:
            import json as _json

            sessions[request.session_id]["last_accessed"] = time.time()
            environment: Environment = sessions[request.session_id]["environment"]
            results: list[ToolResultMessage] = []

            for tc_input in request.tool_calls:
                func = tc_input.function
                raw_args = func.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        args = _json.loads(raw_args)
                    except _json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args

                tool_call = ToolCall(
                    id=tc_input.id,
                    name=func.get("name", ""),
                    arguments=args,
                    requestor="assistant",
                )
                tool_msg: ToolMessage = environment.get_response(tool_call)
                results.append(
                    ToolResultMessage(
                        tool_call_id=tool_msg.id,
                        content=tool_msg.content or "",
                        requestor=tool_msg.requestor,
                        error=tool_msg.error,
                    )
                )

            return ExecuteToolsResponse(results=results)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/agent/turn")
    async def agent_turn(request: AgentChatRequest) -> AgentTurnResponse:
        """
        Run a **complete agent turn** — including all tool-call loops — and
        return the final text message together with every intermediate step.

        This is the high-level alternative to calling
        `POST /v1/agent/chat/completions` + `POST /v1/tool/execute` in a loop.
        The server handles the full cycle internally:

        ```
        LLM call
          └─ if tool_calls → execute tools → LLM call again (repeat)
          └─ if stop       → return final_message
        ```

        A `session_id` (from `POST /v1/session`) is required so that tool
        calls execute against the correct live environment.

        **Response fields**
        - `steps`: list of `{tool_calls, tool_results}` rounds (empty when no
          tools were called).
        - `final_message`: the agent's concluding text message.
        - `is_stop`: `True` when the message contains a stop signal.

        Append `steps[*].tool_calls + steps[*].tool_results + final_message`
        to your conversation history after each call.
        """
        import json as _json

        if request.session_id is None:
            raise HTTPException(
                status_code=422,
                detail="'session_id' is required for /v1/agent/turn.",
            )
        if request.session_id not in sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found.",
            )

        try:
            from tau2.user.base import OUT_OF_SCOPE, STOP, TRANSFER

            sessions[request.session_id]["last_accessed"] = time.time()
            environment: Environment = sessions[request.session_id]["environment"]

            # Working copy of the message history – extended as tools execute
            current_messages = list(request.messages)
            steps: list[AgentTurnStep] = []
            total_usage = UsageInfo()

            while True:
                assistant_msg = _run_agent(
                    config, current_messages, environment=environment
                )

                # Accumulate usage
                if assistant_msg.usage:
                    total_usage.prompt_tokens += assistant_msg.usage.get(
                        "prompt_tokens", 0
                    )
                    total_usage.completion_tokens += assistant_msg.usage.get(
                        "completion_tokens", 0
                    )
                    total_usage.total_tokens += assistant_msg.usage.get(
                        "total_tokens", 0
                    )

                if not assistant_msg.tool_calls:
                    # Final text message – exit loop
                    content = assistant_msg.content or ""
                    is_stop = (
                        STOP in content
                        or TRANSFER in content
                        or OUT_OF_SCOPE in content
                    )

                    # Build ChoiceMessage for the final message
                    final_msg = ChoiceMessage(
                        role="assistant",
                        content=assistant_msg.content,
                        tool_calls=None,
                    )
                    return AgentTurnResponse(
                        model=config.agent_llm,
                        steps=steps,
                        final_message=final_msg,
                        finish_reason="stop",
                        usage=total_usage,
                        is_stop=is_stop,
                    )

                # ── Tool-call round ──────────────────────────────────────────
                # Serialise the tool calls to OpenAI format
                tc_outputs: list[ToolCallOutput] = []
                for tc in assistant_msg.tool_calls:
                    tc_outputs.append(
                        ToolCallOutput(
                            id=tc.id or f"call_{uuid.uuid4().hex[:16]}",
                            function=FunctionCall(
                                name=tc.name,
                                arguments=_json.dumps(tc.arguments),
                            ),
                        )
                    )

                # Execute each tool call against the live environment
                tool_results: list[ToolResultMessage] = []
                for tc in assistant_msg.tool_calls:
                    tool_msg: ToolMessage = environment.get_response(tc)
                    tool_results.append(
                        ToolResultMessage(
                            tool_call_id=tool_msg.id,
                            content=tool_msg.content or "",
                            requestor=tool_msg.requestor,
                            error=tool_msg.error,
                        )
                    )

                steps.append(
                    AgentTurnStep(tool_calls=tc_outputs, tool_results=tool_results)
                )

                # Extend working history: assistant message with tool_calls +
                # the tool result messages, so the next LLM call sees them.
                assistant_dict: dict = {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tco.id,
                            "type": "function",
                            "function": {
                                "name": tco.function.name,
                                "arguments": tco.function.arguments,
                            },
                        }
                        for tco in tc_outputs
                    ],
                }
                current_messages = (
                    current_messages
                    + [ChatMessageInput(**assistant_dict)]
                    + [
                        ChatMessageInput(
                            role="tool",
                            tool_call_id=r.tool_call_id,
                            content=r.content,
                            requestor=r.requestor,
                        )
                        for r in tool_results
                    ]
                )

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/agent/chat/completions")
    async def agent_chat_completions(
        request: AgentChatRequest,
    ) -> ChatCompletionResponse:
        """
        Generate the **agent's** next message given a conversation history.

        The agent plays the role of a customer-service representative for the
        domain configured at server startup.

        Pass a `session_id` (from POST /v1/session) so the agent uses the
        session's live environment for correct tool schemas.

        If `finish_reason` is `"tool_calls"`, call POST /v1/tool/execute with
        the same `session_id`, append the returned tool messages, then call
        this endpoint again.

        `is_stop` is `True` when the agent has included a stop signal
        (`###STOP###` or `###TRANSFER###`).
        """
        try:
            environment: Optional[Environment] = None
            if request.session_id is not None:
                if request.session_id not in sessions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session '{request.session_id}' not found.",
                    )
                environment = sessions[request.session_id]["environment"]
                sessions[request.session_id]["last_accessed"] = time.time()

            assistant_msg = _run_agent(
                config, request.messages, environment=environment
            )

            from tau2.user.base import OUT_OF_SCOPE, STOP, TRANSFER

            content = assistant_msg.content or ""
            is_stop = STOP in content or TRANSFER in content or OUT_OF_SCOPE in content

            return _assistant_message_to_response(
                assistant_msg=assistant_msg,
                model=config.agent_llm,
                is_stop=is_stop,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/user/chat/completions")
    async def user_chat_completions(
        request: UserChatRequest,
    ) -> ChatCompletionResponse:
        """
        Generate the **user simulator's** next message given a conversation
        history and a task scenario.

        Supply exactly one of `task_id` or `scenario` in the request body.
        Optionally pass a `session_id` to reuse its environment's user tools.

        `is_stop` is `True` when the user response contains a stop signal
        (`###STOP###`, `###TRANSFER###`, or `###OUT-OF-SCOPE###`).
        """
        try:
            environment: Optional[Environment] = None
            if request.session_id is not None:
                if request.session_id not in sessions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Session '{request.session_id}' not found.",
                    )
                environment = sessions[request.session_id]["environment"]
                sessions[request.session_id]["last_accessed"] = time.time()

            scenario = _resolve_scenario(config, request)
            user_msg = _run_user(
                config, scenario, request.messages, environment=environment
            )
            is_stop = UserSimulator.is_stop(user_msg)
            return _user_message_to_response(
                user_msg=user_msg,
                model=config.user_llm,
                is_stop=is_stop,
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Entry point for direct execution (uses all defaults)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    domain = sys.argv[1] if len(sys.argv) > 1 else "airline"
    cfg = ChatServerConfig(domain=domain)
    uvicorn.run(create_app(cfg), host="127.0.0.1", port=CHAT_SERVER_PORT)
