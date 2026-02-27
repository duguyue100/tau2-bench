"""
simulate_conversation.py
------------------------
Example: step-by-step conversation simulation using the tau2 chat server.

Start the server first:
    tau2 chat-server --domain airline --agent-llm gpt-4o-mini --user-llm gpt-4o-mini

Then run this script:
    python examples/simulate_conversation.py --task-id 3
    python examples/simulate_conversation.py --scenario "I want to cancel my flight."

Conversation flow
-----------------
1. POST /v1/session        {"task_id": "3"}
   -> {"session_id": "..."}   (initialises environment with task's initial_state)

2. POST /v1/agent/chat/completions  {"session_id": "...", "messages": []}
   -> Agent sends opening greeting.

3. POST /v1/user/chat/completions   {"session_id": "...", "task_id": "3",
                                     "messages": [agent greeting]}
   -> User simulator replies.

4. POST /v1/agent/chat/completions  {"session_id": "...", "messages": [...]}
   -> Agent responds.
   If finish_reason == "tool_calls":
     POST /v1/tool/execute  {"session_id": "...", "tool_calls": [...]}
     -> Append role="tool" messages to history, call agent endpoint again.

5. Repeat from step 3 until is_stop == True or max turns reached.

6. DELETE /v1/session/{session_id}
"""

import argparse
import json
import sys

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8002"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def call(method: str, base_url: str, path: str, **kwargs) -> dict:
    resp = requests.request(method, f"{base_url}{path}", timeout=120, **kwargs)
    resp.raise_for_status()
    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


def create_session(base_url: str, task_id: str) -> str:
    data = call("POST", base_url, "/v1/session", json={"task_id": task_id})
    return data["session_id"]


def delete_session(base_url: str, session_id: str) -> None:
    requests.delete(f"{base_url}/v1/session/{session_id}", timeout=10)


def call_agent(base_url: str, session_id: str, messages: list[dict]) -> dict:
    return call(
        "POST",
        base_url,
        "/v1/agent/chat/completions",
        json={"session_id": session_id, "messages": messages},
    )


def call_user(
    base_url: str,
    session_id: str,
    messages: list[dict],
    task_id: str | None,
    scenario: str | None,
) -> dict:
    body: dict = {"session_id": session_id, "messages": messages}
    if task_id is not None:
        body["task_id"] = task_id
    else:
        body["scenario"] = scenario
    return call("POST", base_url, "/v1/user/chat/completions", json=body)


def execute_tools(base_url: str, session_id: str, tool_calls: list[dict]) -> list[dict]:
    """Execute tool calls and return a list of role='tool' message dicts."""
    data = call(
        "POST",
        base_url,
        "/v1/tool/execute",
        json={"session_id": session_id, "tool_calls": tool_calls},
    )
    # Convert ToolResultMessage objects into OpenAI-style tool messages
    tool_msgs = []
    for r in data["results"]:
        tool_msgs.append(
            {
                "role": "tool",
                "tool_call_id": r["tool_call_id"],
                "content": r["content"],
                "requestor": r["requestor"],
            }
        )
    return tool_msgs


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_turn(role: str, msg: dict, is_stop: bool = False) -> None:
    stop_tag = "  [STOP]" if is_stop else ""
    print(f"\n[{role.upper()}]{stop_tag}")
    content = msg.get("content")
    if content:
        print(f"  {content}")
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        print("  Tool calls:")
        for tc in tool_calls:
            fn = tc.get("function", {})
            print(f"    -> {fn.get('name')}({fn.get('arguments')})")


def print_tool_results(results: list[dict]) -> None:
    print("\n[ENV]")
    for r in results:
        error_tag = " [ERROR]" if r.get("error") else ""
        print(f"  tool_call_id={r['tool_call_id']}{error_tag}: {r['content']}")


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def simulate(
    task_id: str | None,
    scenario: str | None,
    base_url: str = DEFAULT_BASE_URL,
    max_turns: int = 20,
) -> None:
    # Check server health
    try:
        health = requests.get(f"{base_url}/health", timeout=5).json()
        cfg = requests.get(f"{base_url}/config", timeout=5).json()
    except requests.ConnectionError:
        print(f"ERROR: Cannot reach the chat server at {base_url}.")
        print("Start it with:  tau2 chat-server --domain <domain>")
        sys.exit(1)

    print(
        f"Server: {health['app_health']}  |  "
        f"Domain: {cfg['domain']}  |  "
        f"Agent LLM: {cfg['agent_llm']}  |  "
        f"User LLM: {cfg['user_llm']}"
    )

    # A session is always required so the environment stays alive across turns.
    # When using --scenario (no task_id), we still need a task to initialise
    # the environment; task "0" is used as a placeholder.
    init_task_id = task_id if task_id is not None else "0"
    session_id: str = create_session(base_url, init_task_id)
    if task_id is not None:
        print(f"Task ID: {task_id}  |  Session: {session_id}")
    else:
        print(f"Scenario mode (env init from task '0')  |  Session: {session_id}")
        print(f"Scenario: {scenario!r}")
    print("-" * 60)

    history: list[dict] = []

    try:
        for _turn in range(max_turns):
            # ── Agent turn ────────────────────────────────────────────────
            agent_resp = call_agent(base_url, session_id, history)
            agent_msg = agent_resp["choices"][0]["message"]
            agent_stop = agent_resp.get("is_stop", False)
            finish_reason = agent_resp["choices"][0]["finish_reason"]

            history.append(agent_msg)
            print_turn("agent", agent_msg, agent_stop)

            if agent_stop:
                print("\n[Simulation ended: agent signalled stop]")
                break

            # Execute tool calls if the agent requested them
            while finish_reason == "tool_calls":
                tool_results = execute_tools(
                    base_url, session_id, agent_msg["tool_calls"]
                )
                print_tool_results(tool_results)
                history.extend(tool_results)

                # Let agent continue after seeing tool results
                agent_resp = call_agent(base_url, session_id, history)
                agent_msg = agent_resp["choices"][0]["message"]
                agent_stop = agent_resp.get("is_stop", False)
                finish_reason = agent_resp["choices"][0]["finish_reason"]

                history.append(agent_msg)
                print_turn("agent", agent_msg, agent_stop)

                if agent_stop:
                    break

            if agent_stop:
                print("\n[Simulation ended: agent signalled stop]")
                break

            # ── User turn ─────────────────────────────────────────────────
            user_resp = call_user(
                base_url, session_id, history, task_id=task_id, scenario=scenario
            )
            user_msg = user_resp["choices"][0]["message"]
            user_msg["role"] = "user"
            user_stop = user_resp.get("is_stop", False)

            history.append(user_msg)
            print_turn("user", user_msg, user_stop)

            if user_stop:
                print("\n[Simulation ended: user signalled stop]")
                break
        else:
            print(f"\n[Simulation ended: reached max turns ({max_turns})]")

    finally:
        if session_id:
            delete_session(base_url, session_id)
            print(f"\n[Session {session_id} deleted]")

    print("\n" + "=" * 60)
    print("Full history:")
    print(json.dumps(history, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Simulate a tau2 conversation step by step via the chat server."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--task-id",
        default="3",
        help="Task ID to simulate (resolved by the server). Default: '3'.",
    )
    group.add_argument(
        "--scenario",
        help="Raw scenario string passed directly to the user simulator.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of conversation turns. Default: 20.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Chat server base URL. Default: {DEFAULT_BASE_URL}.",
    )
    args = parser.parse_args()

    simulate(
        task_id=args.task_id if args.scenario is None else None,
        scenario=args.scenario,
        base_url=args.base_url,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
