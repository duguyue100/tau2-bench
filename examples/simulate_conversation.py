"""
simulate_conversation.py
------------------------
Example: step-by-step conversation simulation using the tau2 chat server.

Start the server first:
    tau2 chat-server --domain airline --agent-llm gpt-4o-mini --user-llm gpt-4o-mini

Then run this script:
    python examples/simulate_conversation.py --task-id 3
    python examples/simulate_conversation.py --scenario "I want to cancel my flight."

To demonstrate TTL-based session eviction, start the server with a short TTL
and pass --wait-for-eviction to this script:
    tau2 chat-server --domain airline --agent-llm gpt-4o-mini --user-llm gpt-4o-mini \\
        --session-ttl 10
    python examples/simulate_conversation.py --task-id 3 --wait-for-eviction

Conversation flow
-----------------
1. POST /v1/session        {"task_id": "3"}
   -> {"session_id": "..."}   (initialises environment with task's initial_state)

2. POST /v1/agent/turn  {"session_id": "...", "messages": []}
   -> Agent sends opening greeting (no tools called on first turn).

3. POST /v1/user/chat/completions   {"session_id": "...", "task_id": "3",
                                     "messages": [agent greeting]}
   -> User simulator replies.

4. POST /v1/agent/turn  {"session_id": "...", "messages": [...]}
   -> Server runs the full agent turn internally (LLM → tool calls → LLM → …)
      and returns the final text message plus all intermediate tool steps.
   -> Append steps[*].tool_calls + steps[*].tool_results + final_message to history.

5. Repeat from step 3 until is_stop == True or max turns reached.

6. Session is left to expire via server-side TTL.  Use GET /v1/session/{id}
   to check whether it is still alive (200 = alive, 404 = evicted / gone).
"""

import argparse
import sys
import time
import json

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


def session_alive(base_url: str, session_id: str) -> bool:
    """Return True if the session exists on the server, False if it is gone."""
    resp = requests.get(f"{base_url}/v1/session/{session_id}", timeout=10)
    if resp.status_code == 200:
        return True
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return False  # unreachable


def call_agent_turn(base_url: str, session_id: str, messages: list[dict]) -> dict:
    """Run a full agent turn (LLM + tool loops) in one request."""
    return call(
        "POST",
        base_url,
        "/v1/agent/turn",
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


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_agent_turn(turn: dict) -> None:
    """Print a /v1/agent/turn response."""
    is_stop = turn.get("is_stop", False)
    # Print intermediate tool steps
    for i, step in enumerate(turn.get("steps", []), 1):
        print(f"\n[AGENT - tool round {i}]")
        for tc in step.get("tool_calls", []):
            fn = tc.get("function", {})
            print(f"  -> {fn.get('name')}({fn.get('arguments')})")
        print("  [ENV results]")
        for r in step.get("tool_results", []):
            error_tag = " [ERROR]" if r.get("error") else ""
            print(f"    tool_call_id={r['tool_call_id']}{error_tag}: {r['content']}")
    # Final message
    stop_tag = "  [STOP]" if is_stop else ""
    print(f"\n[AGENT]{stop_tag}")
    content = turn.get("final_message", {}).get("content")
    if content:
        print(f"  {content}")


def print_user_turn(msg: dict, is_stop: bool = False) -> None:
    stop_tag = "  [STOP]" if is_stop else ""
    print(f"\n[USER]{stop_tag}")
    content = msg.get("content")
    if content:
        print(f"  {content}")


# ---------------------------------------------------------------------------
# Build history from an agent turn response
# ---------------------------------------------------------------------------


def history_from_agent_turn(turn: dict) -> list[dict]:
    """Return the list of messages to append to history from a turn response."""
    msgs = []
    for step in turn.get("steps", []):
        # assistant message with tool_calls
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": step["tool_calls"],
            }
        )
        # tool result messages
        for r in step["tool_results"]:
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "content": r["content"],
                    "requestor": r["requestor"],
                }
            )
    # Final text message
    msgs.append(turn["final_message"])
    return msgs


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def wait_for_eviction(
    base_url: str,
    session_id: str,
    poll_interval: float = 2.0,
    timeout: float = 120.0,
) -> bool:
    """
    Poll GET /v1/session/{session_id} until it returns 404 (evicted) or
    *timeout* seconds elapse.

    Returns True if the session was evicted, False if it was still alive after
    *timeout* seconds.
    """
    deadline = time.time() + timeout
    print(
        f"\n[Waiting for TTL eviction of session {session_id} "
        f"(polling every {poll_interval:.0f}s, timeout {timeout:.0f}s)...]"
    )
    while time.time() < deadline:
        alive = session_alive(base_url, session_id)
        if not alive:
            print(
                f"[Session {session_id} has been evicted by the server (TTL expired)]"
            )
            return True
        remaining = deadline - time.time()
        print(
            f"  Session still alive. Checking again in {poll_interval:.0f}s "
            f"({remaining:.0f}s remaining)..."
        )
        time.sleep(poll_interval)
    print(f"[Timeout: session {session_id} is still alive after {timeout:.0f}s]")
    return False


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------


def simulate(
    task_id: str | None,
    scenario: str | None,
    base_url: str = DEFAULT_BASE_URL,
    max_turns: int = 20,
    wait_for_eviction: bool = False,
    eviction_timeout: float = 120.0,
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
    session_ttl = cfg.get("session_ttl", 0)
    if session_ttl > 0:
        print(
            f"Session TTL: {session_ttl}s (server will evict idle sessions automatically)"
        )
    else:
        print("Session TTL: disabled")

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

    for _turn in range(max_turns):
        # ── Agent turn ────────────────────────────────────────────────
        # /v1/agent/turn handles the full LLM→tool→LLM loop internally.
        agent_turn = call_agent_turn(base_url, session_id, history)
        agent_stop = agent_turn.get("is_stop", False)

        print_agent_turn(agent_turn)
        history.extend(history_from_agent_turn(agent_turn))

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
        print_user_turn(user_msg, user_stop)

        if user_stop:
            print("\n[Simulation ended: user signalled stop]")
            break
    else:
        print(f"\n[Simulation ended: reached max turns ({max_turns})]")

    # Session is intentionally NOT deleted here.  The server will evict it
    # automatically once the TTL expires.  Use GET /v1/session/{id} to check.
    print(f"\n[Session {session_id} left on server (TTL-based eviction)]")
    if session_ttl > 0:
        print(
            f"  It will be evicted automatically after ~{session_ttl}s of inactivity."
        )
    print(f"  Check manually: GET {base_url}/v1/session/{session_id}")

    if wait_for_eviction:
        if session_ttl <= 0:
            print(
                "\nWARNING: --wait-for-eviction has no effect when the server's "
                "session_ttl is 0 (eviction disabled). Start the server with "
                "--session-ttl <seconds> to enable it."
            )
        else:
            _poll_for_eviction(base_url, session_id, eviction_timeout)

    print("\n" + "=" * 60)
    print("Full history:")
    print(json.dumps(history, indent=2))


def _poll_for_eviction(base_url: str, session_id: str, timeout: float = 120.0) -> None:
    """Poll until the session disappears or timeout elapses."""
    poll_interval = 2.0
    deadline = time.time() + timeout
    print(
        f"\n[Waiting for TTL eviction of session {session_id} "
        f"(polling every {poll_interval:.0f}s, timeout {timeout:.0f}s)...]"
    )
    while time.time() < deadline:
        alive = session_alive(base_url, session_id)
        if not alive:
            print(
                f"[Session {session_id} has been evicted by the server (TTL expired)]"
            )
            return
        remaining = deadline - time.time()
        print(
            f"  Session still alive. Checking again in {poll_interval:.0f}s "
            f"({remaining:.0f}s remaining)..."
        )
        time.sleep(poll_interval)
    print(f"[Timeout: session {session_id} is still alive after {timeout:.0f}s]")


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
    parser.add_argument(
        "--wait-for-eviction",
        action="store_true",
        help=(
            "After the conversation ends, poll GET /v1/session/{id} every 2s "
            "until the server evicts the session (TTL expired) or "
            "--eviction-timeout elapses.  Requires the server to be started "
            "with --session-ttl <N>."
        ),
    )
    parser.add_argument(
        "--eviction-timeout",
        type=float,
        default=120.0,
        help=(
            "Maximum seconds to wait for eviction when --wait-for-eviction is "
            "set. Default: 120."
        ),
    )
    args = parser.parse_args()

    simulate(
        task_id=args.task_id if args.scenario is None else None,
        scenario=args.scenario,
        base_url=args.base_url,
        max_turns=args.max_turns,
        wait_for_eviction=args.wait_for_eviction,
        eviction_timeout=args.eviction_timeout,
    )


if __name__ == "__main__":
    main()
