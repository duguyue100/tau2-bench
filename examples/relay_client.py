#!/usr/bin/env python3
"""Run a shared relay conversation with external OpenAI-compatible endpoints.

This script:
1) Creates one shared relay session in tau2 chat server.
2) Alternates turns between external user-model and agent-model endpoints.
3) Sends each generated turn back to tau2 relay endpoint.

Example:
  python examples/relay_client.py \
    --domain retail \
    --user-model gpt-5-mini \
    --agent-model gpt-5 \
    --user-endpoint-base https://api.openai.com \
    --agent-endpoint-base https://api.openai.com
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any

from tau2.registry import registry


def _http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    request = urllib.request.Request(
        url=url,
        data=payload,
        headers=req_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed request to {url}: {exc}") from exc


def _parse_observation(observation: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for raw_line in observation.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        role, content = line.split(":", 1)
        role = role.strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        messages.append({"role": role, "content": content.strip()})
    return messages


def _generate_turn(
    endpoint_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
) -> dict[str, Any]:
    req_messages = [{"role": "system", "content": system_prompt}] + messages
    request_body: dict[str, Any] = {
        "model": model,
        "messages": req_messages,
        "temperature": temperature,
        "stream": False,
    }
    if tools and tool_choice != "none":
        request_body["tools"] = tools
        request_body["tool_choice"] = tool_choice

    response = _http_json(
        method="POST",
        url=f"{endpoint_base.rstrip('/')}/v1/chat/completions",
        body=request_body,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        message = response["choices"][0]["message"]
    except Exception as exc:
        raise RuntimeError(
            f"Invalid OpenAI response from {endpoint_base}: {json.dumps(response, indent=2)}"
        ) from exc

    result: dict[str, Any] = {}
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        result["content"] = content.strip()

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        if len(tool_calls) > 1:
            tool_calls = [tool_calls[0]]
        result["tool_calls"] = tool_calls

    if not result:
        raise RuntimeError("Model returned neither content nor tool_calls")
    return result


def _load_domain_metadata(
    domain: str,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    env = registry.get_env_constructor(domain)()
    policy = env.get_policy()
    agent_tools = [tool.openai_schema for tool in env.get_tools()]
    try:
        user_tools = [tool.openai_schema for tool in (env.get_user_tools() or [])]
    except ValueError:
        user_tools = []
    return policy, agent_tools, user_tools


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Relay client for tau2 chat server")
    parser.add_argument("--tau2-base-url", default="http://127.0.0.1:8005")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--task-split-name", default="base")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--full-observation", action="store_true", default=False)

    parser.add_argument("--user-endpoint-base", required=True)
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--user-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--user-temperature", type=float, default=0.7)
    parser.add_argument(
        "--user-tool-choice",
        choices=["auto", "required", "none"],
        default="auto",
    )

    parser.add_argument("--agent-endpoint-base", required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--agent-temperature", type=float, default=0.1)
    parser.add_argument(
        "--agent-tool-choice",
        choices=["auto", "required", "none"],
        default="auto",
    )

    parser.add_argument(
        "--user-system-prompt",
        default=(
            "You are the end user in a customer support chat. "
            "Respond naturally as the user, concise but realistic."
        ),
    )
    parser.add_argument(
        "--agent-system-prompt",
        default=(
            "You are the customer support agent. "
            "Help the user resolve their issue and follow policy."
        ),
    )
    parser.add_argument(
        "--include-domain-policy",
        action="store_true",
        default=False,
        help="Append tau2 domain policy text to both system prompts.",
    )

    args = parser.parse_args()

    user_api_key = _require_env(args.user_api_key_env)
    agent_api_key = _require_env(args.agent_api_key_env)
    domain_policy, agent_tools, user_tools = _load_domain_metadata(args.domain)

    user_system_prompt = args.user_system_prompt
    agent_system_prompt = args.agent_system_prompt
    if args.include_domain_policy:
        policy_block = f"\n\nDomain policy:\n{domain_policy}"
        user_system_prompt += policy_block
        agent_system_prompt += policy_block

    create_body: dict[str, Any] = {
        "domain": args.domain,
        "task_split_name": args.task_split_name,
        "max_steps": args.max_steps,
        "full_observation": args.full_observation,
    }
    if args.task_id:
        create_body["task_id"] = args.task_id

    create_response = _http_json(
        method="POST",
        url=f"{args.tau2_base_url.rstrip('/')}/v1/relay/sessions",
        body=create_body,
    )

    session_id = create_response["session_id"]
    next_turn = create_response.get("next_turn", "user")
    observation = create_response.get("initial_observation", "")

    print(f"relay session: {session_id}")
    print(f"initial next_turn: {next_turn}")

    terminated = False
    reward = 0.0
    try:
        for turn_idx in range(1, args.max_turns + 1):
            if terminated:
                break
            history = _parse_observation(observation)
            if next_turn == "user":
                generated = _generate_turn(
                    endpoint_base=args.user_endpoint_base,
                    api_key=user_api_key,
                    model=args.user_model,
                    messages=history,
                    system_prompt=user_system_prompt,
                    temperature=args.user_temperature,
                    tools=user_tools,
                    tool_choice=args.user_tool_choice,
                )
                if generated.get("tool_calls"):
                    tool_name = generated["tool_calls"][0]["function"]["name"]
                    print(f"[{turn_idx}] user -> tool_call:{tool_name}")
                else:
                    print(f"[{turn_idx}] user -> {generated['content']}")
                user_message = {"role": "user", **generated}
                relay_response = _http_json(
                    method="POST",
                    url=f"{args.tau2_base_url.rstrip('/')}/v1/relay/user/chat/completions",
                    body={
                        "session_id": session_id,
                        "messages": [user_message],
                    },
                )
            elif next_turn == "agent":
                generated = _generate_turn(
                    endpoint_base=args.agent_endpoint_base,
                    api_key=agent_api_key,
                    model=args.agent_model,
                    messages=history,
                    system_prompt=agent_system_prompt,
                    temperature=args.agent_temperature,
                    tools=agent_tools,
                    tool_choice=args.agent_tool_choice,
                )
                if generated.get("tool_calls"):
                    tool_name = generated["tool_calls"][0]["function"]["name"]
                    print(f"[{turn_idx}] agent -> tool_call:{tool_name}")
                else:
                    print(f"[{turn_idx}] agent -> {generated['content']}")
                agent_message = {"role": "assistant", **generated}
                relay_response = _http_json(
                    method="POST",
                    url=f"{args.tau2_base_url.rstrip('/')}/v1/relay/agent/chat/completions",
                    body={
                        "session_id": session_id,
                        "messages": [agent_message],
                    },
                )
            else:
                raise RuntimeError(f"Unknown next_turn: {next_turn}")

            observation = relay_response.get("observation", "")
            next_turn = relay_response.get("next_turn")
            terminated = bool(relay_response.get("terminated", False))
            reward = float(relay_response.get("reward", 0.0))

        print(f"terminated={terminated}, reward={reward}")
        return 0
    finally:
        try:
            _http_json(
                method="DELETE",
                url=f"{args.tau2_base_url.rstrip('/')}/v1/sessions/{session_id}",
            )
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
