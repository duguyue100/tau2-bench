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
        if role not in {"system", "user", "assistant", "tool"}:
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
) -> str:
    req_messages = [{"role": "system", "content": system_prompt}] + messages
    response = _http_json(
        method="POST",
        url=f"{endpoint_base.rstrip('/')}/v1/chat/completions",
        body={
            "model": model,
            "messages": req_messages,
            "temperature": temperature,
            "stream": False,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(
            f"Invalid OpenAI response from {endpoint_base}: {json.dumps(response, indent=2)}"
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Model returned empty content")
    return content.strip()


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

    parser.add_argument("--agent-endpoint-base", required=True)
    parser.add_argument("--agent-model", required=True)
    parser.add_argument("--agent-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--agent-temperature", type=float, default=0.1)

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

    args = parser.parse_args()

    user_api_key = _require_env(args.user_api_key_env)
    agent_api_key = _require_env(args.agent_api_key_env)

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
                text = _generate_turn(
                    endpoint_base=args.user_endpoint_base,
                    api_key=user_api_key,
                    model=args.user_model,
                    messages=history,
                    system_prompt=args.user_system_prompt,
                    temperature=args.user_temperature,
                )
                print(f"[{turn_idx}] user -> {text}")
                relay_response = _http_json(
                    method="POST",
                    url=f"{args.tau2_base_url.rstrip('/')}/v1/relay/user/chat/completions",
                    body={
                        "session_id": session_id,
                        "messages": [{"role": "user", "content": text}],
                    },
                )
            elif next_turn == "agent":
                text = _generate_turn(
                    endpoint_base=args.agent_endpoint_base,
                    api_key=agent_api_key,
                    model=args.agent_model,
                    messages=history,
                    system_prompt=args.agent_system_prompt,
                    temperature=args.agent_temperature,
                )
                print(f"[{turn_idx}] agent -> {text}")
                relay_response = _http_json(
                    method="POST",
                    url=f"{args.tau2_base_url.rstrip('/')}/v1/relay/agent/chat/completions",
                    body={
                        "session_id": session_id,
                        "messages": [{"role": "assistant", "content": text}],
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
