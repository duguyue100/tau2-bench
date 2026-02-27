#!/bin/bash

# Start the Tau2 chat server.
#
# Usage:
#   ./scripts/start_chat_server.sh --domain <domain> [options]
#
# All arguments are forwarded to `tau2 chat-server`. Run
#   tau2 chat-server --help
# for the full list of options (--agent-llm, --user-llm, --port, etc.).

echo "Starting the Tau2 chat server..."
tau2 chat-server "$@"
