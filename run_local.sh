#!/usr/bin/env bash
# Start the chat app using the project's virtualenv if it exists.
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python run.py "$@"
else
  exec python3 run.py "$@"
fi
