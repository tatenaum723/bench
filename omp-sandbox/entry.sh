#!/bin/sh
# HOME is a tmpfs; give omp its natives and config, start the model bridge, run the command.
set -e
mkdir -p "$HOME/.omp/natives" "$HOME/.omp/agent"
cp -rsn /opt/omp-natives/. "$HOME/.omp/natives/" 2>/dev/null || true
[ -d /cfg ] && cp /cfg/*.yml "$HOME/.omp/agent/" 2>/dev/null || true
python3 /opt/bridge.py /sock/model.sock &
exec "$@"
