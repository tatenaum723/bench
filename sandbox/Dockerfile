# Execution sandbox for model-generated code. Runs with --network none,
# read-only root, no capabilities, non-root user, pids/memory/cpu limits.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.4.2 flask==3.1.2 \
 && useradd -m -u 10001 runner
USER runner
WORKDIR /work
