# omp agent sandbox: the local model drives omp exactly as Tate would, but every
# command it runs executes here -- no network, read-only root, no capabilities,
# non-root. The ONLY route out is a unix socket bridged to the SGLang server.
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs git ripgrep procps ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir pytest==8.4.2 flask==3.1.2 \
 && useradd -m -u 10001 runner
COPY omp /usr/local/bin/omp
COPY natives /opt/omp-natives
COPY bridge.py /opt/bridge.py
COPY entry.sh /opt/entry.sh
RUN chmod 755 /usr/local/bin/omp /opt/entry.sh && chmod -R a+rX /opt/omp-natives
USER runner
WORKDIR /work
ENTRYPOINT ["/opt/entry.sh"]
