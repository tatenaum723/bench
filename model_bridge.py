"""Host side of the sandbox's only network path: unix socket -> SGLang TCP port.

The omp container runs with --network none. It reaches the model through a
unix socket mounted at /sock/model.sock, which this bridge forwards to
127.0.0.1:<port> on the host. Nothing else on the host or the internet is
reachable from inside the container.
"""
import asyncio
import os
import threading


class ModelBridge:
    def __init__(self, sock_dir, port=30000):
        self.sock_dir = sock_dir
        self.path = os.path.join(sock_dir, "model.sock")
        self.port = port
        self.loop = None
        self.thread = None
        self.server = None

    async def _handle(self, r, w):
        try:
            ur, uw = await asyncio.open_connection("127.0.0.1", self.port)
        except OSError:
            w.close()
            return

        async def pipe(a, b):
            try:
                while data := await a.read(65536):
                    b.write(data)
                    await b.drain()
            except Exception:
                pass
            finally:
                try:
                    b.close()
                except Exception:
                    pass
        await asyncio.gather(pipe(r, uw), pipe(ur, w))

    def start(self):
        os.makedirs(self.sock_dir, exist_ok=True)
        os.chmod(self.sock_dir, 0o755)
        if os.path.exists(self.path):
            os.remove(self.path)
        ready = threading.Event()

        def run():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            async def main():
                self.server = await asyncio.start_unix_server(self._handle, path=self.path)
                os.chmod(self.path, 0o777)   # container user is uid 10001
                ready.set()
                async with self.server:
                    await self.server.serve_forever()
            try:
                self.loop.run_until_complete(main())
            except asyncio.CancelledError:
                pass
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        if not ready.wait(10):
            raise RuntimeError("model bridge failed to start")
        return self

    def stop(self):
        if self.loop and self.server:
            self.loop.call_soon_threadsafe(self.server.close)
        if os.path.exists(self.path):
            os.remove(self.path)
