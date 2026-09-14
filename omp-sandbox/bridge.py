"""TCP 127.0.0.1:30000 inside the container -> unix socket /sock/model.sock.
Runs in a --network none container, so this is the only path to the model."""
import asyncio, sys
SOCK = sys.argv[1] if len(sys.argv) > 1 else "/sock/model.sock"
async def handle(r, w):
    try:
        ur, uw = await asyncio.open_unix_connection(SOCK)
    except OSError:
        w.close(); return
    async def pipe(a, b):
        try:
            while data := await a.read(65536):
                b.write(data); await b.drain()
        except Exception:
            pass
        finally:
            try: b.close()
            except Exception: pass
    await asyncio.gather(pipe(r, uw), pipe(ur, w))
async def main():
    srv = await asyncio.start_server(handle, "127.0.0.1", 30000)
    async with srv:
        await srv.serve_forever()
asyncio.run(main())
