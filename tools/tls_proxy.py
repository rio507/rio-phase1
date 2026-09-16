"""TLS in front of the dashboard, so a phone can be asked for permissions.

    python -m tools.tls_proxy                      # 8443 (TLS) -> 8888 (plain)
    python -m tools.tls_proxy --listen 9443 --to 8888
    python -m tools.tls_proxy --check              # is the cert usable? exit 0/1

WHY A TERMINATOR RATHER THAN `uvicorn --ssl-keyfile`
-----------------------------------------------------
uvicorn can serve one protocol per process, and this process loads Qwen3-VL.
Turning the existing listener into an HTTPS one would mean either a second
16 GB model server or an HTTP dashboard that no longer exists — and every
selftest, every curl, every playwright run in this repository speaks plain HTTP
to 127.0.0.1:8888. Roughly fifteen suites would have to learn about
certificates to go on testing things that have nothing to do with TLS.

So TLS terminates here and the loopback stays plaintext, which is also how this
is done in production: the edge speaks TLS, the application does not. The phone
gets https://<lan address>:8443 and every local tool keeps working unchanged.

WHY RAW BYTES RATHER THAN AN HTTP PROXY
----------------------------------------
Because the dashboard's most important connection is not HTTP. The frame
transport is a WebSocket, the live session uses one, and an HTTP-aware proxy
has to be taught to pass an Upgrade through. Forwarding bytes needs no such
teaching: the upgrade, the binary frames and the close handshake are all just
bytes, and there is no parser here to get any of them subtly wrong.

The cost is that nothing is added to the request — no X-Forwarded-Proto, no
client address rewriting. Nothing in this application reads either, and the one
place the page cares about the scheme it reads from its own `location`, not
from a header.

WHAT IT DOES NOT DO
-------------------
It does not make the certificate trusted. A self-signed certificate warns on
first visit and tapping through is enough for a secure context, which is all the
permission prompts need. It also does not close port 8888 — plain HTTP is still
there for the loopback and for every local tool. On a machine reachable from
somewhere untrusted, that port is the one to firewall.
"""
import argparse
import asyncio
import os
import ssl
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CERT = REPO / "cert" / "rio-cert.pem"
KEY = REPO / "cert" / "rio-key.pem"

BUF = 65536


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    """One direction, until it ends. Never raises out of the task.

    A connection dying mid-copy is the normal case here, not an error: a phone
    locks, a tab closes, a WebSocket is cut. Every one of those arrives as a
    reset and none of them is worth a traceback in the log the driver's real
    faults are in.
    """
    try:
        while True:
            chunk = await reader.read(BUF)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError,
            ssl.SSLError, OSError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def handle(client_r, client_w, host: str, port: int, stats: dict):
    stats["opened"] += 1
    try:
        up_r, up_w = await asyncio.open_connection(host, port)
    except OSError as e:
        stats["upstream_failed"] += 1
        # The dashboard is not up. Answer in a way a browser can render rather
        # than dropping the socket, because "connection reset" and "the server
        # is not running" look identical on a phone and only one has a fix.
        try:
            body = (b"HTTP/1.1 502 Bad Gateway\r\n"
                    b"Content-Type: text/plain; charset=utf-8\r\n"
                    b"Connection: close\r\n\r\n"
                    b"RIO is not answering on the plain-HTTP port yet.\r\n"
                    b"Start it with: bash boot.sh restart\r\n")
            client_w.write(body)
            await client_w.drain()
        except Exception:
            pass
        try:
            client_w.close()
        except Exception:
            pass
        print(f"[tls] upstream {host}:{port} refused: {e}", flush=True)
        return

    await asyncio.gather(pipe(client_r, up_w), pipe(up_r, client_w))
    stats["closed"] += 1


def ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    # TLS 1.2 is the floor. Anything older is refused by the browsers this has
    # to serve anyway, so allowing it would only widen the surface.
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def check(cert: Path, key: Path) -> int:
    if not cert.exists() or not key.exists():
        print(f"no certificate at {cert}")
        print("make one with: python -m tools.make_cert")
        return 1
    try:
        ssl_context(cert, key)
    except Exception as e:
        print(f"certificate unusable: {type(e).__name__}: {e}")
        return 1
    mode = oct(os.stat(key).st_mode & 0o777)
    print(f"certificate ok: {cert}")
    print(f"private key:    {key}  (mode {mode})")
    if os.stat(key).st_mode & 0o077:
        print("  !! the private key is readable by others; chmod 600 it")
    return 0


async def serve(listen_port: int, to_host: str, to_port: int,
                cert: Path, key: Path):
    stats = {"opened": 0, "closed": 0, "upstream_failed": 0}
    ctx = ssl_context(cert, key)

    server = await asyncio.start_server(
        lambda r, w: handle(r, w, to_host, to_port, stats),
        host="0.0.0.0", port=listen_port, ssl=ctx)

    print(f"[tls] https://0.0.0.0:{listen_port}  ->  http://{to_host}:{to_port}",
          flush=True)
    print(f"[tls] cert {cert}", flush=True)
    print("[tls] the phone will warn once: it is self-signed and nothing has "
          "vouched for it. Tapping through gives a secure context, which is "
          "what the camera, microphone and geolocation prompts need.",
          flush=True)
    async with server:
        await server.serve_forever()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int,
                    default=int(os.getenv("RIO_HTTPS_PORT", "8443")))
    ap.add_argument("--to", type=int,
                    default=int(os.getenv("RIO_PORT", "8888")))
    ap.add_argument("--to-host", default="127.0.0.1")
    ap.add_argument("--cert", default=str(CERT))
    ap.add_argument("--key", default=str(KEY))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    cert, key = Path(a.cert), Path(a.key)
    if a.check:
        return check(cert, key)
    if not cert.exists() or not key.exists():
        print(f"no certificate at {cert}\nmake one with: python -m tools.make_cert")
        return 1

    try:
        asyncio.run(serve(a.listen, a.to_host, a.to, cert, key))
    except KeyboardInterrupt:
        pass
    except OSError as e:
        print(f"[tls] could not listen on {a.listen}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
