from __future__ import annotations

import argparse
import asyncio
import importlib
import os
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse


async def _relay(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    try:
        while chunk := await reader.read(64 * 1024):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError):
        pass


def _connection_handler(
    target_host: str, target_port: int
) -> Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]:
    async def handle(
        client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter
    ) -> None:
        upstream_writer: asyncio.StreamWriter | None = None
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                target_host, target_port
            )
            tasks = {
                asyncio.create_task(_relay(client_reader, upstream_writer)),
                asyncio.create_task(_relay(upstream_reader, client_writer)),
            }
            _, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        except (ConnectionError, OSError):
            pass
        finally:
            client_writer.close()
            if upstream_writer is not None:
                upstream_writer.close()
            await asyncio.gather(
                client_writer.wait_closed(),
                *(
                    [upstream_writer.wait_closed()]
                    if upstream_writer is not None
                    else []
                ),
                return_exceptions=True,
            )

    return handle


async def start_tcp_proxy(
    *,
    listen_host: str,
    listen_port: int,
    target_host: str,
    target_port: int,
) -> asyncio.Server:
    return await asyncio.start_server(
        _connection_handler(target_host, target_port), listen_host, listen_port
    )


def _configured_milvus_target() -> tuple[str, int]:
    importlib.import_module("dailymed_lightrag.config")
    raw_uri = os.getenv("MILVUS_URI") or "http://127.0.0.1:19530"
    parsed = urlparse(raw_uri if "://" in raw_uri else f"tcp://{raw_uri}")
    if not parsed.hostname or not parsed.port:
        raise ValueError("MILVUS_URI must include a host and port")
    return parsed.hostname, parsed.port


async def _serve(listen_host: str, listen_port: int) -> None:
    target_host, target_port = _configured_milvus_target()
    if target_host in {listen_host, "localhost"} and target_port == listen_port:
        raise ValueError("fault proxy target must differ from its listen endpoint")
    server = await start_tcp_proxy(
        listen_host=listen_host,
        listen_port=listen_port,
        target_host=target_host,
        target_port=target_port,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporary TCP fault-injection proxy")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    args = parser.parse_args()
    asyncio.run(_serve(args.listen_host, args.listen_port))


if __name__ == "__main__":
    main()
