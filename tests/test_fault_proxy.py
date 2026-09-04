import asyncio

import pytest

from medication_review_agent.fault_proxy import start_tcp_proxy


@pytest.mark.asyncio
async def test_tcp_fault_proxy_forwards_bytes_until_server_is_closed() -> None:
    async def echo(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            writer.write(await reader.readexactly(4))
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    target = await asyncio.start_server(echo, "127.0.0.1", 0)
    target_port = int(target.sockets[0].getsockname()[1])
    proxy = await start_tcp_proxy(
        listen_host="127.0.0.1",
        listen_port=0,
        target_host="127.0.0.1",
        target_port=target_port,
    )
    proxy_port = int(proxy.sockets[0].getsockname()[1])

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        writer.write(b"ping")
        await writer.drain()
        assert await reader.readexactly(4) == b"ping"
        writer.close()
        await writer.wait_closed()
    finally:
        proxy.close()
        await proxy.wait_closed()
        target.close()
        await target.wait_closed()
