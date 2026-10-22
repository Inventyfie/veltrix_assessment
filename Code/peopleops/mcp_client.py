from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPToolError(RuntimeError):
    pass


async def call_leave_tool(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    requester_id: str,
    db_path: Path,
    index_path: Path,
    today: date,
    write_authorized: bool = False,
    timeout: float = 6,
) -> dict[str, Any]:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "peopleops.mcp_server"],
        env={
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PYTHONPATH")
            if key in os.environ
        }
        | {
            "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
            "PYTHONUTF8": "1",
            "DB_PATH": str(db_path),
            "INDEX_PATH": str(index_path),
            "APP_TODAY": today.isoformat(),
            "TRUSTED_EMPLOYEE_ID": requester_id,
            "WRITE_AUTHORIZED": "1" if write_authorized else "0",
        },
    )

    async def invoke() -> dict[str, Any]:
        async with stdio_client(parameters) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)
                if result.isError:
                    raise MCPToolError("MCP tool execution failed")
                structured = getattr(result, "structuredContent", None)
                if isinstance(structured, dict):
                    return structured
                texts = [
                    item.text for item in result.content
                    if getattr(item, "type", None) == "text"
                ]
                if len(texts) == 1:
                    try:
                        decoded = json.loads(texts[0])
                    except json.JSONDecodeError:
                        pass
                    else:
                        if isinstance(decoded, dict):
                            return decoded
                raise MCPToolError("MCP tool returned an invalid result")

    try:
        return await asyncio.wait_for(invoke(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise MCPToolError("MCP tool timed out") from exc
