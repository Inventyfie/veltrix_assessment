from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import leave
from .database import employee

mcp = FastMCP("Veltrix PeopleOps leave tools", json_response=True)


def _context() -> tuple[Path, Path, str, date, bool]:
    requester_id = os.environ.get("TRUSTED_EMPLOYEE_ID", "")
    db_path = Path(os.environ["DB_PATH"])
    index_path = Path(os.environ["INDEX_PATH"])
    today = date.fromisoformat(os.environ["APP_TODAY"])
    write_authorized = os.environ.get("WRITE_AUTHORIZED") == "1"
    if not requester_id or employee(db_path, requester_id) is None:
        raise RuntimeError("Trusted requester identity is unavailable")
    return db_path, index_path, requester_id, today, write_authorized


@mcp.tool()
def get_leave_balance(employee_id: str, leave_type: str | None = None) -> dict[str, Any]:
    """Return one employee's current leave balance, subject to role access."""
    db_path, _, requester_id, _, _ = _context()
    return leave.get_leave_balance(db_path, requester_id, employee_id, leave_type)


@mcp.tool()
def list_leave_requests(employee_id: str, status: str | None = None) -> dict[str, Any]:
    """List one employee's leave requests, subject to role access."""
    db_path, _, requester_id, _, _ = _context()
    return leave.list_leave_requests(db_path, requester_id, employee_id, status)


@mcp.tool()
def get_team_leave(start_date: str, end_date: str) -> dict[str, Any]:
    """Return overlapping pending or approved leave for direct reports."""
    db_path, _, requester_id, _, _ = _context()
    return leave.get_team_leave(db_path, requester_id, start_date, end_date)


@mcp.tool()
def apply_leave(leave_type: str, start_date: str, end_date: str) -> dict[str, Any]:
    """Validate a leave proposal or execute it with trusted confirmation authorization."""
    db_path, index_path, requester_id, today, write_authorized = _context()
    return leave.apply_leave(
        db_path,
        index_path,
        requester_id,
        leave_type,
        start_date,
        end_date,
        today,
        write_authorized,
    )


@mcp.tool()
def cancel_leave(request_id: str) -> dict[str, Any]:
    """Validate a cancellation proposal or execute it with trusted confirmation authorization."""
    db_path, _, requester_id, today, write_authorized = _context()
    return leave.cancel_leave(db_path, requester_id, request_id, today, write_authorized)


if __name__ == "__main__":
    mcp.run(transport="stdio")
