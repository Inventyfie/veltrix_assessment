from __future__ import annotations

import sqlite3
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .database import connect, employee
from .retrieval import applicable_holidays

UNAUTHORIZED_EMPLOYEE_MESSAGE = "I can't share information about that employee."


def _may_view_employee(
    connection: sqlite3.Connection, requester: sqlite3.Row, target_id: str
) -> sqlite3.Row | None:
    target = connection.execute(
        "SELECT employee_id, manager_id FROM employees WHERE employee_id = ?",
        (target_id,),
    ).fetchone()
    if target is None:
        return None
    if requester["employee_id"] == target_id:
        return target
    if requester["role"] == "hr":
        return target
    if requester["role"] == "manager" and target["manager_id"] == requester["employee_id"]:
        return target
    return None


def get_leave_balance(
    db_path: Path, requester_id: str, employee_id: str, leave_type: str | None = None
) -> dict[str, Any]:
    with connect(db_path) as connection:
        requester = connection.execute(
            "SELECT employee_id, role FROM employees WHERE employee_id = ?", (requester_id,)
        ).fetchone()
        if requester is None or _may_view_employee(connection, requester, employee_id) is None:
            return {"ok": False, "error": UNAUTHORIZED_EMPLOYEE_MESSAGE}
        if leave_type is not None and leave_type not in {"annual", "sick"}:
            return {"ok": False, "error": "leave_type must be annual or sick."}
        sql = (
            "SELECT leave_type, balance_days FROM leave_balances WHERE employee_id = ?"
            + (" AND leave_type = ?" if leave_type else "")
            + " ORDER BY leave_type"
        )
        parameters = (employee_id, leave_type) if leave_type else (employee_id,)
        rows = connection.execute(sql, parameters).fetchall()
        if not rows:
            return {"ok": False, "error": "No leave balance is recorded for that employee."}
        return {
            "ok": True,
            "employee_id": employee_id,
            "balances": [
                {"leave_type": row["leave_type"], "balance_days": row["balance_days"]}
                for row in rows
            ],
        }


def list_leave_requests(
    db_path: Path, requester_id: str, employee_id: str, status: str | None = None
) -> dict[str, Any]:
    with connect(db_path) as connection:
        requester = connection.execute(
            "SELECT employee_id, role FROM employees WHERE employee_id = ?", (requester_id,)
        ).fetchone()
        if requester is None or _may_view_employee(connection, requester, employee_id) is None:
            return {"ok": False, "error": UNAUTHORIZED_EMPLOYEE_MESSAGE}
        if status is not None and status not in {"pending", "approved", "cancelled"}:
            return {"ok": False, "error": "status must be pending, approved, or cancelled."}
        sql = (
            "SELECT request_id, leave_type, start_date, end_date, working_days, status "
            "FROM leave_requests WHERE employee_id = ?"
            + (" AND status = ?" if status else "")
            + " ORDER BY start_date, request_id"
        )
        parameters = (employee_id, status) if status else (employee_id,)
        rows = connection.execute(sql, parameters).fetchall()
        return {
            "ok": True,
            "employee_id": employee_id,
            "requests": [dict(row) for row in rows],
        }


def get_team_leave(
    db_path: Path, requester_id: str, start_date: str, end_date: str
) -> dict[str, Any]:
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return {"ok": False, "error": "Dates must use YYYY-MM-DD."}
    if end < start:
        return {"ok": False, "error": "end_date must be on or after start_date."}
    with connect(db_path) as connection:
        requester = connection.execute(
            "SELECT employee_id, role FROM employees WHERE employee_id = ?", (requester_id,)
        ).fetchone()
        if requester is None or requester["role"] != "manager":
            return {"ok": False, "error": "Only managers may view direct-report team leave."}
        rows = connection.execute(
            """
            SELECT r.employee_id, r.request_id, r.leave_type, r.start_date, r.end_date,
                   r.working_days, r.status
            FROM leave_requests r
            JOIN employees e ON e.employee_id = r.employee_id
            WHERE e.manager_id = ?
              AND r.status IN ('pending', 'approved')
              AND r.start_date <= ? AND r.end_date >= ?
            ORDER BY r.start_date, r.employee_id
            """,
            (requester_id, end.isoformat(), start.isoformat()),
        ).fetchall()
        return {"ok": True, "requests": [dict(row) for row in rows]}


def _validate_apply(
    db_path: Path,
    index_path: Path,
    requester_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    today: date,
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    if leave_type not in {"annual", "sick"}:
        return {"ok": False, "error": "leave_type must be annual or sick."}
    try:
        start, end = date.fromisoformat(start_date), date.fromisoformat(end_date)
    except ValueError:
        return {"ok": False, "error": "Dates must use YYYY-MM-DD."}
    if start.year != 2026 or end.year != 2026:
        return {"ok": False, "error": "Leave dates must both fall in 2026."}
    if start < today:
        return {"ok": False, "error": "start_date must be today or later."}
    if end < start:
        return {"ok": False, "error": "end_date must be on or after start_date."}
    requester = connection.execute(
        "SELECT employee_id, region FROM employees WHERE employee_id = ?",
        (requester_id,),
    ).fetchone()
    if requester is None:
        return {"ok": False, "error": "Requester identity is no longer valid."}
    holidays, calendar_error = applicable_holidays(index_path, "employee", requester["region"])
    if calendar_error:
        # HR and manager roles have no broader restrictions for their own requests.
        role = connection.execute(
            "SELECT role FROM employees WHERE employee_id = ?", (requester_id,)
        ).fetchone()["role"]
        holidays, calendar_error = applicable_holidays(index_path, role, requester["region"])
    if calendar_error or holidays is None:
        return {"ok": False, "error": calendar_error or "Holiday calendar is unresolved."}
    working_days = sum(
        1
        for offset in range((end - start).days + 1)
        if (day := start + timedelta(days=offset)).weekday() < 5
        and day.isoformat() not in holidays
    )
    if working_days == 0:
        return {"ok": False, "error": "The requested dates contain zero working days."}
    balance = connection.execute(
        "SELECT balance_days FROM leave_balances WHERE employee_id = ? AND leave_type = ?",
        (requester_id, leave_type),
    ).fetchone()
    if balance is None:
        return {"ok": False, "error": f"No {leave_type} balance is recorded."}
    if balance["balance_days"] < working_days:
        return {
            "ok": False,
            "error": f"Insufficient {leave_type} balance: {working_days} working days requested, "
            f"{balance['balance_days']} available.",
        }
    overlap = connection.execute(
        """
        SELECT 1 FROM leave_requests
        WHERE employee_id = ? AND status IN ('pending', 'approved')
          AND start_date <= ? AND end_date >= ?
        LIMIT 1
        """,
        (requester_id, end.isoformat(), start.isoformat()),
    ).fetchone()
    if overlap:
        return {"ok": False, "error": "The dates overlap your pending or approved leave request."}
    return {
        "ok": True,
        "employee_id": requester_id,
        "leave_type": leave_type,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "working_days": working_days,
        "balance_days": balance["balance_days"],
    }


def apply_leave(
    db_path: Path,
    index_path: Path,
    requester_id: str,
    leave_type: str,
    start_date: str,
    end_date: str,
    today: date,
    write_authorized: bool,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        if write_authorized:
            connection.execute("BEGIN IMMEDIATE")
        result = _validate_apply(
            db_path, index_path, requester_id, leave_type, start_date, end_date, today, connection
        )
        if not result["ok"]:
            if write_authorized:
                connection.rollback()
            return result
        if not write_authorized:
            return {**result, "proposal_only": True}
        updated = connection.execute(
            """
            UPDATE leave_balances SET balance_days = balance_days - ?
            WHERE employee_id = ? AND leave_type = ? AND balance_days >= ?
            """,
            (result["working_days"], requester_id, leave_type, result["working_days"]),
        )
        if updated.rowcount != 1:
            connection.rollback()
            return {"ok": False, "error": "Insufficient balance; the balance changed before confirmation."}
        request_id = str(uuid.uuid4())
        connection.execute(
            """
            INSERT INTO leave_requests(
                request_id, employee_id, leave_type, start_date, end_date,
                working_days, status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                request_id,
                requester_id,
                leave_type,
                result["start_date"],
                result["end_date"],
                result["working_days"],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
        return {**result, "ok": True, "request_id": request_id, "status": "pending"}


def cancel_leave(
    db_path: Path,
    requester_id: str,
    request_id: str,
    today: date,
    write_authorized: bool,
) -> dict[str, Any]:
    with connect(db_path) as connection:
        if write_authorized:
            connection.execute("BEGIN IMMEDIATE")
        request = connection.execute(
            """
            SELECT request_id, employee_id, leave_type, start_date, end_date, working_days, status
            FROM leave_requests WHERE request_id = ? AND employee_id = ?
            """,
            (request_id, requester_id),
        ).fetchone()
        if request is None:
            if write_authorized:
                connection.rollback()
            return {"ok": False, "error": "That leave request cannot be cancelled."}
        if request["status"] not in {"pending", "approved"}:
            if write_authorized:
                connection.rollback()
            return {"ok": False, "error": "Only pending or approved leave can be cancelled."}
        if date.fromisoformat(request["start_date"]) <= today:
            if write_authorized:
                connection.rollback()
            return {"ok": False, "error": "A request can be cancelled only when its start_date is after today."}
        if not write_authorized:
            return {
                "ok": True,
                "proposal_only": True,
                "request_id": request["request_id"],
                "leave_type": request["leave_type"],
                "start_date": request["start_date"],
                "end_date": request["end_date"],
                "working_days": request["working_days"],
                "status": request["status"],
            }
        updated = connection.execute(
            """
            UPDATE leave_requests SET status = 'cancelled'
            WHERE request_id = ? AND employee_id = ?
              AND status IN ('pending', 'approved') AND start_date > ?
            """,
            (request_id, requester_id, today.isoformat()),
        )
        if updated.rowcount != 1:
            connection.rollback()
            return {"ok": False, "error": "The request changed before confirmation; no balance was restored."}
        connection.execute(
            """
            UPDATE leave_balances SET balance_days = balance_days + ?
            WHERE employee_id = ? AND leave_type = ?
            """,
            (request["working_days"], requester_id, request["leave_type"]),
        )
        connection.commit()
        return {
            "ok": True,
            "request_id": request["request_id"],
            "leave_type": request["leave_type"],
            "start_date": request["start_date"],
            "end_date": request["end_date"],
            "working_days": request["working_days"],
            "status": "cancelled",
        }
