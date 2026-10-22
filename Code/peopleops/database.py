from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=8, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 8000")
    return connection


def initialize_state(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                employee_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pending_proposals (
                session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id),
                employee_id TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS session_context (
                session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id),
                context_hint TEXT NOT NULL DEFAULT ''
            )
            """
        )


def employee(path: Path, employee_id: str) -> sqlite3.Row | None:
    with connect(path) as connection:
        return connection.execute(
            "SELECT employee_id, region, role, manager_id, join_date, level "
            "FROM employees WHERE employee_id = ?",
            (employee_id,),
        ).fetchone()


def session_owner(path: Path, session_id: str) -> str | None:
    with connect(path) as connection:
        row = connection.execute(
            "SELECT employee_id FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return None if row is None else str(row["employee_id"])


def establish_session(path: Path, session_id: str, employee_id: str) -> bool:
    with connect(path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT employee_id FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO chat_sessions(session_id, employee_id) VALUES (?, ?)",
                (session_id, employee_id),
            )
            connection.execute(
                "INSERT INTO session_context(session_id, context_hint) VALUES (?, '')",
                (session_id,),
            )
            connection.commit()
            return True
        connection.commit()
        return str(row["employee_id"]) == employee_id
