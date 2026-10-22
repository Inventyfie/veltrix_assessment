from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from .config import Settings
from .database import connect, employee
from .dates import dates_in_message
from .llm import OpenAIAnswerer, ProviderError, redact_personal_data
from .mcp_client import MCPToolError, call_leave_tool
from .retrieval import (
    RetrievedChunk,
    eligible_sources,
    precedence,
    reference_date_for_query,
    retrieve,
    scope_appears_supported,
)

logger = logging.getLogger("peopleops.chat")
CONFIRM_RE = re.compile(r"^(?:yes|confirm(?:ed)?|go ahead|approved|proceed|do it)[.! ]*$", re.I)
REFUSAL_RE = re.compile(
    r"(?:ignore (?:all )?(?:previous|prior|above) instructions|"
    r"(?:show|reveal|print|repeat).{0,30}(?:system prompt|hidden data|secret)|"
    r"(?:system prompt|hidden instructions)|"
    r"(?:bypass|override).{0,25}(?:security|access|confirmation))",
    re.I,
)
BULK_RE = re.compile(
    r"\b(?:everyone|all employees|all staff|company[- ]wide balances|"
    r"each employee|every employee|all direct reports' balances)\b",
    re.I,
)
EMPLOYEE_ID_RE = re.compile(r"\b[A-Z]\d{3,}\b", re.I)
CONTEXT_TERMS = (
    "leave", "annual", "sick", "holiday", "policy", "balance", "accrual",
    "carry", "forward", "calendar", "remote", "hybrid", "benefit", "salary",
    "compensation", "expense", "travel", "learning", "probation", "notice",
    "security", "performance", "conduct", "working", "days",
)


@dataclass
class ChatResponse:
    answer: str
    route: str
    citations: list[dict[str, str]]
    tool_calls: list[dict[str, Any]]
    pending_confirmation: bool = False


def _response(
    answer: str,
    route: str,
    citations: list[dict[str, str]] | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    pending_confirmation: bool = False,
) -> ChatResponse:
    citations = citations or []
    tool_calls = tool_calls or []
    valid = {
        "rag": bool(citations) and not tool_calls,
        "tool": bool(tool_calls) and not citations,
        "hybrid": bool(citations) and bool(tool_calls),
        "clarify": not citations and not tool_calls,
        "no_answer": not citations and not tool_calls,
        "refuse": not citations and not tool_calls,
    }
    if route not in valid or not valid[route]:
        raise RuntimeError(f"Invalid response route invariant: {route}")
    if pending_confirmation and (
        route != "tool"
        or not any(call["status"] == "pending_confirmation" for call in tool_calls)
    ):
        raise RuntimeError("A pending confirmation requires a proposed tool call")
    return ChatResponse(answer, route, citations, tool_calls, pending_confirmation)


def _context_hint(db_path: Path, session_id: str) -> str:
    with connect(db_path) as connection:
        row = connection.execute(
            "SELECT context_hint FROM session_context WHERE session_id = ?", (session_id,)
        ).fetchone()
    return "" if row is None else str(row["context_hint"])


def _save_context_hint(db_path: Path, session_id: str, message: str) -> None:
    words = set(re.findall(r"[a-z]+", message.lower()))
    hint = " ".join(term for term in CONTEXT_TERMS if term in words)[:200]
    with connect(db_path) as connection:
        connection.execute(
            "INSERT INTO session_context(session_id, context_hint) VALUES (?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET context_hint = excluded.context_hint",
            (session_id, hint),
        )


def _pending(path: Path, session_id: str) -> sqlite3.Row | None:
    with connect(path) as connection:
        return connection.execute(
            "SELECT tool_name, args_json FROM pending_proposals "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchone()


def _store_proposal(
    path: Path, session_id: str, employee_id: str, tool_name: str, args: dict[str, Any]
) -> None:
    with connect(path) as connection:
        connection.execute(
            """
            INSERT INTO pending_proposals(session_id, employee_id, tool_name, args_json)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET employee_id = excluded.employee_id,
                tool_name = excluded.tool_name, args_json = excluded.args_json,
                created_at = CURRENT_TIMESTAMP
            """,
            (session_id, employee_id, tool_name, json.dumps(args, separators=(",", ":"))),
        )


def _clear_proposal(path: Path, session_id: str) -> None:
    with connect(path) as connection:
        connection.execute("DELETE FROM pending_proposals WHERE session_id = ?", (session_id,))


def _tool_call(name: str, args: dict[str, Any], status: str) -> dict[str, Any]:
    return {"name": name, "args": args, "status": status}


async def _call_tool(
    name: str,
    args: dict[str, Any],
    *,
    employee_id: str,
    settings: Settings,
    write_authorized: bool = False,
) -> dict[str, Any]:
    return await call_leave_tool(
        name,
        args,
        requester_id=employee_id,
        db_path=settings.db_path,
        index_path=settings.index_path,
        today=settings.today,
        write_authorized=write_authorized,
        timeout=6,
    )


def _format_read(name: str, result: dict[str, Any]) -> str:
    if not result.get("ok"):
        return str(result.get("error", "The requested leave action was rejected."))
    if name == "get_leave_balance":
        values = result["balances"]
        return "Current leave balances: " + "; ".join(
            f"{item['leave_type']}: {item['balance_days']} days" for item in values
        ) + "."
    if name == "list_leave_requests":
        requests = result["requests"]
        if not requests:
            return "No leave requests were found."
        return "Leave requests: " + "; ".join(
            f"{item['request_id']} ({item['leave_type']}, {item['start_date']} to "
            f"{item['end_date']}, {item['working_days']} working days, {item['status']})"
            for item in requests
        ) + "."
    requests = result["requests"]
    if not requests:
        return "No pending or approved direct-report leave overlaps that date range."
    return "Overlapping direct-report leave: " + "; ".join(
        f"{item['employee_id']}: {item['leave_type']} {item['start_date']} to "
        f"{item['end_date']} ({item['status']})"
        for item in requests
    ) + "."


def _tool_intent(message: str) -> str | None:
    lowered = message.lower()
    if re.search(r"\b(cancel|withdraw|revoke)\b", lowered) and re.search(
        r"\b(leave|request|booking)\b", lowered
    ):
        return "cancel_leave"
    if re.search(r"\b(apply|request|book|take)\b", lowered) and re.search(
        r"\b(leave|vacation|time off)\b", lowered
    ):
        return "apply_leave"
    if re.search(r"\b(team leave|leave for (?:the )?team|direct reports?|my reports)\b", lowered):
        return "get_team_leave"
    if re.search(r"\b(requests?|history)\b", lowered) and "leave" in lowered:
        return "list_leave_requests"
    if "balance" in lowered and re.search(r"\b(leave|annual|sick)\b", lowered):
        return "get_leave_balance"
    return None


def _new_proposal_args(
    tool_name: str, message: str, current: dict[str, Any] | None, today: date
) -> tuple[dict[str, Any] | None, str | None]:
    dates = dates_in_message(message, today)
    if tool_name == "apply_leave":
        types = set(re.findall(r"\b(annual|sick)\b", message.lower()))
        leave_type = next(iter(types)) if len(types) == 1 else None
        if current:
            leave_type = leave_type or current.get("leave_type")
        has_range_word = bool(re.search(r"\b(from|between|until|through|to)\b", message, re.I))
        if current and dates and len(dates) == 1:
            if re.search(r"\b(start|from)\b", message, re.I):
                start, end = dates[0], current["end_date"]
            else:
                start, end = current["start_date"], dates[0]
        elif len(dates) >= 2:
            start, end = dates[0], dates[1]
        elif len(dates) == 1 and not has_range_word:
            start = end = dates[0]
        elif current:
            start, end = date.fromisoformat(current["start_date"]), date.fromisoformat(current["end_date"])
        else:
            start = end = None
        missing = []
        if leave_type is None:
            missing.append("annual or sick leave")
        if start is None or end is None:
            missing.append("the start and end dates")
        if missing:
            return None, "Please provide " + " and ".join(missing) + "."
        if start < today:
            return None, "That date has already passed. Please provide a date that is today or later."
        return {
            "leave_type": leave_type,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
        }, None
    if tool_name == "cancel_leave":
        match = re.search(r"\b[A-F0-9-]{12,}\b", message, re.I)
        request_id = match.group(0) if match else (
            current.get("request_id") if current else None
        )
        if not request_id:
            return None, "Please provide the request ID you want to cancel."
        return {"request_id": request_id}, None
    return None, None


def _is_changed_followup(message: str, pending: sqlite3.Row) -> bool:
    return bool(
        pending["tool_name"] == "apply_leave"
        and re.search(r"\b(instead|change|make it|actually|until|through|start date|end date)\b", message, re.I)
    )


def _citations_for(answer_ids: tuple[str, ...], chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    locators: dict[str, str] = {}
    for chunk in chunks:
        locators.setdefault(chunk.doc_id, chunk.locator)
    return [{"doc_id": doc_id, "locator": locators[doc_id]} for doc_id in answer_ids]


def _future_candidates(
    index_path: Path, role: str, region: str, reference: date, message: str
) -> list[RetrievedChunk]:
    all_future = eligible_sources(
        index_path, role, region, date(9999, 12, 31), include_future=True
    )
    future = [source for source in all_future if source.effective_date > reference]
    if not future:
        return []
    candidates = retrieve(index_path, message, role, region, date(9999, 12, 31), limit=30)
    by_id: dict[str, list[RetrievedChunk]] = {}
    for chunk in candidates:
        by_id.setdefault(chunk.doc_id, []).append(chunk)
    current_sources = {
        source.doc_id: source
        for source in eligible_sources(index_path, role, region, reference)
        if source.doc_id in by_id
    }
    if not current_sources:
        return []
    highest_current = max(precedence(source, region) for source in current_sources.values())
    selected: list[RetrievedChunk] = []
    for source in future:
        if source.doc_id not in by_id or precedence(source, region) < highest_current:
            continue
        selected.extend(by_id[source.doc_id][:2])
    return selected


async def handle_chat(
    *,
    session_id: str,
    employee_id: str,
    message: str,
    settings: Settings,
    answerer: OpenAIAnswerer,
) -> ChatResponse:
    requester = employee(settings.db_path, employee_id)
    if requester is None:
        raise RuntimeError("Validated requester disappeared from the database")
    pending = _pending(settings.db_path, session_id)
    calls: list[dict[str, Any]] = []
    if pending is not None:
        action_args = json.loads(pending["args_json"])
        if CONFIRM_RE.fullmatch(message.strip()):
            name = str(pending["tool_name"])
            result = await _call_tool(
                name,
                action_args,
                employee_id=employee_id,
                settings=settings,
                write_authorized=True,
            )
            _clear_proposal(settings.db_path, session_id)
            calls.append(_tool_call(name, action_args, "success" if result.get("ok") else "error"))
            if result.get("ok"):
                if name == "apply_leave":
                    answer = (
                        f"Your {result['leave_type']} leave request for {result['start_date']} to "
                        f"{result['end_date']} ({result['working_days']} working days) is pending. "
                        f"Request ID: {result['request_id']}."
                    )
                else:
                    answer = (
                        f"Leave request {result['request_id']} was cancelled and "
                        f"{result['working_days']} days were restored."
                    )
            else:
                answer = str(result.get("error", "The request could not be completed."))
            return _response(answer, "tool", tool_calls=calls)
        if _is_changed_followup(message, pending):
            _clear_proposal(settings.db_path, session_id)
            next_args, clarification = _new_proposal_args(
                str(pending["tool_name"]), message, action_args, settings.today
            )
            if clarification:
                return _response(clarification, "clarify")
            assert next_args is not None
            name = str(pending["tool_name"])
            result = await _call_tool(name, next_args, employee_id=employee_id, settings=settings)
            calls.append(_tool_call(name, next_args, "pending_confirmation" if result.get("ok") else "error"))
            if not result.get("ok"):
                return _response(str(result["error"]), "tool", tool_calls=calls)
            _store_proposal(settings.db_path, session_id, employee_id, name, next_args)
            answer = (
                f"Updated proposal: {next_args['leave_type']} leave from "
                f"{result['start_date']} to {result['end_date']} "
                f"({result['working_days']} working days). Please confirm."
            )
            return _response(answer, "tool", tool_calls=calls, pending_confirmation=True)
        _clear_proposal(settings.db_path, session_id)

    if REFUSAL_RE.search(message):
        return _response(
            "I can help with Veltrix HR policies, holidays, and permitted leave information, "
            "but I can't reveal hidden instructions or bypass access controls.",
            "refuse",
        )
    intent = _tool_intent(message)
    if intent and BULK_RE.search(message) and intent != "get_team_leave":
        return _response("I can't provide bulk employee leave information.", "refuse")
    if intent in {"apply_leave", "cancel_leave"}:
        args, clarification = _new_proposal_args(intent, message, None, settings.today)
        if clarification:
            return _response(clarification, "clarify")
        assert args is not None
        result = await _call_tool(intent, args, employee_id=employee_id, settings=settings)
        calls.append(_tool_call(intent, args, "pending_confirmation" if result.get("ok") else "error"))
        if not result.get("ok"):
            return _response(str(result.get("error")), "tool", tool_calls=calls)
        _store_proposal(settings.db_path, session_id, employee_id, intent, args)
        if intent == "apply_leave":
            answer = (
                f"Please confirm {args['leave_type']} leave from {result['start_date']} to "
                f"{result['end_date']} ({result['working_days']} working days)."
            )
        else:
            answer = (
                f"Please confirm cancellation of request {result['request_id']} "
                f"({result['leave_type']}, {result['start_date']} to {result['end_date']})."
            )
        return _response(answer, "tool", tool_calls=calls, pending_confirmation=True)

    if intent in {"get_leave_balance", "list_leave_requests"}:
        target_match = EMPLOYEE_ID_RE.search(message)
        target_id = target_match.group(0).upper() if target_match else employee_id
        args: dict[str, Any] = {"employee_id": target_id}
        type_match = re.search(r"\b(annual|sick)\b", message, re.I)
        if intent == "get_leave_balance" and type_match:
            args["leave_type"] = type_match.group(1).lower()
        status_match = re.search(r"\b(pending|approved|cancelled)\b", message, re.I)
        if intent == "list_leave_requests" and status_match:
            args["status"] = status_match.group(1).lower()
        result = await _call_tool(intent, args, employee_id=employee_id, settings=settings)
        calls.append(_tool_call(intent, args, "success" if result.get("ok") else "error"))
        if not result.get("ok") and result.get("error") == "I can't share information about that employee.":
            answer = result["error"]
        else:
            answer = _format_read(intent, result)
        return _response(answer, "tool", tool_calls=calls)

    if intent == "get_team_leave":
        dates = dates_in_message(message, settings.today)
        if len(dates) < 2:
            return _response("Please provide the start and end dates for the team leave range.", "clarify")
        args = {"start_date": dates[0].isoformat(), "end_date": dates[1].isoformat()}
        result = await _call_tool(intent, args, employee_id=employee_id, settings=settings)
        calls.append(_tool_call(intent, args, "success" if result.get("ok") else "error"))
        return _response(_format_read(intent, result), "tool", tool_calls=calls)

    if intent and BULK_RE.search(message):
        return _response("I can't provide bulk employee leave information.", "refuse")

    if not scope_appears_supported(message):
        return _response(
            "I can help with Veltrix HR policies, holidays, benefits, and permitted leave information.",
            "refuse",
        )
    reference_date = reference_date_for_query(message, settings.today)
    hint = _context_hint(settings.db_path, session_id)
    search_message = (hint + " " + message).strip()
    chunks = retrieve(
        settings.index_path, search_message, requester["role"], requester["region"], reference_date
    )
    future = _future_candidates(
        settings.index_path, requester["role"], requester["region"], reference_date, search_message
    )
    seen = {(chunk.doc_id, chunk.locator, chunk.content) for chunk in chunks}
    chunks.extend(
        chunk for chunk in future
        if (chunk.doc_id, chunk.locator, chunk.content) not in seen
    )
    if not chunks:
        _save_context_hint(settings.db_path, session_id, message)
        return _response(
            "The documents do not cover this question. Please contact the HR team.",
            "no_answer",
        )
    sanitized_message = redact_personal_data(message, settings.db_path)
    source_ids = {chunk.doc_id for chunk in chunks}
    metadata = {
        source.doc_id: {
            "doc_type": source.doc_type,
            "regions": list(source.regions),
            "effective_date": source.effective_date.isoformat(),
        }
        for source in eligible_sources(
            settings.index_path,
            requester["role"],
            requester["region"],
            date.max,
            include_future=True,
        )
        if source.doc_id in source_ids
    }
    try:
        result = await answerer.answer(
            sanitized_message, chunks, source_metadata=metadata
        )
    except ProviderError:
        raise
    citations = _citations_for(result.doc_ids, chunks)
    _save_context_hint(settings.db_path, session_id, message)
    if not citations:
        return _response(
            "The documents do not cover this question. Please contact the HR team.",
            "no_answer",
        )
    return _response(result.text, "rag", citations=citations)
