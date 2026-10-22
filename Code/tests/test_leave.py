from concurrent.futures import ThreadPoolExecutor
from datetime import date

from peopleops import leave


def test_access_control_for_leave_data_by_role(workspace):
    db = workspace["db"]
    own = leave.get_leave_balance(db, "E1001", "E1001")
    assert own["ok"]
    assert not leave.get_leave_balance(db, "E1001", "E1002")["ok"]
    assert leave.get_leave_balance(db, "E1002", "E1004")["ok"]
    assert not leave.get_leave_balance(db, "E1002", "E1005")["ok"]
    assert leave.get_leave_balance(db, "E1003", "E1005")["ok"]
    missing = leave.get_leave_balance(db, "E1001", "E9999")
    unauthorized = leave.get_leave_balance(db, "E1001", "E1005")
    assert missing["error"] == unauthorized["error"] == leave.UNAUTHORIZED_EMPLOYEE_MESSAGE


def test_team_leave_is_manager_only_and_limited_to_direct_reports(workspace):
    db = workspace["db"]
    result = leave.get_team_leave(db, "E1001", "2026-10-21", "2026-10-30")
    assert not result["ok"]
    request_id = "direct-report-request"
    import sqlite3

    connection = sqlite3.connect(db)
    connection.execute(
        "INSERT INTO leave_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, "E1004", "annual", "2026-10-22", "2026-10-22", 1, "pending", "2026-01-01"),
    )
    connection.commit()
    connection.close()
    result = leave.get_team_leave(db, "E1002", "2026-10-21", "2026-10-30")
    assert result["ok"] and result["requests"][0]["employee_id"] == "E1004"


def test_working_days_overlap_balance_and_atomic_cancellation(workspace):
    db, index = workspace["db"], workspace["index"]
    # Oct 22-26 includes a weekend and the Oct 23 regional holiday: two workdays.
    proposal = leave.apply_leave(
        db, index, "E1001", "annual", "2026-10-22", "2026-10-26",
        date(2026, 10, 21), False,
    )
    assert proposal["ok"] and proposal["working_days"] == 2
    assert leave.get_leave_balance(db, "E1001", "E1001", "annual")["balances"][0]["balance_days"] == 10
    request = leave.apply_leave(
        db, index, "E1001", "annual", "2026-10-22", "2026-10-26",
        date(2026, 10, 21), True,
    )
    assert request["ok"]
    assert leave.get_leave_balance(db, "E1001", "E1001", "annual")["balances"][0]["balance_days"] == 8
    overlap = leave.apply_leave(
        db, index, "E1001", "annual", "2026-10-26", "2026-10-26",
        date(2026, 10, 21), False,
    )
    assert not overlap["ok"] and "overlap" in overlap["error"]
    cancelled = leave.cancel_leave(db, "E1001", request["request_id"], date(2026, 10, 21), True)
    assert cancelled["ok"]
    repeated = leave.cancel_leave(db, "E1001", request["request_id"], date(2026, 10, 21), True)
    assert not repeated["ok"]
    assert leave.get_leave_balance(db, "E1001", "E1001", "annual")["balances"][0]["balance_days"] == 10


def test_apply_rejection_rules_and_future_cancel_boundary(workspace):
    db, index = workspace["db"], workspace["index"]
    today = date(2026, 10, 21)
    cases = [
        ("other", "2026-10-22", "2026-10-22", "leave_type"),
        ("annual", "2025-10-22", "2025-10-22", "2026"),
        ("annual", "2026-10-20", "2026-10-20", "today or later"),
        ("annual", "2026-10-24", "2026-10-25", "zero working days"),
        ("annual", "2026-10-26", "2026-10-22", "on or after"),
        ("annual", "2026-10-22", "2026-10-22", "calendar"),
    ]
    # Temporarily remove the holiday calendar to exercise fail-closed behavior.
    import sqlite3

    for leave_type, start, end, expected in cases[:5]:
        result = leave.apply_leave(db, index, "E1001", leave_type, start, end, today, False)
        assert not result["ok"] and expected.lower() in result["error"].lower()
    connection = sqlite3.connect(index)
    connection.execute("DELETE FROM holidays")
    connection.commit()
    connection.close()
    missing_calendar = leave.apply_leave(db, index, "E1001", "annual", "2026-10-22", "2026-10-22", today, False)
    assert not missing_calendar["ok"] and "calendar" in missing_calendar["error"].lower()

    request_id = "approved-boundary"
    connection = sqlite3.connect(db)
    connection.execute(
        "INSERT INTO leave_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, "E1001", "annual", "2026-10-21", "2026-10-22", 1, "approved", "2026-01-01"),
    )
    connection.commit()
    connection.close()
    result = leave.cancel_leave(db, "E1001", request_id, today, False)
    assert not result["ok"] and "after today" in result["error"]


def test_concurrent_applications_cannot_double_spend(workspace):
    db, index = workspace["db"], workspace["index"]

    def apply():
        return leave.apply_leave(
            db, index, "E1001", "annual", "2026-10-22", "2026-10-22",
            date(2026, 10, 21), True,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: apply(), range(2)))
    assert sum(bool(result["ok"]) for result in results) == 1
    assert leave.get_leave_balance(db, "E1001", "E1001", "annual")["balances"][0]["balance_days"] == 9
