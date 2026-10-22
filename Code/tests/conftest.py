from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import pytest

from peopleops.config import Settings
from peopleops.documents import build_index


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    entries = [
        (
            "POL-GLOBAL-V1",
            "policy_v1.md",
            "policy",
            ["GLOBAL"],
            "2025-01-01",
            "all",
            "# Leave Policy\n## Annual leave\nThe annual leave allowance is 20 days.",
        ),
        (
            "POL-GLOBAL-V2",
            "policy_v2.md",
            "policy",
            ["GLOBAL"],
            "2026-04-01",
            "all",
            "# Leave Policy\n## Annual leave\nThe annual leave allowance is 21 days.",
        ),
        (
            "POL-GLOBAL-V3",
            "policy_v3.md",
            "policy",
            ["GLOBAL"],
            "2027-01-01",
            "all",
            "# Leave Policy\n## Annual leave\nThe annual leave allowance will be 24 days.",
        ),
        (
            "ADD-DE",
            "germany.md",
            "addendum",
            ["DE"],
            "2025-01-01",
            "all",
            "# Germany Leave Addendum\n## Annual leave\nThe annual leave allowance is 30 days.",
        ),
        (
            "FAQ-GLOBAL",
            "faq.html",
            "faq",
            ["GLOBAL"],
            "2026-05-01",
            "all",
            "<h1>Leave FAQ</h1><p>The annual leave allowance is 18 days.</p>",
        ),
        (
            "MGR-ONLY",
            "manager.md",
            "handbook",
            ["GLOBAL"],
            "2026-01-01",
            "managers",
            "# Manager Handbook\n## Private\nRestrictedMarker team confidentiality.",
        ),
        (
            "HR-ONLY",
            "hr.md",
            "policy",
            ["GLOBAL"],
            "2026-01-01",
            "hr",
            "# HR Guidance\n## Private\nHRRestrictedMarker process.",
        ),
        (
            "CAL-IN",
            "india.html",
            "calendar",
            ["IN"],
            "2026-01-01",
            "all",
            "<h1>India Calendar 2026</h1><table><tr><th>Date</th><th>Holiday</th></tr>"
            "<tr><td>2026-10-23</td><td>Example Regional Holiday</td></tr></table>",
        ),
    ]
    manifest = []
    for doc_id, filename, doc_type, regions, effective_date, access, content in entries:
        (corpus / filename).write_text(content, encoding="utf-8")
        manifest.append(
            {
                "doc_id": doc_id,
                "file": filename,
                "title": doc_id,
                "doc_type": doc_type,
                "regions": regions,
                "effective_date": effective_date,
                "access_level": access,
            }
        )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    db_path = tmp_path / "employees.sqlite3"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE employees (
            employee_id TEXT PRIMARY KEY, full_name TEXT NOT NULL, email TEXT NOT NULL,
            phone TEXT NOT NULL, region TEXT NOT NULL, join_date TEXT NOT NULL,
            level TEXT NOT NULL, role TEXT NOT NULL, manager_id TEXT
        );
        CREATE TABLE leave_balances (
            employee_id TEXT NOT NULL, leave_type TEXT NOT NULL, balance_days REAL NOT NULL,
            PRIMARY KEY(employee_id, leave_type)
        );
        CREATE TABLE leave_requests (
            request_id TEXT PRIMARY KEY, employee_id TEXT NOT NULL, leave_type TEXT NOT NULL,
            start_date TEXT NOT NULL, end_date TEXT NOT NULL, working_days INTEGER NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL
        );
        INSERT INTO employees VALUES
            ('E1001', 'Alice Example', 'alice@example.test', '+1 555 0101', 'IN', '2020-01-01', 'L3', 'employee', 'E2001'),
            ('E1002', 'Bob Example', 'bob@example.test', '+1 555 0102', 'IN', '2019-01-01', 'L5', 'manager', NULL),
            ('E1003', 'Carol Example', 'carol@example.test', '+1 555 0103', 'IN', '2018-01-01', 'L6', 'hr', NULL),
            ('E1004', 'Dan Example', 'dan@example.test', '+1 555 0104', 'IN', '2021-01-01', 'L2', 'employee', 'E1002'),
            ('E1005', 'Eve Example', 'eve@example.test', '+1 555 0105', 'DE', '2020-01-01', 'L3', 'employee', 'E2001');
        INSERT INTO leave_balances VALUES
            ('E1001','annual',10), ('E1001','sick',5),
            ('E1002','annual',10), ('E1002','sick',5),
            ('E1003','annual',10), ('E1003','sick',5),
            ('E1004','annual',10), ('E1004','sick',5),
            ('E1005','annual',10), ('E1005','sick',5);
        """
    )
    connection.commit()
    connection.close()
    index_path = tmp_path / "index.sqlite3"
    build_index(manifest_path, corpus, index_path)
    return {
        "root": tmp_path,
        "corpus": corpus,
        "manifest": manifest_path,
        "db": db_path,
        "index": index_path,
    }


@pytest.fixture
def settings(workspace: dict[str, Path]) -> Settings:
    return Settings(
        corpus_dir=workspace["corpus"],
        manifest_path=workspace["manifest"],
        db_path=workspace["db"],
        index_path=workspace["index"],
        today=date(2026, 10, 21),
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        request_timeout=27,
        provider_timeout=12,
    )
