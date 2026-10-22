from datetime import date

from peopleops.documents import build_index
from peopleops.retrieval import (
    applicable_holidays,
    eligible_sources,
    precedence,
    retrieve,
    reference_date_for_query,
)


def test_document_eligibility_and_access_filtering(workspace):
    index = workspace["index"]
    employee_sources = eligible_sources(index, "employee", "IN", date(2026, 10, 21))
    ids = {source.doc_id for source in employee_sources}
    assert "POL-GLOBAL-V2" in ids
    assert "POL-GLOBAL-V3" not in ids
    assert "ADD-DE" not in ids
    assert "MGR-ONLY" not in ids
    assert "HR-ONLY" not in ids

    manager_ids = {
        source.doc_id
        for source in eligible_sources(index, "manager", "IN", date(2026, 10, 21))
    }
    assert "MGR-ONLY" in manager_ids
    assert "HR-ONLY" not in manager_ids
    hr_ids = {
        source.doc_id
        for source in eligible_sources(index, "hr", "IN", date(2026, 10, 21))
    }
    assert {"MGR-ONLY", "HR-ONLY"} <= hr_ids


def test_authority_scope_and_recency_precedence(workspace):
    sources = eligible_sources(workspace["index"], "employee", "IN", date(2026, 12, 31))
    by_id = {source.doc_id: source for source in sources}
    assert precedence(by_id["POL-GLOBAL-V2"], "IN") > precedence(
        next(source for source in sources if source.doc_id == "FAQ-GLOBAL"), "IN"
    )
    regional = next(
        source for source in eligible_sources(
            workspace["index"], "employee", "DE", date(2026, 12, 31)
        )
        if source.doc_id == "ADD-DE"
    )
    global_policy = by_id["POL-GLOBAL-V2"]
    assert precedence(regional, "DE") > precedence(global_policy, "DE")
    assert reference_date_for_query("What was the policy in 2025?", date(2026, 10, 21)) == date(2025, 12, 31)


def test_retrieval_never_returns_restricted_chunks(workspace):
    chunks = retrieve(
        workspace["index"],
        "manager handbook confidentiality RestrictedMarker",
        "employee",
        "IN",
        date(2026, 10, 21),
    )
    assert not any("RestrictedMarker" in chunk.content for chunk in chunks)


def test_calendar_extraction_and_unresolved_tie(workspace):
    days, error = applicable_holidays(workspace["index"], "employee", "IN")
    assert error is None
    assert days == {"2026-10-23"}

    manifest = __import__("json").loads(workspace["manifest"].read_text(encoding="utf-8"))
    (workspace["corpus"] / "india2.html").write_text(
        "<table><tr><td>2026-10-24</td><td>Another Holiday</td></tr></table>",
        encoding="utf-8",
    )
    manifest.append(
        {
            **manifest[-1],
            "doc_id": "CAL-IN-CONFLICT",
            "file": "india2.html",
        }
    )
    workspace["manifest"].write_text(__import__("json").dumps(manifest), encoding="utf-8")
    build_index(workspace["manifest"], workspace["corpus"], workspace["index"])
    days, error = applicable_holidays(workspace["index"], "employee", "IN")
    assert days is None
    assert "conflict" in error.lower()


def test_modified_corpus_rebuilds_without_stale_answers(workspace):
    target = workspace["corpus"] / "policy_v2.md"
    target.write_text(
        "# Leave Policy\n## Annual leave\nThe annual leave allowance is 27 days.",
        encoding="utf-8",
    )
    build_index(workspace["manifest"], workspace["corpus"], workspace["index"])
    chunks = retrieve(
        workspace["index"],
        "annual leave allowance",
        "employee",
        "IN",
        date(2026, 10, 21),
    )
    assert any("27 days" in chunk.content for chunk in chunks)
