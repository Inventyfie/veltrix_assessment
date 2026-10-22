from __future__ import annotations

import re
from datetime import date, timedelta

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
DATE_RE = re.compile(
    rf"\b(20\d{{2}}-\d{{1,2}}-\d{{1,2}}|"
    rf"(?:{MONTHS})\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+20\d{{2}})?|"
    rf"\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS})\.?(?:,?\s+20\d{{2}})?)\b",
    re.I,
)


def _month_number(name: str) -> int:
    from datetime import datetime

    return datetime.strptime(name[:3].title(), "%b").month


def parse_explicit_date(text: str, default_year: int = 2026) -> date | None:
    value = text.strip().replace(",", "").replace(".", "")
    try:
        if re.fullmatch(r"20\d{2}-\d{1,2}-\d{1,2}", value):
            return date.fromisoformat(value)
        month_first = re.fullmatch(
            rf"({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:\s+(20\d{{2}}))?",
            value,
            re.I,
        )
        day_first = re.fullmatch(
            rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})(?:\s+(20\d{{2}}))?",
            value,
            re.I,
        )
        if month_first:
            month, day, year = month_first.groups()
            return date(int(year or default_year), _month_number(month), int(day))
        if day_first:
            day, month, year = day_first.groups()
            return date(int(year or default_year), _month_number(month), int(day))
    except ValueError:
        return None
    return None


def dates_in_message(message: str, today: date) -> list[date]:
    lowered = message.lower()
    if "next week" in lowered:
        next_monday = today + timedelta(days=7 - today.weekday())
        return [next_monday, next_monday + timedelta(days=4)]
    if "next friday" in lowered:
        offset = (4 - today.weekday()) % 7 or 7
        return [today + timedelta(days=offset)]
    relative: list[date] = []
    if re.search(r"\btoday\b", lowered):
        relative.append(today)
    if re.search(r"\btomorrow\b", lowered):
        relative.append(today + timedelta(days=1))
    if relative:
        return relative
    found: list[tuple[int, date]] = []
    for match in DATE_RE.finditer(message):
        parsed = parse_explicit_date(match.group(0))
        if parsed is not None:
            found.append((match.start(), parsed))
    return [value for _, value in sorted(found)]
