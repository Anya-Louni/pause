"""Turn a date phrase into ISO dates, deterministically, and say when it is ambiguous.

The agent never does date arithmetic itself. It calls this, and when the phrase
has more than one honest reading ("next Friday" on a Tuesday, "Friday the 10th"
when the 10th is a Saturday, "at 3" with no am or pm) every reading comes back.
"""

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]  # fmt: skip
_DAY = "|".join(f"{d}|{d[:3]}" for d in WEEKDAYS)
_MONTH = "|".join(f"{m}|{m[:3]}" for m in MONTHS)


@dataclass
class Resolved:
    """Result of resolving one phrase."""

    expression: str
    iso: str | None
    ambiguous: bool
    alternatives: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _weekday(word: str) -> int:
    return next(i for i, d in enumerate(WEEKDAYS) if d.startswith(word[:3]))


def _month(word: str) -> int:
    return next(i for i, m in enumerate(MONTHS) if m.startswith(word[:3])) + 1


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _next_weekday(today: date, target: int, *, allow_today: bool = False) -> date:
    ahead = (target - today.weekday()) % 7
    if ahead == 0 and not allow_today:
        ahead = 7
    return today + timedelta(days=ahead)


def _next_day_of_month(today: date, day: int) -> date | None:
    year, month = today.year, today.month
    for _ in range(13):
        try:
            candidate = date(year, month, day)
        except ValueError:
            candidate = None
        if candidate and candidate >= today:
            return candidate
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return None


def _parse_time(text: str) -> tuple[list[time], str, str]:
    """Pull a clock time out of the phrase. Returns (readings, rest of phrase, note)."""
    for word, value in (("noon", time(12, 0)), ("midday", time(12, 0)), ("midnight", time(0, 0))):
        if re.search(rf"\b{word}\b", text):
            return [value], re.sub(rf"\b(at\s+)?{word}\b", " ", text), ""
    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if m:
        hour, minute = int(m.group(1)) % 12, int(m.group(2) or 0)
        if m.group(3) == "pm":
            hour += 12
        return [time(hour, minute)], text.replace(m.group(0), " "), ""
    m = re.search(r"\b(?:at\s+)?(\d{1,2}):(\d{2})\b", text)
    if m and int(m.group(1)) < 24:
        return [time(int(m.group(1)), int(m.group(2)))], text.replace(m.group(0), " "), ""
    m = re.search(r"\bat\s+(\d{1,2})\b", text)
    if m and 1 <= int(m.group(1)) <= 12:
        hour = int(m.group(1))
        readings = [time(hour % 12, 0), time(hour % 12 + 12, 0)]
        return readings, text.replace(m.group(0), " "), f"'{m.group(0).strip()}' has no am or pm"
    return [], text, ""


def _parse_dates(text: str, today: date) -> tuple[list[date], str]:
    """Return every honest reading of the date part, and a note when there is more than one."""
    text = " ".join(text.split())
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if m:
        return [date(int(m.group(1)), int(m.group(2)), int(m.group(3)))], ""
    if re.search(r"\bday after tomorrow\b", text):
        return [today + timedelta(days=2)], ""
    for word, offset in (("today", 0), ("tonight", 0), ("tomorrow", 1), ("yesterday", -1)):
        if re.search(rf"\b{word}\b", text):
            return [today + timedelta(days=offset)], ""
    m = re.search(r"\bin (\d+) (day|week)s?\b", text)
    if m:
        days = int(m.group(1)) * (7 if m.group(2) == "week" else 1)
        return [today + timedelta(days=days)], ""

    m = (
        re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?(?: of)? ({_MONTH})\b(?: (\d{{4}}))?", text)
        or None
    )
    month_first = re.search(rf"\b({_MONTH}) (\d{{1,2}})(?:st|nd|rd|th)?\b(?:,? (\d{{4}}))?", text)
    if m or month_first:
        if m:
            day, month, year = int(m.group(1)), _month(m.group(2)), m.group(3)
        elif month_first:
            day, month = int(month_first.group(2)), _month(month_first.group(1))
            year = month_first.group(3)
        try:
            if year:
                return [date(int(year), month, day)], ""
            candidate = date(today.year, month, day)
            return [candidate if candidate >= today else date(today.year + 1, month, day)], ""
        except ValueError:
            return [], "that day does not exist in that month"

    weekday = re.search(rf"\b(next |this |on )?({_DAY})\b", text)
    ordinal = re.search(r"\b(?:the )?(\d{1,2})(?:st|nd|rd|th)\b", text)
    if weekday and ordinal:
        target = _weekday(weekday.group(2))
        by_number = _next_day_of_month(today, int(ordinal.group(1)))
        by_name = _next_weekday(today, target, allow_today=True)
        if by_number and by_number.weekday() == target:
            return [by_number], ""
        readings = [d for d in (by_name, by_number) if d]
        name = WEEKDAYS[target].title()
        return readings, f"the next {_ordinal(int(ordinal.group(1)))} is not a {name}"
    if ordinal:
        by_number = _next_day_of_month(today, int(ordinal.group(1)))
        return ([by_number], "") if by_number else ([], "no such day of the month")
    if weekday:
        target = _weekday(weekday.group(2))
        prefix = (weekday.group(1) or "").strip()
        if prefix == "next":
            soonest = _next_weekday(today, target)
            following = soonest + timedelta(days=7)
            # From Monday to the day before, "next Friday" can mean this week's or next week's.
            if soonest.isocalendar().week == today.isocalendar().week:
                return [soonest, following], "'next' can mean this week's or the week after"
            return [soonest], ""
        if today.weekday() == target:
            return [today, today + timedelta(days=7)], "today is that weekday"
        return [_next_weekday(today, target)], ""
    if re.search(r"\bnext week\b", text):
        monday = _next_weekday(today, 0)
        sunday = monday + timedelta(days=6)
        return [], f"'next week' names no day. It runs {monday.isoformat()} to {sunday.isoformat()}"
    return [], "no date found in the phrase"


def resolve_date(expression: str, timezone: str, now: str) -> Resolved:
    """Resolve a date phrase against a fixed 'now'.

    Args:
        expression: The phrase, for example "next Friday at 3pm" or "the 10th".
        timezone: IANA zone name used for the offset, for example "Europe/London".
        now: The current moment as an ISO datetime. Passed in, so results are repeatable.
    """
    zone = ZoneInfo(timezone)
    current = datetime.fromisoformat(now)
    current = current.replace(tzinfo=zone) if current.tzinfo is None else current.astimezone(zone)
    text = expression.lower().strip()
    try:  # a full ISO datetime needs no interpretation
        exact = datetime.fromisoformat(expression.strip())
        if "t" in text or " " in text:
            exact = exact.replace(tzinfo=zone) if exact.tzinfo is None else exact
            return Resolved(expression, exact.isoformat(), False)
    except ValueError:
        pass
    times, rest, time_note = _parse_time(text)
    dates, date_note = _parse_dates(rest, current.date())
    if not dates and times and not date_note.startswith("'next week'"):
        dates, date_note = [current.date()], "no date given, assumed today"

    readings: list[str] = []
    for day in dates:
        if times:
            readings += [datetime.combine(day, t, tzinfo=zone).isoformat() for t in times]
        else:
            readings.append(day.isoformat())
    note = "; ".join(n for n in (date_note, time_note) if n)
    if not readings:
        return Resolved(expression, None, False, [], note or "could not resolve")
    if len(readings) == 1:
        return Resolved(expression, readings[0], False, [], note)
    return Resolved(expression, None, True, readings, note or "more than one reading")
