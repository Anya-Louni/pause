from pause.dates import resolve_date

TZ = "Europe/London"
NOW = "2026-10-06T09:00:00"  # a Tuesday


def r(expression: str, now: str = NOW):
    return resolve_date(expression, TZ, now)


def test_plain_words() -> None:
    assert r("today").iso == "2026-10-06"
    assert r("tomorrow").iso == "2026-10-07"
    assert r("day after tomorrow").iso == "2026-10-08"
    assert r("in 3 days").iso == "2026-10-09"
    assert r("in 2 weeks").iso == "2026-10-20"


def test_weekday_means_the_coming_one() -> None:
    assert r("Friday").iso == "2026-10-09"
    assert r("on monday").iso == "2026-10-12"


def test_next_weekday_is_ambiguous_early_in_the_week() -> None:
    result = r("next Friday")
    assert result.ambiguous and result.iso is None
    assert result.alternatives == ["2026-10-09", "2026-10-16"]
    assert "this week" in result.note


def test_next_weekday_is_clear_when_only_one_reading_exists() -> None:
    saturday = "2026-10-10T09:00:00"
    assert r("next Friday", saturday).iso == "2026-10-16"


def test_same_weekday_today_is_ambiguous() -> None:
    result = r("Tuesday")
    assert result.alternatives == ["2026-10-06", "2026-10-13"]


def test_weekday_and_number_that_disagree() -> None:
    result = r("Friday the 10th")  # the 10th is a Saturday
    assert result.ambiguous
    assert result.alternatives == ["2026-10-09", "2026-10-10"]
    assert result.note == "the next 10th is not a Friday"


def test_weekday_and_number_that_agree() -> None:
    assert r("Friday the 9th").iso == "2026-10-09"
    assert r("the 3rd").iso == "2026-11-03"  # the 3rd of this month has passed


def test_month_names() -> None:
    assert r("10 Oct").iso == "2026-10-10"
    assert r("October 10th").iso == "2026-10-10"
    assert r("3 march").iso == "2027-03-03"
    assert r("1 jan 2028").iso == "2028-01-01"
    assert r("31 feb").iso is None


def test_times() -> None:
    assert r("tomorrow at 3pm").iso == "2026-10-07T15:00:00+01:00"
    assert r("Friday 9:30 am").iso == "2026-10-09T09:30:00+01:00"
    assert r("today at noon").iso == "2026-10-06T12:00:00+01:00"
    assert r("15:45").iso == "2026-10-06T15:45:00+01:00"


def test_bare_hour_has_two_readings() -> None:
    result = r("tomorrow at 3")
    assert result.alternatives == ["2026-10-07T03:00:00+01:00", "2026-10-07T15:00:00+01:00"]
    assert "no am or pm" in result.note


def test_winter_offset() -> None:
    assert r("10 nov at 3pm").iso == "2026-11-10T15:00:00+00:00"


def test_iso_passes_through() -> None:
    assert r("2026-12-01").iso == "2026-12-01"
    assert r("2026-12-01T10:30").iso == "2026-12-01T10:30:00+00:00"


def test_unresolvable() -> None:
    week = r("next week")
    assert week.iso is None and not week.ambiguous and "names no day" in week.note
    nothing = r("sometime soon")
    assert nothing.iso is None and nothing.note == "no date found in the phrase"
