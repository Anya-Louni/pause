"""A simulated workspace: calendar, tasks, contacts and notes in one SQLite file.

Every change writes its inverse to an undo log first, so any action the agent
takes can be rolled back. Nothing here touches a real calendar or sends anything.
"""

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (id TEXT PRIMARY KEY, name TEXT, email TEXT, team TEXT);
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, title TEXT, start TEXT, end TEXT, attendees TEXT, location TEXT);
CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, title TEXT, due TEXT, done INTEGER);
CREATE TABLE IF NOT EXISTS notes (id TEXT PRIMARY KEY, text TEXT, created TEXT);
CREATE TABLE IF NOT EXISTS undo_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, summary TEXT, inverse TEXT, undone INTEGER);
"""

PREFIX = {"contacts": "c", "events": "e", "tasks": "t", "notes": "n"}


class NotFound(LookupError):
    """Raised when an id does not exist."""


class Workspace:
    """One user's simulated workspace."""

    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)

    # Helpers ---------------------------------------------------------------

    def _rows(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    def _next_id(self, table: str) -> str:
        prefix = PREFIX[table]
        ids = [int(r["id"][1:]) for r in self._rows(f"SELECT id FROM {table}")]
        return f"{prefix}{max(ids, default=0) + 1}"

    def _log(self, summary: str, inverse: list[list[Any]], now: str) -> None:
        self._db.execute(
            "INSERT INTO undo_log (at, summary, inverse, undone) VALUES (?, ?, ?, 0)",
            (now, summary, json.dumps(inverse)),
        )

    @staticmethod
    def _event(row: dict[str, Any]) -> dict[str, Any]:
        return {**row, "attendees": json.loads(row["attendees"])}

    # Reads -----------------------------------------------------------------

    def find_contacts(self, name: str) -> list[dict[str, Any]]:
        """Contacts whose name contains the text, case-insensitively."""
        return self._rows(
            "SELECT * FROM contacts WHERE lower(name) LIKE ? ORDER BY id",
            f"%{name.lower().strip()}%",
        )

    def contact(self, contact_id: str) -> dict[str, Any] | None:
        rows = self._rows("SELECT * FROM contacts WHERE id = ?", contact_id)
        return rows[0] if rows else None

    def events(self, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
        """Events that start inside [start, end), ordered by start. Dates compare as ISO text."""
        rows = self._rows("SELECT * FROM events ORDER BY start")
        return [
            self._event(r)
            for r in rows
            if (start is None or r["start"] >= start) and (end is None or r["start"] < end)
        ]

    def event(self, event_id: str) -> dict[str, Any]:
        rows = self._rows("SELECT * FROM events WHERE id = ?", event_id)
        if not rows:
            raise NotFound(f"no event with id {event_id}")
        return self._event(rows[0])

    def tasks(self, include_done: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM tasks" + ("" if include_done else " WHERE done = 0") + " ORDER BY id"
        return self._rows(sql)

    def notes(self) -> list[dict[str, Any]]:
        return self._rows("SELECT * FROM notes ORDER BY id")

    def snapshot(self) -> dict[str, Any]:
        """Everything the web app shows next to the chat."""
        return {
            "events": self.events(),
            "tasks": self.tasks(include_done=True),
            "contacts": self._rows("SELECT * FROM contacts ORDER BY id"),
            "notes": self.notes(),
            "undo": self._rows(
                "SELECT seq, at, summary FROM undo_log WHERE undone = 0 ORDER BY seq DESC LIMIT 10"
            ),
        }

    # Changes ---------------------------------------------------------------

    def create_event(
        self, title: str, start: str, end: str, attendees: list[str], location: str, now: str
    ) -> dict[str, Any]:
        with self._lock:
            unknown = [a for a in attendees if not self.contact(a)]
            if unknown:
                raise NotFound(f"unknown contact ids: {', '.join(unknown)}")
            event_id = self._next_id("events")
            self._db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, title, start, end, json.dumps(attendees), location),
            )
            self._log(
                f"Created event '{title}'", [["DELETE FROM events WHERE id = ?", event_id]], now
            )
            return self.event(event_id)

    def move_event(self, event_id: str, new_start: str, now: str) -> dict[str, Any]:
        with self._lock:
            old = self.event(event_id)
            length = datetime.fromisoformat(old["end"]) - datetime.fromisoformat(old["start"])
            new_end = (datetime.fromisoformat(new_start) + length).isoformat()
            self._db.execute(
                "UPDATE events SET start = ?, end = ? WHERE id = ?", (new_start, new_end, event_id)
            )
            inverse = [
                [
                    "UPDATE events SET start = ?, end = ? WHERE id = ?",
                    old["start"],
                    old["end"],
                    event_id,
                ]
            ]
            self._log(f"Moved event '{old['title']}'", inverse, now)
            return self.event(event_id)

    def delete_event(self, event_id: str, now: str) -> dict[str, Any]:
        with self._lock:
            old = self.event(event_id)
            self._db.execute("DELETE FROM events WHERE id = ?", (event_id,))
            inverse = [
                [
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                    old["id"], old["title"], old["start"], old["end"],
                    json.dumps(old["attendees"]), old["location"],
                ]
            ]  # fmt: skip
            self._log(f"Deleted event '{old['title']}'", inverse, now)
            return old

    def add_task(self, title: str, due: str | None, now: str) -> dict[str, Any]:
        with self._lock:
            task_id = self._next_id("tasks")
            self._db.execute("INSERT INTO tasks VALUES (?, ?, ?, 0)", (task_id, title, due))
            self._log(f"Added task '{title}'", [["DELETE FROM tasks WHERE id = ?", task_id]], now)
            return {"id": task_id, "title": title, "due": due, "done": 0}

    def complete_task(self, task_id: str, now: str) -> dict[str, Any]:
        with self._lock:
            rows = self._rows("SELECT * FROM tasks WHERE id = ?", task_id)
            if not rows:
                raise NotFound(f"no task with id {task_id}")
            self._db.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
            inverse = [["UPDATE tasks SET done = ? WHERE id = ?", rows[0]["done"], task_id]]
            self._log(f"Completed task '{rows[0]['title']}'", inverse, now)
            return {**rows[0], "done": 1}

    def save_note(self, text: str, now: str) -> dict[str, Any]:
        with self._lock:
            note_id = self._next_id("notes")
            self._db.execute("INSERT INTO notes VALUES (?, ?, ?)", (note_id, text, now))
            self._log("Saved a note", [["DELETE FROM notes WHERE id = ?", note_id]], now)
            return {"id": note_id, "text": text, "created": now}

    def undo_last(self) -> str | None:
        """Roll back the most recent change that has not been undone. Returns its summary."""
        with self._lock:
            rows = self._rows("SELECT * FROM undo_log WHERE undone = 0 ORDER BY seq DESC LIMIT 1")
            if not rows:
                return None
            for sql, *args in json.loads(rows[0]["inverse"]):
                self._db.execute(sql, args)
            self._db.execute("UPDATE undo_log SET undone = 1 WHERE seq = ?", (rows[0]["seq"],))
            return rows[0]["summary"]

    def close(self) -> None:
        self._db.close()


def monday_of(day: date) -> date:
    return day - timedelta(days=day.weekday())


def seed(workspace: Workspace, today: date) -> None:
    """Fill an empty workspace with a small, deliberately ambiguous week.

    Two people are called Sam and two meetings are called "Design review", so
    "move the design review" and "invite Sam" each have more than one reading.
    """
    if workspace.events() or workspace.tasks(include_done=True):
        return
    monday = monday_of(today)

    def at(day: int, hour: int, minute: int = 0) -> str:
        return (
            datetime.combine(monday + timedelta(days=day), datetime.min.time())
            .replace(hour=hour, minute=minute)
            .isoformat(timespec="minutes")
        )

    db = workspace._db
    db.executemany(
        "INSERT INTO contacts VALUES (?, ?, ?, ?)",
        [
            ("c1", "Sam Okafor", "sam.okafor@example.com", "Design"),
            ("c2", "Sam Rivera", "sam.rivera@example.com", "Finance"),
            ("c3", "Priya Nair", "priya.nair@example.com", "Design"),
            ("c4", "Jonas Weber", "jonas.weber@example.com", "Engineering"),
            ("c5", "Mei Tanaka", "mei.tanaka@example.com", "Engineering"),
        ],
    )
    db.executemany(
        "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
        [
            ("e1", "Design review", at(0, 10), at(0, 11), '["c1", "c3"]', "Room 2"),
            ("e2", "1:1 with Priya", at(1, 11), at(1, 11, 30), '["c3"]', ""),
            ("e3", "Dentist", at(2, 16), at(2, 16, 30), "[]", "Mill Road Dental"),
            ("e4", "Design review", at(3, 14), at(3, 15), '["c1", "c4"]', "Room 2"),
            ("e5", "Budget sync", at(4, 9, 30), at(4, 10), '["c2"]', ""),
            (
                "e6",
                "Team lunch",
                at(4, 12, 30),
                at(4, 13, 30),
                '["c1", "c2", "c3", "c4", "c5"]',
                "Canteen",
            ),
        ],
    )
    friday = (monday + timedelta(days=4)).isoformat()
    next_monday = (monday + timedelta(days=7)).isoformat()
    db.executemany(
        "INSERT INTO tasks VALUES (?, ?, ?, ?)",
        [
            ("t1", "Send budget draft to Sam", friday, 0),
            ("t2", "Book flights for the offsite", next_monday, 0),
            ("t3", "Review Priya's proposal", None, 0),
        ],
    )
    db.execute(
        "INSERT INTO notes VALUES (?, ?, ?)",
        ("n1", "Offsite is in Lisbon, two weeks from now.", monday.isoformat()),
    )
