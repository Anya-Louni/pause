import pytest
from conftest import NOW

from pause.workspace import NotFound, Workspace


def test_seed_is_deliberately_ambiguous(workspace: Workspace) -> None:
    assert [c["name"] for c in workspace.find_contacts("sam")] == ["Sam Okafor", "Sam Rivera"]
    reviews = [e for e in workspace.events() if e["title"] == "Design review"]
    assert [e["start"] for e in reviews] == ["2026-10-05T10:00", "2026-10-08T14:00"]
    assert len(workspace.tasks()) == 3 and len(workspace.notes()) == 1


def test_events_filter_by_start(workspace: Workspace) -> None:
    friday = workspace.events("2026-10-09", "2026-10-10")
    assert [e["title"] for e in friday] == ["Budget sync", "Team lunch"]


def test_move_keeps_length_and_can_be_undone(workspace: Workspace) -> None:
    moved = workspace.move_event("e3", "2026-10-09T15:00", NOW)
    assert (moved["start"], moved["end"]) == ("2026-10-09T15:00", "2026-10-09T15:30:00")
    assert workspace.undo_last() == "Moved event 'Dentist'"
    assert workspace.event("e3")["start"] == "2026-10-07T16:00"


def test_delete_can_be_undone_with_attendees_intact(workspace: Workspace) -> None:
    workspace.delete_event("e1", NOW)
    with pytest.raises(NotFound):
        workspace.event("e1")
    workspace.undo_last()
    assert workspace.event("e1")["attendees"] == ["c1", "c3"]


def test_create_checks_contacts_and_numbers_ids(workspace: Workspace) -> None:
    with pytest.raises(NotFound, match="c99"):
        workspace.create_event("x", "2026-10-09T10:00", "2026-10-09T11:00", ["c99"], "", NOW)
    event = workspace.create_event("Sync", "2026-10-09T10:00", "2026-10-09T11:00", ["c2"], "", NOW)
    assert event["id"] == "e7"


def test_tasks_and_notes_undo_in_reverse_order(workspace: Workspace) -> None:
    workspace.add_task("Call the bank", "2026-10-09", NOW)
    workspace.complete_task("t1", NOW)
    workspace.save_note("Prefers morning meetings.", NOW)
    assert workspace.undo_last() == "Saved a note"
    assert workspace.undo_last() == "Completed task 'Send budget draft to Sam'"
    assert workspace.undo_last() == "Added task 'Call the bank'"
    assert workspace.undo_last() is None
    assert len(workspace.tasks()) == 3 and len(workspace.notes()) == 1


def test_snapshot_lists_pending_undo_entries(workspace: Workspace) -> None:
    workspace.complete_task("t3", NOW)
    snap = workspace.snapshot()
    assert snap["undo"][0]["summary"] == "Completed task 'Review Priya's proposal'"
    assert {"events", "tasks", "contacts", "notes", "undo"} == set(snap)
