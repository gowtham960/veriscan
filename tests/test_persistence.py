"""Tests for utils/persistence.py — uses a temp DB per test so tests don't
interfere with each other or with a real veriscan.db."""
import pytest


@pytest.fixture
def persistence_module(tmp_path, monkeypatch):
    """Reload the persistence module with DB_PATH pointed at a temp file,
    so each test gets a fresh, isolated database."""
    import utils.persistence as persistence
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(persistence, "DB_PATH", db_path)
    persistence.init_db()
    return persistence


def test_init_db_creates_tables(persistence_module):
    assert persistence_module.get_pending_reviews() == []


def test_persist_run_writes_audit_entries(persistence_module):
    run_id = persistence_module.new_run_id()
    entries = [
        {"node": "ingest", "timestamp": "2026-01-01T00:00:00Z", "input_hash": "abc123",
         "output_summary": "extraction complete", "model_used": "stub", "tokens_used": 0},
        {"node": "route", "timestamp": "2026-01-01T00:00:01Z", "input_hash": "",
         "output_summary": "fact=True", "model_used": "stub", "tokens_used": 0},
    ]
    persistence_module.persist_run(run_id, entries)
    trail = persistence_module.get_audit_trail(run_id)
    assert len(trail) == 2
    assert trail[0]["node"] == "ingest"
    assert trail[1]["node"] == "route"


def test_enqueue_and_get_pending_reviews(persistence_module):
    run_id = persistence_module.new_run_id()
    persistence_module.enqueue_for_review(
        run_id, "some claim", "Unverified", 0.2, 0.5, "low confidence"
    )
    pending = persistence_module.get_pending_reviews()
    assert len(pending) == 1
    assert pending[0]["run_id"] == run_id
    assert pending[0]["status"] == "pending"


def test_resolve_review_removes_from_pending(persistence_module):
    run_id = persistence_module.new_run_id()
    persistence_module.enqueue_for_review(run_id, "claim", "Unverified", 0.2, 0.5, "low confidence")

    assert len(persistence_module.get_pending_reviews()) == 1
    success = persistence_module.resolve_review(run_id, "approved", "looks fine")
    assert success is True
    assert len(persistence_module.get_pending_reviews()) == 0


def test_resolve_nonexistent_review_returns_false(persistence_module):
    success = persistence_module.resolve_review("nonexistent-id", "approved")
    assert success is False
