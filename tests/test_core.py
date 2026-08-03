import logging
import sqlite3

from gatekeeper.core.camera import CameraManager
from gatekeeper.core.config import validate_storage_paths
from gatekeeper.core.database import initialize_database
from gatekeeper.core.logging_setup import configure_logging


def test_validate_storage_paths_creates_required_paths(tmp_path):
    config = {
        "database": {"path": str(tmp_path / "data" / "gatekeeper.db")},
        "storage": {
            "log_folder": str(tmp_path / "logs"),
            "image_folder": str(tmp_path / "images"),
        },
    }

    validate_storage_paths(config)

    assert (tmp_path / "data").is_dir()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "images").is_dir()


def test_initialize_database_creates_tables_and_indexes(tmp_path):
    database_path = tmp_path / "data" / "gatekeeper.db"

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'index')"
            )
        }

    assert {"plates", "events", "settings"}.issubset(names)
    assert {"idx_plate_number", "idx_event_timestamp"}.issubset(names)


def test_camera_health_check_returns_false_without_exception(tmp_path):
    manager = CameraManager(str(tmp_path / "missing-camera"))

    assert manager.health_check() is False


def test_configure_logging_writes_startup_banner(tmp_path):
    log_path = configure_logging(tmp_path / "logs", "0.3.0A")
    logging.shutdown()

    content = log_path.read_text(encoding="utf-8")

    assert "========================================" in content
    assert "GateKeeper AI" in content
    assert "Version 0.3.0A" in content
    assert "Python " in content
