import sqlite3
from pathlib import Path


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS plates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_number TEXT NOT NULL UNIQUE,
        label TEXT,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate_id INTEGER,
        plate_number TEXT,
        event_type TEXT NOT NULL,
        image_path TEXT,
        timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (plate_id) REFERENCES plates(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plate_number ON plates(plate_number)",
    "CREATE INDEX IF NOT EXISTS idx_event_timestamp ON events(timestamp)",
)


def initialize_database(database_path: Path | str) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()
