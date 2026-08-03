"""GateKeeper AI package."""

from pathlib import Path


def get_version() -> str:
    """Return the application version from the repository VERSION file."""
    version_file = Path(__file__).resolve().parents[2] / "VERSION"
    return version_file.read_text(encoding="utf-8").strip()


__all__ = ["get_version"]
