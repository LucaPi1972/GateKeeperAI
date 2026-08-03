import logging
import platform
from pathlib import Path


LOG_FILE_NAME = "gatekeeper.log"


def configure_logging(log_folder: Path | str, version: str) -> Path:
    folder = Path(log_folder)
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / LOG_FILE_NAME

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    logging.info("=" * 40)
    logging.info("GateKeeper AI")
    logging.info("Version %s", version)
    logging.info("Python %s", platform.python_version())
    logging.info("=" * 40)

    return log_path
