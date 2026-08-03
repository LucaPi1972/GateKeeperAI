# Project Rules

Future tasks must always follow this file. Keep the current architecture focused on the GateKeeper AI core unless a task explicitly changes the scope.

## Architecture
- Use a small Python package under `src/gatekeeper`.
- Keep core responsibilities separated into focused modules: configuration, logging, database initialization, camera access, and application startup.
- Avoid adding integrations that are outside the current core scope. Do not implement OCR, GPIO, Home Assistant, or motion detection unless explicitly requested.
- Prefer standard-library functionality where practical to keep the project lightweight for Raspberry Pi deployments.

## Folder structure
- `config/`: YAML configuration files.
- `src/gatekeeper/`: Python application package.
- `src/gatekeeper/core/`: reusable core services such as config validation, logging, database, and camera management.
- `tests/`: automated tests.
- `logs/`: runtime log files, created automatically and ignored by Git.
- `images/`: runtime image storage, created automatically and ignored by Git.
- `data/`: runtime database storage, created automatically and ignored by Git.

## Naming conventions
- Python modules and files use `snake_case`.
- Python classes use `PascalCase`.
- Python functions, methods, variables, and configuration keys use `snake_case`.
- Constants use `UPPER_SNAKE_CASE`.
- Database tables and indexes use lowercase names with underscores when needed.

## Python coding style
- Target Python 3.11 or newer.
- Keep functions small, typed where useful, and focused on one responsibility.
- Never put `try`/`except` blocks around imports.
- Prefer `pathlib.Path` for filesystem paths.
- Avoid broad exception handling unless the caller explicitly requires non-throwing behavior.

## Logging rules
- Application logs must be written under the configured log folder.
- Startup must write the GateKeeper AI banner to `logs/gatekeeper.log`.
- Use Python's `logging` module for application logs.
- Do not log secrets, credentials, or full sensitive configuration payloads.

## Database rules
- Use SQLite for local core persistence.
- Database initialization must be idempotent.
- Required core tables are `plates`, `events`, and `settings`.
- Required indexes are `idx_plate_number` and `idx_event_timestamp`.
- Keep schema changes explicit and versioned through application releases.

## Configuration rules
- Store default configuration in `config/config.yaml`.
- Validate runtime paths before use.
- Automatically create the database parent folder, log folder, and image folder when missing.
- Keep environment-specific secrets out of version-controlled configuration.

## Versioning rules
- Keep the application version synchronized across `VERSION`, `config/config.yaml`, package metadata, and package `__version__`.
- Use the existing alpha suffix format, for example `0.3.0A` for user-facing version strings.
- Update `CHANGELOG.md` for user-visible changes.

## Testing strategy
- Add or update tests for configuration validation, database initialization, and non-throwing hardware abstractions.
- Prefer tests that use temporary directories and do not require Raspberry Pi hardware.
- Run the fastest relevant automated checks before committing.

## Raspberry Pi optimization guidelines
- Keep startup work minimal and deterministic.
- Avoid unnecessary background threads or heavy dependencies in core modules.
- Prefer lazy hardware checks so the app can start without connected peripherals during development.
- Keep file and database operations simple and local-first.
- Avoid CPU-intensive polling loops in core services.
