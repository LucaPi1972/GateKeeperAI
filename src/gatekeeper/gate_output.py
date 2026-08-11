"""GPIO gate-output simulation and progressive field-test LED status."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

LOGGER = logging.getLogger("gatekeeper")
GPIO_PIN = 17
PULSE_SECONDS = 1.0


class GateOutput:
    """Drive Raspberry Pi GPIO17 as a gate simulation and status indicator."""

    def __init__(self, pin: int = GPIO_PIN, pulse_seconds: float = PULSE_SECONDS) -> None:
        self.pin = pin
        self.pulse_seconds = pulse_seconds
        self.available = False
        self.active = False
        self.status = "IDLE"
        self.last_triggered_at = 0.0
        self.last_reason = ""
        self.error = ""
        self._lock = threading.RLock()
        self._gpio: Any | None = None
        self._blink_stop = threading.Event()
        self._blink_thread: threading.Thread | None = None
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on Raspberry Pi runtime
            self.error = str(exc)
            LOGGER.warning("GPIO gate output unavailable: %s", exc)

    def _write(self, value: bool) -> None:
        if self._gpio is not None:
            self._gpio.output(self.pin, self._gpio.HIGH if value else self._gpio.LOW)

    def set_status(self, status: str) -> None:
        """Set the LED indication. Non-confirmed activity uses a progressive blink."""
        status = str(status or "IDLE").upper()
        with self._lock:
            if not self.available or self._gpio is None or self.active:
                self.status = status
                return
            self.status = status
            self._blink_stop.set()
            self._write(False)
            if status in {"IDLE", "NO_PLATE", "READY"}:
                return
            if status == "CONFIRMED":
                self._write(True)
                return
            self._blink_stop = threading.Event()
            stop = self._blink_stop
            self._blink_thread = threading.Thread(
                target=self._progressive_blink,
                args=(stop,),
                daemon=True,
                name="gate-led-status",
            )
            self._blink_thread.start()

    def _progressive_blink(self, stop: threading.Event) -> None:
        """Blink faster as the recognition process approaches confirmation."""
        started = time.monotonic()
        while not stop.is_set():
            elapsed = time.monotonic() - started
            # Recognition normally completes in <2 s: 300 -> 60 ms half-period.
            half_period = max(0.06, min(0.30, 0.30 - (elapsed / 1.8) * 0.24))
            self._write(True)
            if stop.wait(half_period):
                break
            self._write(False)
            if stop.wait(half_period):
                break

    def trigger(self, reason: str = "OCR_CONFIRMED") -> bool:
        with self._lock:
            if not self.available or self._gpio is None or self.active:
                return False
            self._blink_stop.set()
            self.active = True
            self.status = "CONFIRMED"
            self.last_triggered_at = time.time()
            self.last_reason = reason
            gpio = self._gpio
            gpio.output(self.pin, gpio.HIGH)
            threading.Thread(target=self._pulse_off, args=(gpio,), daemon=True, name="gate-gpio-pulse").start()
            LOGGER.info("GATE GPIO%d HIGH for %.1fs reason=%s", self.pin, self.pulse_seconds, reason)
            return True

    def _pulse_off(self, gpio: Any) -> None:
        time.sleep(self.pulse_seconds)
        with self._lock:
            gpio.output(self.pin, gpio.LOW)
            self.active = False
            self.status = "IDLE"
            LOGGER.info("GATE GPIO%d LOW", self.pin)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.available,
                "pin": self.pin,
                "active": self.active,
                "status": self.status,
                "pulse_seconds": self.pulse_seconds,
                "last_triggered_at": self.last_triggered_at,
                "last_reason": self.last_reason,
                "error": self.error,
            }

    def cleanup(self) -> None:
        with self._lock:
            self._blink_stop.set()
            if self._gpio is not None:
                try:
                    self._gpio.output(self.pin, self._gpio.LOW)
                    self._gpio.cleanup(self.pin)
                except Exception:
                    pass
            self.active = False
            self.status = "IDLE"


def start_gate_watch(state: Any, output: GateOutput, refresh_ocr: Any) -> threading.Thread:
    """Watch OCR state, show progressive LED status, and pulse once on confirmation."""
    def worker() -> None:
        was_confirmed = False
        last_status = ""
        while True:
            try:
                payload = refresh_ocr(state)
                status = str(payload.get("status") or "IDLE").upper()
                confirmed = status == "CONFIRMED" and bool(payload.get("stable"))
                if confirmed and not was_confirmed:
                    output.trigger("OCR_CONFIRMED")
                elif status != last_status:
                    output.set_status(status)
                was_confirmed = confirmed
                last_status = status
            except Exception:
                LOGGER.exception("Gate GPIO OCR watcher failed")
                output.set_status("ERROR")
                was_confirmed = False
            time.sleep(0.10)

    thread = threading.Thread(target=worker, daemon=True, name="gate-gpio-watch")
    thread.start()
    return thread
