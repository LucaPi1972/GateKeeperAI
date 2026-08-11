"""GPIO gate-output simulation for field testing."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

LOGGER = logging.getLogger("gatekeeper")
GPIO_PIN = 17
PULSE_SECONDS = 1.0


class GateOutput:
    """Drive a Raspberry Pi GPIO as a safe gate-opening simulation output."""

    def __init__(self, pin: int = GPIO_PIN, pulse_seconds: float = PULSE_SECONDS) -> None:
        self.pin = pin
        self.pulse_seconds = pulse_seconds
        self.available = False
        self.active = False
        self.last_triggered_at = 0.0
        self.last_reason = ""
        self.error = ""
        self._lock = threading.RLock()
        self._gpio: Any | None = None
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)
            self._gpio = GPIO
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on Raspberry Pi runtime
            self.error = str(exc)
            LOGGER.warning("GPIO gate output unavailable: %s", exc)

    def trigger(self, reason: str = "OCR_CONFIRMED") -> bool:
        with self._lock:
            if not self.available or self._gpio is None or self.active:
                return False
            self.active = True
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
            LOGGER.info("GATE GPIO%d LOW", self.pin)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "available": self.available,
                "pin": self.pin,
                "active": self.active,
                "pulse_seconds": self.pulse_seconds,
                "last_triggered_at": self.last_triggered_at,
                "last_reason": self.last_reason,
                "error": self.error,
            }

    def cleanup(self) -> None:
        with self._lock:
            if self._gpio is not None:
                try:
                    self._gpio.output(self.pin, self._gpio.LOW)
                    self._gpio.cleanup(self.pin)
                except Exception:
                    pass
            self.active = False


def start_gate_watch(state: Any, output: GateOutput, refresh_ocr: Any) -> threading.Thread:
    """Watch OCR confirmation and emit one GPIO pulse per confirmed plate lock."""
    def worker() -> None:
        was_confirmed = False
        while True:
            try:
                payload = refresh_ocr(state)
                confirmed = payload.get("status") == "CONFIRMED" and bool(payload.get("stable"))
                if confirmed and not was_confirmed:
                    output.trigger("OCR_CONFIRMED")
                was_confirmed = confirmed
            except Exception:
                LOGGER.exception("Gate GPIO OCR watcher failed")
                was_confirmed = False
            time.sleep(0.25)

    thread = threading.Thread(target=worker, daemon=True, name="gate-gpio-watch")
    thread.start()
    return thread
