from pathlib import Path


class CameraManager:
    def __init__(self, device: str = "/dev/video0") -> None:
        self.device = device

    def health_check(self) -> bool:
        try:
            return Path(self.device).exists()
        except OSError:
            return False
