import time
from typing import Any, Optional


class TaskTracker:
    """Global singleton to track the current inference task progress."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.filename: Optional[str] = None
        self.file_size_mb: float = 0
        self.stage: str = "idle"
        self.stage_label: str = "空闲"
        self.started_at: float = 0
        self.stage_started_at: float = 0
        self.progress_pct: int = 0
        self.error: Optional[str] = None
        self.stages_done: list[str] = []
        self._result: Optional[Any] = None

    def start(self, filename: str, file_size_mb: float):
        self.reset()
        self.active = True
        self.filename = filename
        self.file_size_mb = file_size_mb
        self.started_at = time.time()
        self.set_stage("upload_done", "文件接收完成", 5)

    def set_stage(self, stage: str, label: str, pct: int):
        if self.stage != "idle" and self.stage != stage:
            self.stages_done.append(self.stage)
        self.stage = stage
        self.stage_label = label
        self.progress_pct = pct
        self.stage_started_at = time.time()
        if not self.started_at:
            self.started_at = self.stage_started_at
            self.active = True

    def finish(self, result: Any = None):
        self._result = result
        self.set_stage("done", "处理完成", 100)
        self.active = False

    def fail(self, error: str):
        self.error = error
        self.stage_label = f"失败: {error[:100]}"
        self.active = False

    @property
    def has_result(self) -> bool:
        return self._result is not None

    def take_result(self) -> Optional[Any]:
        """Return the stored result and clear it."""
        r = self._result
        self._result = None
        return r

    def to_dict(self) -> dict:
        elapsed = time.time() - self.started_at if self.started_at else 0
        stage_elapsed = time.time() - self.stage_started_at if self.stage_started_at else 0
        return {
            "active": self.active,
            "filename": self.filename,
            "file_size_mb": round(self.file_size_mb, 1),
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress_pct": self.progress_pct,
            "elapsed_seconds": round(elapsed, 1),
            "stage_elapsed_seconds": round(stage_elapsed, 1),
            "stages_done": self.stages_done,
            "error": self.error,
            "has_result": self.has_result,
        }


tracker = TaskTracker()
