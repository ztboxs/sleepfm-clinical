import torch
import pandas as pd
from fastapi import APIRouter

from api.config import LABEL_MAPPING_PATH
from api.models_manager import manager
from api.task_status import tracker
from api import schemas


router = APIRouter(prefix="/api/v1", tags=["health"])


def _read_cgroup_bytes(path: str) -> int | None:
    try:
        with open(path) as f:
            val = f.read().strip()
            if val == "max":
                return None
            v = int(val)
            return v if v < 2**62 else None
    except Exception:
        return None


def _read_ram_info() -> dict:
    """Read container RAM via cgroup, fallback to /proc/meminfo."""
    try:
        cg_limit = (
            _read_cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
            or _read_cgroup_bytes("/sys/fs/cgroup/memory.max")
        )
        cg_usage = (
            _read_cgroup_bytes("/sys/fs/cgroup/memory/memory.usage_in_bytes")
            or _read_cgroup_bytes("/sys/fs/cgroup/memory.current")
        )

        if cg_limit and cg_usage:
            total = cg_limit / 1024 / 1024
            used = cg_usage / 1024 / 1024
            return {
                "total_mb": round(total),
                "used_mb": round(used),
                "available_mb": round(total - used),
            }

        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = int(parts[1]) / 1024
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        return {
            "total_mb": round(total),
            "used_mb": round(total - available),
            "available_mb": round(available),
        }
    except Exception:
        return {"total_mb": 0, "used_mb": 0, "available_mb": 0}


@router.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_mb": round(props.total_mem / 1024 / 1024) if hasattr(props, "total_mem") else round(props.total_memory / 1024 / 1024),
            "memory_allocated_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024),
            "memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024 / 1024),
        }
    else:
        gpu_info = {"name": "CPU only", "memory_total_mb": 0, "memory_allocated_mb": 0, "memory_reserved_mb": 0}

    return {
        "status": "healthy",
        "gpu": gpu_info,
        "ram": _read_ram_info(),
        "models_loaded": {
            "base_model": manager.base_model is not None,
            "sleep_staging_model": manager.sleep_staging_model is not None,
            "disease_prediction_model": manager.disease_model is not None,
        },
    }


@router.get("/task_status")
async def task_status():
    return tracker.to_dict()


@router.get("/task_result")
async def task_result():
    """Retrieve stored prediction result after task completion."""
    if tracker.active:
        return {"status": "processing", "message": "任务仍在处理中"}
    if not tracker.has_result:
        if tracker.error:
            return {"status": "error", "detail": tracker.error}
        return {"status": "no_result", "message": "没有可用的结果"}
    result = tracker.take_result()
    return result


@router.get("/label_mapping", response_model=schemas.LabelMappingResponse)
async def get_label_mapping():
    df = pd.read_csv(LABEL_MAPPING_PATH)
    mappings = [
        schemas.LabelMappingItem(
            label_idx=int(row["label_idx"]),
            phecode=str(row["phecode"]),
            phenotype=str(row["phenotype"]),
        )
        for _, row in df.iterrows()
    ]
    return schemas.LabelMappingResponse(total=len(mappings), mappings=mappings)
