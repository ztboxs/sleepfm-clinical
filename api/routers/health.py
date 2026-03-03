import torch
import pandas as pd
from fastapi import APIRouter

from api.config import LABEL_MAPPING_PATH
from api.models_manager import manager
from api.task_status import tracker
from api import schemas


router = APIRouter(prefix="/api/v1", tags=["health"])


def _read_ram_info() -> dict:
    """Read system RAM stats from /proc/meminfo (Linux only)."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                if key in ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached"):
                    info[key] = int(parts[1]) / 1024  # kB -> MB
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
