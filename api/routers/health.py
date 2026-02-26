import torch
import pandas as pd
from fastapi import APIRouter

from api.config import LABEL_MAPPING_PATH
from api.models_manager import manager
from api import schemas

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=schemas.HealthResponse)
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024),
            "memory_used_mb": round(torch.cuda.memory_allocated(0) / 1024 / 1024),
        }
    else:
        gpu_info = {"name": "CPU only", "memory_total_mb": 0, "memory_used_mb": 0}

    return schemas.HealthResponse(
        status="healthy",
        gpu=gpu_info,
        models_loaded={
            "base_model": manager.base_model is not None,
            "sleep_staging_model": manager.sleep_staging_model is not None,
            "disease_prediction_model": manager.disease_model is not None,
        },
    )


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
