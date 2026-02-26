from pydantic import BaseModel, Field
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    gpu: dict
    models_loaded: dict


class LabelMappingItem(BaseModel):
    label_idx: int
    phecode: str
    phenotype: str


class LabelMappingResponse(BaseModel):
    total: int
    mappings: list[LabelMappingItem]


class ChannelInfo(BaseModel):
    BAS: list[str] = Field(default_factory=list)
    RESP: list[str] = Field(default_factory=list)
    EKG: list[str] = Field(default_factory=list)
    EMG: list[str] = Field(default_factory=list)


class PreprocessResponse(BaseModel):
    status: str
    channels: ChannelInfo
    duration_seconds: float
    sample_rate: int


class EmbeddingInfo(BaseModel):
    shape: list[int]


class EmbedResponse(BaseModel):
    status: str
    embeddings: dict[str, EmbeddingInfo]
    num_5min_chunks: int
    embed_dim: int


class StageProbabilities(BaseModel):
    Wake: float
    N1: float
    N2: float
    N3: float
    REM: float


class SleepEpoch(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    predicted_stage: str
    probabilities: StageProbabilities


class StageSummaryItem(BaseModel):
    count: int
    percentage: float


class SleepStagingResult(BaseModel):
    total_epochs: int
    epochs: list[SleepEpoch]
    summary: dict[str, StageSummaryItem]


class DiseaseRiskItem(BaseModel):
    rank: int
    label_idx: int
    phecode: str
    phenotype: str
    hazard_score: float


class DiseasePredictionResult(BaseModel):
    top_risks: list[DiseaseRiskItem]
    all_hazards_count: int


class PredictResponse(BaseModel):
    status: str
    sleep_staging: Optional[SleepStagingResult] = None
    disease_prediction: Optional[DiseasePredictionResult] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    detail: str
