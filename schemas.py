from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from typing import Optional

class FileUploadResponse(BaseModel):
    filename: str
    original_name: str
    rows: int
    columns: int
    file_size: int
    file_id: str


class FileRequest(BaseModel):
    filename: str


class TrainRequest(BaseModel):
    filename: str
    model_type: Optional[str] = Field(default="both", description="logistic, randomforest, or both")


class AtRiskRequest(BaseModel):
    filename: str
    threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    customerID: Any
    LogisticRegressionProb: float
    RandomForestProb: float
    ActualChurn: Optional[int] = None


class MetricsResponse(BaseModel):
    ROC_AUC: float
    Precision: float
    Recall: float
    F1: float


class ModelPerformanceResponse(BaseModel):
    LogisticRegression: MetricsResponse
    RandomForest: MetricsResponse


class ChurnPredictionResponse(BaseModel):
    success: bool
    filename: str
    result_filename: Optional[str] = "results"
    num_predictions: int
    predictions: List[PredictionResponse]
    metrics: Optional[Dict[str, MetricsResponse]] = None
    error: Optional[str] = None


class FileListResponse(BaseModel):
    files: List[Dict[str, Any]]


class AtRiskUsersResponse(BaseModel):
    filename: str
    threshold: float
    at_risk_count: int
    at_risk_users: List[PredictionResponse]