from pydantic import BaseModel
from typing import List, Optional

class DataSourceRequest(BaseModel):
    region_name: Optional[str] = None
    satellite_sources: List[str]
    # Representing a bounding box or WKT polygon, simplified for MVP
    bounding_box: Optional[List[float]] = None 

class DataSourceEvaluation(BaseModel):
    spectral_overlap_score: int
    spatial_resolution_score: int
    temporal_density_score: int
    modality_richness_score: int
    data_readiness_score: int
    overall_score: int
    recommendations: List[str]