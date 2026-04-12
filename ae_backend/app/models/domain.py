from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Boolean, Enum
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from app.db.database import Base
import uuid
import enum
from datetime import datetime

class SmSatellite(Base):
    __tablename__ = "sm_satellite"
    # Legacy table mapping (partial) for integration
    norad_number = Column(String, primary_key=True, index=True)
    data_source = Column(String)
    type = Column(String)
    line1 = Column(String)
    line2 = Column(String)
    line3 = Column(String)

class Xiangzhen(Base):
    __tablename__ = "xiangzhen"
    province = Column(String, primary_key=True)
    city = Column(String, primary_key=True)
    county = Column(String, primary_key=True)
    township = Column(String, primary_key=True)
    geometry = Column(Geometry('MULTIPOLYGON', srid=4326))

class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    PREPROCESSING = "PREPROCESSING"
    TRAINING = "TRAINING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AeDataset(Base):
    __tablename__ = "ae_datasets"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_name = Column(String, index=True)
    satellite_sources = Column(JSON) # e.g., ["Sentinel-2"]
    patch_count = Column(Integer, default=0)
    geom = Column(Geometry('POLYGON', srid=4326))
    obs_path_prefix = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

class AeTrainingJob(Base):
    __tablename__ = "ae_training_jobs"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String, ForeignKey("ae_datasets.id"))
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    hyperparameters = Column(JSON)
    current_epoch = Column(Integer, default=0)
    metrics = Column(JSON) # {"loss_rec": [], "loss_uni": []}
    created_at = Column(DateTime, default=datetime.utcnow)
    
    dataset = relationship("AeDataset")

class AeModel(Base):
    __tablename__ = "ae_models"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String, ForeignKey("ae_training_jobs.id"))
    model_name = Column(String)
    evaluation_score = Column(Float)
    weights_obs_key = Column(String)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
