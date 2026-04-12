import os
from pydantic_settings import BaseSettings
import urllib.parse

# Find the project root (2 levels up from app/core)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
env_path = os.path.join(PROJECT_ROOT, ".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "AlphaEarth Training Management API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/ae"
    
    # PostGIS Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "alphaearth_db"
    
    # Huawei Cloud OBS
    HUAWEI_OBS_AK: str = ""
    HUAWEI_OBS_SK: str = ""
    HUAWEI_OBS_SERVER: str = "https://obs.cn-north-4.myhuaweicloud.com"
    HUAWEI_OBS_BUCKET: str = "alphaearth-lake"
    
    # Paths
    DATA_DIR: str = os.path.join(PROJECT_ROOT, "data")
    WEIGHTS_DIR: str = os.path.join(DATA_DIR, "weights")
    RAW_DATA_DIR: str = os.path.join(WEIGHTS_DIR, "raw_data")
    SAMPLES_DIR: str = os.path.join(DATA_DIR, "raw_samples")
    
    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        encoded_password = urllib.parse.quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{self.POSTGRES_USER}:{encoded_password}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    class Config:
        env_file = env_path
        extra = "ignore"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.RAW_DATA_DIR, exist_ok=True)
os.makedirs(settings.SAMPLES_DIR, exist_ok=True)