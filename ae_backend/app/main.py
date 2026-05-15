from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from app.api import pipeline, training, satellites, areas, models, results
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Configure CORS for new domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to ae.img.net etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(
    pipeline.router, 
    prefix=f"{settings.API_V1_STR}/pipeline", 
    tags=["pipeline"]
)
app.include_router(
    training.router, 
    prefix=f"{settings.API_V1_STR}/training", 
    tags=["training"]
)
app.include_router(
    satellites.router, 
    prefix=f"{settings.API_V1_STR}/satellites", 
    tags=["satellites"]
)
app.include_router(
    areas.router, 
    prefix=f"{settings.API_V1_STR}/areas", 
    tags=["areas"]
)
app.include_router(
    models.router,
    prefix=f"{settings.API_V1_STR}/models",
    tags=["models"]
)
app.include_router(
    results.router,
    prefix=f"{settings.API_V1_STR}/results",
    tags=["results"]
)

# Mount results artifacts (GeoJSON, PNGs from change-detection runs)
results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../results"))
if os.path.exists(results_dir):
    app.mount("/results", StaticFiles(directory=results_dir), name="results")

# Mount frontend static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ae_frontend"))
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"), headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
else:
    @app.get("/")
    def root():
        return {"message": "AlphaEarth Training Management API is running", "version": settings.VERSION, "note": "Frontend not found locally."}

