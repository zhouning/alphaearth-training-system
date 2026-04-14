from fastapi import APIRouter, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from app.db.database import get_db, SessionLocal
from app.models.domain import AeTrainingJob, JobStatus, AeDataset
from app.services.trainer import AlphaEarthTrainer
import asyncio
import uuid

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()

class TrainingRequest(BaseModel):
    dataset_id: str = None
    peft_method: str = "linear_probe"  # linear_probe | bitfit | lora | houlsby | geoadapter

async def run_training_job(job_id: str, dataset_id: str, peft_method: str):
    # Wait for the client to establish a WebSocket connection
    await asyncio.sleep(1.0)
    trainer = AlphaEarthTrainer(job_id=job_id, dataset_id=dataset_id, ws_manager=manager, epochs=50, peft_method=peft_method)
    await trainer.run()

@router.post("/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    job_id = str(uuid.uuid4())
    ds_id = request.dataset_id if request.dataset_id else job_id
    peft = request.peft_method if request.peft_method in ("linear_probe", "bitfit", "lora", "houlsby", "geoadapter") else "linear_probe"
    
    # Ensure Dataset exists to satisfy Foreign Key
    existing_ds = db.query(AeDataset).filter(AeDataset.id == ds_id).first()
    if not existing_ds:
        new_ds = AeDataset(id=ds_id, dataset_name=f"Dataset-{ds_id[:6]}")
        db.add(new_ds)
        db.commit()
    
    # Create DB Record
    new_job = AeTrainingJob(
        id=job_id,
        status=JobStatus.PENDING,
        dataset_id=ds_id,
        hyperparameters={"epochs": 50, "learning_rate": 1e-3, "batch_size": 16, "peft_method": peft}
    )
    db.add(new_job)
    db.commit()

    background_tasks.add_task(run_training_job, job_id=job_id, dataset_id=ds_id, peft_method=peft)

    return {"message": "Training started", "job_id": job_id, "dataset_id": ds_id, "peft_method": peft}

@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            if data == "START":
                # Front-end sends START to indicate it's ready.
                await manager.broadcast({"type": "log", "message": "[系统] 监控大屏已连接，等待底层引擎就绪..."})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


