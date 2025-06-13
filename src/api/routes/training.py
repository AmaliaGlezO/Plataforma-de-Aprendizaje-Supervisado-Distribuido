"""
ENDPOINTS DE ENTRENAMIENTO
Este archivo maneja todas las rutas relacionadas con entrenamiento de modelos.
Funciones: iniciar entrenamientos, consultar estado, cancelar trabajos.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from ..schemas.training_schemas import TrainingRequest, TrainingResponse, JobStatus
from typing import List

router = APIRouter(prefix="/api/training", tags=["training"])

@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest):
    """Inicia un nuevo trabajo de entrenamiento distribuido"""
    pass

@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """Sube un dataset para entrenamiento"""
    pass

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Obtiene el estado de un trabajo de entrenamiento"""
    pass

@router.get("/jobs", response_model=List[JobStatus])
async def list_training_jobs():
    """Lista todos los trabajos de entrenamiento"""
    pass

@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """Cancela un trabajo de entrenamiento"""
    pass

@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(job_id: str):
    """Obtiene métricas de un entrenamiento específico"""
    pass

@router.post("/jobs/{job_id}/resume")
async def resume_training_job(job_id: str):
    """Reanuda un trabajo de entrenamiento pausado"""
    pass