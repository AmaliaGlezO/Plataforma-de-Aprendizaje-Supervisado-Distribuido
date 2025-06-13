"""
ENDPOINTS DE ENTRENAMIENTO
Este archivo maneja todas las rutas relacionadas con entrenamiento de modelos.
Funciones: iniciar entrenamientos, consultar estado, cancelar trabajos.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse
from typing import List
import uuid
import os
from datetime import datetime
from pathlib import Path

# Importaciones internas
from ..schemas.training_schemas import TrainingRequest, TrainingResponse, JobStatus
from ...ml_engine.training_orchestrator import TrainingOrchestrator
from ...ray_cluster.cluster_manager import ClusterManager
from ...utils.storage import DatasetStorage
from ...utils.logger import get_logger

router = APIRouter(prefix="/api/training", tags=["training"])
logger = get_logger("training_api")

# Estado en memoria (en producción usar Redis o DB)
training_jobs = {}
dataset_storage = DatasetStorage()

@router.post("/start", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks
):
    """Inicia un nuevo trabajo de entrenamiento distribuido"""
    try:
        # Validar dataset existe
        if not dataset_storage.dataset_exists(request.dataset_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {request.dataset_id} no encontrado"
            )

        # Crear job ID único
        job_id = str(uuid.uuid4())
        
        # Configurar el trabajo
        training_jobs[job_id] = {
            "status": "pending",
            "start_time": datetime.utcnow(),
            "config": request.dict(),
            "metrics": None
        }
        
        # Obtener path del dataset
        dataset_path = dataset_storage.get_dataset_path(request.dataset_id)
        
        # Iniciar entrenamiento en background
        background_tasks.add_task(
            run_distributed_training,
            job_id=job_id,
            dataset_path=dataset_path,
            model_configs=request.model_configs,
            num_workers=request.num_workers
        )
        
        logger.info(f"Iniciando trabajo {job_id} con config: {request}")
        
        return TrainingResponse(
            job_id=job_id,
            status="pending",
            message="Training job started"
        )
        
    except Exception as e:
        logger.error(f"Error iniciando entrenamiento: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

async def run_distributed_training(
    job_id: str,
    dataset_path: str,
    model_configs: List[dict],
    num_workers: int
):
    """Ejecuta el entrenamiento distribuido (en background)"""
    try:
        # Actualizar estado
        training_jobs[job_id]["status"] = "running"
        
        # Iniciar entrenamiento
        orchestrator = TrainingOrchestrator()
        results = orchestrator.execute_distributed_training(
            dataset_path=dataset_path,
            model_configs=model_configs,
            num_workers=num_workers
        )
        
        # Guardar resultados
        training_jobs[job_id].update({
            "status": "completed",
            "end_time": datetime.utcnow(),
            "metrics": {
                "models_trained": len(results),
                "average_accuracy": sum(r['metrics']['accuracy'] for r in results) / len(results)
            },
            "results": results
        })
        
        logger.info(f"Trabajo {job_id} completado exitosamente")
        
    except Exception as e:
        training_jobs[job_id].update({
            "status": "failed",
            "error": str(e),
            "end_time": datetime.utcnow()
        })
        logger.error(f"Error en trabajo {job_id}: {str(e)}")

@router.post("/upload-dataset", status_code=status.HTTP_201_CREATED)
async def upload_dataset(file: UploadFile = File(...)):
    """Sube un dataset para entrenamiento"""
    try:
        # Validar extensión del archivo
        valid_extensions = {".csv", ".parquet", ".json"}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in valid_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Formato de archivo no soportado"
            )
        
        # Guardar dataset
        dataset_id = str(uuid.uuid4())
        file_path = dataset_storage.save_dataset(file, dataset_id)
        
        logger.info(f"Dataset {dataset_id} subido: {file.filename}")
        
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "path": str(file_path),
            "size": os.path.getsize(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error subiendo dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Obtiene el estado de un trabajo de entrenamiento"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job no encontrado"
        )
    
    job = training_jobs[job_id]
    elapsed = (job.get("end_time", datetime.utcnow()) - job["start_time"]).total_seconds()
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        start_time=job["start_time"],
        elapsed_time=elapsed,
        config=job["config"],
        metrics=job.get("metrics")
    )

@router.get("/jobs", response_model=List[JobStatus])
async def list_training_jobs():
    """Lista todos los trabajos de entrenamiento"""
    return [
        JobStatus(
            job_id=job_id,
            status=job["status"],
            start_time=job["start_time"],
            elapsed_time=(job.get("end_time", datetime.utcnow()) - job["start_time"]).total_seconds(),
            config=job["config"]
        )
        for job_id, job in training_jobs.items()
    ]

@router.delete("/jobs/{job_id}", status_code=status.HTTP_202_ACCEPTED)
async def cancel_training_job(job_id: str):
    """Cancela un trabajo de entrenamiento"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job no encontrado"
        )
    
    if training_jobs[job_id]["status"] not in ["pending", "running"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se pueden cancelar trabajos pendientes o en ejecución"
        )
    
    # En una implementación real, aquí se llamaría al ClusterManager
    # para cancelar las tareas Ray asociadas
    training_jobs[job_id]["status"] = "cancelled"
    training_jobs[job_id]["end_time"] = datetime.utcnow()
    
    logger.info(f"Trabajo {job_id} cancelado")
    
    return {"message": f"Job {job_id} cancelled"}

@router.get("/jobs/{job_id}/metrics")
async def get_training_metrics(job_id: str):
    """Obtiene métricas de un entrenamiento específico"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job no encontrado"
        )
    
    if not training_jobs[job_id].get("metrics"):
        raise HTTPException(
            status_code=status.HTTP_425_TOO_EARLY,
            detail="Las métricas no están disponibles aún"
        )
    
    return training_jobs[job_id]["metrics"]

@router.post("/jobs/{job_id}/resume", status_code=status.HTTP_202_ACCEPTED)
async def resume_training_job(job_id: str, background_tasks: BackgroundTasks):
    """Reanuda un trabajo de entrenamiento pausado"""
    if job_id not in training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job no encontrado"
        )
    
    if training_jobs[job_id]["status"] != "paused":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Solo se pueden reanudar trabajos pausados"
        )
    
    # Reanudar el trabajo
    config = training_jobs[job_id]["config"]
    background_tasks.add_task(
        run_distributed_training,
        job_id=job_id,
        dataset_path=dataset_storage.get_dataset_path(config["dataset_id"]),
        model_configs=config["model_configs"],
        num_workers=config["num_workers"]
    )
    
    return {"message": f"Job {job_id} resuming"}