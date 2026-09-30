"""Background research-run endpoints."""

from fastapi import APIRouter, HTTPException

from src.api.deps import get_run_manager
from src.api.schemas import ResearchRunRequest

router = APIRouter(prefix="/api/research", tags=["research"])


@router.post("/runs")
def start_run(request: ResearchRunRequest) -> dict:
    manager = get_run_manager()
    try:
        run_id = manager.start(request.topic, request.max_cycles)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return manager.snapshot(manager.get(run_id))


@router.get("/runs")
def list_runs() -> list[dict]:
    manager = get_run_manager()
    return [manager.snapshot(run) for run in manager.all()]


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    manager = get_run_manager()
    run = manager.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return manager.snapshot(run)