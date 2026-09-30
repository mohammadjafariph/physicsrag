"""Paper library endpoints (from downloader metadata + vector store)."""

import json
import re

from fastapi import APIRouter, HTTPException

from config import settings
from src.api.deps import get_chat_service

router = APIRouter(prefix="/api/papers", tags=["papers"])

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _check_id(paper_id: str) -> None:
    if not _SAFE_ID.match(paper_id) or ".." in paper_id:
        raise HTTPException(status_code=400, detail=f"invalid paper id {paper_id!r}")


@router.get("")
def list_papers() -> list[dict]:
    service = get_chat_service()
    chunk_counts = service.store.paper_chunk_counts()
    papers: list[dict] = []
    for path in sorted(settings.metadata_dir.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        paper_id = meta.get("paper_id") or path.stem
        meta["paper_id"] = paper_id
        meta["chunk_count"] = chunk_counts.get(paper_id, 0)
        papers.append(meta)
    papers.sort(
        key=lambda p: p.get("updated") or p.get("published") or "", reverse=True
    )
    return papers


@router.get("/{paper_id}")
def get_paper(paper_id: str) -> dict:
    _check_id(paper_id)
    path = settings.metadata_dir / f"{paper_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="paper not found")
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail=f"unreadable metadata: {error}") from error
    meta["paper_id"] = meta.get("paper_id") or paper_id
    service = get_chat_service()
    meta["chunk_count"] = len(service.store.get_paper_chunks(paper_id))
    return meta