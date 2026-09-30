"""RAG chat endpoints (with-citations JSON and SSE stream)."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.deps import get_chat_service
from src.api.schemas import ChatRequest, ChatResponse
from src.llm.base import LLMError

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    service = get_chat_service()
    try:
        return service.answer(request.question, k=request.k or None)
    except LLMError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events: citations -> token* -> done (or error)."""
    service = get_chat_service()

    def event_stream():
        try:
            for event in service.answer_stream(request.question, k=request.k or None):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except LLMError as error:
            payload = json.dumps({"type": "error", "error": str(error)})
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")