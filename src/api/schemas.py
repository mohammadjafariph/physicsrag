"""Request/response schemas for the API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    k: int = Field(default=0, ge=0, le=30, description="0 -> server default")


class SourceOut(BaseModel):
    n: int
    chunk_id: str
    paper_id: str
    title: str
    section: str
    page: int
    arxiv_url: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    citations: list[SourceOut]
    evidence_count: int
    provider: str
    model: str


class ResearchRunRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=200)
    max_cycles: int = Field(default=3, ge=1, le=10)


class SettingsUpdate(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    persist: bool = True


class HealthOut(BaseModel):
    status: str
    papers: int
    chunks: int
    topics: int
    provider: str
    model: str