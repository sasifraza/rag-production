from pydantic import BaseModel, Field
from typing import List


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=8, ge=1, le=20)


class SourceItem(BaseModel):
    source: str
    chunk_id: int
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem]