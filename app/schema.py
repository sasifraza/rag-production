from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class SourceDocument(BaseModel):
    content: str
    metadata: dict = {}


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    query: str


class HealthResponse(BaseModel):
    status: str = "ok"