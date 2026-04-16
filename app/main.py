from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.schema import QueryRequest, QueryResponse, SourceDocument, HealthResponse
from app.pipeline.rag_pipeline import RAGPipeline

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    pipeline = RAGPipeline()
    print("Pipeline ready")
    yield
    print("Shutting down")


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    try:
        response = pipeline.run(body.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=response.answer,
        sources=[
            SourceDocument(
                content=doc.page_content,
                metadata=doc.metadata,
            )
            for doc in response.sources
        ],
        query=response.query,
    )