from fastapi import FastAPI
from app.schema import AskRequest
from app.rag_pipeline import RAGPipeline

app = FastAPI(title="Production RAG API")

pipeline = RAGPipeline()


@app.get("/")
def home():
    return {"message": "RAG API is running"}


@app.post("/ask")
def ask(request: AskRequest):
    return pipeline.ask(request.question, request.top_k)