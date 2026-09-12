"""FastAPI entry point for the RAG pipeline demo."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest, query, status

app = FastAPI(
    title="RAG Pipeline Demo",
    description="Upload a PDF, embed it 3 ways, and compare retrieval quality.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, prefix="/api")
app.include_router(query.router,  prefix="/api")
app.include_router(status.router, prefix="/api")


@app.get("/")
def root():
    return {"message": "RAG Pipeline API is running. Visit /docs for the API explorer."}
