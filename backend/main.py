"""FastAPI entry point for the RAG pipeline demo."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import ingest, query, status

allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://frontend-n052z6lo9-ravib-projects.vercel.app,https://*.vercel.app",
).split(",")

app = FastAPI(
    title="RAG Pipeline Demo",
    description="Upload a PDF, embed it 3 ways, and compare retrieval quality.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+",
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
