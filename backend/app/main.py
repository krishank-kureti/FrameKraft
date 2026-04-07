from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.api.pipeline import router as pipeline_router
import os

app = FastAPI(title="FrameKraft API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-app-name.vercel.app", "http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "FrameKraft Brain"}

@app.get("/")
async def serve_frontend():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    index_path = os.path.join(base_dir, "frontend", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "Frontend not found. Ensure frontend/index.html exists."}

app.include_router(pipeline_router)