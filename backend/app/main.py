from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="FrameKraft", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

from app.api.pipeline import router as pipeline_router
app.include_router(pipeline_router)

@app.get("/")
async def root():
    index_path = os.path.join(frontend_dir, "framekraft.html")
    return FileResponse(index_path)
