from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.pipeline import router as pipeline_router

app = FastAPI(title="FrameKraft API", version="0.1.0")

# KEEP THIS: It allows your Vercel frontend to talk to this Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# A simple health check so you know the server is alive
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "FrameKraft Brain"}

# Your actual ML logic
app.include_router(pipeline_router)