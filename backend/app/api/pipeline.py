from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.core.pipeline import run_pipeline_for_image
import io

router = APIRouter()

@router.post("/api/pipeline")
async def pipeline(
    image: UploadFile = File(...),
    mood: str = Form(...),
    vibe: str = Form(...)
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        result = run_pipeline_for_image(
            image_stream=io.BytesIO(image_bytes),
            mood=mood,
            vibe=vibe
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
