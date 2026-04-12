from fastapi import APIRouter, File, UploadFile, Form
from typing import Optional
from core.pipeline import process_content

router = APIRouter()

@router.post("/analyze/text")
async def analyze_text(text: str = Form(...)):
    """
    Endpoint for analyzing raw text.
    """
    result = await process_content(content_type="text", data=text)
    return result

@router.post("/analyze/file")
async def analyze_file(file: UploadFile = File(...)):
    """
    Endpoint for analyzing media/files (image, video, audio).
    """
    # Read first bytes or just pass the file to pipeline
    content = await file.read()
    
    # Simple content type resolution
    content_type = "document"
    if "image" in file.content_type:
        content_type = "image"
    elif "video" in file.content_type:
        content_type = "video"
    elif "audio" in file.content_type:
        content_type = "audio"
        
    result = await process_content(
        content_type=content_type, 
        data=content, 
        filename=file.filename
    )
    return result
