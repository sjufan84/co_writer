# FastAPI endpoints for the LinerGenV1 project.
from fastapi import APIRouter, UploadFile, File
from app.services.clone_service import clone_liner_vocals

router = APIRouter()

@router.post("/clone_vocals")
async def clone_vocals_endpoint(audio_file: UploadFile = File(...)):
    """
    Clones the vocals from an audio file.

    Args:
    audio_file (UploadFile): The audio file to clone vocals from.

    Returns:
    str: A Base64 string of the cloned vocals audio file.
    """
    return await clone_liner_vocals(audio_file)
