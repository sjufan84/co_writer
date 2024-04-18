# FastAPI endpoints for the LinerGenV1 project.
from fastapi import APIRouter, UploadFile, File
from app.services.clone_service import clone_liner_vocals
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    # Check to see if the file is a .webm file
    logger.info(f"Received file: {audio_file.filename}")

    cloned_vocals = await clone_liner_vocals(audio_file)

    response = {"cloned_vocals": cloned_vocals}

    logger.info(f"Returning cloned vocals: {response}")

    return response
