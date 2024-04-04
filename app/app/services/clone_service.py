import numpy as np
import base64
from scipy.io.wavfile import write
from io import BytesIO
from fastapi import UploadFile
from app.app_utils import clone_vocals

async def clone_liner_vocals(audio_file: UploadFile):
    """
    Clones the vocals from an audio file.

    Args:
    audio_file (UploadFile): The audio file to clone vocals from.

    Returns:
    str: A Base64 string of the cloned vocals audio file.
    """
    audio_array, sr = await clone_vocals(audio_file)

    # Convert numpy array to byte stream
    byte_stream = BytesIO()
    write(byte_stream, sr, np.array(audio_array, dtype=np.int16))

    # Convert byte stream to Base64 string
    base64_audio = base64.b64encode(byte_stream.getvalue()).decode('utf-8')

    return base64_audio
