""" Backend API for chatbot """
import os
import openai
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY2")
openai.organization = os.getenv("OPENAI_ORG2")

playht_api_key = os.getenv("PLAY_HT_KEY")
playht_user_id = os.getenv("PLAY_HT_ID")

# Initialize FastAPI app
app = FastAPI()


# Generate audio stream
async def generate_audio_stream(completion):
    """ Generate audio stream from Play.ht API """
    async with httpx.AsyncClient(timeout=400) as client:
        payload = {
            "quality": "draft",
            "output_format": "mp3",
            "speed": 1,
            "sample_rate": 24000,
            "text": f"{completion}",
            "voice": "mark"
        }
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {playht_api_key}",
            "X-USER-ID": f"{playht_user_id}"
        }
        response = await client.post("https://play.ht/api/v2/tts/stream", json=payload, headers=headers)
        for block in response.iter_bytes(1024):
            yield block

# API endpoint to handle chat and audio streaming
@app.get("/chat", response_class=StreamingResponse)
async def chat(query: str):
    FULL_RESPONSE = ""
    for completion in openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ],
        stream=True,
        max_tokens=150
    ):
        FULL_RESPONSE += completion.choices[0].delta.get("content", "")
    
    return StreamingResponse(generate_audio_stream(FULL_RESPONSE), media_type="audio/mpeg")
