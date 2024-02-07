import logging
from typing import List, Union
import os
import asyncio
import pandas as pd
import numpy as np
import requests
import librosa
# import json
import streamlit as st
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

async def get_llm_inputs(
    prompt: str = None, artist: str = "Dave Matthews"
):
    """ Get the inputs for the LLM model """
    # Set up Mistral client
    api_key = os.getenv("MISTRAL_API_KEY")
    client = MistralAsyncClient(api_key=api_key)
    mistral_model = "mistral-medium"

    initial_message = [
        ChatMessage(
            role = "system", content = f"""You are a famous artist in the style of {artist},
            the famous musician, engaging with the user in a 'co-writing'
            session.  Your job is to help create a text prompt for a music generating
            model based on your previous prompt {st.session_state.current_prompt}
            the current user's prompt {prompt}, and your most recent chat_response
            {st.session_state.chat_history[-1]}.
            Here are some examples of prompts for the model:

            Example Prompt 1: 'a funky house with 80s hip hop vibes'
            Example Prompt 2: 'a chill song with influences from lofi, chillstep and downtempo'
            Example Prompt 3: 'a catchy beat for a podcast intro'

            Remember, these are just examples.  You should
            craft the prompt based on {artist}' signature style, the chat history
            and the message that the user provides.  Focus on the instruments, the
            tempo, the mood, and the genre of the song you want to create.  Your prompt
            should be similar to the examples above.  Focus on the key components of the
            clip, keeping your prompt concise, focused, and to the point.
            """
        )
    ]

    user_message = [
        ChatMessage(
            role = "user", content = f"""Please provide a prompt for the music gen model based on the
            previous prompt {st.session_state.current_prompt} so far and and my prompt {prompt}."""
        )
    ]
    messages = initial_message + user_message

    chat_response = await client.chat(
        model=mistral_model,
        messages=messages,
        temperature=0.5,
        max_tokens=250
    )

    prompt = chat_response.choices[0].message.content

    st.session_state.current_prompt = prompt

    # prompt_message = ChatMessage(
    #    role = "system", content = f"""I've created this prompt for the music gen model:
    #    {prompt}"""
    # )
    # st.session_state.chat_history.append(prompt_message)

    return prompt

async def generate_text_music(prompt: str = None):
    """ Get a response from the music gen model
    based on text, no music """
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    prompt = await get_llm_inputs(prompt=prompt)
    logger.info(prompt)
    API_URL = "https://j8q5ioorjuh9ce3u.us-east-1.aws.endpoints.huggingface.cloud"
    payload = {
        "inputs" : prompt,
        "parameters" : {
            "do_sample": True,
            "temperature": 0.7,
            "duration": 12,
            "guidance_scale": 3
        }
    }
    print(payload)
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response

async def generate_audio_prompted_music(audio_clip: Union[np.array, List], prompt: str = None):
    """ Get a response from the music gen model
    based on text, no music """
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    API_URL = "https://j8q5ioorjuh9ce3u.us-east-1.aws.endpoints.huggingface.cloud"
    prompt = await get_llm_inputs(prompt=prompt)
    # Check to see if the audio clip is an array.  If so, convert to a list
    # so that it can be serialized to JSON
    if isinstance(audio_clip, np.ndarray):
        audio_clip = audio_clip.tolist()
    payload = {
        "inputs" : prompt,
        "parameters" : {
            "do_sample": True,
            "temperature": 0.7,
            "duration": 15,
            "guidance_scale": 3,
            "audio": audio_clip
        }
    }

    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response

def load_audio_clip(audio_file: str):
    """ Load an audio clip from a file """
    audio, sr = librosa.load(audio_file, sr=32000)
    return audio, sr

# Create a function to split the audio clip into 10 second clips
def split_audio_clip(audio_clip: np.array, clip_length: int = 10):
    """ Split an audio clip into smaller clips """
    # Get the length of the audio clip
    clip_length = clip_length * 32000
    clip_count = len(audio_clip) // clip_length
    split_clips = np.array_split(audio_clip, clip_count)

    return split_clips

def create_audio_clip_df(audio_clips: List[np.array], sr: int):
    """ Create a dataframe of audio clips """
    audio_clips_df = pd.DataFrame()
    # Keep every 3rd clip
    for i, clip in enumerate(audio_clips):
        if i % 5 == 0:
            clip_series = pd.Series(clip)
            audio_clips_df = pd.concat([audio_clips_df, clip_series], axis=1)
    # Save the dataframe to a csv file
    audio_clips_df.to_csv("app/audios/audio_clips.csv", index=False)
    return "Audio clips dataframe created successfully."

def convert_to_wav(audio_clip: np.array, sr: int, file_name: str = "generated_music"):
    """ Convert an audio clip to a wav file """
    audio_clip = (audio_clip * 32767).astype(np.int16)
    # Save the audio clip to a wav file
    librosa.output.write_wav(f"app/audios/instrumentals/{file_name}.wav", audio_clip, sr)
    return audio_clip

if __name__ == "__main__":
    audio_clip = load_audio_clip("app/audios/instrumentals/fc_instrumental.wav")
    split_clips = split_audio_clip(audio_clip[0])
    create_audio_clip_df(split_clips, audio_clip[1])

