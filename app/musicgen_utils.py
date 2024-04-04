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

client = MistralAsyncClient(api_key=os.getenv("MISTRAL_API_KEY"))

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
            role = "system", content = f"""You are a musician
            engaging with the user in a 'co-writing'
            session.  Your job is to help create a text prompt for a music generating
            model based on your previous prompt {st.session_state.current_prompt}
            the current user's prompt {prompt}, and your most recent chat_response
            {st.session_state.chat_history[-1]}.
            Here are some examples of prompts for the model:

            Example Prompt 1: 'A dynamic blend of hip-hop and orchestral elements,
            with sweeping strings and brass, evoking the vibrant energy of the city.'
            Example Prompt 2: 'Violins and synths that inspire awe
            at the finiteness of life and the universe.'
            Example Prompt 3: 'Rock with saturated guitars, a
            heavy bass line and crazy drum break and fills.'

            Remember, these are just examples.  You should craft your own prompt
            based on the user's input and your previous prompt.  The prompt should be concise
            and focused, however, similar to the examples above.  Each new clip will build on the previous
            one, so highlight any requested changes or new directions in the prompt.
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
        temperature=0.7,
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
            "duration": 10,
            "guidance_scale": 3,
            "audio": None
        }
    }
    print(payload)
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response

async def generate_audio_prompted_music(
        audio_clip: Union[np.array, List], prompt: str = None
):
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
            "duration": 10,
            "guidance_scale": 3,
            "audio": audio_clip,
        }
    }
    print(payload)
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    return response
