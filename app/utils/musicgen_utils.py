import logging
from typing import List
import os
import asyncio
# import numpy as np
import requests
import json
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
    mistral_model = "mistral-small"

    initial_message = [
        ChatMessage(
            role = "system", content = f"""You are a famous artist in the style of {artist},
            the famous musician, engaging with the user in a 'co-writing'
            session.  Your job is to help create a text prompt for a music generating
            model based on your previous prompt {st.session_state.current_prompt}
            and the prompt {prompt} that the user provides.
            Here are some examples of prompts for the model:

            Example Prompt 1: 'a funky house with 80s hip hop vibes'
            Example Prompt 2: 'a chill song with influences from lofi, chillstep and downtempo'
            Example Prompt 3: 'a catchy beat for a podcast intro'

            Remember, these are just examples.  You should
            craft the prompt based on {artist}' signature style, the chat history
            and the message that the user provides.  The music gen model does not know you are an artist
            in the style of {artist} so you may need to provide some context for the model.  Keep your prompt
            as clear and concise as possible while still conveying the appropriate information to the model.
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
    )

    prompt = chat_response.choices[0].message.content

    st.session_state.current_prompt = prompt

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

'''async def generate_audio_music(prompt: str = None, audio_clip: Union[np.array, None] = None):
    """ Get a response from the music gen model
    based on text, no music """
    auth_token = os.getenv("HUGGINGFACE_TOKEN")
    if not prompt:
        prompt = await get_llm_inputs()
    API_URL =
    payload = {
        "inputs" : prompt,
    }
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    return response.json()'''

if __name__ == "__main__":
    text_prompt = "a funky house with 80s hip hop vibes"
    output = asyncio.run(generate_text_music(prompt=text_prompt))
    print(output)
