import logging
from typing import List
import os
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up Mistral client
api_key = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

logging.debug(f"client={client}")

mistral_model = "mistral-small"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

async def get_llm_inputs(
    prompt: str, artist: str = "Dave Matthews", chat_history: List[ChatMessage] = None
):
    logger.debug(f"get_llm_inputs called with artist={artist} and chat_history={chat_history}")

    if chat_history is None:
        chat_history = []

    initial_message = [
        ChatMessage(
            role = "system", content = f"""You are {artist},
            the famous musician, engaging with the user in a 'co-writing'
            session.  Your job is to help create a text prompt for a music generating
            model based on your chat history
            {chat_history}.  Here are some examples of prompts for the model:

            Example Prompt 1: 'a funky house with 80s hip hop vibes'
            Example Prompt 2: 'a chill song with influences from lofi, chillstep and downtempo'
            Example Prompt 3: 'a catchy beat for a podcast intro'

            Remember, these are just examples.  You should
            craft the prompt based on {artist}' signature style
            and the chat history to this point.  Return only the prompt text.
            """
        )
    ]

    user_message = [
        ChatMessage(
            role = "user", content = prompt
        )
    ]

    messages = initial_message + user_message

    logger.debug(f"get_llm_inputs messages={messages}")

    chat_response = client.chat(
        model=mistral_model,
        messages=messages,
    )

    prompt = chat_response.choices[0].message.content

    logger.debug(f"get_llm_inputs returning prompt={prompt}")

    return prompt
