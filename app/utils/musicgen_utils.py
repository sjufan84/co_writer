import logging
from typing import List
import os
import streamlit as st
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up Mistral client
api_key = os.getenv("MISTRAL_API_KEY")
print(type(api_key))

client = MistralAsyncClient(api_key=api_key)

print(client)

logging.debug(f"client={client}")

mistral_model = "mistral-tiny"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

async def get_llm_inputs(
    prompt: str, artist: str = "Dave Matthews", chat_history: List[ChatMessage] = None
):

    if chat_history is None:
        chat_history = st.session_state.chat_history

    initial_message = [
        ChatMessage(
            role = "system", content = f"""You are a famous artist in the style of {artist},
            the famous musician, engaging with the user in a 'co-writing'
            session.  Your job is to help create a text prompt for a music generating
            model based the chat history so far {chat_history}
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
            role = "user", content = f"""Please provide a prompt for the music gen model based on our
            conversation so far and and my prompt {prompt}."""
        )
    ]

    messages = initial_message + user_message

    chat_response = await client.chat(
        model=mistral_model,
        messages=messages,
    )

    prompt = chat_response.choices[0].message.content

    response_message = [
        ChatMessage(
            role = "system", content = f"""You have created the following prompt for the model:
            {prompt}
            """
        )
    ]

    # Add the response message to the chat history
    chat_history.extend(response_message)

    st.session_state.chat_history = chat_history

    return prompt
