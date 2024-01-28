import logging
from typing import List
import os
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

mistral_model = "mistral-tiny"

async def get_llm_inputs(artist: str = "Dave Matthews", chat_history: List[ChatMessage] = None):
    logger.debug(f"get_llm_inputs called with artist={artist} and chat_history={chat_history}")

    if chat_history is None:
        chat_history = []

    messages = [
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

    chat_response = client.chat(
        model=mistral_model,
        messages=messages,
    )

    prompt = chat_response.choices[0].message.content

    logger.debug(f"get_llm_inputs returning prompt={prompt}")

    return prompt
