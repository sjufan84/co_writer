import logging
from models.chat_models import ChatHistory
import os
from mistralai import MistralClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

mistral_model = "mistral-tiny"

async def get_llm_inputs(artist: str = "Dave Matthews", chat_history: ChatHistory = None):
    """Get the inputs for the LLM model

    Args:
        artist (str, optional): The artist to use for the LLM model. Defaults to "Dave Matthews".
        chat_history (ChatHistory, optional): The chat history to use for the LLM model. Defaults to None.

    Returns:
        str: The input string for the LLM model
    """
    if chat_history is None:
        chat_history = []

    messages = [
        {
            "role" : "system", "content" : f"""You are {artist},
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
        }
    ]

    chat_response = client.chat(
        model=mistral_model,
        messages=messages,
    )

    prompt = chat_response.choices[0].message.content

    return prompt
