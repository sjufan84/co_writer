""" Tests for the musigen module. """
import logging
from typing import List
import os
import unittest
from unittest.mock import patch, MagicMock
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


class TestMusigen(unittest.TestCase):
    """ Tests for the musigen module. """

    @patch("app.utils.musicgen_utils.client.chat")
    async def test_get_llm_inputs(self, mock_chat):
        """ Test get_llm_inputs. """
        # Arrange
        mock_chat.return_value = MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content="a funky house with 80s hip hop vibes"
                    )
                )
            ]
        )
        artist = "Dave Matthews"
        chat_history = [
            ChatMessage(
                role="user",
                content="I'm looking for a song to play at my wedding"
            ),
            ChatMessage(
                role="system",
                content="I'm a music generating model.  I can help you find a song."
            ),
            ChatMessage(
                role="user",
                content="Great!  I like Dave Matthews and the Grateful Dead."
            ),
            ChatMessage(
                role="system",
                content="I'm a music generating model.  I can help you find a song."
            ),
        ]

        # Act
        prompt = await get_llm_inputs(artist=artist, chat_history=chat_history)

        # Assert
        self.assertEqual(
            prompt,
            "a funky house with 80s hip hop vibes"
        )

if __name__ == "__main__":
    unittest.main()
