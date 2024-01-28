from pyantic import BaseModel, Field
from typing import List

class ChatMessage:
    content: str = Field(..., description="The content of the message")
    role: str = Field(..., description="The role of the message (either 'user' or 'assistant')")

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = Field(..., description="The chat history")
