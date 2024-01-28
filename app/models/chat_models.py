from pydantic import BaseModel, Field
from typing import List
from mistralai.models.chat_completion import ChatMessage

class ChatHistory(BaseModel):
    messages: List[ChatMessage] = Field([], description="The chat history")
