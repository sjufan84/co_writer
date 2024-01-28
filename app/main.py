import asyncio
import streamlit as st
from mistralai.models.chat_completion import ChatMessage
from utils.musicgen_utils import get_llm_inputs

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

async def main():
    text_input = st.text_input("Chat input", value="I'm looking for a song to play at my wedding")
    artist = st.text_input("Artist", value="Dave Matthews")

    prompt_button = st.button("Get prompt")
    if prompt_button:
        # Add the user's input to the chat history
        st.session_state.chat_history.append(ChatMessage(role="user", content=text_input))
        prompt = await get_llm_inputs(artist=artist, prompt=text_input)
        st.write(prompt)

if __name__ == "__main__":
    asyncio.run(main())
