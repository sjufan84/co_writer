import asyncio
import os
import numpy as np
# import json
import streamlit as st
from mistralai.models.chat_completion import ChatMessage
from utils.musicgen_utils import generate_text_music
from mistralai.client import MistralClient
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# Set up Mistral client
api_key = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = [
        "chat_history", "current_audio_clip", "chat_mode", "prompt_chat_history", "current_prompt"
    ]
    default_values = [
        [], None, "text", None, None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_cowriter_session_variables()

async def main():
    """ Main function for the chat page """
    chat_mode_toggle = st.sidebar.radio(
        "Chat Mode", ["Text", "Audio"], index=0
    )
    if len(st.session_state.chat_history) >= 3:
        st.session_state.prompt_chat_history = st.session_state.chat_history[-3:]
    else:
        st.session_state.prompt_chat_history = st.session_state.chat_history
    new_prompt = [
        ChatMessage(
            role = "system", content = f"""You are Dave Matthews,
            the famous musician and songwriter, engaged in a
            co-writing session with the user who could be a fellow musician or fan.  The goal
            of the session is to help the user feel as though you are right alongside them,
            helping them craft their song with Dave's style and personality.  Do not break character.
            Your most recent chat history is {st.session_state.prompt_chat_history}.
            """)
    ]

    # Display markdown with animation
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                    -o-animation: fadeIn ease 3s; -ms-animation:
                    fadeIn ease 3s;">
                    </div>""", unsafe_allow_html=True)

    # Check if there are any messages in the session state
    if len(st.session_state.chat_history) == 0:
        logging.debug("No messages in session state.")
        st.warning(
            "The audio generation does take some time, especially upon start.  As we scale,\
            we will continue to increase our compute thus speeding up the process dramatically.  However, for\
            demo purposes, we are not utilizing large amounts of GPU resources."
        )

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history:
        logging.debug(f"Displaying message: {message}")
        if message.role == "assistant":
            with st.chat_message(message.role, avatar="🎸"):
                st.markdown(message.content)
        elif message.role == "user":
            with st.chat_message(message.role):
                st.markdown(message.content)

    # Accept user input
    if prompt := st.chat_input("Hey friend, let's start writing!"):
        logging.debug(f"Received user input: {prompt}")
        # Add user message to chat history
        st.session_state.chat_history.append(ChatMessage(role="user", content=prompt))
        if len(st.session_state.prompt_chat_history) <= 3:
            new_prompt = st.session_state.chat_history
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎸"):
            message_placeholder = st.empty()
            full_response = ""
        st.write(st.session_state.chat_history)
        st.write(new_prompt)
        response = client.chat_stream(
            model="mistral-small",
            messages= new_prompt,
            temperature=0.75,
            max_tokens=350,
        )
        for chunk in response:
            if chunk.choices[0].finish_reason == "stop":
                logging.debug("Received 'stop' signal from response.")
                break
            full_response += chunk.choices[0].delta.content
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        logging.debug(f"Received response: {full_response}")
        st.session_state.chat_history.append(ChatMessage(role="assistant", content=full_response))
        logging.debug(f"Chat history: {st.session_state.chat_history}")
        if chat_mode_toggle == "Audio":
            with st.spinner("Composing your audio...  I'll be back shortly!"):
                audio_clip = await generate_text_music(prompt=prompt)
                st.write(audio_clip)
                audio_clip_json = audio_clip.json()
                st.session_state.current_audio_clip = audio_clip_json
                logging.info(f"Current clip: {st.session_state.current_audio_clip}")
                logging.debug("Rerunning app after composing audio.")
                st.rerun()
    if st.session_state.current_audio_clip:
        audio_array = np.array(st.session_state.current_audio_clip[0]["generated_text"]).flatten()
        sample_rate = st.session_state.current_audio_clip[0]["sampling_rate"]
        st.write(audio_array[:100])
        st.write(sample_rate)
        st.audio(data=audio_array, sample_rate=sample_rate)

if __name__ == "__main__":
    asyncio.run(main())
