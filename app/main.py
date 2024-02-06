import asyncio
import os
import numpy as np
# import pandas as pd
# from pathlib import Path
# import json
from IPython.display import Audio
import streamlit as st
from mistralai.models.chat_completion import ChatMessage
from utils.musicgen_utils import generate_text_music, generate_audio_prompted_music
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

# @TODO - set up selectbox for audio samples for the
# user to choose from
# df = pd.read_csv(Path("app/audio_samples.csv"))

# list1 = df["Sample 0"].tolist()
# list2 = df["Sample 1"].tolist()

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = [
        "chat_history", "current_audio_clip", "chat_mode", "prompt_chat_history", "current_prompt"
    ]
    default_values = [
        [], None, "text", [], None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_cowriter_session_variables()

async def main():
    """ Main function for the chat page """
    st.write(f"Current prompt: {st.session_state.current_prompt}")
    st.sidebar.write(f"Current chat mode: {st.session_state.chat_mode}")
    chat_mode_toggle = st.sidebar.radio(
        "Chat Mode", ["Text", "Audio"], index=0
    )
    new_prompt = [
        ChatMessage(
            role = "system", content = """You are Dave Matthews,
            the famous musician and songwriter, engaged in a
            co-writing session with the user who could be a fellow musician or fan.  The goal
            of the session is to help the user feel as though you are right alongside them,
            helping them craft their song with Dave's style and personality.  Do not break character.
            Your most recent chat history is detailed in the next prompts.  Keep your answers concise
            and open-ended to help the user feel as though they are in a real conversation with you.
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
        if len(st.session_state.chat_history) >= 3:
            st.session_state.prompt_chat_history = st.session_state.chat_history[-3:]
        else:
            st.session_state.prompt_chat_history = st.session_state.chat_history
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎸"):
            message_placeholder = st.empty()
            full_response = ""
        response = client.chat_stream(
            model="mistral-medium",
            messages=new_prompt + st.session_state.prompt_chat_history,
            temperature=0.75,
            max_tokens=750,
        )
        st.write(f"Messages: {new_prompt + st.session_state.prompt_chat_history}")
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
                if st.session_state.current_audio_clip:
                    logger.debug("Generating audio with audio prompt.")
                    audio_clip = await generate_audio_prompted_music(
                        prompt=prompt,
                        audio_clip=np.array(
                            st.session_state.current_audio_clip[0]["generated_text"]).flatten())
                    st.session_state.current_audio_clip = None
                else:
                    audio_clip = await generate_text_music(prompt=prompt)
                audio_clip_json = audio_clip.json()
                st.session_state.current_audio_clip = audio_clip_json
                logging.info(f"Current clip: {st.session_state.current_audio_clip}")
                logging.debug("Rerunning app after composing audio.")
                st.rerun()
    if st.session_state.current_audio_clip is not None:
        try:
            audio_array = st.session_state.current_audio_clip[0]["generated_text"]
            st.write(Audio(audio_array, rate=32000))
            logging.debug(f"Audio array: {audio_array}, success!")
        except Exception as e:
            try:
                logging.error(f"Failed to play audio: {e}")
                st.error("Failed to play audio with Audio, trying alternative method.")
                audio_array = st.session_state.current_audio_clip[0]["generated_text"]
                st.audio(audio_array.flatten(), format="audio/wav", sample_rate=32000)
            except Exception as e:
                logging.error(f"Failed to play audio with alternative method: {e}")
                st.error("Failed to play audio with alternative method.")
    else:
        logging.debug("No audio clip available.")
        st.write("No audio clip available.")
if __name__ == "__main__":
    asyncio.run(main())
