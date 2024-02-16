import asyncio
import os
import sys
import numpy as np
from IPython.display import Audio
import streamlit as st
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from dotenv import load_dotenv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import clone_vocals
from musicgen_utils import generate_audio_prompted_music, generate_text_music
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

load_dotenv()

# Set up Mistral client
api_key = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

# Create a list of audio paths from the "./audios/vocal_clips" directory
audio_paths = [f"./audios/vocal_clips/{audio}" for audio in os.listdir("./audios/vocal_clips")]

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = [
        "chat_history", "current_audio_clip", "chat_mode", "prompt_chat_history", "current_prompt",
        "uploaded_vocals", "current_cloned_vocals"
    ]
    default_values = [
        [], None, "text", [], None, None, None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_cowriter_session_variables()

async def main():
    """ Main function for the chat page """
    st.write(f"Current prompt: {st.session_state.current_prompt}")
    st.sidebar.write(f"Current chat mode: {st.session_state.chat_mode}")
    st.write(st.session_state.current_cloned_vocals)
    chat_mode_toggle = st.sidebar.radio(
        "Chat Mode", ["Text", "Audio", "Voice Clone"], index=0
    )

    if chat_mode_toggle == "Audio":
        if st.session_state.current_audio_clip is not None:
            with st.sidebar.container():
                st.markdown("Current audio clip:")
                st.write(Audio(np.array(st.session_state.current_audio_clip[0]["generated_text"]), rate=32000))

    if chat_mode_toggle == "Voice Clone":
        with st.sidebar.container():
            audio_clips_select = st.selectbox(
                "Select a vocal clip to clone", audio_paths
            )
            if audio_clips_select:
                st.audio(audio_clips_select, format="audio/wav")
            clone_button = st.button("Clone Vocals")
            if clone_button and audio_clips_select:
                with st.spinner("Cloning your vocals...  This will take a minute.  I'll be back shortly!"):
                    audio_path = audio_clips_select
                    st.session_state.current_cloned_vocals = await clone_vocals(audio_path)
                    logger.debug(f"Cloned vocals: {st.session_state.current_cloned_vocals}")
            elif clone_button and not audio_clips_select:
                st.error("Please select a vocal clip to clone.")
        if st.session_state.current_cloned_vocals:
            with st.sidebar.container():
                st.markdown("Your cloned vocals:")
                st.write(st.session_state.current_cloned_vocals[1][0])
                st.audio(
                    st.session_state.current_cloned_vocals[1][1], format="audio/wav",
                    sample_rate=st.session_state.current_cloned_vocals[1][0]
                )
    new_prompt = ChatMessage(
        role = "system", content = """You are Dave Matthews,
        the famous musician and songwriter, engaged in a
        co-writing session with the user who could be a fellow musician or fan.  The goal
        of the session is to help the user feel as though you are right alongside them,
        helping them craft their song with Dave's style and personality.  Do not break character.
        Your most recent chat history is detailed in the next prompts.  Keep your answers concise
        and open-ended to help the user feel as though they are in a real conversation with you.
        """)

    # Display markdown with animation
    st.markdown("""<div class="text-container;" style="animation: fadeIn ease 3s;
                    -webkit-animation: fadeIn ease 3s; -moz-animation: fadeIn ease 3s;
                    -o-animation: fadeIn ease 3s; -ms-animation:
                    fadeIn ease 3s;">
                    </div>""", unsafe_allow_html=True)

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
            messages=[new_prompt] + st.session_state.prompt_chat_history,
            temperature=0.75,
            max_tokens=750,
        )
        st.write(f"Messages: {[new_prompt] + st.session_state.prompt_chat_history}")
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
                            st.session_state.current_audio_clip[0]["generated_text"][0])
                    )
                    st.session_state.current_audio_clip = None
                    audio_clip_json = audio_clip.json()
                    st.session_state.current_audio_clip = audio_clip_json
                    logging.info(f"Current clip: {st.session_state.current_audio_clip}")
                    logging.debug("Rerunning app after composing audio.")
                    st.rerun()
                else:
                    audio_clip = await generate_text_music(prompt=prompt)
                    audio_clip_json = audio_clip.json()
                    st.session_state.current_audio_clip = audio_clip_json
                    logging.info(f"Current clip: {st.session_state.current_audio_clip}")
                    logging.debug("Rerunning app after composing audio.")
                    st.rerun()
            st.rerun()

if __name__ == "__main__":
    asyncio.run(main())
