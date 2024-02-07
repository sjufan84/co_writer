import asyncio
import os
import sys
import numpy as np
import pandas as pd
from IPython.display import Audio
import streamlit as st
from audio_recorder_streamlit import audio_recorder
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

# Create a function to initialize the session state variables
def init_cowriter_session_variables():
    """ Initialize session state variables """
    # Initialize session state variables
    session_vars = [
        "chat_history", "current_audio_clip", "chat_mode", "prompt_chat_history", "current_prompt",
        "recorded_vocals", "current_cloned_vocals"
    ]
    default_values = [
        [], None, "text", [], None, None, None
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_cowriter_session_variables()


def audio_clips_selectbox():
    clips_df = pd.read_csv("./audios/audio_clips.csv")
    clip_options = clips_df.columns.tolist()
    selected_clip = st.selectbox("Select an audio clip to use as a prompt:", clip_options)
    if selected_clip:
        st.audio(np.array(clips_df[selected_clip].values), format="audio/wav", sample_rate=32000)
        st.session_state.original_clip = np.array(clips_df[selected_clip].values)
    else:
        return None

async def main():
    """ Main function for the chat page """
    st.write(f"Current prompt: {st.session_state.current_prompt}")
    st.sidebar.write(f"Current chat mode: {st.session_state.chat_mode}")
    st.write(st.session_state.current_cloned_vocals)
    chat_mode_toggle = st.sidebar.radio(
        "Chat Mode", ["Text", "Existing Audio", "New Audio", "Voice Clone"], index=0
    )
    st.write(st.session_state.current_audio_clip)

    if chat_mode_toggle == "Existing Audio" and st.session_state.current_audio_clip is None:
        with st.sidebar.container():
            st.markdown("**Select an audio clip to use as the initial prompt:**")
            audio_clips_selectbox()
    elif chat_mode_toggle == "Existing Audio" and st.session_state.current_audio_clip is not None:
        with st.sidebar.container():
            st.markdown("Current audio clip:")
            st.audio(
                st.session_state.current_audio_clip[0]["generated_text"],
                format="audio/wav", sample_rate=32000
            )
    
    if chat_mode_toggle == "New Audio":
        if st.session_state.current_audio_clip:
            st.sidebar.markdown("**Current audio clip:**")
            st.audio(
                st.session_state.current_audio_clip[0]["generated_text"],
                format="audio/wav", sample_rate=32000
            )

    if chat_mode_toggle == "Voice Clone":
        st.sidebar.markdown("**Record your vocals to be cloned:**")
        with st.sidebar.container():
            audio_bytes = audio_recorder()
            if audio_bytes:
                st.session_state.recorded_vocals = audio_bytes
                logger.debug(f"Recorded vocals: {audio_bytes}")
        if st.session_state.recorded_vocals:
            with st.sidebar.container():
                st.markdown("Your recorded vocals:")
                st.audio(st.session_state.recorded_vocals, format="audio/wav")
                clone_button = st.sidebar.button("Clone Vocals")
                if clone_button:
                    with st.spinner("Cloning your vocals...  This will take a minute.  I'll be back shortly!"):
                        audio_path = "./audios/recorded_audio.wav"
                        with open(audio_path, "wb") as f:
                            f.write(st.session_state.recorded_vocals)
                        st.session_state.current_cloned_vocals = clone_vocals(audio_path)
                        logger.debug(f"Cloned vocals: {st.session_state.current_cloned_vocals}")
        if st.session_state.current_cloned_vocals:
            with st.sidebar.container():
                st.markdown("Your cloned vocals:")
                # st.write(st.session_state.current_cloned_vocals[1])
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
        if chat_mode_toggle == "New Audio":
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

        elif chat_mode_toggle == "Existing Audio":
            if st.session_state.original_clip is not None:
                with st.spinner("Composing your audio...  I'll be back shortly!"):
                    audio_clip = await generate_audio_prompted_music(
                        prompt=prompt,
                        audio_clip=np.array(st.session_state.original_clip)
                    )
                    audio_clip_json = audio_clip.json()
                    st.session_state.current_audio_clip = audio_clip_json
                    logging.info(f"Current clip: {st.session_state.current_audio_clip}")
                    logging.debug("Rerunning app after composing audio.")
                    st.rerun()
            else:
                with st.spinner("Composing your audio...  I'll be back shortly!"):
                    audio_clip = await generate_text_music(
                        prompt=prompt, audio_clip=st.session_state.current_audio_clip[0]["generated_text"]
                    )
                    audio_clip_json = audio_clip.json()
                    st.session_state.current_audio_clip = audio_clip_json
                    logging.info(f"Current clip: {st.session_state.current_audio_clip}")
                    logging.debug("Rerunning app after composing audio.")
                    st.rerun()
if __name__ == "__main__":
    asyncio.run(main())
