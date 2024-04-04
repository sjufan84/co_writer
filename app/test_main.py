import asyncio
import streamlit as st
import logging
import os
from app_utils import clone_vocals

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Create a list of audio paths from the "./audios/vocal_clips" directory
audio_paths = [f"./audios/vocal_clips/{audio}" for audio in os.listdir("./audios/vocal_clips")]

if "cloned_vocals" not in st.session_state:
    st.session_state.cloned_vocals = None
if "current_cloned_vocals" not in st.session_state:
    st.session_state.current_cloned_vocals = None

async def main():
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
        st.audio(
            st.session_state.current_cloned_vocals[1][1], format="audio/wav",
            sample_rate=st.session_state.current_cloned_vocals[1][0]
        )

if __name__ == "__main__":
    asyncio.run(main())
