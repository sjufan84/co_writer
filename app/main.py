import asyncio
import os
import streamlit as st
from mistralai.models.chat_completion import ChatMessage
# from utils.musicgen_utils import get_llm_inputs
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
        "chat_history"
    ]
    default_values = [
        []
    ]
    for var, default_value in zip(session_vars, default_values):
        if var not in st.session_state:
            st.session_state[var] = default_value

init_cowriter_session_variables()

async def main():
    """ Main function for the chat page """
    new_prompt = [
        ChatMessage(
            role = "system", content = f"""You are Dave Matthews,
            the famous musician and songwriter, engaged in a
            co-writing session with the user who could be a fellow musician or fan.  The goal
            of the session is to help the user feel as though you are right alongside them,
            helping them craft their song with Dave's style and personality.  Do not break character.
            Your conversation so far is {st.session_state.chat_history}.
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

    # Add a blank line
    st.text("")
    st.write(st.session_state.chat_history)

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
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Load the prophet image for the avatar
        # Display assistant response in chat message container
        with st.chat_message("assistant", avatar="🎸"):
            message_placeholder = st.empty()
            full_response = ""
        response = client.chat_stream(
            model="mistral-small",
            messages= new_prompt + [ChatMessage(role = m.role, content = m.content)
                                    for m in st.session_state.chat_history],
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
        # Add assistant message to chat history
        st.session_state.chat_history.append(ChatMessage(role="assistant", content=full_response))

if __name__ == "__main__":
    asyncio.run(main())
