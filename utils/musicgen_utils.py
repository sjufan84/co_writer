import logging
import os
import streamlit as st
from mistralai import MistralClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Mistral client
api_key = os.environ.get("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)


