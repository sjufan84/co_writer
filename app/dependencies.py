from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_openai_api_key():
    """ Function to get the OpenAI API key. """
    return os.getenv("OPENAI_KEY")

def get_openai_org():
    """ Function to get the OpenAI organization. """
    return os.getenv("OPENAI_ORG")

def get_openai_client():
    """ Get the OpenAI client. """
    return OpenAI(
        api_key=get_openai_api_key(), organization=get_openai_org(),
        max_retries=3, timeout=10
    )
