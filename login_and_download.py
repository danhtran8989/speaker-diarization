from dotenv import load_dotenv
import os
from huggingface_hub import login

def hf_login():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the Hugging Face token from .env
    token = os.getenv("HF_TOKEN")
    
    if token:
        login(token=token)
        print("Successfully logged in to Hugging Face Hub!")
    else:
        print("Error: HF_TOKEN not found in .env file. Please add it and try again.")
