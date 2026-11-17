
import os
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables from .env
load_dotenv()

# Read API key from local environment or Streamlit Cloud secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        GEMINI_API_KEY = None

# Model configuration
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is missing. Configure it in .env "
        "or Streamlit Secrets."
    )
