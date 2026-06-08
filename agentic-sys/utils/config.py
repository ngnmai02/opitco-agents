# config.py
import os
from dotenv import load_dotenv

load_dotenv()  # reads .env from the current project directory

API_KEY = os.getenv("AITTA_API")
