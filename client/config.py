# client/config.py
import logging

# Set up the logger
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)