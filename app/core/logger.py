"""
Module: logger.py

Contains the logger configuration for the FastAPI app.

Classes:
- COLORS: Pydantic model for color codes for different log levels.

Functions:
- None

Usage:
- Import the logger object from this module into the main FastAPI app, along with COLORS if you wish to use color.

Author: Adam Haile  
Date: 10/10/2024
"""

import logging
from pydantic import BaseModel

from .settings import get_settings

logger = logging.getLogger("uvicorn.error")


if get_settings().ENVIRONMENT == "local":

    class COLORS(BaseModel):
        DEBUG: str = "\033[34m"  # Blue
        INFO: str = "\033[32m"  # Green
        WARNING: str = "\033[33m"  # Yellow
        ERROR: str = "\033[31m"  # Red
        CRITICAL: str = "\033[41m"  # Red Background
        RESET: str = "\033[0m"

else:

    class COLORS(BaseModel):
        DEBUG: str = ""
        INFO: str = ""
        WARNING: str = ""
        ERROR: str = ""
        CRITICAL: str = ""
        RESET: str = ""
