"""
Module: settings.py

Contains the settings for the FastAPI application.

Classes:
- Settings: The settings for the FastAPI application.

Functions:
- get_settings() -> Settings: Gets the settings for the FastAPI application.

Usage:
- Import the get_settings function from this module into the main FastAPI app.

Author: Adam Haile  
Date: 9/27/2024
"""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings

abs_path_env = os.path.abspath("../../.env")


class Settings(BaseSettings):
    """
    The settings for the FastAPI application.

    Attributes:
    - API_TOKEN (str): The API token for the FastAPI application.
    - APP_NAME (str): The name of the FastAPI application.
    - ENVIRONMENT (str): The environment for the FastAPI application.
    - DEVICE (str): The device to use for processing (options: "cpu", "cuda").
    - BASE_URL (str): The base URL for the FastAPI application.
    - MODEL_PATH (str): The path to the model for the LLM if using a local model.
    - ROSIE_LLM (str): The URL for the Rosie LLM.

    Usage:
    - settings = get_settings().APP_NAME

    Author: Adam Haile  
    Date: 9/27/2024
    """

    API_TOKEN: str
    APP_NAME: str = "RosieRAG"
    ENVIRONMENT: str = "Rosie"
    DEVICE: str = "cuda"
    BASE_URL: str = ""
    MODEL_PATH: str = "/home/hailea/Llama-3.2-3B-Instruct.gguf"
    ROSIE_LLM: str = "http://dh-dgxh100-2.hpc.msoe.edu:8000/v1"
    TOKENIZER_PATH: str = os.path.abspath("./app/models/tokenizer.pkl")

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """
    Gets the settings for the FastAPI application.

    Returns:
    - Settings: The settings for the FastAPI application.

    Usage:
    - get_settings()

    Author: Adam Haile  
    Date: 9/27/2024
    """
    return Settings()
