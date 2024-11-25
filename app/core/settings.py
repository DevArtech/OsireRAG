import os
from functools import lru_cache

from pydantic_settings import BaseSettings

abs_path_env = os.path.abspath("../../.env")


class Settings(BaseSettings):
    API_TOKEN: str
    APP_NAME: str = "RosieRAG"
    ENVIRONMENT: str = "Rosie"
    DEVICE: str = "cuda"
    BASE_URL: str = ""
    MODEL_PATH: str = "/home/hailea/Llama-3.2-3B-Instruct.gguf"
    ROSIE_LLM: str = "http://dh-dgxh100-2.hpc.msoe.edu:8000/v1"

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
