import os
from functools import lru_cache


from pydantic_settings import BaseSettings

abs_path_env = os.path.abspath("../../.env")


class Settings(BaseSettings):
    API_TOKEN: str
    SALT: str

    # Feel free to modify these for your platform
    APP_NAME: str = "Rosie FastAPI Template"
    APP_VERSION: str = "0.0.0"
    APP_DESC: str = (
        "A FastAPI template for building APIs on Rosie, the MSOE supercomputer. \nDeveloped by: Adam Haile - 2024"
    )

    ENVIRONMENT: str = "Rosie"
    DEVICE: str = "cuda"
    BASE_URL: str = ""

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
