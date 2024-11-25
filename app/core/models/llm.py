"""
Module: llm.py

Classes:
- LLM: Pydantic model for the LLM model.

Functions:
- None

Usage:
- Import the predefined LLM element from this module (ensures only one instance of the LLM model is running).

Author: Adam Haile
Date: 10/30/2024
"""

from openai import OpenAI
from llama_cpp import Llama
from pydantic import BaseModel
from typing import Iterator, Union

from core.settings import get_settings
from core.logger import logger, COLORS


class LLM(BaseModel):
    """
    Pydantic model for the LLM model.

    Attributes:
    - llm: Union[Llama, OpenAI]: The Llama/OpenAI model.

    Methods:
    - __init__: Initializes the LLM model.
    - prompt: Prompts the LLM model with a given prompt.
    - stream_prompt: Prompts the LLM model with a given prompt and streams the output.

    Usage:
    - Create an instance of this class to interact with the LLM model.

    Author: Adam Haile
    Date: 10/30/2024
    """

    llm: Union[Llama, OpenAI] = None  # Llama/OpenAI

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()
        # Initialize the LLM model depending on the environment
        if get_settings().ENVIRONMENT == "local":
            logger.warning(
                f"{COLORS().WARNING}Initializing LLM Model - This may take a second{COLORS().RESET}"
            )
            self.llm = Llama(
                model_path=get_settings().MODEL_PATH,
                n_gpu_layers=-1 if get_settings().DEVICE == "cuda" else 0,
                seed=1337,
                n_ctx=128000,
                verbose=False,
            )
            logger.info(f"{COLORS().INFO}LLM Model Initialized{COLORS().RESET}")
        else:
            logger.info("Connected to Rosie Meta LLaMa Model")
            self.llm = OpenAI(base_url=get_settings().ROSIE_LLM, api_key="not_used")

    def prompt(self, prompt: str, max_length: int = 128000) -> str:
        """
        Prompts the LLM model with a given prompt.

        Args:
        - prompt: str: The prompt to send to the LLM model.
        - max_length: int: The maximum length of the output.

        Returns:
        - str: The output from the LLM model.

        Usage:
        - llm.prompt(prompt, max_length = 128000)

        Author: Adam Haile
        Date: 10/30/2024
        """

        # Prompt the LLM model with the given prompt depending on the environment
        if get_settings().ENVIRONMENT == "local":
            # Text parsing for if the model is a local llama-cpp-python model
            return (
                self.llm(prompt, max_tokens=max_length, stop=["Query:"], echo=True)[
                    "choices"
                ][0]["text"]
                .split("Answer:")[1]
                .strip()
            )
        else:
            # Querying and parsing for if using the Rosie Meta LLM.
            return (
                self.llm.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    max_tokens=max_length,
                    stream=False,
                )
                .choices[0]
                .message.content
            )

    def stream_prompt(self, prompt: str, max_length: int = 128000) -> Iterator[str]:
        """
        Prompts the LLM model with a given prompt and streams the output.

        Args:
        - prompt: str: The prompt to send to the LLM model.
        - max_length: int: The maximum length of the output.

        Returns:
        - Iterator[str]: The output from the LLM model.

        Usage:
        - for token in llm.stream_prompt(prompt, max_length = 128000):
              print(token)

        Author: Adam Haile
        Date: 10/30/2024
        """

        # Prompt the LLM model with the given prompt depending on the environment
        if get_settings().ENVIRONMENT == "local":
            # Text parsing for if the model is a local llama-cpp-python model
            for token in self.llm(
                prompt, max_tokens=max_length, stop=["Query:"], echo=True, stream=True
            ):
                yield token["choices"][0]["text"]
        else:
            # Querying and parsing for if using the Rosie Meta LLM.
            for token in self.llm.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_length,
                stream=True,
            ):
                if token.choices[0].delta.content:
                    yield token.choices[0].delta.content

        return


llm = LLM()
