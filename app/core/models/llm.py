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

import copy
import json
import pickle
from openai import OpenAI
from llama_cpp import Llama
from pydantic import BaseModel
from transformers import AutoTokenizer
from typing import Iterator, Union, Optional, List, Dict, ClassVar, Tuple

from app.core.settings import get_settings
from app.core.logger import logger, COLORS


class LLM(BaseModel):
    """
    Pydantic model for the LLM model.

    Attributes:
    - llm: Union[Llama, OpenAI]: The Llama/OpenAI model.

    Methods:
    - __init__: Initializes the LLM model.
    - _tokenize: Validates a message is within token length and fixes it if not.
    - prompt: Prompts the LLM model with a given prompt.
    - stream_prompt: Prompts the LLM model with a given prompt and streams the output.

    Usage:
    - Create an instance of this class to interact with the LLM model.

    Author: Adam Haile
    Date: 10/30/2024
    """

    llm: ClassVar[Union[Llama, OpenAI]] = None
    tokenizer: ClassVar[AutoTokenizer] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        super().__init__()
        # Load the Llama 3.1 70b tokenizer
        with open(get_settings().TOKENIZER_PATH, "rb") as file:
            LLM.tokenizer = pickle.load(file)

        # Initialize the LLM model depending on the environment
        if get_settings().ENVIRONMENT == "local":
            logger.warning(
                f"{COLORS().WARNING}Initializing LLM Model - This may take a second{COLORS().RESET}"
            )

            # Initialize the Llama model
            LLM.llm = Llama(
                model_path=get_settings().MODEL_PATH,
                n_gpu_layers=-1 if get_settings().DEVICE == "cuda" else 0,
                seed=1337,
                n_ctx=128000,
                verbose=False,
            )

            logger.info(f"{COLORS().INFO}LLM Model Initialized{COLORS().RESET}")
        else:
            LLM.llm = OpenAI(base_url=get_settings().HPC_LLM, api_key="not_used")
            logger.info("Connected to HPC LLM Model")

    def _tokenize(
        self, messages: List[Dict[str, str]], l: int = 100000
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Validates a message is within token length and fixes it if not.

        Args:
        - `message (List[Dict[str, str]])`: The message to tokenize.

        Returns:
        - List[Dict[str, str]]: The corrected message.

        Usage:
        - `llm.tokenize(message)`
        """
        messages_c = copy.deepcopy(messages)
        logger.info(f"Original messages: {messages_c}")
        tokenized = len(
            self.tokenizer(json.dumps(messages_c), add_special_tokens=True)["input_ids"]
        )
        logger.info(f"Message length: {tokenized}")
        while tokenized > l:
            # Remove messages in accordance to if they are the system message or not
            pop = 1 if messages_c[0].get("role") == "system" else 0
            popped_message = messages_c.pop(pop)

            # If the last one popped was a user message, also remove the assistant
            if popped_message.get("role") == "user":
                messages_c.pop(pop)

            tokenized = len(
                self.tokenizer(json.dumps(messages_c), add_special_tokens=True)[
                    "input_ids"
                ]
            )

        logger.info(f"Final message length: {tokenized}")
        logger.info(f"Final messages: {messages_c}")
        return (
            messages_c,
            tokenized,
        )  # Return the corrected message and the token length

    def prompt(self, prompt: str, max_length: int = 128000) -> str:
        """
        Prompts the LLM model with a given prompt.

        Args:
        - `prompt (str)`: The prompt to send to the LLM model.
        - `max_length (int)`: The maximum length of the output.

        Returns:
        - str: The output from the LLM model.

        Usage:
        - `llm.prompt(prompt, max_length = 128000)`

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
            # Querying and parsing for if using the HPC LLM.
            return (
                self.llm.chat.completions.create(
                    model=get_settings().REMOTE_MODEL,
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

    def stream_prompt(
        self,
        prompt: str,
        max_length: int = 128000,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]:
        """
        Prompts the LLM model with a given prompt and streams the output.

        Args:
        - `prompt (str)`: The prompt to send to the LLM model.
        - `max_length (int)`: The maximum length of the output.

        Returns:
        - Iterator[str]: The output from the LLM model.

        Usage:
        - ```
        for token in llm.stream_prompt(prompt, max_length = 128000):
              print(token)
        ```

        Author: Adam Haile
        Date: 10/30/2024
        """

        # Prompt the LLM model with the given prompt depending on the environment
        if get_settings().ENVIRONMENT == "local":
            # Text parsing for if the model is a local llama-cpp-python model

            # Create a full prompt with the history
            full_prompt = ""
            if history:
                for item in history:
                    for k, v in item.items():
                        if k == "role" and v == "user":
                            full_prompt += "Query: "
                        elif k == "role" and v == "assistant":
                            full_prompt += "Answer: "
                        elif k == "content":
                            full_prompt += f"{v}\n"

                full_prompt += prompt

            # Stream the LLM model output
            for token in self.llm(
                full_prompt,
                max_tokens=max_length,
                stop=["Query:"],
                echo=True,
                stream=True,
            ):
                yield token["choices"][0]["text"]
        else:
            # Querying and parsing for if using the HPC LLM.

            # Create a full prompt with the history
            full_messages = []
            if history:
                full_messages = history

            full_messages.pop()  # Remove the pre-inserted user message from the history
            full_messages.append({"role": "user", "content": prompt})

            full_messages, tokens_used = self._tokenize(full_messages)

            # Stream the LLM model output
            for token in self.llm.chat.completions.create(
                model=get_settings().REMOTE_MODEL,
                messages=full_messages,
                max_tokens=max_length - tokens_used,
                stream=True,
            ):
                if token.choices[0].delta.content:
                    yield token.choices[0].delta.content

        return


llm = LLM()
