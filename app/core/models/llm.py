from openai import OpenAI
from llama_cpp import Llama
from typing import Iterator

from core.settings import get_settings
from core.logger import logger, COLORS


class LLM:
    def __init__(self) -> None:
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
        if get_settings().ENVIRONMENT == "local":
            return (
                self.llm(prompt, max_tokens=max_length, stop=["Query:"], echo=True)[
                    "choices"
                ][0]["text"]
                .split("Answer:")[1]
                .strip()
            )
        else:
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
        if get_settings().ENVIRONMENT == "local":
            for token in self.llm(
                prompt, max_tokens=max_length, stop=["Query:"], echo=True, stream=True
            ):
                yield token["choices"][0]["text"]
        else:
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
