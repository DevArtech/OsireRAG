"""
Module: llm.py (Router)

This module contains the FastAPI router for the large language model. The large language model is a
module that allows users to interact with the RosieRAG model and recieve LLM summarizations of the chunks.

Classes:
- Prompt: A recieve contract for the prompt endpoint.
- RAGPrompt: A recieve contract for the RAG prompt endpoint.

Functions:
- craft_rag_prompt: Crafts a new RAG prompt for the model.
- rag_prompt: Streams the RAG response to the user.
- prompt: FastAPI endpoint for prompting the LLM.
- rag: FastAPI endpoint for prompting the model using RAG.

Attributes:
- router: The FastAPI router object.
- kb: The KnowledgeBase object.

Author: Adam Haile  
Date: 10/30/2024
"""

import json
import textwrap
from fastapi import APIRouter
from dataclasses import dataclass
from pydantic import BaseModel, field_validator
from fastapi.responses import StreamingResponse
from typing import Iterator, Tuple, List, Optional, Dict, Any

from app.core.logger import logger
from app.core.models.llm import llm
from app.core.models.chunker import Chunk
from app.core.settings import get_settings
from app.core.models.knowledge_base import KnowledgeBase, SearchParameters


@dataclass
class Prompt:
    """
    A recieve contract for the prompt endpoint.

    Attributes:
    - prompt (str): The prompt to send to the model.
    - stream (bool): Whether to stream the response.

    Methods:
    - None

    Author: Adam Haile  
    Date: 10/30/2024
    """

    prompt: str
    stream: bool = False


class RAGPrompt(BaseModel):
    """
    A recieve contract for the RAG prompt endpoint.

    Attributes:
    - project_name (str): The name of the project.
    - vectorstore_name (str): The name of the vectorstore.
    - model_name (str): The name of the model.
    - params (SearchParameters): The search parameters.
    - stream (bool): Whether to stream the response.

    Methods:
    - None

    Author: Adam Haile  
    Date: 10/30/2024
    """

    project_name: str
    vectorstore_name: str
    model_name: str
    params: SearchParameters
    stream: bool = False
    conversation_history: Optional[List[Dict[str, str]]] = None

    def __init__(self, **data: Any) -> None:
        """
        Custom initialization method for the RAGPrompt class.

        Args:
        - `data (Dict)`: The data to initialize the RAGPrompt object with.

        Returns:
        - None

        Usage:
        - `rag_prompt = RAGPrompt(project_name="project", vectorstore_name="vs", model_name="model", params=params)`

        Author: Adam Haile  
        Date: 12/2/2024
        """
        super().__init__(**data)
        
        # Create a deep copy of the conversation history
        self.conversation_history = [{k: v for k, v in item.items()} for item in self.conversation_history]


router = APIRouter(prefix="/llm", tags=["llm"])
kb = KnowledgeBase()


def craft_rag_prompt(prompt: RAGPrompt) -> Tuple[str, List[Tuple[Chunk, float]]]:
    """
    Crafts a new RAG prompt for the model.

    Args:
    - `prompt (RAGPrompt)`: The RAG prompt to craft.

    Returns:
    - str: The new RAG prompt.
    - List[Tuple[Chunk, float]]: The list of documents and their scores.

    Usage:
    - `craft_rag_prompt(prompt)`

    Author: Adam Haile  
    Date: 10/30/2024
    """

    # Log the received RAG prompt and search for documents
    logger.info(f"Received RAG prompt: {prompt.params.query}")
    documents = kb.search(
        project_name=prompt.project_name,
        vectorstore_name=prompt.vectorstore_name,
        model_name=prompt.model_name,
        params=prompt.params,
    )
    logger.info(
        f"Retrieved documents: {'\n\n'.join([doc[0].content for doc in documents])}"
    )

    # Craft the new RAG prompt
    new_prompt = textwrap.dedent(
        f"""Contextual Information is below:
            ------------------------------------------
            {"\n\n".join([doc[0].content for doc in documents])}
            ------------------------------------------
            Use the context information, and not prior knowledge, to answer the query prompted by the user.
            """
    )

    # Apply special prompting if using local llama-cpp-python model
    if get_settings().ENVIRONMENT == "local":
        new_prompt += textwrap.dedent(
            """
            \nEnd your response with \"Query:\"
            Query: What was the average stock price of Amazon in 2020?
            Answer: The average stock price per share of Amazon in 2020 was $3,311.37.
            """
        )

    # Append the query to the prompt
    new_prompt += f"\n\nQuery: {prompt.params.query}"

    # Apply special prompting if using local llama-cpp-python model
    if get_settings().ENVIRONMENT == "local":
        new_prompt += "\n" + "Answer:"

    return (
        new_prompt,
        documents,
    )


def rag_prompt(prompt: RAGPrompt) -> Iterator[str]:
    """
    Streams the RAG response to the user.

    Args:
    - `prompt (RAGPrompt)`: The RAG prompt to stream.

    Returns:
    - Iterator[str]: The stream of the RAG response.

    Usage:
    - `rag_prompt(prompt)`

    Author: Adam Haile  
    Date: 10/30/2024
    """

    # Get the RAG prompt and documents
    contextual_prompt, documents = craft_rag_prompt(prompt)

    # Prompt the LLM and yield the results
    response = ""
    generator = llm.stream_prompt(
        contextual_prompt, history=prompt.conversation_history
    )
    for item in generator:
        yield item.replace('"', '\\"')
        response += item.replace('"', '\\"')

    logger.info(f"Model Response: {response}")

    # ID the chunks and yield them
    yield "<|C|>"
    for doc in documents:
        yield json.dumps(doc[0].model_dump())

    return


@router.post("/prompt/")
async def prompt(prompt: Prompt) -> StreamingResponse:
    """
    FastAPI endpoint for prompting the LLM.

    Args:
    - `prompt (Prompt)`: The prompt to send to the model.

    Returns:
    - StreamingResponse: The stream of the LLM response.

    Usage:
    - POST /llm/prompt/

    Author: Adam Haile  
    Date: 10/30/2024
    """

    # Apply special prompting if using local llama-cpp-python model
    if get_settings().ENVIRONMENT == "local":
        query = f'End your response with "Query:" Query: {prompt.prompt} Answer:'
    else:
        query = prompt.prompt

    if prompt.stream:
        # Stream the response using JSON encoding
        def json_encoder(query):
            try:
                generator = llm.stream_prompt(query)
                current_item = next(generator)
                yield current_item.lstrip().replace('"', '\\"')
                current_item = next(generator)

                while True:
                    try:
                        yield current_item.replace('"', '\\"')
                        current_item = next(generator)
                    except StopIteration:
                        yield current_item.rstrip().replace('"', '\\"')
                        break

            except StopIteration:
                pass

        return StreamingResponse(json_encoder(query), media_type="text/event-stream")

    # Statically prompt the LLM response
    return llm.prompt(query)


@router.post("/rag/")
async def rag(prompt: RAGPrompt) -> StreamingResponse:
    """
    FastAPI endpoint for prompting the model using RAG.

    Args:
    - `prompt (RAGPrompt)`: The RAG prompt to send to the model.

    Returns:
    - StreamingResponse: The stream of the RAG response.

    Usage:
    - POST /llm/rag/

    Author: Adam Haile  
    Date: 10/30/2024
    """

    # Stream the RAG response
    if prompt.stream:
        return StreamingResponse(rag_prompt(prompt), media_type="text/plain")

    # Statically prompt the RAG response
    contextual_prompt, _ = craft_rag_prompt(prompt)
    return llm.prompt(contextual_prompt)
