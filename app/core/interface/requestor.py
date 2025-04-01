"""
Module: requestor.py

Contains the functions that handle the requests for a Gradio application.

Classes:
- None

Functions:
- new_knowledge_base(project: str, vs: str, model: str) -> dict: Creates a new knowledge base.
- add_webpages(project: str, vs: str, model: str, urls: str) -> None: Adds webpages to the knowledge base.
- add_documents(project: str, vs: str, model: str, documents: List[str]) -> None: Adds documents to the knowledge base.
- query(project: str, vs: str, model: str, query: str, history: Optional[List[Dict[str, str]]] = None, n_results: int = 10, rerank: bool = True, temperature: float = 0.7, allow_no_results: bool = False, threshold: float = None) -> Iterator[str]: Queries the knowledge base for an LLM response.

Usage:
- Import the functions from this module into your Gradio application.

Author: Adam Haile  
Date: 10/20/2024
"""

import io
import gradio as gr
from gradio import exceptions
from fastapi import UploadFile
from typing import List, Tuple, Iterator, Dict, Optional

from app.core.models.knowledge_base import KnowledgeBase, DocumentArgs

from app.routes.llm import rag_prompt, RAGPrompt, SearchParameters

kb = KnowledgeBase()


def new_knowledge_base(project: str, vs: str, model: str) -> None:
    """
    Creates a new knowledge base.

    Args:
    - `project (str)`: The name of the project.
    - `vs (str)`: The name of the vector store.
    - `model (str)`: The name of the model.

    Returns:
    - None

    Raises:
    - ValueError: If the knowledge base cannot be created.

    Usage:
    - `new_knowledge_base(project, vs, model)`

    Author: Adam Haile
    Date: 10/20/2024
    """
    try:
        kb.create_kb(project, vs, model)
        return {"response": "Knowledge base created successfully"}
    except ValueError as e:
        raise exceptions.Error(str(e))


def add_webpages(
    project: str,
    vs: str,
    model: str,
    urls: str,
    n: int = 7,
    char_len: int = 10000,
    overlap: int = 50,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> Tuple[None, None]:
    """
    Adds webpages to the knowledge base.

    Args:
    - `project (str)`: The name of the project.
    - `vs (str)`: The name of the vector store.
    - `model (str)`: The name of the model.
    - `urls (str)`: A string with comma separated URLs of the webpages.

    Returns:
    - None

    Raises:
    - ValueError: If the webpages cannot be added.

    Usage:
    - `add_webpages(project, vs, model, urls)`

    Author: Adam Haile
    Date: 10/20/2024
    """
    if urls:
        urls = [
            url.strip() for url in urls.split(",")
        ]  # Split the URLs for each webpage
        gr.Info(f"Adding {len(urls)} webpage(s)...")

        try:
            kb.add_webpages(
                DocumentArgs(
                    project_name=project,
                    vectorstore_name=vs,
                    model_name=model,
                    n=n,
                    chunk_len=char_len,
                    chunk_overlap=overlap,
                    k1=k1,
                    b=b,
                    epsilon=epsilon,
                ),
                urls,
            )
            gr.Info(f"{len(urls)} Webpage(s) added successfully.")
            return None, None, None, None, None
        except ValueError as e:
            raise exceptions.Error(str(e))

    return None, None, None, None, None


def add_documents(
    project: str,
    vs: str,
    model: str,
    documents: List[str],
    n: int = 7,
    char_len: int = 10000,
    overlap: int = 50,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> Tuple[None, None]:
    """
    Adds documents to the knowledge base.

    Args:
    - `project (str)`: The name of the project.
    - `vs (str)`: The name of the vector store.
    - `model (str)`: The name of the model.
    - `documents (List[str])`: A list of file paths for the documents.

    Returns:
    - None

    Raises:
    - ValueError: If the documents cannot be added.

    Usage:
    - `add_documents(project, vs, model, documents)`

    Author: Adam Haile
    Date: 10/20/2024
    """
    print(f"{project}, {vs}, {model}, {documents}, {n}, {char_len}, {overlap}, {k1}, {b}, {epsilon}")
    if documents:
        # Read the documents into a bytestream and create a list of UploadFile objects
        files = [
            UploadFile(
                file=io.BytesIO(f.read()),
                filename=doc.replace("\\", "/").split("/")[-1],
            )
            for doc in documents
            for f in [open(doc, "rb")]
        ]
        gr.Info(f"Adding {len(files)} document(s)...")

        try:
            kb.add_documents(
                DocumentArgs(
                    project_name=project,
                    vectorstore_name=vs,
                    model_name=model,
                    n=n,
                    chunk_len=char_len,
                    chunk_overlap=overlap,
                    k1=k1,
                    b=b,
                    epsilon=epsilon,
                ),
                files,
            )
            gr.Info(f"{len(files)} Files(s) added successfully.")
            return None, None, None, None, None
        except ValueError as e:
            raise exceptions.Error(str(e))

    return None, None, None, None, None


def query(
    project: str,
    vs: str,
    model: str,
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    n_results: int = 10,
    rerank: bool = True,
    temperature: float = 0.7,
    allow_no_results: bool = True,
    threshold: float = None
) -> Iterator[str]:
    """
    Queries the knowledge base for an LLM response.

    Args:
    - `project (str)`: The name of the project.
    - `vs (str)`: The name of the vector store.
    - `model (str)`: The name of the model.
    - `query (str)`: The query to search for.
    - `history (Optional[List[Dict[str, str]]])`: The conversation history.
    - `n_results (int)`: Number of results to return from search.
    - `rerank (bool)`: Whether to rerank the search results.
    - `temperature (float)`: The temperature for the model's generation.
    - `allow_no_results (bool)`: Whether to allow an empty response if no documents are found.
    - `threshold (float)`: Minimum similarity score for retrieved documents (None = no threshold).

    Returns:
    - Iterator[str]: A generator that yields the LLM response.

    Usage:
    - ```
    for response in query(project, vs, model, query):
        print(response)
    ```

    Author: Adam Haile
    Date: 10/20/2024
    """
    prompt = RAGPrompt(
        project_name=project,
        vectorstore_name=vs,
        model_name=model,
        params=SearchParameters(
            query=query,
            n_results=n_results,
            filter={},
            rerank=rerank,
            allow_no_results=allow_no_results,
            threshold=threshold
        ),
        stream=True,
        conversation_history=history,
        temperature=temperature,
    )

    # Yield each chunk as it's received from the RAGPrompt generator
    for chunk in rag_prompt(prompt):
        if chunk:
            yield chunk
