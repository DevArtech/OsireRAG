"""
Module: requestor.py

Contains the functions that handle the requests for a Gradio application.

Classes:
- None

Functions:
- new_knowledge_base(project: str, vs: str, model: str) -> dict: Creates a new knowledge base.
- add_webpages(project: str, vs: str, model: str, urls: str) -> None: Adds webpages to the knowledge base.
- add_documents(project: str, vs: str, model: str, documents: List[str]) -> None: Adds documents to the knowledge base.
- query(project: str, vs: str, model: str, query: str) -> Iterator[str]: Queries the knowledge base for an LLM response.

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


def add_webpages(project: str, vs: str, model: str, urls: str) -> Tuple[None, None]:
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
            kb.add_webpages(project, vs, model, urls)
            gr.Info(f"{len(urls)} Webpage(s) added successfully.")
            return None, None
        except ValueError as e:
            raise exceptions.Error(str(e))

    return None, None


def add_documents(
    project: str, vs: str, model: str, documents: List[str]
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
                    project_name=project, vectorstore_name=vs, model_name=model
                ),
                files,
            )
            gr.Info(f"{len(files)} Files(s) added successfully.")
            return None, None
        except ValueError as e:
            raise exceptions.Error(str(e))

    return None, None


def query(project: str, vs: str, model: str, query: str, history: Optional[List[Dict[str, str]]] = None) -> Iterator[str]:
    """
    Queries the knowledge base for an LLM response.

    Args:
    - `project (str)`: The name of the project.
    - `vs (str)`: The name of the vector store.
    - `model (str)`: The name of the model.
    - `query (str)`: The query to search for.

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
        params=SearchParameters(query=query, n_results=10, filter={}, rerank=True),
        stream=True,
        conversation_history=history,
    )

    # Yield each chunk as it's recieved from the RAGPrompt generator
    for chunk in rag_prompt(prompt):
        if chunk:
            yield chunk
