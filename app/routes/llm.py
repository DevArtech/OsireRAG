import json
import textwrap
from typing import Iterator
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

from core.logger import logger
from core.models.llm import llm
from core.settings import get_settings
from core.models.knowledge_base import KnowledgeBase, SearchParameters


class Prompt(BaseModel):
    prompt: str
    stream: bool = False


class RAGPrompt(BaseModel):
    project_name: str
    vectorstore_name: str
    model_name: str
    params: SearchParameters
    stream: bool = False

    class Config:
        protected_namespaces = ()


router = APIRouter(prefix="/llm", tags=["llm"])
kb = KnowledgeBase()


def craft_rag_prompt(prompt: RAGPrompt) -> str:
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

    new_prompt = textwrap.dedent(
        f"""Contextual Information is below:
            ------------------------------------------
            {"\n\n".join([doc[0].content for doc in documents])}
            ------------------------------------------
            Use the context information, and not prior knowledge, to answer the query prompted by the user.
            """
    )

    if get_settings().ENVIRONMENT == "local":
        new_prompt += textwrap.dedent(
            """
            \nEnd your response with \"Query:\"
            Query: What was the average stock price of Amazon in 2020?
            Answer: The average stock price per share of Amazon in 2020 was $3,311.37.
            """
        )

    new_prompt += f"\n\nQuery: {prompt.params.query}"

    if get_settings().ENVIRONMENT == "local":
        new_prompt += "\n" + "Answer:"

    return (
        new_prompt,
        documents,
    )


def rag_prompt(prompt: RAGPrompt) -> Iterator[str]:
    contextual_prompt, documents = craft_rag_prompt(prompt)

    response = ""
    generator = llm.stream_prompt(contextual_prompt)
    for item in generator:
        yield item.replace('"', '\\"')
        response += item.replace('"', '\\"')

    logger.info(f"Model Response: {response}")

    yield "<|C|>"
    for doc in documents:
        yield json.dumps(doc[0].model_dump())

    return


@router.post("/prompt/")
async def prompt(prompt: Prompt) -> StreamingResponse:
    if get_settings().ENVIRONMENT == "local":
        query = f'End your response with "Query:" Query: {prompt.prompt} Answer:'
    else:
        query = prompt.prompt

    if prompt.stream:

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

    return llm.prompt(query)


@router.post("/rag/")
async def rag(prompt: RAGPrompt) -> StreamingResponse:
    if prompt.stream:
        return StreamingResponse(rag_prompt(), media_type="text/plain")

    contextual_prompt, _ = craft_rag_prompt(prompt)
    return llm.prompt(contextual_prompt)
