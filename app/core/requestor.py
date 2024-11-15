import io
from typing import List
from gradio import exceptions
from fastapi import UploadFile

from core.settings import get_settings
from core.models.knowledge_base import KnowledgeBase
from routes.llm import rag_prompt, RAGPrompt, SearchParameters

kb = KnowledgeBase()


def new_knowledge_base(project: str, vs: str, model: str):
    try:
        kb.create_kb(project, vs, model)
        return {"response": "Knowledge base created successfully"}
    except ValueError as e:
        raise exceptions.Error(str(e))


def add_documents(project: str, vs: str, model: str, documents: List[str]):
    files = [
        UploadFile(
            file=io.BytesIO(f.read()), filename=doc.replace("\\", "/").split("/")[-1]
        )
        for doc in documents
        for f in [open(doc, "rb")]
    ]

    try:
        return {"response": kb.add_documents(project, vs, model, files)}
    except ValueError as e:
        raise exceptions.Error(str(e))


def query(project: str, vs: str, model: str, query: str):

    prompt = RAGPrompt(
        project_name=project,
        vectorstore_name=vs,
        model_name=model,
        params=SearchParameters(query=query, n_results=10, filter={}, rerank=True),
        stream=True,
    )

    for chunk in rag_prompt(prompt):
        if chunk:
            yield chunk
