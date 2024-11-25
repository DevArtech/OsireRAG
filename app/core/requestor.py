import io
import gradio as gr
from typing import List
from gradio import exceptions
from fastapi import UploadFile

from core.models.knowledge_base import KnowledgeBase, DocumentArgs
from routes.llm import rag_prompt, RAGPrompt, SearchParameters

kb = KnowledgeBase()


def new_knowledge_base(project: str, vs: str, model: str):
    try:
        kb.create_kb(project, vs, model)
        return {"response": "Knowledge base created successfully"}
    except ValueError as e:
        raise exceptions.Error(str(e))


def add_webpages(project: str, vs: str, model: str, urls: str):
    if urls:
        urls = [url.strip() for url in urls.split(",")]
        gr.Info(f"Adding {len(urls)} webpage(s)...")

        try:
            kb.add_webpages(project, vs, model, urls)
            gr.Info(f"{len(urls)} Webpage(s) added successfully.")
            return None, None
        except ValueError as e:
            raise exceptions.Error(str(e))

    return None, None


def add_documents(project: str, vs: str, model: str, documents: List[str]):
    if documents:
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
