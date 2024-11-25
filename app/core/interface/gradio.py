"""
Module: gradio.py

This module is responsible for creating and managing the Gradio interface for the RosieRAG application.

Classes:
- None

Functions:
- update_project: Updates the project, vectorstore, and model dropdowns based on the selected project.
- rag_query: Queries the RosieRAG model and returns the response.
- create_knowledge_base: Creates a new knowledge base in RosieRAG.
- refresh_all: Refreshes the project, vectorstore, and model dropdowns.

Attributes:
- css: The CSS styling for the Gradio interface.
- io: The Gradio interface object.

Author: Adam Haile
Date: 11/25/2024
"""

import os
import json
import gradio as gr
from typing import Tuple, Iterator

from app.core.interface.requestor import (
    query,
    add_documents,
    new_knowledge_base,
    add_webpages,
)

css = """
.left-bar {width: 12.5% !important; min-width: 0 !important; flex-grow: 1.1 !important;}
.right-bar {width: 85% !important; flex-grow: 3.5 !important;}
.send-button {position: absolute; z-index: 99; right: 10px; height: 100%; background: none; min-width: 0 !important;}
footer {display: none !important;}
"""


def update_project(project: str) -> Tuple[str, gr.update, gr.update, str, str]:
    """
    Updates the project, vectorstore, and model dropdowns based on the selected project.

    Args:
    - project (str): The selected project.

    Returns:
    - project (str): The selected project.
    - vs_list (gr.update): The updated vectorstore dropdown.
    - model_list (gr.update): The updated model dropdown.
    - selected_vs (str): The selected vectorstore.
    - selected_model (str): The selected model.

    Usage:
    - update_project(project)

    Author: Adam Haile
    Date: 11/25/2024
    """

    # Collect the vectorstores for the given project
    vs_list = [
        i
        for i in os.listdir(f"./.rosierag/{project}")
        if os.path.isdir(f"./.rosierag/{project}/{i}")
        and any(
            fname.endswith(".faiss")
            for fname in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
        )
    ]

    # Collect the keyword models for the given project
    model_list = [
        i
        for i in os.listdir(f"./.rosierag/{project}")
        if os.path.isdir(f"./.rosierag/{project}/{i}")
        and not any(
            fname.endswith(".faiss")
            for fname in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
        )
    ]

    # Get the top vectorstore and model
    selected_vs = None
    if vs_list:
        selected_vs = vs_list[0]

    selected_model = None
    if model_list:
        selected_model = model_list[0]

    return (
        project,
        gr.update(choices=vs_list),
        gr.update(choices=model_list),
        selected_vs,
        selected_model,
    )


def rag_query(
    user_query: str, project: str, vs: str, model: str
) -> Iterator[gr.update, gr.update, None]:
    """
    Queries the RosieRAG model and returns the response.

    Args:
    - user_query (str): The user's query.
    - project (str): The selected project.
    - vs (str): The selected vectorstore.
    - model (str): The selected model.

    Returns:
    - gr.update: The updated chatbot interface.
    - gr.update: The updated chunks interface.
    - None

    Usage:
    - rag_query(user_query, project, vs, model)

    Author: Adam Haile
    Date: 11/25/2024
    """
    # Query the RosieRAG model
    response = query(project=project, vs=vs, model=model, query=user_query)

    # Process the response
    text_response = ""
    for token in response:

        # Check if the response contains a token identifying the end of the LLM response
        # and the start of the chunks
        if "<|C|>" in text_response + token:

            # Remove the token and return the response
            res = (text_response + token).replace("<|C|>", "")
            text_response = res
            yield gr.update(
                value=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": text_response},
                ]
            ), gr.update(value=None), None
            break
        else:
            # Append the token to the response
            text_response += token
            yield gr.update(
                value=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": text_response},
                ]
            ), gr.update(value=None), None

    json_value = ""
    for token in response:
        json_value += token

    # Fix the JSON formatting
    json_value = json.loads("[" + json_value.replace("}}{", "}},{") + "]")

    yield gr.update(
        value=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": text_response},
        ]
    ), gr.update(value=json_value), None


def create_knowledge_base(
    project: str, vs: str, model: str
) -> Tuple[gr.update, gr.update, gr.update, str, str, str]:
    """
    Creates a new knowledge base.

    Args:
    - project (str): The name of the project.
    - vs (str): The name of the vectorstore.
    - model (str): The name of the model.

    Returns:
    - project_update (gr.update): The updated project dropdown.
    - vs_update (gr.update): The updated vectorstore dropdown.
    - model_update (gr.update): The updated model dropdown.
    - project (str): The selected project.
    - vs (str): The selected vectorstore.
    - model (str): The selected model.

    Usage:
    - create_knowledge_base(project, vs, model)

    Author: Adam Haile
    Date: 11/25/2024
    """
    # Create the knowledge base
    res = new_knowledge_base(project=project, vs=vs, model=model)

    # Check if the knowledge base was created successfully
    if res["response"] == "Knowledge base created successfully":
        # Notify the user
        gr.Info(f"Knowledge base '{project}/{vs}+{model}' created successfully.")

        # Get the updated dropdowns
        project_update = gr.update(choices=os.listdir("./.rosierag"), value=project)
        vs_update = gr.update(
            choices=[
                i
                for i in os.listdir(f"./.rosierag/{project}")
                if os.path.isdir(f"./.rosierag/{project}/{i}")
                and any(
                    fname.endswith(".faiss")
                    for fname in os.listdir(
                        f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                    )
                )
            ],
            value=vs,
        )
        model_update = gr.update(
            choices=[
                i
                for i in os.listdir(f"./.rosierag/{project}")
                if os.path.isdir(f"./.rosierag/{project}/{i}")
                and not any(
                    fname.endswith(".faiss")
                    for fname in os.listdir(
                        f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                    )
                )
            ],
            value=model,
        )

        return project_update, vs_update, model_update, project, vs, model
    else:
        # Notify the user of the error
        gr.Error(f"Error: {res['response']}")

    return None, None, None, None, None, None


def refresh_all() -> (
    Tuple[gr.update, gr.update, gr.update, gr.update, gr.update, gr.update]
):
    """
    Refreshes the project, vectorstore, and model dropdowns.

    Args:
    - None

    Returns:
    - project_update (gr.update): The updated project dropdown.
    - vs_update (gr.update): The updated vectorstore dropdown.
    - model_update (gr.update): The updated model dropdown.
    - selected_project (gr.update): The selected project.
    - selected_vs (gr.update): The selected vectorstore.
    - selected_model (gr.update): The selected model.

    Usage:
    - refresh_all()

    Author: Adam Haile
    Date: 11/25/2024
    """
    # Get the projects, vectorstores, and models
    projects = os.listdir("./.rosierag")
    vs = (
        [
            i
            for i in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}")
            if os.path.isdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
            and any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.rosierag")) > 0
        else []
    )
    models = (
        [
            i
            for i in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}")
            if os.path.isdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
            and not any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.rosierag")) > 0
        else []
    )

    # Create the updated dropdowns
    project_update = gr.update(choices=projects)
    vs_update = gr.update(choices=vs)
    model_update = gr.update(choices=models)

    # Get the top project, vectorstore, and model
    selected_project = gr.update(value=projects[0] if projects else None)
    selected_vs = gr.update(value=vs[0] if vs else None)
    selected_model = gr.update(value=models[0] if models else None)

    return (
        project_update,
        vs_update,
        model_update,
        selected_project,
        selected_vs,
        selected_model,
    )


with gr.Blocks() as projects:
    # Get the projects, vectorstores, and models
    new_project = gr.State()
    new_vs = gr.State()
    new_model = gr.State()
    selected_project = gr.State(
        value=(
            os.listdir("./.rosierag")[0] if len(os.listdir("./.rosierag")) > 0 else None
        )
    )
    vss = (
        [
            i
            for i in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}")
            if os.path.isdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
            and any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.rosierag")) > 0
        else []
    )
    selected_vs = gr.State(value=vss[0] if vss else None)
    models = (
        [
            i
            for i in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}")
            if os.path.isdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
            and not any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.rosierag")) > 0
        else []
    )
    selected_model = gr.State(value=models[0] if models else None)

    # Create the project, vectorstore, and model interface with dropdowns
    with gr.Row():
        with gr.Column():
            with gr.Row():
                project_dropdown = gr.Dropdown(
                    choices=os.listdir("./.rosierag"), label="Projects"
                )
                vs_dropdown = gr.Dropdown(
                    choices=(
                        [
                            i
                            for i in os.listdir(
                                f"./.rosierag/{os.listdir('./.rosierag')[0]}"
                            )
                            if os.path.isdir(
                                f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                            )
                            and any(
                                fname.endswith(".faiss")
                                for fname in os.listdir(
                                    f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                                )
                            )
                        ]
                        if len(os.listdir("./.rosierag")) > 0
                        else []
                    ),
                    label="Vectorstores",
                )
                keyword_model_dropdown = gr.Dropdown(
                    choices=(
                        [
                            i
                            for i in os.listdir(
                                f"./.rosierag/{os.listdir('./.rosierag')[0]}"
                            )
                            if os.path.isdir(
                                f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                            )
                            and not any(
                                fname.endswith(".faiss")
                                for fname in os.listdir(
                                    f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}"
                                )
                            )
                        ]
                        if len(os.listdir("./.rosierag")) > 0
                        else []
                    ),
                    label="Keyword Models",
                )

                # Create the refresh button
                refresh = gr.Button("Refresh")
                refresh.click(
                    fn=refresh_all,
                    inputs=[],
                    outputs=[
                        project_dropdown,
                        vs_dropdown,
                        keyword_model_dropdown,
                        selected_project,
                        selected_vs,
                        selected_model,
                    ],
                )

        # Create the new project, vectorstore, and model interface column
        with gr.Column():
            with gr.Row():
                new_project = gr.Textbox(label="New Project")
                new_vs = gr.Textbox(label="New Vectorstore")
                new_model = gr.Textbox(label="New Keyword Model")
            with gr.Row():
                btn = gr.Button("Create Knowledge Base")
                btn.click(
                    create_knowledge_base,
                    inputs=[new_project, new_vs, new_model],
                    outputs=[
                        project_dropdown,
                        vs_dropdown,
                        keyword_model_dropdown,
                        selected_project,
                        selected_vs,
                        selected_model,
                    ],
                )

    # Update the project, vectorstore, and model dropdowns based on the selected project
    project_dropdown.change(
        fn=update_project,
        inputs=project_dropdown,
        outputs=[
            selected_project,
            vs_dropdown,
            keyword_model_dropdown,
            selected_vs,
            selected_model,
        ],
    )
    vs_dropdown.change(lambda vs: vs, inputs=vs_dropdown, outputs=selected_vs)

    keyword_model_dropdown.change(
        fn=lambda kw_model: kw_model,
        inputs=keyword_model_dropdown,
        outputs=selected_vs,
    )


with gr.Blocks() as home:
    # Create the home interface w/ document ingestion, webpage ingestion, chat interface, and retrieved chunks
    with gr.Row():

        # Create the left bar which contains the document and webpage ingestion
        with gr.Column(elem_classes="left-bar"):
            gr.Markdown("## Documents")
            file = gr.File(
                label="Upload documents",
                file_types=[".txt", ".pdf", "text"],
                file_count="multiple",
            )
            pages = gr.Textbox(
                label="Get a webpage", placeholder="Enter URL(s) separated by commas"
            )
            upload_pages = gr.Button("Submit URLs")

        # Create the right bar which contains the chat interface and retrieved chunks
        with gr.Column(elem_classes="right-bar"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("## RAG Chat Interface")
                    chatbot = gr.Chatbot(height=425, type="messages")
                    with gr.Row():
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Message RosieRAG...",
                            scale=6,
                            submit_btn=True,
                        )

                with gr.Column(scale=2):
                    gr.Markdown("## Retrieved Chunks")
                    chunks = gr.JSON(label="Chunks", value=None)

            # Query the RosieRAG model based on the user's input
            textbox.submit(
                fn=rag_query,
                inputs=[textbox, selected_project, selected_vs, selected_model],
                outputs=[chatbot, chunks, textbox],
            )

        # Add the documents to the knowledge base
        file.change(
            fn=add_documents,
            inputs=[selected_project, selected_vs, selected_model, file],
            outputs=[chatbot, textbox],
        )

        # Add the webpages to the knowledge base
        upload_pages.click(
            fn=add_webpages,
            inputs=[selected_project, selected_vs, selected_model, pages],
            outputs=[chatbot, textbox],
        )

# Create the Gradio interface and connect the home and projects interfaces
with gr.Blocks(title="RosieRAG", css=css) as io:
    gr.TabbedInterface([home, projects], ["Home", "Projects"])

io.queue()
