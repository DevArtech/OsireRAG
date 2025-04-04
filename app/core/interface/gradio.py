"""
Module: gradio.py

This module is responsible for creating and managing the Gradio interface for the OsireRAG application.

Classes:
- None

Functions:
- update_project: Updates the project, vectorstore, and model dropdowns based on the selected project.
- rag_query: Queries the OsireRAG model and returns the response.
- create_knowledge_base: Creates a new knowledge base in OsireRAG.
- refresh_all: Refreshes the project, vectorstore, and model dropdowns.
- clear_conversation: Clears the conversation history, chatbot display, and chunks display.

Attributes:
- css: The CSS styling for the Gradio interface.
- io: The Gradio interface object.

Author: Adam Haile  
Date: 11/25/2024
"""

import os
import json
import gradio as gr
from app.core.settings import get_settings
from typing import Tuple, Iterator, List, Dict

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

/* Custom checkbox styling */
.rerank-checkbox input[type="checkbox"] {
    width: 1.5rem !important;
    height: 1.5rem !important;
    margin-top: 1.75rem !important;
}

.rerank-checkbox span {
    padding-top: 1.75rem !important;
    font-size: 1rem !important;
}
"""


def update_project(project: str) -> Tuple[str, gr.update, gr.update, str, str]:
    """
    Updates the project, vectorstore, and model dropdowns based on the selected project.

    Args:
    - `project (str)`: The selected project.

    Returns:
    - str: The selected project.
    - gr.update: The updated vectorstore dropdown.
    - gr.update: The updated model dropdown.
    - str: The selected vectorstore.
    - str: The selected model.

    Usage:
    - `update_project(project)`

    Author: Adam Haile
    Date: 11/25/2024
    """

    try:
        # Collect the vectorstores for the given project
        vs_list = []
        if os.path.exists(f"./.osirerag/{project}"):
            vs_list = [
                i
                for i in os.listdir(f"./.osirerag/{project}")
                if os.path.isdir(f"./.osirerag/{project}/{i}")
                and any(
                    fname.endswith(".faiss")
                    for fname in os.listdir(f"./.osirerag/{project}/{i}")
                    if os.path.isdir(f"./.osirerag/{project}/{i}")
                )
            ]

        # Collect the keyword models for the given project
        model_list = []
        if os.path.exists(f"./.osirerag/{project}"):
            model_list = [
                i
                for i in os.listdir(f"./.osirerag/{project}")
                if os.path.isdir(f"./.osirerag/{project}/{i}")
                and not any(
                    fname.endswith(".faiss")
                    for fname in os.listdir(f"./.osirerag/{project}/{i}")
                    if os.path.isdir(f"./.osirerag/{project}/{i}")
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
    except Exception as e:
        gr.Error(f"Error updating project: {str(e)}")
        return project, gr.update(choices=[]), gr.update(choices=[]), None, None


def rag_query(
    user_query: str,
    project: str,
    vs: str,
    model: str,
    conversation_history: List[Dict[str, str]],
    n_results: int = 10,
    rerank: bool = True,
    system_prompt: str = "",
    temperature: float = 0.7,
    allow_no_results: bool = True,
    threshold: float = None
) -> Iterator[Tuple[gr.update, gr.update, gr.update, None]]:
    """
    Queries the OsireRAG model and returns the response.

    Args:
    - `user_query (str)`: The user's query.
    - `project (str)`: The selected project.
    - `vs (str)`: The selected vectorstore.
    - `model (str)`: The selected model.
    - `conversation_history (List[Dict[str, str]])`: The conversation history.
    - `n_results (int)`: Number of results to return from search.
    - `rerank (bool)`: Whether to rerank the search results.
    - `system_prompt (str)`: An optional system prompt to prepend to the conversation.
    - `temperature (float)`: The temperature for the model's generation.
    - `allow_no_results (bool)`: Whether to allow an empty response if no documents are found.
    - `threshold (float)`: Minimum similarity score for retrieved documents (None = no threshold).

    Returns:
    - gr.update: The updated chatbot interface.
    - gr.update: The updated chunks interface.
    - gr.update: The updated conversation history.
    - None

    Usage:
    - `rag_query(user_query, project, vs, model, conversation_history, n_results, rerank, system_prompt, temperature, allow_no_results, threshold)`

    Author: Adam Haile
    Date: 11/25/2024
    """
    # Add the user's query to the conversation history
    conversation_history.append({"role": "user", "content": user_query})

    # Prepare history for backend query, including system prompt if provided
    backend_history = conversation_history.copy()
    if system_prompt and system_prompt.strip():
        backend_history.insert(0, {"role": "system", "content": system_prompt.strip()})

    # Query the OsireRAG model
    response = query(
        project=project,
        vs=vs,
        model=model,
        query=user_query,
        history=backend_history,
        n_results=n_results,
        rerank=rerank,
        temperature=temperature,
        allow_no_results=allow_no_results,
        threshold=threshold
    )

    # Process the response
    assistant_response = ""
    for token in response:

        # Check if the response contains a token identifying the end of the LLM response
        # and the start of the chunks
        if "<|C|>" in assistant_response + token:

            # Remove the token and return the response
            res = (assistant_response + token).replace("<|C|>", "")
            assistant_response = res

            if (
                not conversation_history
                or conversation_history[-1]["role"] != "assistant"
            ):
                conversation_history.append({"role": "assistant", "content": ""})

            conversation_history[-1]["content"] = assistant_response

            yield gr.update(value=conversation_history), gr.update(
                value=None
            ), gr.update(value=conversation_history), None
            break
        else:
            # Append the token to the response
            assistant_response += token

            if (
                not conversation_history
                or conversation_history[-1]["role"] != "assistant"
            ):
                conversation_history.append({"role": "assistant", "content": ""})

            conversation_history[-1]["content"] = assistant_response

            yield gr.update(value=conversation_history), gr.update(
                value=None
            ), gr.update(value=conversation_history), None

    json_value = ""
    for token in response:
        json_value += token

    # Fix the JSON formatting
    json_value = json.loads("[" + json_value.replace("}}{", "}},{") + "]")

    yield gr.update(value=conversation_history), gr.update(value=json_value), gr.update(
        value=conversation_history
    ), None


def clear_conversation() -> Tuple[List, None, None, None, None, None]:
    """
    Clears the conversation history, chatbot display, chunks display,
    and input fields (textbox, file upload, URL input).

    Returns:
    - List: An empty list for the conversation history.
    - None: None to clear the chatbot display.
    - None: None to clear the chunks display.
    - None: None to clear the chat input textbox.
    - None: None to clear the file input.
    - None: None to clear the URL input textbox.
    """
    return [], None, None, None, None, None


def create_knowledge_base(
    project: str, vs: str, model: str
) -> Tuple[gr.update, gr.update, gr.update, str, str, str]:
    """
    Creates a new knowledge base.

    Args:
    - `project (str)`: The name of the project.
    - `vs (str)`: The name of the vectorstore.
    - `model (str)`: The name of the model.

    Returns:
    - gr.update: The updated project dropdown.
    - gr.update: The updated vectorstore dropdown.
    - gr.update: The updated model dropdown.
    - str: The selected project.
    - str: The selected vectorstore.
    - str: The selected model.

    Usage:
    - `create_knowledge_base(project, vs, model)`

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
        project_update = gr.update(choices=os.listdir("./.osirerag"), value=project)
        
        # Fix: Check if directories exist before attempting to list them
        try:
            vs_paths = []
            if os.path.exists(f"./.osirerag/{project}"):
                vs_paths = [
                    i
                    for i in os.listdir(f"./.osirerag/{project}")
                    if os.path.isdir(f"./.osirerag/{project}/{i}")
                    and any(
                        fname.endswith(".faiss")
                        for fname in os.listdir(f"./.osirerag/{project}/{i}")
                        if os.path.isdir(f"./.osirerag/{project}/{i}")
                    )
                ]
            vs_update = gr.update(choices=vs_paths, value=vs)
            
            model_paths = []
            if os.path.exists(f"./.osirerag/{project}"):
                model_paths = [
                    i
                    for i in os.listdir(f"./.osirerag/{project}")
                    if os.path.isdir(f"./.osirerag/{project}/{i}")
                    and not any(
                        fname.endswith(".faiss")
                        for fname in os.listdir(f"./.osirerag/{project}/{i}")
                        if os.path.isdir(f"./.osirerag/{project}/{i}")
                    )
                ]
            model_update = gr.update(choices=model_paths, value=model)
            
            return project_update, vs_update, model_update, project, vs, model
        except Exception as e:
            gr.Error(f"Error updating dropdowns: {str(e)}")
            return project_update, gr.update(), gr.update(), project, vs, model
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
    - gr.update: The updated project dropdown.
    - gr.update: The updated vectorstore dropdown.
    - gr.update: The updated model dropdown.
    - gr.update: The selected project.
    - gr.update: The selected vectorstore.
    - gr.update: The selected model.

    Usage:
    - `refresh_all()`

    Author: Adam Haile
    Date: 11/25/2024
    """
    # Get the projects, vectorstores, and models
    projects = os.listdir("./.osirerag")
    vs = (
        [
            i
            for i in os.listdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}")
            if os.path.isdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}")
            and any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.osirerag")) > 0
        else []
    )
    models = (
        [
            i
            for i in os.listdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}")
            if os.path.isdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}")
            and not any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.osirerag")) > 0
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
            os.listdir("./.osirerag")[0] if len(os.listdir("./.osirerag")) > 0 else None
        )
    )
    vss = (
        [
            i
            for i in os.listdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}")
            if os.path.isdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}")
            and any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.osirerag")) > 0
        else []
    )
    selected_vs = gr.State(value=vss[0] if vss else None)
    models = (
        [
            i
            for i in os.listdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}")
            if os.path.isdir(f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}")
            and not any(
                fname.endswith(".faiss")
                for fname in os.listdir(
                    f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                )
            )
        ]
        if len(os.listdir("./.osirerag")) > 0
        else []
    )
    selected_model = gr.State(value=models[0] if models else None)

    # Create the project, vectorstore, and model interface with dropdowns
    with gr.Row():
        with gr.Column():
            with gr.Row():
                project_dropdown = gr.Dropdown(
                    choices=os.listdir("./.osirerag"), label="Projects"
                )
                vs_dropdown = gr.Dropdown(
                    choices=(
                        [
                            i
                            for i in os.listdir(
                                f"./.osirerag/{os.listdir('./.osirerag')[0]}"
                            )
                            if os.path.isdir(
                                f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                            )
                            and any(
                                fname.endswith(".faiss")
                                for fname in os.listdir(
                                    f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                                )
                            )
                        ]
                        if len(os.listdir("./.osirerag")) > 0
                        else []
                    ),
                    label="Vectorstores",
                )
                keyword_model_dropdown = gr.Dropdown(
                    choices=(
                        [
                            i
                            for i in os.listdir(
                                f"./.osirerag/{os.listdir('./.osirerag')[0]}"
                            )
                            if os.path.isdir(
                                f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                            )
                            and not any(
                                fname.endswith(".faiss")
                                for fname in os.listdir(
                                    f"./.osirerag/{os.listdir('./.osirerag')[0]}/{i}"
                                )
                            )
                        ]
                        if len(os.listdir("./.osirerag")) > 0
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
        outputs=selected_model,
    )

with gr.Blocks() as params:
    sentences = gr.State(value=7)
    chunk_len = gr.State(value=10000)
    chunk_overlap = gr.State(value=50)
    k1 = gr.State(value=1.5)
    b = gr.State(value=0.75)
    epsilon = gr.State(value=0.25)
    n_results = gr.State(value=10)
    rerank = gr.State(value=True)
    system_prompt = gr.State(value="") # Default system prompt
    temperature = gr.State(value=0.7)  # Default temperature
    allow_no_results = gr.State(value=True) # Default allow_no_results
    threshold = gr.State(value=None) # Default threshold value for RRF
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                "## Parameters\nThese are additional parameters you can modify for your OsireRAG model."
            )
            
            gr.Markdown("### Ingestion Parameters")
            with gr.Row():
                sentences_in = gr.Number(value=7, label="Number of Sentences per Chunk")
                chunk_len_in = gr.Number(value=10000, label="Chunk Length")
                chunk_overlap_in = gr.Number(value=50, label="Chunk Overlap")
            with gr.Row():
                k1_in = gr.Number(
                    value=1.5, label="k1 - Term Frequency Saturation", step=0.01
                )
                b_in = gr.Number(
                    value=0.75, label="b - Length Normalization", step=0.01
                )
                epsilon_in = gr.Number(
                    value=0.25, label="epsilon - Smoothing Parameter for IDF", step=0.01
                )
            
            gr.Markdown("### Search Parameters") 
            with gr.Row():
                n_results_in = gr.Number(
                    value=10,
                    label="Number of Results",
                    step=1,
                    minimum=1
                )
                rerank_in = gr.Checkbox(
                    value=True,
                    label="Rerank Results",
                    elem_classes=["rerank-checkbox"]
                )
                allow_no_results_in = gr.Checkbox(
                    value=True,
                    label="Allow No Results",
                    elem_classes=["rerank-checkbox"] # Reuse style for alignment
                )
            
            gr.Markdown("### System Prompt")
            with gr.Row():
                system_prompt_in = gr.Textbox(
                    label="System Prompt",
                    placeholder="Your LLM system prompt here...",
                    value="You are a knowledgeable AI assistant with access to relevant background information. Directly answer the user's question. Be concise and to the point.",
                    lines=3
                )

    # Bind state updates
    sentences_in.change(lambda s: s, inputs=sentences_in, outputs=sentences)
    chunk_len_in.change(lambda c: c, inputs=chunk_len_in, outputs=chunk_len)
    chunk_overlap_in.change(lambda o: o, inputs=chunk_overlap_in, outputs=chunk_overlap)
    k1_in.change(lambda k: k, inputs=k1_in, outputs=k1)
    b_in.change(lambda b: b, inputs=b_in, outputs=b)
    epsilon_in.change(lambda e: e, inputs=epsilon_in, outputs=epsilon)
    n_results_in.change(lambda n: n, inputs=n_results_in, outputs=n_results)
    rerank_in.change(lambda r: r, inputs=rerank_in, outputs=rerank)
    system_prompt_in.change(lambda sp: sp, inputs=system_prompt_in, outputs=system_prompt)
    allow_no_results_in.change(lambda anr: anr, inputs=allow_no_results_in, outputs=allow_no_results) # Bind state update

    gr.Markdown("### Generation Parameters")
    with gr.Row():
        temperature_in = gr.Slider(minimum=0.0, maximum=2.0, step=0.1, value=0.7, label="Temperature")
        threshold_in = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=None, label="Similarity Threshold", info="Minimum similarity score for retrieved documents (0 = no threshold)")

    # Bind state updates
    temperature_in.change(lambda t: t, inputs=temperature_in, outputs=temperature)
    threshold_in.change(lambda t: t, inputs=threshold_in, outputs=threshold)

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
            clear_btn = gr.Button("Clear")

        # Create the right bar which contains the chat interface and retrieved chunks
        with gr.Column(elem_classes="right-bar"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("## RAG Chat Interface")
                    chatbot = gr.Chatbot(height=425, type="messages", latex_delimiters=[{"left": "$", "right": "$", "display": True}])

                    # Conversation history state
                    conversation_history = gr.State(value=[])

                    with gr.Row():
                        textbox = gr.Textbox(
                            show_label=False,
                            placeholder="Message OsireRAG...",
                            scale=6,
                            submit_btn=True,
                        )

                with gr.Column(scale=2):
                    gr.Markdown("## Retrieved Chunks")
                    chunks = gr.JSON(label="Chunks", value=None)

            # Query the OsireRAG model based on the user's input
            textbox.submit(
                fn=rag_query,
                inputs=[
                    textbox,
                    selected_project,
                    selected_vs,
                    selected_model,
                    conversation_history,
                    n_results,
                    rerank,
                    system_prompt,
                    temperature,
                    allow_no_results,
                    threshold
                ],
                outputs=[chatbot, chunks, conversation_history, textbox],
            )

            # Wire the clear button
            clear_btn.click(
                fn=clear_conversation,
                inputs=[],
                outputs=[conversation_history, chatbot, chunks, textbox, file, pages],
                show_progress=False
            )

        # Add the documents to the knowledge base
        file.change(
            fn=add_documents,
            inputs=[
                selected_project,
                selected_vs,
                selected_model,
                file,
                sentences,
                chunk_len,
                chunk_overlap,
                k1,
                b,
                epsilon,
            ],
            outputs=[chatbot, textbox, file, pages, upload_pages],
        )

        # Add the webpages to the knowledge base
        upload_pages.click(
            fn=add_webpages,
            inputs=[
                selected_project,
                selected_vs,
                selected_model,
                pages,
                sentences,
                chunk_len,
                chunk_overlap,
                k1,
                b,
                epsilon,
            ],
            outputs=[chatbot, textbox, file, pages, upload_pages],
        )

# Create the Gradio interface and connect the home and projects interfaces
with gr.Blocks(title="OsireRAG", css=css, head=f"<link rel='stylesheet' href='https://dh-ood.hpc.msoe.edu{get_settings().BASE_URL}/theme.css?token={get_settings().API_TOKEN}'>") as io:
    gr.Markdown("## <div style='display: flex; align-items: center; gap: 0.5rem;'><img src='https://i.imgur.com/zWuIP3A.png' width='64' height='64'> OsireRAG</div>")
    gr.TabbedInterface([home, projects, params], ["Home", "Projects", "Parameters"])

io.queue()
