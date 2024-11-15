import os
import json
import gradio as gr
from fastapi import FastAPI, status
from fastapi.responses import Response, RedirectResponse


from api.api import api_router
from core.logger import logger
from core.settings import get_settings
from core.requestor import query, add_documents, new_knowledge_base
from core.middleware.token_validator import TokenValidationMiddleware

os.makedirs("./.rosierag", exist_ok=True)

if get_settings().ENVIRONMENT == "local":
    rosie_path = "http://localhost:8080"
else:
    rosie_path = "https://dh-ood.hpc.msoe.edu" + get_settings().BASE_URL + "/"

app = FastAPI(
    title="RosieRAG", root_path=get_settings().BASE_URL, redirect_slashes=True
)
app.add_middleware(TokenValidationMiddleware)

app.include_router(api_router)

css = """
.left-bar {width: 12.5% !important; min-width: 0 !important; flex-grow: 1.1 !important;}
.right-bar {width: 85% !important; flex-grow: 3.5 !important;}
.send-button {position: absolute; z-index: 99; right: 10px; height: 100%; background: none; min-width: 0 !important;}
"""


@api_router.get("/ping/", tags=["admin"])
async def health_check():
    return Response(status_code=status.HTTP_200_OK)


def update_project(project):
    vs_list = [
        i
        for i in os.listdir(f"./.rosierag/{project}")
        if os.path.isdir(f"./.rosierag/{project}/{i}")
        and any(
            fname.endswith(".faiss")
            for fname in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
        )
    ]

    model_list = [
        i
        for i in os.listdir(f"./.rosierag/{project}")
        if os.path.isdir(f"./.rosierag/{project}/{i}")
        and not any(
            fname.endswith(".faiss")
            for fname in os.listdir(f"./.rosierag/{os.listdir('./.rosierag')[0]}/{i}")
        )
    ]

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


def upload_documents(files, project, vs, model):
    try:
        gr.Info(
            f"Starting upload of {len(files)} file(s) to '{project}'/'{vs}+{model}'..."
        )
        add_documents(project=project, vs=vs, model=model, documents=files)
        gr.Info(f"{len(files)} file(s) uploaded successfully.")
    except Exception as e:
        gr.Error(f"Upload failed: {str(e)}")


def rag_query(user_query, project, vs, model):
    responses = query(project=project, vs=vs, model=model, query=user_query)
    text_response = ""

    for token in responses:
        if "<|C|>" in text_response + token:
            res = (text_response + token).replace("<|C|>", "")
            text_response = res
            yield gr.update(
                value=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": text_response},
                ]
            ), gr.update(value=None)
            break
        else:
            text_response += token
            yield gr.update(
                value=[
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": text_response},
                ]
            ), gr.update(value=None)

    json_value = ""
    for token in responses:
        json_value += token

    json_value = json.loads("[" + json_value.replace("}}{", "}},{") + "]")

    yield gr.update(
        value=[
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": text_response},
        ]
    ), gr.update(value=json_value)


def create_knowledge_base(project, vs, model):
    res = new_knowledge_base(project=project, vs=vs, model=model)
    if res["response"] == "Knowledge base created successfully":
        gr.Info(f"Knowledge base '{project}/{vs}+{model}' created successfully.")

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
        gr.Error(f"Error: {res['response']}")

    return None, None, None, None, None, None


with gr.Blocks() as projects:
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
    with gr.Row():
        with gr.Column(elem_classes="left-bar"):
            gr.Markdown("## Documents")
            file = gr.File(
                label="Upload documents",
                file_types=[".txt", ".pdf", "text"],
                file_count="multiple",
            )
            gr.Textbox(label="Get a webpage")

        with gr.Column(elem_classes="right-bar"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("## RAG Chat Interface")
                    chatbot = gr.Chatbot(height=350, type="messages")
                    with gr.Row():
                        textbox = gr.Textbox(
                            show_label=False, placeholder="Message RosieRAG...", scale=6
                        )
                        button = gr.Button(
                            "",
                            elem_classes="send-button",
                            icon="https://uxwing.com/wp-content/themes/uxwing/download/communication-chat-call/send-white-icon.png",
                        )

                with gr.Column(scale=2):
                    gr.Markdown("## Retrieved Chunks")
                    chunks = gr.JSON(label="Chunks", value=None)

            textbox.submit(
                fn=rag_query,
                inputs=[textbox, selected_project, selected_vs, selected_model],
                outputs=[chatbot, chunks],
            )

            button.click(
                fn=rag_query,
                inputs=[textbox, selected_project, selected_vs, selected_model],
                outputs=[chatbot, chunks],
            )

        file.change(
            fn=upload_documents,
            inputs=[file, selected_project, selected_vs, selected_model],
            outputs=chatbot,
        )

with gr.Blocks(css=css) as io:
    gr.TabbedInterface([home, projects], ["Home", "Projects"])


app = gr.mount_gradio_app(
    app,
    io,
    path="/",
    root_path=get_settings().BASE_URL,
    app_kwargs={"redirect_slashes": True},
)


@app.on_event("startup")
async def startup_event():
    logger.info(f"RosieRAG is running at: {rosie_path}")
