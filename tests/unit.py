import io
import os
import sys
import uuid
import pytest
import shutil
from fastapi import HTTPException, UploadFile

# Set environment variables and path
os.environ["PREFIX"] = "rr"
os.environ["API_KEY"] = "z2lYZ846AE0AqW48M6oIdTHcgPPPWimf"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

proj = str(uuid.uuid4())

from src import api

# from src.routes import documents, web
# from src.models.documents import Document


def test_extract_params():
    injectible = "/documents/{project_name}/retrieve/{file_name}"
    injected = "/documents/test/retrieve/test.txt"
    res_injectible, res_injected, res_params = api.extract_params(injectible, injected)
    assert res_injectible == injected
    assert res_injected == injected
    assert res_params == {"project_name": "test", "file_name": "test.txt"}


# def test_validate_token():
#     assert api.validate_token(api.api_token) == api.api_token


# def test_validate_token_fails():
#     with pytest.raises(HTTPException) as exc_info:
#         api.validate_token("rr-1234567890")

#     assert exc_info.value.status_code == 401
#     assert exc_info.value.detail == "Invalid APIKey"


# @pytest.mark.asyncio
# async def test_list_documents_no_project():
#     response = await documents.list_documents()
#     assert response == {"response": "No project loaded."}


# @pytest.mark.asyncio
# async def test_load_documents():
#     response = await documents.load_documents(proj)
#     assert response == {"response": f"Project {proj} loaded successfully."}


# @pytest.mark.asyncio
# async def test_list_documents():
#     await documents.load_documents(proj)
#     response = await documents.list_documents()
#     assert response == {"response": []}


# @pytest.mark.asyncio
# async def test_upload_documents():
#     await documents.load_documents(proj)
#     upload_file = UploadFile(
#         filename="test.txt", file=io.BytesIO(b"This is a test file.")
#     )

#     response = await documents.upload_documents(upload_file)
#     assert response == {"response": "Document uploaded successfully."}


# @pytest.mark.asyncio
# async def test_upload_documents_no_project():
#     documents.documents.project_name = None
#     upload_file = UploadFile(
#         filename="test.txt", file=io.BytesIO(b"This is a test file.")
#     )

#     response = await documents.upload_documents(upload_file)
#     assert response == {"response": "No project loaded."}


# @pytest.mark.asyncio
# async def test_list_uploaded_documents():
#     await documents.load_documents(proj)
#     upload_file = UploadFile(
#         filename="test.txt", file=io.BytesIO(b"This is a test file.")
#     )
#     await documents.upload_documents(upload_file)

#     response = await documents.list_documents()
#     assert (
#         response["response"][0].directory
#         == str(os.getcwd()).replace("\\", "/") + "/.rosierag/" + proj + "/test.txt"
#     )


# @pytest.mark.asyncio
# async def test_retrieve_document():
#     await documents.load_documents(proj)
#     upload_file = UploadFile(
#         filename="test.txt", file=io.BytesIO(b"This is a test file.")
#     )
#     await documents.upload_documents(upload_file)

#     response = await documents.retrieve_document(0)
#     assert (
#         response["response"]["directory"]
#         == str(os.getcwd()).replace("\\", "/") + "/.rosierag/" + proj + "/test.txt"
#     )
#     assert response["response"]["content"] == "This is a test file."


# @pytest.mark.asyncio
# async def test_retrieve_document_no_project():
#     documents.documents.project_name = None
#     response = await documents.retrieve_document(0)
#     assert response == {"response": "No project loaded."}


# @pytest.mark.asyncio
# async def test_delete_document():
#     await documents.load_documents(proj)
#     upload_file = UploadFile(
#         filename="test.txt", file=io.BytesIO(b"This is a test file.")
#     )
#     await documents.upload_documents(upload_file)

#     response = await documents.delete_documents(
#         [
#             Document(
#                 directory=str(os.getcwd()).replace("\\", "/")
#                 + "/.rosierag/"
#                 + proj
#                 + "/test.txt"
#             )
#         ]
#     )
#     assert response == {"response": "Documents deleted successfully."}


# @pytest.mark.asyncio
# async def test_delete_documents_no_project():
#     documents.documents.project_name = None
#     response = await documents.delete_documents(
#         [
#             Document(
#                 directory=str(os.getcwd()).replace("\\", "/")
#                 + "/.rosierag/"
#                 + proj
#                 + "/test.txt"
#             )
#         ]
#     )
#     assert response == {"response": "No project loaded."}


# @pytest.mark.asyncio
# async def test_upload_web():
#     web.documents.project_name = proj
#     response = await web.upload_web(["https://www.google.com"])
#     assert response == {"response": "1 web pages uploaded successfully."}


# @pytest.mark.asyncio
# async def test_upload_web_no_project():
#     web.documents.project_name = None
#     response = await web.upload_web(["https://www.google.com"])
#     assert response == {"response": "No project loaded."}


# @pytest.mark.asyncio
# async def test_upload_web_fails():
#     web.documents.project_name = proj
#     with pytest.raises(ValueError) as exc_info:
#         await web.upload_web(["https://www.google.com/404"])

#     assert (
#         str(exc_info.value) == "Failed to download page https://www.google.com/404: 404"
#     )


# @pytest.fixture(scope="session", autouse=True)
# def cleanup():
#     yield  # This will run the tests first
#     project_dir = os.path.join(os.getcwd(), ".rosierag", proj)
#     if os.path.exists(project_dir):
#         shutil.rmtree(project_dir)
#     print(f"Cleaned up {project_dir}")
