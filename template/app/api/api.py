from fastapi import APIRouter
from fastapi.responses import JSONResponse

from routes import example

api_router = APIRouter()


# This is both a GET and a POST because the Rosie OOD performs a POST request by
# default to supply the API token in the body, not in the query parameters, including
# GET requests allows for this to be flexible for if the user refreshes the page.
@api_router.api_route("/", methods=["GET", "POST"], include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        status_code=200, content={"message": f"Welcome to the Rosie FastAPI Template"}
    )


api_router.include_router(example.router)
