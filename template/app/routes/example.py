from typing import List
from fastapi import APIRouter
from fastapi.responses import Response, JSONResponse

from core.models import Example

router = APIRouter(prefix="/test", tags=["test"])

example_data: List[Example] = []


@router.get(
    "/get/", responses={200: {"model": List[Example], "description": "Example data"}}
)
async def example_get() -> JSONResponse:
    """
    This is an example GET endpoint that returns example data which has been added, modified, or deleted.
    """
    return JSONResponse(
        status_code=200,
        content={"response": [example.model_dump() for example in example_data]},
    )


@router.post(
    "/post/",
    responses={
        201: {"description": "Example added successfully"},
        400: {"description": "Example already exists."},
    },
)
async def example_post(example: Example) -> JSONResponse:
    """
    This is an example POST endpoint that adds an example to the example data.
    """
    if example.id not in [example.id for example in example_data]:
        example_data.append(example)
    else:
        return JSONResponse(
            status_code=400, content={"response": "Example already exists."}
        )

    return JSONResponse(
        status_code=201, content={"response": "Example added successfully."}
    )


@router.put(
    "/put/",
    responses={
        200: {"description": "Example updated successfully"},
        404: {"description": "Example not found."},
    },
)
async def example_put(example: Example) -> JSONResponse:
    """
    This is an example PUT endpoint that updates an example in the example data.
    """
    for i, ex in enumerate(example_data):
        if ex.id == example.id:
            example_data[i] = example
            return JSONResponse(
                status_code=200, content={"response": "Example updated successfully."}
            )

    return JSONResponse(404, content={"response": "Example not found."})


@router.delete(
    "/delete/",
    responses={
        204: {"description": "Example deleted successfully"},
        404: {"description": "Example not found."},
    },
)
async def example_delete(id: int) -> Response:
    """
    This is an example DELETE endpoint that deletes an example from the example data.
    """
    found = False
    for i, ex in enumerate(example_data):
        if ex.id == id:
            example_data.pop(i)
            return Response(status_code=204)

    return JSONResponse(status_code=404, content={"response": "Example not found."})
