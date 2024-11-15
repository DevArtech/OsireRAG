from fastapi import FastAPI, status, Form
from fastapi.responses import Response, HTMLResponse, RedirectResponse, JSONResponse

from api.api import api_router
from core.logger import logger
from core.settings import get_settings
from core.middleware.token_validator import TokenValidationMiddleware, validate_token

app = FastAPI(
    title=get_settings().APP_NAME,
    version=get_settings().APP_VERSION,
    description=get_settings().APP_DESC,
    root_path=get_settings().BASE_URL,
    redirect_slashes=True,
)
app.add_middleware(TokenValidationMiddleware)


@app.get("/ping/", tags=["admin"])
async def health_check():
    return Response(status_code=status.HTTP_200_OK)


@app.get("/login/", tags=["admin"], include_in_schema=False)
async def login_page():
    html_content = f"""
    <html>
        <head>
            <title>Password Input Form</title>
        </head>
        <body>
            <h2>Enter Your Password</h2>
            <form action="{get_settings().BASE_URL}/submit-token" method="post">
                <label for="token">Password:</label>
                <input type="password" id="token" name="token" required>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return HTMLResponse(html_content)


@app.post("/submit-token/", tags=["admin"], include_in_schema=False)
async def submit_token(token: str = Form(...)):
    result = validate_token(token)
    if result:
        response = RedirectResponse(
            url=get_settings().BASE_URL + "/docs", status_code=302
        )
        response.set_cookie(
            key="apitoken",
            value=token,
            max_age=86400,
            secure=True,
            path=get_settings().BASE_URL + "/",
        )
        return response

    return JSONResponse(status_code=401, content={"detail": "Invalid token"})


app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    logger.info(
        f"Running {get_settings().APP_NAME} on: https://dh-ood.hpc.msoe.edu{get_settings().BASE_URL}"
    )
