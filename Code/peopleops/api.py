from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .chat import handle_chat
from .config import load_settings
from .database import connect, establish_session, employee, initialize_state
from .documents import build_index
from .llm import OpenAIAnswerer, ProviderError
from .mcp_client import MCPToolError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("uvicorn.access").disabled = True
logger = logging.getLogger("peopleops.api")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)


def _validate_database(db_path) -> None:
    expected = {
        "employees": {"employee_id", "full_name", "email", "phone", "region", "join_date", "level", "role", "manager_id"},
        "leave_balances": {"employee_id", "leave_type", "balance_days"},
        "leave_requests": {"request_id", "employee_id", "leave_type", "start_date", "end_date", "working_days", "status", "created_at"},
    }
    with connect(db_path) as connection:
        for table, columns in expected.items():
            actual = {
                row["name"] for row in connection.execute(f"PRAGMA table_info({table})")
            }
            if not columns.issubset(actual):
                raise RuntimeError(f"Required database table schema is missing: {table}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = load_settings()
    app.state.ready = False
    app.state.answerer = OpenAIAnswerer(app.state.settings)

    async def prepare() -> None:
        settings = app.state.settings
        await asyncio.to_thread(initialize_state, settings.db_path)
        await asyncio.to_thread(_validate_database, settings.db_path)
        await asyncio.to_thread(
            build_index, settings.manifest_path, settings.corpus_dir, settings.index_path
        )
        app.state.ready = True
        logger.info("service_ready documents_indexed")

    app.state.prepare_task = asyncio.create_task(prepare())
    yield
    if not app.state.prepare_task.done():
        app.state.prepare_task.cancel()
        try:
            await app.state.prepare_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Veltrix PeopleOps Assistant", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, __: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "Missing or invalid request body."})


@app.exception_handler(ProviderError)
async def provider_failure(_: Request, __: ProviderError) -> JSONResponse:
    logger.error("infrastructure_failure component=provider")
    return JSONResponse(
        status_code=503,
        content={"detail": "The assistant is temporarily unavailable; please retry."},
    )


@app.exception_handler(MCPToolError)
async def mcp_failure(_: Request, __: MCPToolError) -> JSONResponse:
    logger.error("infrastructure_failure component=mcp")
    return JSONResponse(
        status_code=503,
        content={"detail": "The assistant is temporarily unavailable; please retry."},
    )


@app.exception_handler(Exception)
async def unexpected_failure(_: Request, __: Exception) -> JSONResponse:
    logger.error("infrastructure_failure component=application")
    return JSONResponse(
        status_code=503,
        content={"detail": "The assistant is temporarily unavailable; please retry."},
    )


@app.get("/health")
async def health(request: Request) -> JSONResponse:
    if not getattr(request.app.state, "ready", False):
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/chat")
async def chat(
    payload: ChatRequest,
    request: Request,
    x_employee_id: str | None = Header(default=None, alias="X-Employee-Id"),
) -> JSONResponse:
    if not x_employee_id or len(x_employee_id) > 100:
        return JSONResponse(status_code=401, content={"detail": "Valid X-Employee-Id required."})
    settings = request.app.state.settings
    if employee(settings.db_path, x_employee_id) is None:
        return JSONResponse(status_code=401, content={"detail": "Valid X-Employee-Id required."})
    if not request.app.state.ready:
        return JSONResponse(status_code=503, content={"detail": "The assistant is not ready."})
    if not establish_session(settings.db_path, payload.session_id, x_employee_id):
        return JSONResponse(status_code=403, content={"detail": "Session belongs to another employee."})
    try:
        response = await asyncio.wait_for(
            handle_chat(
                session_id=payload.session_id,
                employee_id=x_employee_id,
                message=payload.message,
                settings=settings,
                answerer=request.app.state.answerer,
            ),
            timeout=settings.request_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("infrastructure_failure component=request_timeout")
        return JSONResponse(
            status_code=503,
            content={"detail": "The assistant could not finish within the request time limit."},
        )
    return JSONResponse(
        status_code=200,
        content={
            "request_id": str(uuid4()),
            "answer": response.answer,
            "route": response.route,
            "citations": response.citations,
            "tool_calls": response.tool_calls,
            "pending_confirmation": response.pending_confirmation,
        },
    )


def run() -> None:
    import uvicorn

    uvicorn.run("peopleops.api:app", host="0.0.0.0", port=8000, access_log=False)


if __name__ == "__main__":
    run()
