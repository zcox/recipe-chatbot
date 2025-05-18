from __future__ import annotations

"""FastAPI application entry-point for the recipe chatbot."""

from pathlib import Path
from typing import Final, List, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.utils import get_agent_response  # noqa: WPS433 import from parent

# -----------------------------------------------------------------------------
# Application setup
# -----------------------------------------------------------------------------

APP_TITLE: Final[str] = "Recipe Chatbot"
app = FastAPI(title=APP_TITLE)

# Serve static assets (currently just the HTML) under `/static/*`.
STATIC_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -----------------------------------------------------------------------------
# Request / response models
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """Schema for a single message in the chat history."""
    role: str = Field(..., description="Role of the message sender (system, user, or assistant).")
    content: str = Field(..., description="Content of the message.")

class ChatRequest(BaseModel):
    """Schema for incoming chat messages."""

    messages: List[ChatMessage] = Field(..., description="The entire conversation history.")


class ChatResponse(BaseModel):
    """Schema for the assistant's reply returned to the front-end."""

    messages: List[ChatMessage] = Field(..., description="The updated conversation history.")


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest) -> ChatResponse:  # noqa: WPS430
    """Main conversational endpoint.

    It proxies the user's message list to the underlying agent and returns the updated list.
    """
    # Convert Pydantic models to simple dicts for the agent
    request_messages: List[Dict[str, str]] = [msg.model_dump() for msg in payload.messages]

    try:
        updated_messages_dicts = get_agent_response(request_messages)
    except Exception as exc:  # noqa: BLE001 broad; surface as HTTP 500
        # In production you would log the traceback here.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    # Convert dicts back to Pydantic models for the response
    response_messages: List[ChatMessage] = [ChatMessage(**msg) for msg in updated_messages_dicts]
    return ChatResponse(messages=response_messages)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:  # noqa: WPS430
    """Serve the chat UI."""

    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Frontend not found. Did you forget to build it?",
        )

    return HTMLResponse(html_path.read_text(encoding="utf-8")) 