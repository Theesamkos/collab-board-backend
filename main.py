"""
CollabBoard AI Service v2
--------------------------
Single-layer AI endpoint powered by claude-sonnet-4-6 via LangChain.

Eliminates the brittle regex layer in favour of a single, robust
/api/v2/ai-command endpoint that understands the full range of natural
language whiteboard commands and returns precise tool calls for the
frontend to execute.

LangSmith tracing is enabled automatically when the environment provides:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=<langsmith key>
  LANGCHAIN_PROJECT=collab-board
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CollabBoard AI Service",
    description="claude-sonnet-4-6 powered whiteboard command interpreter.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Tool definitions (OpenAI format; LangChain converts to Anthropic) ─────────

TOOLS: list[dict[str, Any]] = [
    # ── Shape creation ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "createStickyNote",
            "description": "Create a sticky note on the whiteboard. Use for notes, labels, text items.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":   {"type": "string",  "description": "Text content of the note."},
                    "x":      {"type": "number",  "description": "Left edge X position (0–1000)."},
                    "y":      {"type": "number",  "description": "Top edge Y position (0–700)."},
                    "color":  {"type": "string",  "description": "Background color: hex or name."},
                    "width":  {"type": "number",  "description": "Width in pixels (default 200)."},
                    "height": {"type": "number",  "description": "Height in pixels (default 150)."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "createRectangle",
            "description": "Create a rectangle or square shape on the whiteboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x":      {"type": "number", "description": "Left edge X position."},
                    "y":      {"type": "number", "description": "Top edge Y position."},
                    "width":  {"type": "number", "description": "Width in pixels (default 200)."},
                    "height": {"type": "number", "description": "Height in pixels (default 140)."},
                    "color":  {"type": "string", "description": "Fill color: hex or name."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "createCircle",
            "description": "Create a circle or ellipse shape on the whiteboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x":      {"type": "number", "description": "Center X position."},
                    "y":      {"type": "number", "description": "Center Y position."},
                    "radius": {"type": "number", "description": "Radius in pixels (default 60)."},
                    "color":  {"type": "string", "description": "Fill color: hex or name."},
                },
                "required": [],
            },
        },
    },
    # ── Object manipulation ───────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "moveObject",
            "description": "Move an existing object to new absolute coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objectId": {"type": "string", "description": "ID of the object to move."},
                    "x":        {"type": "number", "description": "New X position."},
                    "y":        {"type": "number", "description": "New Y position."},
                },
                "required": ["objectId", "x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deleteObject",
            "description": "Delete a specific object by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objectId": {"type": "string", "description": "ID of the object to delete."},
                },
                "required": ["objectId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "updateText",
            "description": "Update the text content of an existing sticky note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objectId": {"type": "string", "description": "ID of the sticky note."},
                    "text":     {"type": "string", "description": "New text content."},
                },
                "required": ["objectId", "text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "changeColor",
            "description": "Change the fill color of any object (shape or sticky note).",
            "parameters": {
                "type": "object",
                "properties": {
                    "objectId": {"type": "string", "description": "ID of the object."},
                    "color":    {"type": "string", "description": "New color: hex or name."},
                },
                "required": ["objectId", "color"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clearBoard",
            "description": "Remove ALL objects from the board. Use only when the user explicitly asks to clear everything.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    # ── Layout ────────────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "arrangeInGrid",
            "description": "Rearrange all objects on the board into a neat grid layout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {"type": "number", "description": "Number of grid columns (default 3)."},
                    "spacing": {"type": "number", "description": "Pixel gap between objects (default 240)."},
                },
                "required": [],
            },
        },
    },
    # ── Camera / viewport ─────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "setZoom",
            "description": "Control the board zoom level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["in", "out", "reset", "set"],
                        "description": "'in'=zoom in 25%, 'out'=zoom out 25%, 'reset'=100% + center, 'set'=exact level.",
                    },
                    "level": {
                        "type": "number",
                        "description": "Zoom multiplier for action='set' (e.g. 2.0 = 200%). Range: 0.25–4.0.",
                    },
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "panView",
            "description": "Pan (scroll) the board viewport in a direction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right", "up", "down"],
                        "description": "Direction to pan.",
                    },
                    "amount": {
                        "type": "number",
                        "description": "Distance to pan in pixels (default 200).",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fitToView",
            "description": "Zoom and pan to fit all objects on the board into the visible viewport. Use for 'show me everything', 'zoom to fit', 'fit all'.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resetView",
            "description": "Reset viewport to default position: 100% zoom, centered at origin.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ── System prompt factory ─────────────────────────────────────────────────────

def build_system_prompt(board_state: list[dict]) -> str:
    return f"""You are an AI assistant for CollabBoard, a real-time collaborative whiteboard.
Your job is to translate natural language commands into precise tool calls.
Always call tools — never just describe what you would do.

═══ CANVAS ════════════════════════════════════════════════════════════════════
The canvas is approximately 1200×800 pixels.

POSITIONING GUIDE (use these as starting coordinates):
  top-left:     x≈80,  y≈80
  top-center:   x≈550, y≈80
  top-right:    x≈900, y≈80
  center-left:  x≈80,  y≈300
  center:       x≈550, y≈300
  center-right: x≈900, y≈300
  bottom-left:  x≈80,  y≈500
  bottom-center:x≈550, y≈500
  bottom-right: x≈900, y≈500

SPACING:
  "in a row" (horizontal): add 220–240px to x for each successive object
  "in a column" (vertical): add 170–190px to y for each successive object
  "evenly spaced": distribute across the canvas with equal gaps
  Avoid placing objects at the exact same (x, y) — offset each by at least 220px

═══ COLORS ════════════════════════════════════════════════════════════════════
yellow=#FFDD57  red=#EF4444    blue=#3B82F6   green=#22C55E
purple=#8B5CF6  orange=#F97316 pink=#EC4899   teal=#14B8A6
white=#F8FAFC   black=#1F2937  gray=#6B7280   gold=#F59E0B
lime=#84CC16    indigo=#6366F1 coral=#F87171  emerald=#10B981

═══ DEFAULT SIZES ═════════════════════════════════════════════════════════════
  sticky note : 200×150px  (color: yellow #FFDD57)
  rectangle   : 200×140px  (color: blue #3B82F6)
  circle      : radius 60px → 120×120px  (color: green #22C55E)
  "large"     : multiply default by 1.5
  "small"     : multiply default by 0.6

═══ RULES ═════════════════════════════════════════════════════════════════════
- Emit MULTIPLE tool calls in a single response when creating several objects.
- For "3 blue squares in a row": emit 3 × createRectangle with x spacing 220px.
- For "arrange in grid" / "grid layout": use arrangeInGrid.
- For "center the board" / "fit to view" / "show everything": use fitToView.
- For camera/zoom/pan commands: use setZoom or panView.
- For "reset" (view): use resetView.
- Prefer updateText over delete+create when editing existing text content.
- For "clear the board"/"delete everything": use clearBoard (one call only).
- NEVER invent objectIds. Only use IDs present in the board state below.
- When "selected" objects are mentioned, operate on objects where selected=true,
  or on all objects if none are explicitly selected in the board state.

═══ BOARD STATE ═══════════════════════════════════════════════════════════════
{board_state if board_state else "[]"}
"""

# ── Pydantic models ───────────────────────────────────────────────────────────

class AICommandRequest(BaseModel):
    command: str
    board_state: list[dict] = []


class ToolCallResult(BaseModel):
    name: str
    args: dict[str, Any]


class AICommandResponse(BaseModel):
    handler: str = "langchain"
    tool_calls: list[ToolCallResult]
    message: str | None = None


# ── Shared handler ────────────────────────────────────────────────────────────

async def run_ai_command(body: AICommandRequest) -> AICommandResponse:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    llm = ChatAnthropic(
        model="claude-sonnet-4-6",
        temperature=0,
        api_key=api_key,  # type: ignore[arg-type]
    )

    model_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(content=build_system_prompt(body.board_state)),
        HumanMessage(content=body.command),
    ]

    response = await model_with_tools.ainvoke(messages)

    tool_calls = [
        ToolCallResult(name=tc["name"], args=tc.get("args") or {})
        for tc in (response.tool_calls or [])
    ]

    return AICommandResponse(
        handler="langchain",
        tool_calls=tool_calls,
        message=str(response.content) if not tool_calls else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/recognize-intent", response_model=AICommandResponse)
async def recognize_intent(body: AICommandRequest) -> AICommandResponse:
    return await run_ai_command(body)


@app.post("/api/v2/ai-command", response_model=AICommandResponse)
async def ai_command_v2(body: AICommandRequest) -> AICommandResponse:
    return await run_ai_command(body)


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": "2.0.0"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
