"""
CollabBoard AI Service v4
--------------------------
Single-layer AI endpoint powered by claude-sonnet-4-6 via LangChain.

# Version 4.0.0 - Added summarizeBoard feature

v4 adds: summarizeBoard (AI-generated board analysis with key points, risks, action items)
Full board state is accepted and injected into the system prompt so Claude
can target specific objects by ID.

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
    version="4.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Tool definitions ──────────────────────────────────────────────────────────

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
                    "text":   {"type": "string", "description": "Optional label text inside the rectangle."},
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
            "name": "updateObject",
            "description": (
                "Update one or more properties of an existing object. "
                "Use for: 'make the circle red', 'resize the rectangle', "
                "'move the sticky note', 'change the selected item to blue'. "
                "Only include properties you want to change."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objectId": {"type": "string", "description": "ID of the object to update (from board state)."},
                    "color":    {"type": "string", "description": "New fill color (hex or name). Omit if not changing."},
                    "text":     {"type": "string", "description": "New text content. Omit if not changing."},
                    "x":        {"type": "number", "description": "New X position. Omit if not moving."},
                    "y":        {"type": "number", "description": "New Y position. Omit if not moving."},
                    "width":    {"type": "number", "description": "New width in pixels. Omit if not resizing."},
                    "height":   {"type": "number", "description": "New height in pixels. Omit if not resizing."},
                },
                "required": ["objectId"],
            },
        },
    },
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
            "name": "deleteObjects",
            "description": (
                "Delete one or more objects by their IDs. "
                "Use for: 'delete the yellow rectangle', 'remove all circles', "
                "'clear the selected items'. Emit one call with all IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objectIds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of object IDs to delete.",
                    },
                },
                "required": ["objectIds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "updateText",
            "description": "Update the text content of an existing object.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objectId": {"type": "string", "description": "ID of the object."},
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
            "description": "Remove ALL objects from the board. Use only when explicitly asked to clear everything.",
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
    {
        "type": "function",
        "function": {
            "name": "alignObjects",
            "description": (
                "Align a set of objects along an axis. "
                "Use for: 'align all circles to the left', 'center the sticky notes vertically', "
                "'align the rectangles to the top'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "alignment": {
                        "type": "string",
                        "enum": ["left", "right", "top", "bottom", "center-x", "center-y"],
                        "description": (
                            "Alignment edge/axis. "
                            "left/right align the left or right edges; "
                            "top/bottom align the top or bottom edges; "
                            "center-x centers horizontally; center-y centers vertically."
                        ),
                    },
                    "objectIds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of objects to align. If omitted, applies to all non-connector objects.",
                    },
                },
                "required": ["alignment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distributeObjects",
            "description": (
                "Distribute objects with even spacing between them. "
                "Use for: 'distribute the rectangles horizontally', 'space the sticky notes evenly'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["horizontal", "vertical"],
                        "description": "horizontal = even x-spacing; vertical = even y-spacing.",
                    },
                    "objectIds": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of objects to distribute. If omitted, applies to all.",
                    },
                },
                "required": ["direction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "createTemplate",
            "description": (
                "Create a pre-built multi-object template. "
                "Use for: 'SWOT analysis', 'kanban board', 'user journey map', "
                "'decision matrix', 'Eisenhower matrix'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "templateType": {
                        "type": "string",
                        "enum": ["swot", "kanban", "userJourney", "decisionMatrix"],
                        "description": (
                            "swot = 4-quadrant SWOT analysis grid; "
                            "kanban = 3-column board (To Do / In Progress / Done); "
                            "userJourney = 5-stage awareness→advocacy map; "
                            "decisionMatrix = Eisenhower 2x2 urgency/importance grid."
                        ),
                    },
                    "x": {"type": "number", "description": "Starting X position (default 80)."},
                    "y": {"type": "number", "description": "Starting Y position (default 80)."},
                },
                "required": ["templateType"],
            },
        },
    },
    # ── Board analysis ────────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "summarizeBoard",
            "description": (
                "Analyze the current board content and generate a structured summary. "
                "Use when the user asks: 'summarize the board', 'what are the action items?', "
                "'give me a summary', 'analyze this board', '/summarize'. "
                "Read all object text from the board state, then fill in the summary fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "A concise title (5–10 words) capturing the board's overall purpose.",
                    },
                    "key_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3–5 main ideas or themes extracted from the board content.",
                    },
                    "risks": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "2–3 risks, concerns, or challenges visible on the board.",
                    },
                    "action_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "3–5 concrete next steps derived from the board content.",
                    },
                },
                "required": ["title", "key_points", "risks", "action_items"],
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
            "description": "Zoom and pan to fit all objects into the visible viewport.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resetView",
            "description": "Reset viewport to default: 100% zoom, centered at origin.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# ── System prompt factory ─────────────────────────────────────────────────────

def format_board_state(board_state: list[dict]) -> str:
    if not board_state:
        return "The board is empty."
    lines = []
    for obj in board_state:
        parts = [
            f"id={obj.get('id','?')}",
            f"type={obj.get('type','?')}",
            f"x={obj.get('x',0):.0f}",
            f"y={obj.get('y',0):.0f}",
            f"w={obj.get('width',0):.0f}",
            f"h={obj.get('height',0):.0f}",
        ]
        if obj.get("color"):
            parts.append(f"color={obj['color']}")
        if obj.get("text"):
            text_preview = str(obj["text"])[:40].replace("\n", " ")
            parts.append(f'text="{text_preview}"')
        if obj.get("selected"):
            parts.append("SELECTED")
        lines.append("  " + "  ".join(parts))
    return "\n".join(lines)


def build_system_prompt(board_state: list[dict]) -> str:
    return f"""You are an AI assistant for CollabBoard, a real-time collaborative whiteboard.
Translate natural language commands into precise tool calls. Always call tools — never just describe.

═══ CANVAS ════════════════════════════════════════════════════════════════════
Canvas size: ~1200×800 pixels.

POSITIONING GUIDE:
  top-left:     x≈80,  y≈80    top-center:   x≈550, y≈80    top-right:  x≈900, y≈80
  center-left:  x≈80,  y≈300   center:       x≈550, y≈300   center-right: x≈900, y≈300
  bottom-left:  x≈80,  y≈500   bottom-center: x≈550, y≈500  bottom-right: x≈900, y≈500

SPACING: "in a row" → +220px x per item | "in a column" → +180px y per item

═══ COLORS ════════════════════════════════════════════════════════════════════
yellow=#FFDD57  red=#EF4444    blue=#3B82F6   green=#22C55E
purple=#8B5CF6  orange=#F97316 pink=#EC4899   teal=#14B8A6
white=#F8FAFC   black=#1F2937  gray=#6B7280   gold=#F59E0B
lime=#84CC16    indigo=#6366F1 coral=#F87171  emerald=#10B981

═══ DEFAULT SIZES ═════════════════════════════════════════════════════════════
  sticky note: 200×150   rectangle: 200×140   circle radius: 60
  "large" = ×1.5         "small" = ×0.6

═══ COMMAND RULES ═════════════════════════════════════════════════════════════
CREATION:
- Emit MULTIPLE tool calls for multiple objects (3 circles → 3 × createCircle, x+220px each).
- Rectangles and circles can carry text via the text field.

MANIPULATION (uses board state object IDs below):
- "change the selected object to red" → updateObject for objects marked SELECTED.
- "make all circles bigger" → emit multiple updateObject calls (one per circle id), width/height ×1.5.
- "delete the yellow rectangle" → deleteObjects with the matching id.
- "move the top sticky note down" → updateObject with new y.
- NEVER invent IDs. Only use IDs present in the board state.
- When "selected" is mentioned, operate on SELECTED objects only.
- When a type/color filter is mentioned ("all circles", "the red rectangle"), match from board state.

LAYOUT:
- "arrange in grid" / "grid layout" → arrangeInGrid.
- "align left/right/top/bottom" → alignObjects with objectIds from board state.
- "distribute evenly / space out" → distributeObjects with objectIds.

TEMPLATES (createTemplate tool — one call, frontend renders multi-object template):
- "SWOT analysis" / "SWOT board" → createTemplate templateType=swot
- "kanban" / "kanban board" → createTemplate templateType=kanban
- "user journey" / "journey map" → createTemplate templateType=userJourney
- "decision matrix" / "Eisenhower matrix" → createTemplate templateType=decisionMatrix

CAMERA:
- "fit to view" / "show everything" → fitToView
- "zoom in/out" → setZoom
- "reset" (view) → resetView

ANALYSIS:
- "summarize" / "summary" / "action items" / "analyze the board" / "/summarize" → summarizeBoard
  (read the board state above and synthesize real content — do not leave fields empty)

═══ BOARD STATE ═══════════════════════════════════════════════════════════════
{format_board_state(board_state)}
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
    return {"status": "ok", "version": "4.0.0"}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
