from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import re
from fastapi.middleware.cors import CORSMiddleware

# --- Pydantic Models ---
class RecognizeRequest(BaseModel):
    command: str

class RecognizeResponse(BaseModel):
    intent: str
    entities: Dict[str, Any]
    confidence: float
    handler: str

# --- FastAPI App ---
app = FastAPI(
    title="Intent Recognition Microservice",
    description="A lightweight FastAPI service that uses rule-based regex matching to parse simple whiteboard commands.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

# --- Intent Recognition Logic ---
@app.post("/recognize-intent", response_model=RecognizeResponse)
def recognize_intent(request: RecognizeRequest):
    command = request.command.lower()

    # Define patterns for intents
    patterns = {
        'CREATE': r"create a (?P<color>\w+)? ?(?P<type>rectangle|square|circle|sticky note)",
        'DELETE': r"delete the selected|delete everything|remove all rectangles",
        'UPDATE': r"change color to (?P<color>\w+)|resize to (?P<size>\d+x\d+)|set text to (?P<text>.+)",
        'CLEAR': r"clear the board",
        'UNDO': r"undo|redo",
        'ZOOM': r"zoom in|zoom to 150%",
        'SELECT': r"select all|select all circles",
    }

    for intent, pattern in patterns.items():
        match = re.search(pattern, command)
        if match:
            entities = {k: v for k, v in match.groupdict().items() if v is not None}
            return RecognizeResponse(
                intent=intent,
                entities=entities,
                confidence=1.0,
                handler="local",
            )

    # If no pattern matches, forward to LangChain
    return RecognizeResponse(
        intent="UNKNOWN",
        entities={},
        confidence=0.0,
        handler="forward_to_langchain",
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}
