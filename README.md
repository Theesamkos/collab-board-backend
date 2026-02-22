# CollabBoard AI Service - Backend

This repository contains the backend service for **CollabBoard**, a real-time collaborative whiteboard application. This service is a Python-based FastAPI application that provides a powerful AI agent to interpret natural language commands and manipulate the whiteboard state.

**Live Frontend:** [https://collab-board-peach.vercel.app](https://collab-board-peach.vercel.app)
**Live Backend Health Check:** [https://collab-board-backend-production-d852.up.railway.app/health](https://collab-board-backend-production-d852.up.railway.app/health)

---

## AI Features

The core of this service is an AI agent powered by **Claude 3.5 Sonnet** and orchestrated with **LangChain**. The agent is equipped with a suite of tools that allow it to perform a wide range of actions on the whiteboard, from creating simple shapes to generating complex templates and summarizing board content.

### Key AI Capabilities:

- **Object Creation & Manipulation:** The AI can create, update, and delete all object types (sticky notes, shapes, text).
- **Layout & Arrangement:** The AI can arrange, align, and distribute objects on the board to create clean, organized layouts.
- **Template Generation:** The AI can generate pre-defined templates, such as a SWOT analysis, with a single command.
- **Board Summarization:** The AI can analyze the entire board state and generate a concise summary, identifying key points, risks, and action items.

---

## Tech Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/)
- **AI Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM:** [Claude 3.5 Sonnet](https://www.anthropic.com/news/claude-3-5-sonnet)
- **Deployment:** [Railway](https://railway.app/)
- **Monitoring & Tracing:** [LangSmith](https://www.langchain.com/langsmith)

---

## API Endpoints

### `POST /recognize-intent`

This is the main endpoint for interacting with the AI agent.

- **Request Body:**
  ```json
  {
    "board_objects": { ... },
    "user_prompt": "Your natural language command"
  }
  ```
- **Response:** An array of tool calls that the frontend can execute to update the board state.

### `GET /health`

Returns the current status and version of the service.

- **Response:**
  ```json
  {
    "status": "ok",
    "version": "4.0.0"
  }
  ```

---

##  Getting Started (Local Development)

To run this service locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Theesamkos/collab-board-backend.git
    cd collab-board-backend
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root of the project and add your Anthropic API key:
    ```
    ANTHROPIC_API_KEY="your-api-key"
    ```

4.  **Run the server:**
    ```bash
    uvicorn main:app --reload
    ```
    The server will be available at `http://127.0.0.1:8000`.

---

## ☁️ Deployment

This service is automatically deployed to [Railway](https://railway.app/) on every push to the `main` branch. The deployment is configured using a `Dockerfile` to ensure a consistent and reliable build process.

### LangSmith Integration

LangSmith is used for tracing and monitoring the AI agent's performance. It is enabled by setting the following environment variables in the Railway service:

- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<your_langsmith_api_key>`
- `LANGCHAIN_PROJECT=collab-board-backend`

This provides invaluable insights into the AI's behavior, helping to debug prompts, optimize performance, and monitor costs.
