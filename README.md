# LangGraph RAG Demo

A conversational AI agent built with **LangGraph** that combines retrieval-augmented generation (RAG) with tool use. The agent can answer questions from a local knowledge base and perform arithmetic — all with a live Streamlit UI showing the tool call trace.

## Features

- **RAG search** — queries a FAISS vector index built from local `.txt` documents
- **Arithmetic tools** — add, multiply, divide
- **LangGraph agent loop** — the LLM decides when to call tools and loops until it has a final answer
- **Streamlit UI** — chat interface with a sidebar showing every tool call, its inputs, and its output
- **Groq LLM** — fast inference via `ChatGroq`
- **Google Gemini embeddings** — `gemini-embedding-001` for document and query embedding

## Architecture

```
User prompt
    │
    ▼
┌─────────────┐     tool calls      ┌─────────────┐
│  llm_call   │ ─────────────────► │  tool_node  │
│  (ChatGroq) │ ◄───────────────── │             │
└─────────────┘   tool results      └─────────────┘
    │
    │ no more tool calls
    ▼
 Final answer
```

**Tools available to the agent:**

| Tool | Description |
|------|-------------|
| `search_docs` | Semantic search over the FAISS knowledge base |
| `add` | Adds two integers |
| `multiply` | Multiplies two integers |
| `divide` | Divides two integers |

## Project Structure

```
.
├── agent.py          # LangGraph agent definition (state, tools, graph)
├── app.py            # Streamlit UI
├── ingest.py         # One-time script to build the FAISS index
├── main.py           # CLI entrypoint placeholder
├── sample_docs/      # Source .txt documents for the knowledge base
│   └── ai_concepts.txt
├── faiss_index/      # Generated FAISS index (created by ingest.py)
├── pyproject.toml
└── .env.example
```

## Setup

### Prerequisites

- Python 3.12+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd "LangGraph RAG"
uv sync
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GOOGLE_API_KEY=your_google_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

- Get a Google API key from [Google AI Studio](https://aistudio.google.com/)
- Get a Groq API key from [console.groq.com](https://console.groq.com/)

### 3. Build the knowledge base index

Run this once to embed your documents and create the FAISS index:

```bash
uv run python ingest.py
```

This reads all `.txt` files from `sample_docs/`, splits them into ~500-character chunks, embeds them with Gemini, and saves the index to `faiss_index/`.

### 4. Launch the Streamlit app

```bash
uv run streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Usage

- **Ask AI/ML questions** — e.g. "What is RAG?" or "Explain how transformers work"
- **Do math** — e.g. "What is 42 multiplied by 7?"
- **Combine both** — e.g. "If embeddings have 768 dimensions, how many numbers is that for 5 documents?"

The sidebar shows a live trace of every tool the agent called, including inputs and outputs, for each conversation turn.

## Adding Documents

Add `.txt` files to `sample_docs/` and re-run `ingest.py` to rebuild the index:

```bash
uv run python ingest.py
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Agent graph orchestration |
| `langchain` | LLM/tool abstractions |
| `langchain-groq` | Groq LLM integration |
| `langchain-google-genai` | Gemini embedding model |
| `langchain-community` | FAISS vector store integration |
| `faiss-cpu` | Local vector similarity search |
| `streamlit` | Web UI |
| `python-dotenv` | Environment variable loading |
