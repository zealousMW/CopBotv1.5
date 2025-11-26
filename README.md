# CopBot v1.5

CopBot v1.5 is an experimental assistant built to help Indian law enforcement officers and citizens query legal materials (Police Act 1861, IPC codes, standing orders, FIR templates and emergency contacts) using semantic search and an LLM-powered router. The project combines a Python backend that extracts, indexes and serves documents via vector search (FAISS + LlamaIndex) with a Next.js frontend located in the `chatbot/` folder.

This README documents project purpose, quick start steps, API endpoints, data layout, development notes, and recommended next steps.

## Features

- Semantic search over legal/police documents (PDF/CSV)
- Uses HuggingFace embeddings + FAISS for vector stores
- Routes queries across multiple indexed sources with a multi-selector router
- REST endpoints for querying, reindexing and admin operations
- Next.js frontend scaffold in `chatbot/`

## Repository layout

- `app.py` — Primary Flask server that loads indexes from `settings.json`, exposes query endpoints and admin actions.
- `final.py` — Alternate/experimental backend with additional admin endpoints and sample PDF/CSV processing code.
- `chatbot/` — Next.js frontend (React + Tailwind-based components) that can be used to call the backend APIs.
- `data/` — Source documents used to build indexes (PDFs and CSVs).
- `storage/` — Persisted vector store directories created by the backend (one directory per category).
- `settings.json` — Configuration mapping core categories to document paths and descriptions.
- `README.md` — This file.

## Quick start — Backend (Python)

Note: The project depends on native and ML libraries (PyTorch, FAISS, PyMuPDF) which have platform-specific install instructions. The instructions below are a practical starting point on Windows; if you run into build issues, consider using WSL or a Linux environment.

1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install core Python packages (adjust versions as needed):

```powershell
pip install --upgrade pip
pip install flask flask-cors llama-index google-genai transformers torch faiss-cpu pymupdf pandas
```

Notes:

- On Windows `faiss-cpu` may require wheels or WSL; if you cannot install it, run in WSL or Linux.
- `google-genai` usage requires Google Cloud API access and credentials.

3. Provide your Google API key. Remove hard-coded keys in code and use an environment variable instead:

```powershell
$env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```

4. Start the backend (development mode):

```powershell
python app.py
# or
python final.py
```

The server runs on port `5000` by default.

## Quick start — Frontend (chatbot)

1. From the `chatbot/` folder install dependencies and run the dev server:

```powershell
cd chatbot
npm install
npm run dev
```

2. Open `http://localhost:3000` and ensure the frontend is calling the backend API at `http://localhost:5000` (update fetch URLs as needed).

## Important API endpoints

Backend (examples in `app.py` and `final.py`):

- `GET /query?query=...` : Query the agent and return an LLM-generated response.
- `POST /reset` : Reset the agent's internal state.
- `GET /get_files` or `/admin/files` : Return configured core files and metadata.
- `POST /replace/<file_key>` or `/admin/update-core-file` : Replace or upload a core file (PDF). Admin operation.
- `POST /edit_description/<file_key>` : Edit the description for a CORE_FILES entry (updates `settings.json`).
- `POST /rebuild_indexes` : Delete `./storage/` and rebuild all indexes from configured files.
- `POST /admin/upload` and `/admin/reindex` (in `final.py`) : upload arbitrary files and reindex them.

All admin endpoints currently have no authentication; add auth before exposing to untrusted networks.

## Data & index storage

- Source docs are under `data/` (e.g. `police_act_1861.pdf`, `pso.pdf`, `IPC_codes.pdf`, `fir.pdf`, `emergency_numbers.pdf`).
- Indexed vector stores are persisted under `./storage/<category>/`.
- The `storage/*` directories contain FAISS and LlamaIndex persistence files.

## Development notes and recommendations

- Move secrets out of source: remove hard-coded `GOOGLE_API_KEY` values and read from environment variables or a secrets manager.
- Add a `requirements.txt` or `pyproject.toml` for reproducible installs. Example:

```
# requirements.txt (suggested)
flask
flask-cors
llama-index
google-genai
transformers
torch
faiss-cpu
pymupdf
pandas
```

- Consider Dockerizing the backend to avoid platform-specific dependency issues (FAISS, PyTorch).
- Add authentication (JWT/API key) to admin routes before deployment.
- Add unit/integration tests for indexing, endpoints, and basic query flows.
- Sanitize and validate file uploads to avoid processing malicious files.

## Security & privacy notes

- Do not store API keys in source. Set `GOOGLE_API_KEY` via environment.
- Running Flask with `debug=True` is unsafe for production.
- Be careful with user-uploaded documents — they may contain sensitive data.
