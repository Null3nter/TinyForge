# TinyForge

Lightweight, fast, locally hosted mini-version of ForgeGUI.
Designed to run comfortably on a laptop with ~32 GB RAM.

**Core focus:** speed, low disk footprint, reliability.

## Stack (Step 1)

- **Backend**: Single FastAPI app (`backend/app.py`)
- Talks to local **ComfyUI** (`http://127.0.0.1:8188`) and **Ollama** (`http://127.0.0.1:11434`)
- Saves generated assets under `/outputs`
- Aggressive memory cleanup between tasks

## Quick Start (Backend)

```bash
# 1. Clone
git clone https://github.com/Null3nter/TinyForge.git
cd TinyForge

# 2. Create venv & install
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Make sure ComfyUI and Ollama are running locally
#    ComfyUI default: http://127.0.0.1:8188
#    Ollama default:  http://127.0.0.1:11434

# 4. Run backend
cd backend
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

API docs: http://127.0.0.1:8000/docs

## Current Endpoints (Step 1)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status + ComfyUI/Ollama reachability |
| POST | `/generate/image` | Queue a fast 2D generation via ComfyUI |
| POST | `/generate/luau` | Generate Luau / UI tree JSON via Ollama |
| GET | `/outputs` | List saved files in `/outputs` |
| GET | `/outputs/{filename}` | Download a generated file |
| POST | `/cleanup` | Force memory cleanup |

## Folder Structure

```
TinyForge/
├── backend/
│   └── app.py          # Single-file FastAPI backend
├── outputs/            # Generated images + scripts
├── requirements.txt
├── README.md
└── .gitignore
```

Next steps (coming):
- Step 2: Lean ComfyUI workflow JSON for icons/UI
- Step 3: Clean frontend
- Step 4: Roblox Studio plugin
