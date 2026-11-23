"""
TinyForge Backend – Step 1
Lightweight FastAPI app that talks to local ComfyUI + Ollama.
Designed for laptop use: small footprint, fast defaults, memory cleanup.
"""

from __future__ import annotations

import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default fast model for Luau generation (user can override)
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

app = FastAPI(
    title="TinyForge",
    description="Lightweight local AI backend for UI/icons + Roblox Luau generation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only tool; tighten later if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------

def force_cleanup() -> dict[str, Any]:
    """Aggressively free Python memory. Safe no-op if CUDA/torch not present."""
    collected = gc.collect()
    extra: dict[str, Any] = {"gc_collected": collected}

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            extra["cuda_cache_cleared"] = True
        else:
            extra["cuda_cache_cleared"] = False
    except ImportError:
        extra["cuda_cache_cleared"] = False

    return extra


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

class ComfyUIClient:
    """Minimal ComfyUI HTTP client."""

    def __init__(self, base_url: str = COMFYUI_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def is_alive(self) -> bool:
        try:
            r = await self.client.get(f"{self.base_url}/system_stats")
            return r.status_code == 200
        except Exception:
            return False

    async def queue_prompt(self, workflow: dict, client_id: str | None = None) -> str:
        """Submit a workflow and return the prompt_id."""
        payload = {"prompt": workflow}
        if client_id:
            payload["client_id"] = client_id
        r = await self.client.post(f"{self.base_url}/prompt", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["prompt_id"]

    async def get_history(self, prompt_id: str) -> dict:
        r = await self.client.get(f"{self.base_url}/history/{prompt_id}")
        r.raise_for_status()
        return r.json()

    async def wait_for_completion(
        self, prompt_id: str, poll_interval: float = 0.8, timeout: float = 180.0
    ) -> dict:
        """Poll until the prompt finishes or times out."""
        start = time.time()
        while time.time() - start < timeout:
            history = await self.get_history(prompt_id)
            if prompt_id in history:
                return history[prompt_id]
            await asyncio_sleep(poll_interval)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout}s")

    async def download_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = await self.client.get(f"{self.base_url}/view", params=params)
        r.raise_for_status()
        return r.content

    async def close(self):
        await self.client.aclose()


class OllamaClient:
    """Minimal Ollama HTTP client."""

    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def is_alive(self) -> bool:
        try:
            r = await self.client.get(f"{self.base_url}/api/tags")
            return r.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[str]:
        r = await self.client.get(f"{self.base_url}/api/tags")
        r.raise_for_status()
        data = r.json()
        return [m["name"] for m in data.get("models", [])]

    async def generate(
        self,
        prompt: str,
        model: str = DEFAULT_OLLAMA_MODEL,
        system: str | None = None,
        temperature: float = 0.4,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        r = await self.client.post(f"{self.base_url}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    async def close(self):
        await self.client.aclose()


# Helper so we don't need to import asyncio at top level just for sleep
async def asyncio_sleep(seconds: float):
    import asyncio
    await asyncio.sleep(seconds)


comfy = ComfyUIClient()
ollama = OllamaClient()


@app.on_event("shutdown")
async def shutdown():
    await comfy.close()
    await ollama.close()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    negative_prompt: str = ""
    width: int = Field(512, ge=64, le=2048)
    height: int = Field(512, ge=64, le=2048)
    steps: int = Field(6, ge=1, le=30)          # low default for speed
    cfg: float = Field(1.5, ge=0.0, le=15.0)   # low for distilled/turbo models
    seed: int = Field(-1, description="-1 = random")
    # Placeholder – real workflow comes in Step 2
    workflow_override: Optional[dict] = None


class ImageGenResponse(BaseModel):
    success: bool
    prompt_id: Optional[str] = None
    filename: Optional[str] = None
    path: Optional[str] = None
    message: str = ""


class LuauGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = DEFAULT_OLLAMA_MODEL
    temperature: float = Field(0.35, ge=0.0, le=1.5)
    system_prompt: Optional[str] = None


class LuauGenResponse(BaseModel):
    success: bool
    content: str = ""
    model: str = ""
    filename: Optional[str] = None
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    comfyui: bool
    ollama: bool
    ollama_models: list[str] = []
    output_dir: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    comfy_ok = await comfy.is_alive()
    ollama_ok = await ollama.is_alive()
    models: list[str] = []
    if ollama_ok:
        try:
            models = await ollama.list_models()
        except Exception:
            pass

    return HealthResponse(
        status="ok" if (comfy_ok or ollama_ok) else "degraded",
        comfyui=comfy_ok,
        ollama=ollama_ok,
        ollama_models=models,
        output_dir=str(OUTPUT_DIR),
    )


@app.post("/cleanup")
async def cleanup():
    info = force_cleanup()
    return {"success": True, **info}


@app.get("/outputs")
async def list_outputs():
    files = []
    for p in sorted(OUTPUT_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if p.is_file() and p.name != ".gitkeep":
            files.append(
                {
                    "name": p.name,
                    "size": p.stat().st_size,
                    "modified": p.stat().st_mtime,
                }
            )
    return {"files": files}


@app.get("/outputs/{filename}")
async def get_output(filename: str):
    # Prevent path traversal
    safe = Path(filename).name
    path = OUTPUT_DIR / safe
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.post("/generate/image", response_model=ImageGenResponse)
async def generate_image(req: ImageGenRequest):
    """
    Queue a generation job to ComfyUI.

    NOTE (Step 1):
    A real lean workflow JSON will be provided in Step 2.
    For now we accept an optional workflow_override so you can already
    test the client + polling + file saving path.
    """
    if not await comfy.is_alive():
        raise HTTPException(
            status_code=503,
            detail="ComfyUI is not reachable at " + COMFYUI_URL,
        )

    if req.workflow_override is None:
        return ImageGenResponse(
            success=False,
            message=(
                "No workflow provided yet. "
                "Step 2 will ship a lean default workflow for icons/UI. "
                "You can already pass a full ComfyUI workflow JSON in "
                "`workflow_override` to test the pipeline."
            ),
        )

    try:
        client_id = str(uuid.uuid4())
        prompt_id = await comfy.queue_prompt(req.workflow_override, client_id=client_id)
        result = await comfy.wait_for_completion(prompt_id)

        # Extract first image from outputs
        outputs = result.get("outputs", {})
        image_info = None
        for node_id, node_out in outputs.items():
            if "images" in node_out and node_out["images"]:
                image_info = node_out["images"][0]
                break

        if not image_info:
            return ImageGenResponse(
                success=False,
                prompt_id=prompt_id,
                message="ComfyUI finished but no images were found in outputs.",
            )

        img_bytes = await comfy.download_image(
            filename=image_info["filename"],
            subfolder=image_info.get("subfolder", ""),
            folder_type=image_info.get("type", "output"),
        )

        # Save under our own outputs folder with a clean name
        ext = Path(image_info["filename"]).suffix or ".png"
        safe_name = f"icon_{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
        out_path = OUTPUT_DIR / safe_name
        out_path.write_bytes(img_bytes)

        force_cleanup()

        return ImageGenResponse(
            success=True,
            prompt_id=prompt_id,
            filename=safe_name,
            path=str(out_path),
            message="Image generated and saved.",
        )
    except Exception as e:
        force_cleanup()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/luau", response_model=LuauGenResponse)
async def generate_luau(req: LuauGenRequest):
    """Generate clean Luau + optional JSON UI tree via Ollama."""
    if not await ollama.is_alive():
        raise HTTPException(
            status_code=503,
            detail="Ollama is not reachable at " + OLLAMA_URL,
        )

    system = req.system_prompt or (
        "You are an expert Roblox Luau engineer. "
        "Write clean, modern Luau code and/or JSON UI trees "
        "(ScreenGui, Frame, ImageLabel, TextButton, UICorner, UIListLayout, etc.). "
        "Prefer short, readable, production-ready snippets. "
        "Do not add long explanations unless the user asks."
    )

    try:
        content = await ollama.generate(
            prompt=req.prompt,
            model=req.model,
            system=system,
            temperature=req.temperature,
        )

        # Optionally save the script
        safe_name = f"luau_{int(time.time())}_{uuid.uuid4().hex[:8]}.lua"
        out_path = OUTPUT_DIR / safe_name
        out_path.write_text(content, encoding="utf-8")

        force_cleanup()

        return LuauGenResponse(
            success=True,
            content=content,
            model=req.model,
            filename=safe_name,
            message="Luau / UI tree generated.",
        )
    except Exception as e:
        force_cleanup()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    return {
        "name": "TinyForge",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
