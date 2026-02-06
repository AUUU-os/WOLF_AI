"""
WOLF_AI FastAPI Server - The Pack Command Center

Run with: python -m api.server
Or: uvicorn api.server:app --reload
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from .config import API_HOST, API_PORT, BRIDGE_PATH, WOLF_ROOT
from .auth import verify_api_key

# ==================== MODELS ====================

class HowlRequest(BaseModel):
    message: str
    to: str = "pack"
    frequency: str = "medium"

class HowlResponse(BaseModel):
    status: str
    howl: dict

class HuntRequest(BaseModel):
    target: str
    assigned_to: str = "hunter"

class WilkRequest(BaseModel):
    message: str
    mode: str = "chat"
    stream: bool = False

class StatusResponse(BaseModel):
    status: str
    pack: dict
    timestamp: str

# ==================== LIFESPAN ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🐺 WOLF_AI Command Center                               ║
    ║   ═══════════════════════════════════════                 ║
    ║                                                           ║
    ║   The pack is ready. AUUUUUUUUUUUUUUUUUU!                 ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    # Log startup howl
    _howl_to_bridge("system", "pack", "API Server started. Pack Command Center online.", "high")
    yield
    # Shutdown
    _howl_to_bridge("system", "pack", "API Server shutting down.", "medium")

# ==================== APP ====================

app = FastAPI(
    title="WOLF_AI Command Center",
    description="Control your wolf pack from anywhere",
    version="1.0.0",
    lifespan=lifespan
)

# CORS - configured for security with mobile access
# Set WOLF_CORS_ORIGINS in .env to restrict (comma-separated)
import os
cors_origins_env = os.getenv("WOLF_CORS_ORIGINS", "")
if cors_origins_env:
    cors_origins = [o.strip() for o in cors_origins_env.split(",")]
else:
    # Default: allow localhost and common local network patterns
    cors_origins = [
        "http://localhost:*",
        "http://127.0.0.1:*",
        "http://192.168.*.*:*",  # Local network
        "https://*.ngrok.io",
        "https://*.ngrok-free.app",
        "https://*.trycloudflare.com",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For mobile/tunnel access - API key provides security
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# Serve dashboard
dashboard_path = WOLF_ROOT / "dashboard"
if dashboard_path.exists():
    app.mount("/static", StaticFiles(directory=str(dashboard_path)), name="static")

# ==================== HELPERS ====================

def _howl_to_bridge(from_wolf: str, to: str, message: str, frequency: str = "medium") -> dict:
    """Write howl to bridge."""
    howl_data = {
        "from": from_wolf,
        "to": to,
        "howl": message,
        "frequency": frequency,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    howls_file = BRIDGE_PATH / "howls.jsonl"
    howls_file.parent.mkdir(parents=True, exist_ok=True)

    with open(howls_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(howl_data, ensure_ascii=False) + "\n")

    return howl_data

def _get_recent_howls(limit: int = 20) -> List[dict]:
    """Get recent howls from bridge."""
    howls_file = BRIDGE_PATH / "howls.jsonl"

    if not howls_file.exists():
        return []

    with open(howls_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    howls = []
    for line in lines[-limit:]:
        try:
            howls.append(json.loads(line.strip()))
        except json.JSONDecodeError:
            continue

    return howls

def _get_pack_state() -> dict:
    """Get current pack state."""
    state_file = BRIDGE_PATH / "state.json"

    if not state_file.exists():
        return {"pack_status": "dormant", "wolves": {}}

    with open(state_file, "r", encoding="utf-8") as f:
        return json.load(f)

# ==================== ROUTES ====================

@app.get("/")
async def root():
    """Root endpoint - redirect to dashboard."""
    return {"message": "WOLF_AI Command Center", "dashboard": "/dashboard", "docs": "/docs"}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve mobile-friendly dashboard."""
    dashboard_file = WOLF_ROOT / "dashboard" / "index.html"
    if dashboard_file.exists():
        return FileResponse(dashboard_file)
    return HTMLResponse("<h1>Dashboard not found. Run setup first.</h1>")

# ==================== PACK ROUTES ====================

@app.get("/api/status", response_model=StatusResponse)
async def get_status(api_key: str = Depends(verify_api_key)):
    """Get pack status."""
    state = _get_pack_state()
    return StatusResponse(
        status="ok",
        pack=state,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

@app.post("/api/awaken")
async def awaken_pack(api_key: str = Depends(verify_api_key)):
    """Awaken the pack."""
    try:
        from core.pack import awaken_pack as do_awaken
        pack = do_awaken()
        return {"status": "ok", "message": "Pack awakened!", "pack": pack.status_report()}
    except Exception as e:
        _howl_to_bridge("api", "pack", f"Awaken failed: {str(e)}", "high")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/hunt")
async def start_hunt(request: HuntRequest, api_key: str = Depends(verify_api_key)):
    """Start a hunt (task)."""
    try:
        from core.pack import get_pack
        pack = get_pack()
        pack.hunt(request.target, request.assigned_to)
        return {"status": "ok", "target": request.target, "assigned_to": request.assigned_to}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== HOWL ROUTES ====================

@app.post("/api/howl", response_model=HowlResponse)
async def send_howl(request: HowlRequest, api_key: str = Depends(verify_api_key)):
    """Send a howl to the pack."""
    howl = _howl_to_bridge("commander", request.to, request.message, request.frequency)
    return HowlResponse(status="ok", howl=howl)

@app.get("/api/howls")
async def get_howls(limit: int = 20, api_key: str = Depends(verify_api_key)):
    """Get recent howls."""
    howls = _get_recent_howls(limit)
    return {"status": "ok", "count": len(howls), "howls": howls}

# ==================== WILK ROUTES ====================

@app.post("/api/wilk")
async def ask_wilk(request: WilkRequest, api_key: str = Depends(verify_api_key)):
    """Ask WILK (Dolphin) a question."""
    try:
        from modules.wilk import get_wilk
        wilk = get_wilk(request.mode)
        response = wilk.ask(request.message)
        return {"status": "ok", "mode": request.mode, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/wilk/status")
async def wilk_status(api_key: str = Depends(verify_api_key)):
    """Check WILK/Ollama status."""
    try:
        from modules.wilk.dolphin import get_client
        client = get_client()
        alive = client.is_alive()
        return {"status": "ok", "ollama_alive": alive, "model": client.model}
    except Exception as e:
        return {"status": "error", "ollama_alive": False, "error": str(e)}

# ==================== GITHUB SYNC ====================

@app.post("/api/sync")
async def github_sync(api_key: str = Depends(verify_api_key)):
    """Sync with GitHub (git pull)."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=str(WOLF_ROOT),
            capture_output=True,
            text=True,
            timeout=30
        )
        _howl_to_bridge("api", "pack", f"GitHub sync: {result.stdout or 'up to date'}", "medium")
        return {"status": "ok", "output": result.stdout, "error": result.stderr}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MEMORY ROUTES ====================

class MemoryStoreRequest(BaseModel):
    key: str
    value: str
    category: str = "general"

@app.post("/api/memory/store")
async def store_memory(request: MemoryStoreRequest, api_key: str = Depends(verify_api_key)):
    """Store information in memory."""
    try:
        from memory.store import get_memory
        memory = get_memory()
        memory.set(request.key, request.value, namespace=request.category)
        return {"status": "ok", "key": request.key, "category": request.category}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/recall")
async def recall_memory(key: str = None, category: str = None, api_key: str = Depends(verify_api_key)):
    """Recall information from memory."""
    try:
        from memory.store import get_memory
        memory = get_memory()

        if key:
            value = memory.get(key, namespace=category or "general")
            return {"status": "ok", "key": key, "data": value}
        elif category:
            data = memory.get_namespace(category)
            return {"status": "ok", "category": category, "data": data}
        else:
            data = memory.get_all()
            return {"status": "ok", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== TRACK ROUTES ====================

@app.get("/api/track/search")
async def track_search(pattern: str, type: str = "files", api_key: str = Depends(verify_api_key)):
    """Search for files or code."""
    try:
        from modules.track import get_tracker
        tracker = get_tracker()

        if type == "files":
            results = tracker.find(pattern)
        elif type == "content":
            results = tracker.grep(pattern)
        elif type == "functions":
            results = tracker.find_functions(pattern)
        elif type == "classes":
            results = tracker.find_classes(pattern)
        else:
            results = tracker.find(pattern)

        return {
            "status": "ok",
            "type": type,
            "pattern": pattern,
            "results": [r.to_dict() for r in results[:20]]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEBSOCKET ====================

connected_clients: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to WOLF_AI Command Center"
        })

        while True:
            data = await websocket.receive_json()

            # Handle commands via WebSocket
            if data.get("type") == "howl":
                howl = _howl_to_bridge(
                    "commander",
                    data.get("to", "pack"),
                    data.get("message", ""),
                    data.get("frequency", "medium")
                )
                await websocket.send_json({"type": "howl_sent", "howl": howl})

            elif data.get("type") == "get_status":
                state = _get_pack_state()
                await websocket.send_json({"type": "status", "pack": state})

            elif data.get("type") == "get_howls":
                howls = _get_recent_howls(data.get("limit", 20))
                await websocket.send_json({"type": "howls", "howls": howls})

    except WebSocketDisconnect:
        connected_clients.remove(websocket)

# ==================== RUN ====================

def run():
    """Run the server."""
    import uvicorn
    print(f"\n🐺 Starting WOLF_AI Command Center on http://{API_HOST}:{API_PORT}")
    print(f"📱 Dashboard: http://localhost:{API_PORT}/dashboard")
    print(f"📚 API Docs: http://localhost:{API_PORT}/docs\n")
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    run()
