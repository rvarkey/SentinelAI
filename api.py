"""
SentinelAI - Week 3: FastAPI REST API
======================================
Wraps the agent_v3 pipeline in a proper REST API.

Endpoints:
    POST /analyse          → analyse a URL right now, compare to baseline if exists
    POST /baseline         → build (or rebuild) a baseline for a URL
    GET  /baseline/{url}   → check if a baseline exists for a URL
    DELETE /baseline/{url} → delete a stored baseline
    GET  /health           → health check (Ollama + ChromaDB status)

Run:
    pip install fastapi uvicorn
    uvicorn api:app --reload --port 8000

Then test it:
    curl -X POST http://localhost:8000/analyse \
         -H "Content-Type: application/json" \
         -d '{"url": "https://github.com/login"}'

Or open the auto-generated docs in your browser:
    http://localhost:8000/docs
"""

import asyncio
import base64
import json
import os
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
import chromadb
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from playwright.async_api import async_playwright
from pydantic import BaseModel, HttpUrl

# Import all agent logic from agent_v3
# This keeps the API layer thin — it only handles HTTP concerns
from agent_v3 import (
    get_db,
    build_baseline,
    check_url,
    count_stored_snapshots,
    baseline_doc_ids,
    VISION_MODEL,
    EMBED_MODEL,
    BASELINE_SNAPSHOTS,
)


# ─────────────────────────────────────────────
# STARTUP / SHUTDOWN
# ─────────────────────────────────────────────
#
# FastAPI lifespan handles startup and shutdown logic.
# We initialise ChromaDB once on startup and share it
# across all requests — no need to reconnect every call.

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise shared resources on startup."""
    print("Starting SentinelAI API...")

    # Verify Ollama is reachable
    try:
        ollama.list()
        print(f"✅ Ollama connected — vision: {VISION_MODEL}, embed: {EMBED_MODEL}")
    except Exception as e:
        print(f"⚠️  Ollama not reachable: {e}")
        print("   Start Ollama before sending requests")

    # Initialise ChromaDB — stored in app state so all routes can access it
    app.state.db = get_db()
    print(f"✅ ChromaDB ready — {app.state.db.count()} baseline documents stored")

    yield  # API is now running

    print("Shutting down SentinelAI API")


# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="SentinelAI",
    description="AI-powered login page risk detection API",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────
#
# Pydantic models define the shape of request bodies and responses.
# FastAPI uses these to validate input and auto-generate API docs.

class AnalyseRequest(BaseModel):
    url: str
    force_baseline: bool = False    # if True, rebuild baseline before checking

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://github.com/login",
                "force_baseline": False,
            }
        }


class BaselineRequest(BaseModel):
    url: str

    class Config:
        json_schema_extra = {
            "example": {"url": "https://github.com/login"}
        }


class VisionResult(BaseModel):
    is_login_page: bool
    service_name: str
    legitimacy_score: int
    risk_level: str
    red_flags: list[str]
    summary: str
    is_multistep: bool = False


class MemoryResult(BaseModel):
    has_baseline: bool
    similarity: Optional[float]
    drift_detected: bool
    snapshot_count: Optional[int] = None
    baseline_stored_at: Optional[str] = None
    threshold: Optional[float] = None


class AnalyseResponse(BaseModel):
    url: str
    risk_level: str             # final risk level (vision + memory combined)
    trust_score: int
    is_login_page: bool
    service_name: str
    summary: str
    red_flags: list[str]
    vision: VisionResult
    memory: MemoryResult
    analysed_at: str


class BaselineResponse(BaseModel):
    url: str
    status: str
    snapshots_stored: int
    message: str


class HealthResponse(BaseModel):
    status: str
    ollama: str
    chromadb: str
    baselines_stored: int
    models: list[str]


# ─────────────────────────────────────────────
# HELPER: combine vision + memory into final risk
# ─────────────────────────────────────────────

def compute_final_risk(vision: dict, memory: dict) -> str:
    """
    Combine vision and memory signals into a final risk level.
    Memory drift always upgrades to HIGH regardless of vision score.
    """
    if memory.get("drift_detected"):
        return "HIGH"
    return vision.get("risk_level", "MEDIUM")


def build_response(url: str, vision: dict, memory: dict) -> AnalyseResponse:
    """Build a clean AnalyseResponse from raw agent outputs."""
    final_risk = compute_final_risk(vision, memory)

    red_flags = list(vision.get("red_flags", []))
    if memory.get("drift_detected"):
        red_flags.append("Visual drift from known-good baseline")

    return AnalyseResponse(
        url=url,
        risk_level=final_risk,
        trust_score=vision.get("legitimacy_score", 5),
        is_login_page=vision.get("is_login_page", False),
        service_name=vision.get("service_name", "Unknown"),
        summary=vision.get("summary", ""),
        red_flags=red_flags,
        vision=VisionResult(**{
            "is_login_page": vision.get("is_login_page", False),
            "service_name": vision.get("service_name", "Unknown"),
            "legitimacy_score": vision.get("legitimacy_score", 5),
            "risk_level": vision.get("risk_level", "MEDIUM"),
            "red_flags": vision.get("red_flags", []),
            "summary": vision.get("summary", ""),
            "is_multistep": vision.get("is_multistep", False),
        }),
        memory=MemoryResult(**memory),
        analysed_at=datetime.now().isoformat(),
    )


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Check if Ollama and ChromaDB are reachable and ready.
    Call this first to verify your setup is working.
    """
    ollama_status = "ok"
    model_names = []

    try:
        models = ollama.list()
        model_names = [m.model for m in models.models]
    except Exception as e:
        ollama_status = f"error: {e}"

    db_status = "ok"
    baselines = 0
    try:
        baselines = app.state.db.count()
    except Exception as e:
        db_status = f"error: {e}"

    overall = "ok" if ollama_status == "ok" and db_status == "ok" else "degraded"

    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        chromadb=db_status,
        baselines_stored=baselines,
        models=model_names,
    )


@app.post("/analyse", response_model=AnalyseResponse, tags=["Detection"])
async def analyse(request: AnalyseRequest):
    """
    Analyse a URL for login page risk.

    - If a baseline exists: compares against it and reports similarity + drift
    - If no baseline exists: analyses vision only (no memory comparison)
    - Set force_baseline=true to rebuild the baseline before checking

    Returns a structured risk report with trust score, red flags, and
    a combined risk level from both vision analysis and memory comparison.
    """
    url = request.url

    try:
        collection = app.state.db

        # Optionally rebuild baseline first
        if request.force_baseline:
            await build_baseline(url, collection)

        # Run the full check pipeline from agent_v3
        result = await check_url(url, collection)

        return build_response(url, result["vision"], result["memory"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/baseline", response_model=BaselineResponse, tags=["Baseline"])
async def create_baseline(request: BaselineRequest):
    """
    Build or rebuild a baseline for a URL.

    Visits the URL BASELINE_SNAPSHOTS times and stores each description
    as a separate snapshot in ChromaDB. Future /analyse calls will
    compare against these snapshots.

    This is a slow operation (30-90 seconds depending on hardware).
    """
    url = request.url

    try:
        collection = app.state.db
        await build_baseline(url, collection)
        stored = count_stored_snapshots(url, collection)

        return BaselineResponse(
            url=url,
            status="ok",
            snapshots_stored=stored,
            message=f"Baseline built with {stored} snapshots for {url}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baseline/{encoded_url}", response_model=BaselineResponse, tags=["Baseline"])
async def get_baseline(encoded_url: str):
    """
    Check if a baseline exists for a URL.

    Pass the URL encoded as a path segment:
        GET /baseline/https%3A%2F%2Fgithub.com%2Flogin
    """
    url = urllib.parse.unquote(encoded_url)

    try:
        collection = app.state.db
        stored = count_stored_snapshots(url, collection)

        if stored == 0:
            return BaselineResponse(
                url=url,
                status="not_found",
                snapshots_stored=0,
                message=f"No baseline stored for {url}",
            )

        return BaselineResponse(
            url=url,
            status="ok",
            snapshots_stored=stored,
            message=f"{stored} baseline snapshots stored for {url}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/baseline/{encoded_url}", tags=["Baseline"])
async def delete_baseline(encoded_url: str):
    """
    Delete all stored baseline snapshots for a URL.

    Use this when a legitimate page has been redesigned and the
    old baseline is no longer valid.
    """
    url = urllib.parse.unquote(encoded_url)

    try:
        collection = app.state.db
        ids = baseline_doc_ids(url)

        # Only delete IDs that actually exist
        existing = collection.get(ids=ids)
        if not existing["ids"]:
            raise HTTPException(
                status_code=404,
                detail=f"No baseline found for {url}"
            )

        collection.delete(ids=existing["ids"])

        return {
            "url": url,
            "status": "deleted",
            "snapshots_deleted": len(existing["ids"]),
            "message": f"Deleted {len(existing['ids'])} baseline snapshots for {url}",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/baselines", tags=["Baseline"])
async def list_baselines():
    """
    List all URLs that have stored baselines.
    """
    try:
        collection = app.state.db
        all_docs = collection.get(include=["metadatas"])

        # Group by URL and count snapshots
        url_counts: dict[str, dict] = {}
        for meta in all_docs["metadatas"]:
            url = meta.get("url", "unknown")
            if url not in url_counts:
                url_counts[url] = {
                    "url": url,
                    "snapshots": 0,
                    "latest_stored_at": meta.get("stored_at", "unknown"),
                }
            url_counts[url]["snapshots"] += 1
            # Track the most recent snapshot date
            if meta.get("stored_at", "") > url_counts[url]["latest_stored_at"]:
                url_counts[url]["latest_stored_at"] = meta.get("stored_at", "")

        return {
            "total_urls": len(url_counts),
            "total_snapshots": collection.count(),
            "baselines": list(url_counts.values()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
