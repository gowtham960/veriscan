"""
VeriScan API — thin FastAPI layer over the LangGraph pipeline + persistence.

Endpoints:
  POST /verify           run a claim/video through the graph, persist, return result
  GET  /reviews          list pending human-review items
  POST /reviews/{run_id}/resolve   approve/reject a flagged run
  GET  /health           basic liveness check
"""
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from graph import build_graph
from utils.persistence import (
    init_db, persist_run, enqueue_for_review, get_pending_reviews,
    resolve_review, new_run_id,
)

os.environ.setdefault("LANGSMITH_PROJECT", "veriscan")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="VeriScan API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev / demo only — tighten for real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

graph_app = build_graph()
init_db()


class ReviewResolution(BaseModel):
    status: str  # "approved" | "rejected"
    notes: str = ""


def _run_pipeline(source_type: str, raw_input: str) -> dict:
    run_id = new_run_id()
    result = graph_app.invoke(
        {"source_type": source_type, "raw_input": raw_input, "audit_log": []},
        config={"run_name": "veriscan_pipeline", "tags": [source_type], "metadata": {"internal_run_id": run_id}},
    )
    persist_run(run_id, result["audit_log"])

    if result["requires_human_review"]:
        reasons = []
        if result.get("confidence_score", 1.0) < 0.6:
            reasons.append("low confidence")
        if result.get("faithfulness_score", 1.0) < 0.85:
            reasons.append("low faithfulness")
        if result.get("metadata", {}).get("injection_scan", {}).get("flagged"):
            reasons.append("injection flagged")
        enqueue_for_review(
            run_id, raw_input, result["final_verdict"],
            result.get("confidence_score", 0.0), result.get("faithfulness_score", 0.0),
            "; ".join(reasons) or "unspecified",
        )

    return {
        "run_id": run_id,
        "routing_reason": result.get("routing_reason", ""),
        "final_verdict": result.get("final_verdict", ""),
        "confidence_score": result.get("confidence_score", 0.0),
        "faithfulness_score": result.get("faithfulness_score", 0.0),
        "requires_human_review": result.get("requires_human_review", False),
        "fact_check_result": result.get("fact_check_result"),
        "media_check_result": result.get("media_check_result"),
        "audit_log": result.get("audit_log", []),
        "injection_flagged": result.get("metadata", {}).get("injection_scan", {}).get("flagged", False),
    }


@app.post("/verify")
async def verify_text(source_type: str = Form(...), raw_input: str = Form(None), file: UploadFile = File(None)):
    if source_type == "video_upload":
        if file is None:
            raise HTTPException(400, "file is required for source_type=video_upload")
        dest = UPLOAD_DIR / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        return _run_pipeline("video_upload", str(dest))

    if not raw_input or not raw_input.strip():
        raise HTTPException(400, "raw_input is required for text_claim/url")
    return _run_pipeline(source_type, raw_input)


@app.get("/reviews")
async def list_reviews():
    return get_pending_reviews()


@app.post("/reviews/{run_id}/resolve")
async def resolve(run_id: str, body: ReviewResolution):
    ok = resolve_review(run_id, body.status, body.notes)
    if not ok:
        raise HTTPException(404, "run_id not found in review queue")
    return {"resolved": True}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
