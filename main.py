import json
from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from schemas import IncidentRequest, ReceiptRequest
from database import (
    init_db,
    get_run_state,
    check_incidents_replay,
    save_run_state,
    check_receipt_replay,
    record_receipt
)
from agent import IncidentAgent
from otlp_builder import build_otlp_trace

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

app = FastAPI(title="GA5 Incident Agent", version="2.0", lifespan=lifespan)
agent = IncidentAgent()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"status": "ok", "service": "ga5-incident-agent"}

@app.post("/v2/incidents")
async def create_incident(request: Request):
    raw_bytes = await request.body()
    try:
        data = json.loads(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    profile = data.get("profile")
    if profile != "ga5-incident-agent/v2":
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported profile: {profile}"}
        )

    run_id = data.get("runId")
    if not run_id:
        return JSONResponse(status_code=400, content={"error": "Missing runId"})

    # Check Idempotency & Conflict
    replay_status, stored_state = check_incidents_replay(run_id, raw_bytes)
    if replay_status == "CONFLICT":
        return JSONResponse(status_code=409, content={"error": "Changed-content conflict"})
    elif replay_status == "REPLAY" and stored_state:
        # Build stored response based on status
        current_status = stored_state.get("status", "waiting")
        if current_status in ["completed", "failed"]:
            resp = {
                "runId": stored_state["runId"],
                "status": current_status,
                "diagnosis": stored_state["diagnosis"],
                "chosenEffect": stored_state.get("chosenEffectTool", "") if current_status == "completed" else "",
                "suppressed": stored_state.get("suppressed", []),
                "actionLog": stored_state.get("actionLog", []),
                "receiptLog": stored_state.get("receiptLog", []),
                "otlp": build_otlp_trace(stored_state.get("traceSpans", []))
            }
        else:
            resp = {
                "runId": stored_state["runId"],
                "status": "waiting",
                "diagnosis": stored_state["diagnosis"],
                "dispatches": stored_state.get("dispatchesSent", []),
                "approvals": [stored_state["approvalPending"]] if stored_state.get("approvalPending") else []
            }
        return JSONResponse(status_code=200, content=resp)

    # Validate schema
    try:
        req_obj = IncidentRequest.model_validate(data)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    # Plan incident with LLM
    headers_dict = dict(request.headers)
    state, response = agent.plan_incident(req_obj, headers_dict)

    # Persist state
    save_run_state(run_id, raw_bytes, response["status"], state)

    return JSONResponse(status_code=200, content=response)


@app.post("/v2/incidents/{runId}/receipts")
async def post_receipt(runId: str, request: Request):
    raw_bytes = await request.body()
    try:
        data = json.loads(raw_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    receipt_id = data.get("receiptId")
    if not receipt_id:
        return JSONResponse(status_code=400, content={"error": "Missing receiptId"})

    state = get_run_state(runId)
    if not state:
        return JSONResponse(status_code=404, content={"error": f"Run {runId} not found"})

    # Check Receipt Idempotency & Conflict
    rec_status = check_receipt_replay(receipt_id, runId, raw_bytes)
    if rec_status == "CONFLICT":
        return JSONResponse(status_code=409, content={"error": "Receipt changed-content conflict"})
    elif rec_status == "REPLAY":
        # Return current stored state response
        current_status = state.get("status", "waiting")
        if current_status in ["completed", "failed"]:
            resp = {
                "runId": state["runId"],
                "status": current_status,
                "diagnosis": state["diagnosis"],
                "chosenEffect": state.get("chosenEffectTool", "") if current_status == "completed" else "",
                "suppressed": state.get("suppressed", []),
                "actionLog": state.get("actionLog", []),
                "receiptLog": state.get("receiptLog", []),
                "otlp": build_otlp_trace(state.get("traceSpans", []))
            }
        else:
            resp = {
                "runId": state["runId"],
                "status": "waiting",
                "diagnosis": state["diagnosis"],
                "dispatches": state.get("dispatchesSent", []),
                "approvals": [state["approvalPending"]] if state.get("approvalPending") else []
            }
        return JSONResponse(status_code=200, content=resp)

    # Record new receipt
    record_receipt(receipt_id, runId, raw_bytes)

    # Validate receipt request object
    try:
        receipt_req = ReceiptRequest.model_validate(data)
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": str(e)})

    # Process state machine step
    updated_state, response = agent.process_receipt(state, receipt_req)

    # Save updated state
    save_run_state(runId, raw_bytes, response["status"], updated_state)

    return JSONResponse(status_code=200, content=response)


@app.get("/v2/incidents/{runId}")
def get_incident(runId: str):
    state = get_run_state(runId)
    if not state:
        return JSONResponse(status_code=404, content={"error": f"Run {runId} not found"})

    current_status = state.get("status", "waiting")
    if current_status in ["completed", "failed"]:
        resp = {
            "runId": state["runId"],
            "status": current_status,
            "diagnosis": state["diagnosis"],
            "chosenEffect": state.get("chosenEffectTool", "") if current_status == "completed" else "",
            "suppressed": state.get("suppressed", []),
            "actionLog": state.get("actionLog", []),
            "receiptLog": state.get("receiptLog", []),
            "otlp": build_otlp_trace(state.get("traceSpans", []))
        }
    else:
        resp = {
            "runId": state["runId"],
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": state.get("dispatchesSent", []),
            "approvals": [state["approvalPending"]] if state.get("approvalPending") else []
        }
    return JSONResponse(status_code=200, content=resp)
