import pytest
import json
import os
import uuid
from fastapi.testclient import TestClient
from main import app
from database import init_db

client = TestClient(app)

def test_full_incident_flow():
    if os.path.exists("incidents.db"):
        os.remove("incidents.db")
    init_db()

    run_id = f"test_run_{uuid.uuid4().hex[:8]}"

    payload = {
        "profile": "ga5-incident-agent/v2",
        "runId": run_id,
        "agentName": "incident-response",
        "publicMarker": "pub_marker_test",
        "sensitive": {"accessToken": "secret_token_123"},
        "incident": {
            "incidentId": "inc_001",
            "title": "High CPU utilization on auth-service",
            "service": "auth-service",
            "severity": "SEV-1",
            "transcript": "[ev_101] Customer reported 500 errors.\n[ev_102] CPU usage spiked to 99%.\n[ev_103] Memory leak suspected in worker process.",
            "allowedRootCauses": ["cpu_exhaustion", "memory_leak", "db_deadlock"]
        },
        "toolCatalog": [
            {"name": "query_metrics", "description": "Query CPU/memory metrics", "inputSchema": {"type": "object"}},
            {"name": "rollback_deployment", "description": "Rollback auth-service deployment", "inputSchema": {"type": "object"}}
        ],
        "policy": {
            "maximumDiagnostics": 3,
            "effectTools": ["rollback_deployment"],
            "approvalRequiredFor": ["rollback_deployment"],
            "doNotExport": ["sensitive"]
        }
    }

    # 1. POST /v2/incidents
    res = client.post("/v2/incidents", json=payload)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    body = res.json()
    assert body["runId"] == run_id
    assert body["status"] == "waiting"
    assert "diagnosis" in body
    assert len(body["dispatches"]) >= 1

    first_dispatch = body["dispatches"][0]
    action_id = first_dispatch["actionId"]
    call_id = first_dispatch["callId"]

    # 2. Replay POST /v2/incidents (Idempotency check)
    res_replay = client.post("/v2/incidents", json=payload)
    assert res_replay.status_code == 200
    assert res_replay.json() == body

    # 3. POST /v2/incidents/{runId}/receipts (Diagnostic Outcome)
    receipt_payload = {
        "receiptId": "rec_diag_01",
        "outcomes": [
            {
                "actionId": action_id,
                "callId": call_id,
                "attempt": 1,
                "status": 200,
                "resultClass": "diagnosis_confirmed",
                "nonce": "nonce_uuid_123"
            }
        ]
    }
    res_receipt = client.post(f"/v2/incidents/{run_id}/receipts", json=receipt_payload)
    assert res_receipt.status_code == 200
    rec_body = res_receipt.json()
    
    # Since rollback_deployment requires approval, expect approvals list in response
    assert rec_body["status"] == "waiting"
    assert len(rec_body["approvals"]) == 1
    appr_id = rec_body["approvals"][0]["approvalId"]

    # 4. POST approval receipt
    approval_receipt_payload = {
        "receiptId": "rec_appr_01",
        "approvals": [
            {
                "approvalId": appr_id,
                "decision": "approved",
                "nonce": "nonce_appr_456"
            }
        ]
    }
    res_appr = client.post(f"/v2/incidents/{run_id}/receipts", json=approval_receipt_payload)
    assert res_appr.status_code == 200
    appr_body = res_appr.json()
    assert len(appr_body["dispatches"]) == 1
    effect_dispatch = appr_body["dispatches"][0]
    effect_action_id = effect_dispatch["actionId"]
    effect_call_id = effect_dispatch["callId"]

    # 5. POST effect receipt (Outcome of effect execution)
    effect_receipt_payload = {
        "receiptId": "rec_eff_01",
        "outcomes": [
            {
                "actionId": effect_action_id,
                "callId": effect_call_id,
                "attempt": 1,
                "status": 200,
                "resultClass": "effect_applied",
                "nonce": "nonce_eff_789"
            }
        ]
    }
    res_eff = client.post(f"/v2/incidents/{run_id}/receipts", json=effect_receipt_payload)
    assert res_eff.status_code == 200
    final_body = res_eff.json()
    assert final_body["status"] == "completed"
    assert final_body["chosenEffect"] == "rollback_deployment"
    assert "otlp" in final_body
    assert len(final_body["otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"]) >= 4

    # 6. GET /v2/incidents/{runId}
    res_get = client.get(f"/v2/incidents/{run_id}")
    assert res_get.status_code == 200
    assert res_get.json()["status"] == "completed"

    print("SUCCESS: All incident agent flow tests passed cleanly!")

if __name__ == "__main__":
    test_full_incident_flow()
