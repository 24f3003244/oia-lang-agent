from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# --- OpenAI Structured Output Schemas ---

class DiagnosticCallSpec(BaseModel):
    toolName: str = Field(description="Name of the diagnostic tool from the tool catalog")
    arguments: Dict[str, Any] = Field(description="Incident-specific arguments matching tool input schema")
    evidence: List[str] = Field(description="Citation of 1 to 4 evidence IDs supporting this specific diagnostic call")

class DiagnosisAndPlan(BaseModel):
    rootCause: str = Field(description="Selected root cause, exactly one value from allowedRootCauses")
    evidence: List[str] = Field(description="Two to four evidence IDs cited from the transcript (e.g. ['ev_101', 'ev_102'])")
    diagnosticCalls: List[DiagnosticCallSpec] = Field(description="1 to 3 necessary diagnostic tool calls to confirm root cause")
    effectToolName: str = Field(description="Selected recovery effect tool from policy.effectTools")
    effectArguments: Dict[str, Any] = Field(description="Arguments for the recovery effect tool matching its schema")


# --- API Request Schemas ---

class IncidentData(BaseModel):
    incidentId: Optional[str] = None
    title: Optional[str] = None
    service: Optional[str] = None
    severity: Optional[str] = None
    transcript: str
    allowedRootCauses: List[str]

class PolicyData(BaseModel):
    maximumDiagnostics: int = 3
    effectTools: List[str]
    approvalRequiredFor: List[str] = []
    doNotExport: List[str] = []

class IncidentRequest(BaseModel):
    profile: str
    runId: str
    agentName: str
    publicMarker: str
    sensitive: Optional[Dict[str, Any]] = None
    incident: IncidentData
    toolCatalog: List[Dict[str, Any]]
    policy: PolicyData


# --- Receipt Request Schemas ---

class ReceiptOutcome(BaseModel):
    actionId: str
    callId: str
    attempt: int
    status: int
    resultClass: Optional[str] = None
    nonce: Optional[str] = None
    errorType: Optional[str] = None

class ApprovalReceipt(BaseModel):
    approvalId: str
    decision: str
    nonce: Optional[str] = None

class ReceiptRequest(BaseModel):
    receiptId: str
    outcomes: Optional[List[ReceiptOutcome]] = None
    approvals: Optional[List[ApprovalReceipt]] = None
