"""
VeriScan — shared state schema for the LangGraph app.
This mirrors the architecture doc's AgentState definition.
"""
from typing import TypedDict, Literal, Optional, Annotated
import operator


class AgentResult(TypedDict, total=False):
    verdict: str
    confidence: float
    evidence: list
    status: Literal["success", "failed", "timeout", "skipped"]
    latency_ms: int
    cost_usd: float


class MediaAgentResult(TypedDict, total=False):
    visual_verdict: str
    visual_confidence: float
    visual_evidence: list
    audio_verdict: str
    audio_confidence: float
    audio_evidence: list
    status: Literal["success", "partial", "failed", "timeout", "skipped"]
    latency_ms: int
    cost_usd: float


class AuditEntry(TypedDict, total=False):
    node: str
    timestamp: str
    input_hash: str
    output_summary: str
    model_used: str
    tokens_used: int


class AgentState(TypedDict, total=False):
    # Input
    source_type: Literal["text_claim", "url", "video_upload"]
    raw_input: str

    # Step 0 — extraction outputs
    transcript: Optional[str]
    key_frames: Optional[list]
    audio_track: Optional[str]
    metadata: dict

    # Step 1 — routing decision
    route_fact_check: bool
    route_media_check: bool
    routing_reason: str

    # Step 2 — parallel agent outputs
    fact_check_result: Optional[AgentResult]
    media_check_result: Optional[MediaAgentResult]

    # Step 3 — reconciliation
    final_verdict: str
    confidence_score: float
    evidence_summary: list

    # Step 3.5 — faithfulness guardrail
    faithfulness_score: float
    unsupported_claims: list
    verdict_revised: bool

    # Governance — Annotated with operator.add so parallel nodes (fact_check_agent,
    # media_check_agent) can each append an entry in the same superstep without conflict
    audit_log: Annotated[list, operator.add]
    requires_human_review: bool
