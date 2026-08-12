"""
Faithfulness / Hallucination Guardrail — real implementation.

Decomposes the reconciled verdict into individual claims and checks each one
against the collected evidence, using Groq (cheap) rather than Claude, since
this runs on every single request and the model-tiering strategy reserves the
expensive model for the lowest-volume, highest-stakes node (there isn't one
here yet — see the honest note in graph.py about reconcile_node currently
being pure rule-based logic with no LLM call at all).

LangSmith is used for TRACING/logging this check, not as a required dependency
for it to function: if LANGSMITH_API_KEY isn't set, tracing is simply a no-op
and the faithfulness logic still runs for real via Groq. This mirrors the
separation described in the architecture doc — LangSmith is an observability
overlay, not something the guardrail logic depends on to work.
"""
import os
import re
import json
import time

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):
        """No-op fallback if langsmith isn't installed — decorator becomes a passthrough."""
        def decorator(func):
            return func
        return decorator

from utils.security import check_tool_allowed


GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are a faithfulness/groundedness checker inside a larger \
orchestration system. You will be given a verdict statement and a list of \
evidence items. Break the verdict down into its individual factual claims, and \
for EACH claim, determine whether it is directly supported by at least one \
evidence item.

Rules:
- A claim is "supported" only if a specific evidence item backs it up — not \
because it sounds plausible.
- Be strict: general statements ("no checks were applicable") don't need \
evidence, but specific verdicts (Corroborated/Contradicted/Authentic/etc.) do.
- Do not add your own outside knowledge — only judge against the evidence given.

Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "claims": [
    {"claim": "<the specific claim text>", "supported": true|false, "reason": "<short reason>"}
  ],
  "overall_score": <float 0.0-1.0, fraction of claims that are supported>
}"""


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:200]}")
    return json.loads(match.group(0))


@traceable(name="faithfulness_guardrail", run_type="chain")
def run_faithfulness_check(final_verdict: str, evidence_summary: list) -> dict:
    """Returns {'faithfulness_score': float, 'unsupported_claims': list, 'verdict_revised': bool}."""
    start = time.time()

    if not final_verdict or not final_verdict.strip():
        return {"faithfulness_score": 1.0, "unsupported_claims": [], "verdict_revised": False}

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key or Groq is None:
        # Degraded fallback: same naive check as the original stub, clearly labeled.
        unsupported = [] if evidence_summary else ["[STUB — GROQ_API_KEY not set; no real faithfulness check run] final_verdict has no supporting evidence"]
        score = 0.6 if unsupported else 0.75  # capped below 'confident' since this isn't a real check
        return {"faithfulness_score": score, "unsupported_claims": unsupported, "verdict_revised": bool(unsupported)}

    try:
        check_tool_allowed("faithfulness_guardrail", "groq_chat")
        client = Groq(api_key=groq_key)

        evidence_block = "\n".join(f"- {e}" for e in evidence_summary) if evidence_summary else "(no evidence provided)"

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Verdict:\n{final_verdict}\n\nEvidence:\n{evidence_block}"},
            ],
            temperature=0.0,
            max_tokens=800,
        )

        parsed = _extract_json(response.choices[0].message.content)
        claims = parsed.get("claims", [])
        unsupported = [c["claim"] for c in claims if not c.get("supported", True)]
        score = float(parsed.get("overall_score", 1.0 if not unsupported else 0.5))

        return {
            "faithfulness_score": round(score, 2),
            "unsupported_claims": unsupported,
            "verdict_revised": bool(unsupported),
        }

    except Exception as e:
        # Fail safe: an error in the guardrail itself should not silently pass
        # the verdict through as trustworthy — treat it as low-confidence.
        return {
            "faithfulness_score": 0.3,
            "unsupported_claims": [f"[faithfulness check errored: {str(e)[:150]}]"],
            "verdict_revised": True,
        }
