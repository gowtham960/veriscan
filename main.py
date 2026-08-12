"""Quick runner to sanity-check the graph end to end with a few input types."""
import os
from graph import build_graph
from utils.persistence import init_db, persist_run, enqueue_for_review, new_run_id

# LangGraph is built on LangChain's Runnable protocol, so it inherits tracing
# automatically once LANGSMITH_TRACING/LANGSMITH_API_KEY are set in the
# environment — no wrapper code needed for the graph itself. This just groups
# traces under a named project instead of the default one.
os.environ.setdefault("LANGSMITH_PROJECT", "veriscan")

app = build_graph()
init_db()


def run(source_type: str, raw_input: str):
    print(f"\n{'=' * 70}\nINPUT: source_type={source_type!r}, raw_input={raw_input!r}\n{'=' * 70}")
    run_id = new_run_id()

    # internal_run_id in trace metadata lets you jump from a SQLite audit_log
    # row to the matching LangSmith trace (and back) using the same ID.
    result = app.invoke(
        {"source_type": source_type, "raw_input": raw_input, "audit_log": []},
        config={"run_name": "veriscan_pipeline", "tags": [source_type], "metadata": {"internal_run_id": run_id}},
    )

    # Persist the audit trail for this run (immutable, append-only)
    persist_run(run_id, result["audit_log"])

    # If flagged, actually enqueue it for human review — not just a boolean nobody acts on
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

    print(f"\nRun ID: {run_id}")
    print(f"Routing: {result['routing_reason']}")
    print(f"\n--- FINAL VERDICT ---\n{result['final_verdict']}")
    print(f"\nOverall confidence: {result['confidence_score']}")
    print(f"Faithfulness score: {result['faithfulness_score']}")
    print(f"Requires human review: {result['requires_human_review']}")
    print(f"\nAudit trail ({len(result['audit_log'])} entries, persisted to veriscan.db):")
    for e in result["audit_log"]:
        print(f"  [{e['node']}] {e['output_summary']}")
    return run_id, result


if __name__ == "__main__":
    # 1. Plain text claim — should only route to fact-check
    run("text_claim", "The Eiffel Tower was built in 1889 for the World's Fair.")

    # 2. Video upload — should route to both fact-check and media-check
    run("video_upload", "test_video.mp4")
