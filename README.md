# VeriScan — LangGraph Skeleton (Step 1)
![CI](https://github.com/gowtham960/veriscan/actions/workflows/ci.yml/badge.svg)
This is the working end-to-end skeleton from the architecture doc: ingestion → routing →
parallel fan-out (Fact-Triangulation Agent + Media Authenticity Agent) → reconciliation →
faithfulness guardrail → policy gate.

All agent logic is **stubbed** (dummy verdicts, no real API calls) so you can verify the
graph wiring, conditional routing, and parallel fan-out/fan-in work correctly before
plugging in real intelligence.

## Setup

```bash
pip install langgraph langchain-core python-dotenv groq tavily-python langsmith fastapi uvicorn python-multipart
export GROQ_API_KEY=gsk_...      # free tier available at console.groq.com
export TAVILY_API_KEY=tvly_...   # free tier available at tavily.com (~1,000 searches/month)

# Optional — only needed for paid deepfake/voice-clone detection APIs.
# Without these, the Media Authenticity Agent still runs on free ffprobe/ffmpeg
# metadata forensics and returns a real (if lower-confidence) verdict.
export HIVE_API_KEY=...          # optional, AI-image/video detection
export VOICE_CLONE_API_KEY=...   # optional, voice-clone detection

# Optional — LangSmith tracing (pure observability overlay; the faithfulness
# guardrail runs on Groq regardless of whether this is set)
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_TRACING=true
```

Also requires **ffmpeg** installed on the system (`ffprobe`/`ffmpeg` on PATH) for the
Media Authenticity Agent's free metadata and audio-extraction checks.

All API keys above are optional for running the skeleton — if any are missing, the
corresponding agent returns a clearly-labeled stub/degraded result (`status: "skipped"`
or a lower-confidence real result) instead of failing, so the graph always runs
end-to-end at $0 cost by default.

## Run

```bash
python main.py
```

This runs two sample inputs:
1. A plain text claim — only the Fact-Triangulation Agent should run
2. A video upload — both agents should run in parallel

You'll see the routing decision, the parallel agents firing, the reconciled verdict, and
the full audit trail for each run.

## Key implementation note

`audit_log` in `state.py` uses `Annotated[list, operator.add]`. This is required because
`fact_check_agent` and `media_check_agent` run in the **same LangGraph superstep** (true
parallel execution) and both write to `audit_log`. Without the reducer, LangGraph raises
`InvalidUpdateError` since it can't tell how to merge two concurrent writes to the same key.
Any state field written by more than one parallel node needs this pattern.

## Next steps (from the architecture doc's build order)

3. ~~Implement Fact-Triangulation Agent fully~~ ✅ done — `agents/fact_check.py` uses
   Tavily (search) + Groq/Llama (synthesis), not Claude, to keep this high-volume node
   cheap. A source-credibility guardrail is layered on top of the model's self-reported
   ratings. Falls back to a labeled stub when API keys are missing.
4. ~~Implement Media Authenticity Agent~~ ✅ done — `agents/media_check.py` uses FREE,
   local signal extraction by default: ffprobe metadata forensics (AI-generator tool
   signatures, encoder tags) and C2PA manifest presence scanning, plus real ffmpeg
   audio-track extraction with a coarse duration/readability check. Paid APIs (Hive
   for AI-image detection, a voice-clone detector) are wired as optional plug-ins —
   only called if their env vars are set, so the default path costs $0.
5. ~~Add timeout/retry/partial-failure handling per node~~ ✅ done —
   `utils/resilience.py` wraps both agent calls with a timeout + exponential-backoff
   retry decorator. **Note**: the first version used `ThreadPoolExecutor`, which
   turned out to be a real bug — its worker threads are non-daemon, so a genuinely
   hung call (e.g. a network request that never returns) would prevent the whole
   Python process from exiting even after the wrapper "gave up" and returned a
   timeout result. Rewrote it with raw daemon `threading.Thread` instead. Verified
   with an actual hanging function: the graph now completes and correctly sets
   `requires_human_review=True` when an agent times out, instead of hanging forever.
6. ~~Add security layer~~ ✅ done — `utils/security.py` implements prompt-injection
   scanning (regex-based, flags rather than silently strips — so real content isn't
   hidden from human review), output sanitization (strips script tags/control chars
   before untrusted content re-enters a prompt), and a per-agent tool allowlist.
   Wired into ingestion (transcript scanned before any agent sees it) and the
   fact-check agent (search results sanitized + scanned before entering the LLM
   prompt). Policy gate now escalates to human review on a flagged injection.

   **Bugs caught during adversarial testing** (asked "are you sure that's the only
   bug?" and stress-tested further instead of assuming the first fix was enough):
   - Original regex used a single optional modifier group (`?`), so "IGNORE ALL
     PREVIOUS INSTRUCTIONS" (two modifiers) slipped through. Fixed with `*`.
   - The "you are now" pattern was too broad and **false-positived** on innocent
     text like "You are now viewing our privacy policy" — fixed by requiring
     role-assignment continuation (`you are now a/an/free/unrestricted/...`).

   **Known, unfixed limitations** (documented rather than papered over):
   - Vocabulary gaps: phrasing like "disregard the guidelines above" or
     "disregard everything I told you before" isn't caught — the pattern list
     doesn't cover every synonym/phrasing of an injection attempt.
   - Unicode homoglyph attacks bypass detection entirely (e.g. a Cyrillic "о"
     substituted into "ignore" defeats the ASCII-based regex).

   Regex-based detection is fundamentally a losing blocklist game against a
   creative attacker. The correct production fix is a small classifier or
   LLM-based prompt-injection guard model, not an ever-growing pattern list —
   this is flagged here as a real architectural next step, not silently ignored.
7. ~~Wire the Faithfulness Guardrail to a real eval~~ ✅ done — `agents/faithfulness.py`
   decomposes the verdict into individual claims and checks each against the
   collected evidence via Groq (kept cheap, same reasoning as the fact-check agent).
   Wrapped with LangSmith's `@traceable` decorator for tracing — this is a no-op
   passthrough if `LANGSMITH_API_KEY` isn't set, so LangSmith is an observability
   overlay, not a hard dependency for the guardrail to actually function.

   **Honest testing limitation**: the sandbox this was built in only allows network
   access to `api.anthropic.com` and package registries — `api.groq.com` and
   `api.tavily.com` are NOT reachable from it. So only the no-key fallback path and
   the JSON-parsing/scoring logic (tested in isolation with mock LLM output) were
   actually verified here. **The real Groq API call itself has not been tested and
   needs verification once you run this locally with a real `GROQ_API_KEY`.**
8. ~~Add audit logging persistence + human-review escalation flow~~ ✅ done —
   `utils/persistence.py` uses SQLite (not Supabase — free, zero external network
   dependency, and portable SQL means swapping to Postgres/Supabase later is a
   connection-string change, not a rewrite). Two tables: an immutable `audit_log`
   and a `review_queue` that flagged runs actually land in — this makes
   `requires_human_review=True` a concrete, actionable queue instead of a boolean
   nothing consumes. `main.py` now persists every run and enqueues flagged ones.
   Verified end-to-end: ran the graph, queried the DB independently to confirm
   the data survived, then resolved a pending review and confirmed the queue
   count dropped.
9. ~~Add LangSmith tracing~~ ✅ done — `@traceable` added to `run_fact_check`,
   `_search` (retrieval traced separately from synthesis), and `run_media_check`
   (faithfulness was already traced in Step 7). The graph itself needs no wrapper
   code: LangGraph is built on LangChain's `Runnable` protocol, so it inherits
   tracing automatically once `LANGSMITH_TRACING`/`LANGSMITH_API_KEY` are set —
   `main.py` just sets a project name and passes our internal SQLite `run_id` as
   trace metadata (`internal_run_id`), so a flagged run in the review queue can
   be traced back to its exact LangSmith run and vice versa.

   **Honest testing limitation** (same category as Step 7): `api.smith.langchain.com`
   isn't reachable from this sandbox, so only the no-key no-op path was verified —
   confirmed the graph still runs cleanly with tracing config attached. **Whether
   traces actually appear correctly in a real LangSmith dashboard has not been
   verified and needs checking once you run this locally with a real key.**
10. ~~Frontend~~ ✅ done — `api.py` (FastAPI, wraps the graph + persistence, serves
    the frontend as static files so it's one process to run) and
    `frontend/index.html` (single self-contained file, no build step).

    **Design**: built around the product's actual substance rather than a generic
    dashboard — a "chain of custody" trace renders the real `audit_log` as the
    hero interaction, and a rotated case-file stamp (VERIFIED/FLAGGED) marks the
    verdict. Dark ink palette (`#101820`) with a teal-verified/amber-flagged
    accent system, Space Grotesk headlines + IBM Plex Mono for evidence/IDs/hashes
    (ties the typography to the audit-trail concept). Includes a Review Queue tab
    wired to the Step 8 persistence layer — approve/reject actually updates SQLite.

    **Honest limitations**:
    - Verified functionally end-to-end (server boots, static file serves, `/verify`
      and `/reviews` both work against a live running instance, HTML/CSS parses
      with no structural errors) — but **the visual design has NOT been checked
      by eye**. Both a headless browser (Playwright) and the Google Fonts CDN
      require network access this sandbox doesn't have, so I could not screenshot
      it or confirm fonts actually load. **Open `frontend/index.html` yourself
      (with `api.py` running) and visually check it before you demo this.**
    - No live "agent status" polling — the UI shows a single loading spinner and
      then the full result, not a per-agent live status view (that would need
      the graph to stream intermediate state, which isn't wired up).
    - No auth — anyone with network access to the API can submit/resolve reviews.
      Fine for a local demo, not for any real deployment.

## Running tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite (29 tests) runs entirely against fallback/stub paths — no API
keys required — so it also runs in CI without secrets. It covers:
- `tests/test_security.py` — injection detection, sanitization, tool allowlist,
  including regression tests for the two real bugs found during adversarial
  testing (multi-modifier injection bypass, "you are now" false positive)
- `tests/test_resilience.py` — timeout/retry behavior, including a regression
  test for the daemon-thread bug (non-daemon `ThreadPoolExecutor` threads would
  have blocked process exit on a genuinely hung call)
- `tests/test_persistence.py` — SQLite audit log + review queue CRUD
- `tests/test_graph.py` — end-to-end graph wiring: conditional routing, parallel
  fan-out/fan-in, audit log accumulation, injection escalation to human review

**Not covered by these tests**: real Groq/Tavily/LangSmith API responses (would
need real keys as CI secrets), frontend rendering, or Whisper transcription.

CI (`.github/workflows/ci.yml`) runs this suite automatically on every push/PR
to `main`.

## Running the full stack

```bash
python api.py          # or: uvicorn api:app --reload
# then open http://localhost:8000 in a browser
```

## Files

- `state.py` — shared `AgentState` schema (TypedDict)
- `graph.py` — node definitions + graph assembly
- `main.py` — quick test runner
