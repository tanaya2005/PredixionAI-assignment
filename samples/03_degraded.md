# Degraded Mode Execution Sample (`03_degraded.json`)

## Research Question
> "Compare Qdrant vs Milvus vector databases for RAG pipelines."

## Status
`status="degraded"`

## Cause of Degradation
Primary search provider (Tavily) failed with HTTP 500 / rate-limit exhaustion, triggering the automatic fallback chain to DuckDuckGo. The fallback produced reduced candidate diversity, yielding only a single surviving verified document.

## Contract Integrity & Safeguards
1. **Never 500:** Even when primary providers fail and candidate volume drops, the pipeline completes gracefully and produces a fully validated schema.
2. **Explicit Limitation:** The limitation block records: `"Primary search provider returned HTTP 500 / 429; pipeline fell back to secondary search engine."`
3. **Hard Confidence Overrides Applied:**
   - **Single Domain:** Score capped at `<= 0.60`.
   - **Single Finding:** Level hard-capped at `"low"` (`score < 0.50`).
4. **Concrete Next Steps:** The model outputs actionable recovery instructions: `"Retry query once primary search provider rate limits reset."`
