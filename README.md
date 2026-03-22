# SatyaCheck: Automated Fact-Checker for Vernacular News

Production-style backend pipeline for multilingual misinformation detection, built for the HPE x Intel GenAI Workshop.

## Problem

Social media misinformation in India spreads faster than manual fact-checking can scale.

## Objective

Process thousands of posts/minute while improving fact retrieval accuracy by avoiding irrelevant and outdated context.

## What This Project Demonstrates

1. Naive approach (send full facts database each time)
2. Optimized approach (ScaleDown query-aware compression + claim compaction)
3. Measured improvements in tokens, cost, and throughput
4. Real API backend that can be deployed and integrated with UI

## Core Design

### Naive Mode

- Input post goes directly to verification.
- Entire facts database is considered during matching (including outdated/superseded facts).
- High token usage, higher cost, more conflict risk.

### Smart Mode

- Step 1: Compress the post to strip conversational fluff and keep factual claims.
- Step 1.5: Apply multilingual claim normalization templates for Hindi, Tamil, and Bengali (digit normalization + domain term canonicalization).
- Step 2: Retrieve only active candidate facts by lexical relevance.
- Step 3: Use ScaleDown again to keep only query-relevant candidate evidence.
- Step 4: Run deterministic evidence scoring plus contradiction guards (numeric/year mismatch checks) to avoid wrong verdicts on conflicting values.
- Step 5: If still UNVERIFIABLE, fetch real-time trusted-source evidence (RSS/news) and re-evaluate stance with citations.
- Result: lower token usage, lower cost, less confusion from unrelated facts.

### Hybrid RAG Retrieval (Now Enabled)

- Retrieval is now metadata-aware hybrid RAG, not plain keyword matching.
- Candidate ranking combines:
	- lexical relevance,
	- source trust weighting,
	- freshness from date metadata,
	- optional semantic similarity (`ENABLE_SEMANTIC_RAG=1`).
- This improves old-vs-current fact selection and reduces stale evidence conflicts.

## Real-Time Coverage

- The pipeline does not rely only on predefined facts.
- For new/current claims with weak local matches, smart mode queries trusted live sources and includes citation links in response.
- If trusted sources do not provide explicit stance, the system returns UNVERIFIABLE rather than guessing.
- The live feed UI can ingest current social/news items through backend social and news endpoints.

## Backend Endpoints

- `GET /` basic service metadata
- `GET /health` health + facts count
- `POST /fact-check` smart mode default (keeps current frontend compatibility)
- `POST /fact-check/smart` explicit smart mode
- `POST /fact-check/naive` explicit naive mode
- `POST /benchmark` batch benchmark with parallel workers
- `GET /live-news` live headlines (or fallback feed)
- `GET /social-feed` monitored social stream (configured X/Instagram handles via RSS)

## Request/Response Examples

### Single Check

`POST /fact-check/smart`

```json
{
	"text": "नई नीति के तहत 12 लाख तक कोई टैक्स नहीं लगेगा"
}
```

Response contains:

- `result` (VERDICT + REASON)
- `tokens_before`, `tokens_after`, `token_savings_pct`
- `cost_estimate_usd`, `cost_if_naive_usd`, `cost_saved_usd`
- `latency_ms`, `compression_latency_ms`, `verifier_latency_ms`
- `claim_compaction` details

### Batch Benchmark

`POST /benchmark`

```json
{
	"mode": "smart",
	"workers": 50,
	"posts": [
		"UPI now has 2% charges for every user",
		"India won T20 World Cup 2024"
	]
}
```

Response contains throughput/min, aggregate token and cost metrics, and per-post outputs.

## Setup

1. Create virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill keys.
3. Run API.

```powershell
cd backend
pip install -r requirements.txt
copy ..\.env.example ..\.env
..\venv\Scripts\python.exe -m uvicorn main:app --reload
```

## Required Environment Variables

- `SCALEDOWN_API_KEY` required for real compression
- `NEWS_API_KEY` optional for live news feed
- `APP_API_KEY` optional API security key; if set, clients must send `x-api-key`
- `ENABLE_SEMANTIC_RAG` optional semantic retrieval toggle (`0`/`1`)
- `SEMANTIC_MODEL_NAME` optional embedding model id
- `RAG_TOP_K_SMART` and `RAG_TOP_K_NAIVE` candidate limits

If keys are missing, the backend uses safe fallback behavior so local development still runs.

## Submission Demo Flow (Workshop)

1. Run `POST /fact-check/naive` on 3-5 claims.
2. Run `POST /fact-check/smart` on the same claims.
3. Show `tokens_before vs tokens_after`, latency, and cost fields.
4. Run `POST /benchmark` with 50 workers and 100+ posts.
5. Report:
	 - cost/query before vs after
	 - token/query before vs after
	 - throughput posts/min
	 - verdict quality on known true/false claims

## Accuracy Evaluation Pack

- Gold set: `scripts/eval_claims.json`
- Evaluator: `scripts/evaluate_accuracy.py`
- Run:

```powershell
..\venv\Scripts\python.exe scripts\evaluate_accuracy.py
```

- Outputs:
	- `scripts/evaluation_report_smart.json`
	- `scripts/evaluation_report_naive.json`

These reports include accuracy, macro-F1, confusion matrix, by-language accuracy, confidence, and per-claim traces.

## Submission Report (One-Command)

To produce rubric-ready evidence for qualification (naive vs smart comparison, throughput, cost, token savings, accuracy, and pass/fail checklist):

```powershell
..\venv\Scripts\python.exe scripts\generate_submission_report.py
```

Generated files:

- `scripts/submission_report.md`
- `scripts/submission_report.json`

## Deployment Notes

- Keep API keys in cloud secret manager, never hardcode in source.
- Add CORS and auth if exposing public endpoints.
- Run with process manager (Gunicorn/Uvicorn workers or container orchestration).
- Add logging + tracing for latency and failure diagnostics.

## Project Structure

- `backend/main.py` FastAPI entrypoints
- `backend/pipeline.py` naive/smart pipelines and deterministic evidence engine
- `backend/facts_loader.py` fact DB loading and normalization
- `backend/news_ingestion.py` live feed ingestion
- `backend/data/facts_db.json` verified fact base with topic/source/date/status
- `frontend/fact_checker.html` real UI wired to backend APIs

