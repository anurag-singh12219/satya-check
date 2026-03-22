# SatyaCheck Project Document

## 1. Project Summary

SatyaCheck is a multilingual automated fact-checking system for vernacular social/news claims.
It verifies claims in Hindi, Tamil, Bengali, and English as:

- TRUE
- FALSE
- UNVERIFIABLE

The system is designed for high-throughput processing with measurable cost and token efficiency.

## 2. Problem Statement

Manual fact-checking is too slow for viral misinformation. A scalable pipeline is needed to process thousands of claims per minute while avoiding confusion from irrelevant or outdated facts.

## 3. Approach

### Naive Mode

- Uses broad/full fact context for each check.
- Serves as baseline for comparison.

### Smart Mode

- Compacts claim text.
- Retrieves relevant fact candidates first.
- Uses ScaleDown-aware context reduction.
- Applies deterministic contradiction checks.
- Falls back to live evidence when local facts are insufficient.

## 4. Tech Stack

- Python
- FastAPI
- Pydantic
- Requests
- Uvicorn
- HTML/CSS/JavaScript (frontend)
- ScaleDown API integration

## 5. Repository Structure

- `backend/` API, pipeline, retrieval, evidence modules
- `frontend/` web UI
- `scripts/` evaluation and submission report generators
- `.env.example` env template
- `README.md` detailed usage

## 6. Setup and Run

From project root:

```powershell
cd backend
pip install -r requirements.txt
copy ..\.env.example ..\.env
..\venv\Scripts\python.exe -m uvicorn main:app --reload
```

## 7. API Endpoints

- `GET /health`
- `POST /fact-check`
- `POST /fact-check/smart`
- `POST /fact-check/naive`
- `POST /benchmark`
- `GET /live-news`
- `GET /social-feed`

## 8. Evaluation and Benchmark Commands

From project root:

```powershell
..\venv\Scripts\python.exe scripts\evaluate_accuracy.py
..\venv\Scripts\python.exe scripts\generate_submission_report.py
```

Generated artifacts:

- `scripts/evaluation_report_smart.json`
- `scripts/evaluation_report_naive.json`
- `scripts/submission_report.md`
- `scripts/submission_report.json`

## 9. Submission Guidance

- Include this repository as public GitHub URL.
- Keep generated reports in the repo for reviewer verification.
- `submission_report` files are optional for upload forms, but strongly recommended in GitHub because they show measurable proof (throughput, token savings, cost, accuracy).

## 10. Final Checklist

- [ ] Public GitHub repository created
- [ ] `.env` not committed
- [ ] API runs locally
- [ ] Evaluation report generated
- [ ] Submission report generated
- [ ] README and this document updated
