# Contributing

## Setup

1. Create and activate a virtual environment.
2. Install backend dependencies:
   - `cd backend`
   - `pip install -r requirements.txt`
3. Copy env template:
   - `copy ..\\.env.example ..\\.env`
4. Run API:
   - `..\\venv\\Scripts\\python.exe -m uvicorn main:app --reload`

## Development Guidelines

- Keep API responses backward compatible where possible.
- Add measurable outputs for performance and cost-impact changes.
- Prefer deterministic, evidence-based verdicting over unsupported guesses.
- Keep multilingual normalization changes covered in `scripts/eval_claims.json`.

## Validation Before PR

- `..\\venv\\Scripts\\python.exe scripts\\evaluate_accuracy.py`
- `..\\venv\\Scripts\\python.exe scripts\\generate_submission_report.py`

## Security

- Never commit `.env` or secret keys.
- Use `.env.example` for placeholder variables only.
