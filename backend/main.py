from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import APP_API_KEY, SOCIAL_INSTAGRAM_HANDLES, SOCIAL_X_HANDLES
from facts_loader import load_facts_records
from pipeline import fact_check_pipeline, run_batch
from realtime_evidence import fetch_social_feed

app = FastAPI(title="SatyaCheck API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

facts_records = load_facts_records()


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not APP_API_KEY:
        return
    if x_api_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class PostRequest(BaseModel):
    text: str = Field(..., min_length=3, description="Post or claim to verify")


class BatchRequest(BaseModel):
    posts: list[str] = Field(..., min_length=1, max_length=1000)
    mode: str = Field(default="smart", pattern="^(smart|naive)$")
    workers: int = Field(default=20, ge=1, le=100)


@app.get("/")
def home():
    return {
        "message": "SatyaCheck Backend Running",
        "endpoints": [
            "/fact-check",
            "/fact-check/smart",
            "/fact-check/naive",
            "/benchmark",
            "/live-news",
            "/social-feed",
        ],
    }


@app.get("/health")
def health():
    active_count = len([f for f in facts_records if f.get("status") == "active"])
    return {"status": "ok", "facts_loaded": len(facts_records), "active_facts": active_count}


@app.post("/fact-check")
def check(req: PostRequest, _: None = Depends(_require_api_key)):
    # Keep default route compatible with existing frontend.
    return fact_check_pipeline(req.text, facts_records, mode="smart")


@app.post("/fact-check/smart")
def check_smart(req: PostRequest, _: None = Depends(_require_api_key)):
    return fact_check_pipeline(req.text, facts_records, mode="smart")


@app.post("/fact-check/naive")
def check_naive(req: PostRequest, _: None = Depends(_require_api_key)):
    return fact_check_pipeline(req.text, facts_records, mode="naive")


@app.post("/benchmark")
def benchmark(req: BatchRequest, _: None = Depends(_require_api_key)):
    try:
        return run_batch(req.posts, facts_records, mode=req.mode, max_workers=req.workers)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/live-news")
def live_news(_: None = Depends(_require_api_key)):
    from news_ingestion import fetch_indian_news

    return {"posts": fetch_indian_news()}


@app.get("/social-feed")
def social_feed(_: None = Depends(_require_api_key)):
    rows = fetch_social_feed(
        x_handles=SOCIAL_X_HANDLES,
        instagram_handles=SOCIAL_INSTAGRAM_HANDLES,
        limit_per_handle=6,
    )
    return {
        "count": len(rows),
        "x_handles": SOCIAL_X_HANDLES,
        "instagram_handles": SOCIAL_INSTAGRAM_HANDLES,
        "posts": rows,
    }