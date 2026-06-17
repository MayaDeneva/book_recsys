"""FastAPI swipe API. `create_app` is injectable (tested with fakes); `get_app`
wires the real artifact-backed services for uvicorn."""
from typing import Union

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from book_recsys.api.sessions import SessionStore


class SessionReq(BaseModel):
    liked: list = []
    lam: float = Field(default=1.0, ge=0.0)  # penalty strength; negative would reward disliked
    k: int = 10


class SwipeReq(BaseModel):
    session_id: str
    book_id: Union[int, str]
    action: str


def create_app(rec_service, feed_service, session_store) -> FastAPI:
    app = FastAPI(title="Book Swipe")

    def cards(book_ids):
        return [{"book_id": b, "label": rec_service.label(b)} for b in book_ids]

    def feed_for(session):
        return cards(
            feed_service.next(session.liked,
                              session.disliked,
                              session.seen,
                              k=session.k,
                              lam=session.lam))

    @app.get("/search")
    def search(q: str, limit: int = Query(default=20, ge=1, le=200)):
        return cards(rec_service.search(q, limit))

    @app.post("/session")
    def session(req: SessionReq):
        sid = session_store.create(req.liked, req.lam, req.k)
        return {"session_id": sid, "cards": feed_for(session_store.get(sid))}

    @app.post("/swipe")
    def swipe(req: SwipeReq):
        try:
            s = session_store.apply(req.session_id, req.book_id, req.action)
        except KeyError:
            raise HTTPException(status_code=404, detail="unknown session")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return {"cards": feed_for(s), "reading_list": cards(s.reading_list)}

    return app


def _find(name: str) -> str:  # pragma: no cover
    import glob
    for pat in (f"artifacts/{name}", f"../artifacts/{name}", f"../../artifacts/{name}"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under artifacts/ (run notebooks/07_models.ipynb)")


def get_app() -> FastAPI:  # pragma: no cover
    """Production app for `uvicorn book_recsys.api.app:get_app --factory`."""
    import os

    import joblib
    import numpy as np
    import pandas as pd
    from fastapi.staticfiles import StaticFiles

    from book_recsys.ui.feed import FeedService
    from book_recsys.ui.service import RecommenderService

    models = joblib.load(_find("models.joblib"))
    catalog = pd.read_parquet(_find("catalog.parquet"))
    emb = np.load(_find("embeddings.npy"))

    hybrid = models["hybrid_cf_content"]
    rec_service = RecommenderService(catalog, {"hybrid": hybrid}, models["similar"])
    feed_service = FeedService(hybrid, emb, catalog["book_id"].tolist())

    app = create_app(rec_service, feed_service, SessionStore())
    web = os.path.join(os.path.dirname(__file__), "..", "ui", "web")
    app.mount("/", StaticFiles(directory=web, html=True), name="web")  # serves the SPA
    return app
