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


class ChatReq(BaseModel):
    message: str
    session_id: Union[str, None] = None
    use_history: bool = False   # blend the session's liked books into the query (UC3)


def create_app(rec_service, feed_service, session_store, overview=None) -> FastAPI:
    app = FastAPI(title="Book Swipe")

    def cards(book_ids):
        # short label cards — search results and the reading list
        return [{"book_id": b, "label": rec_service.label(b)} for b in book_ids]

    def feed_cards(book_ids):
        # rich cards (full synopsis) for the book you're deciding on
        return [rec_service.card(b) for b in book_ids]

    def feed_for(session):
        return feed_cards(
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

    @app.post("/chat")
    def chat(req: ChatReq):
        if overview is None:
            raise HTTPException(status_code=503, detail="LLM chat unavailable (is Ollama running?)")
        history, history_titles = [], []
        if req.use_history and req.session_id:
            try:
                liked = session_store.get(req.session_id).liked
            except KeyError:
                liked = []
            history = list(liked)
            history_titles = [rec_service.card(b)["title"] for b in history]
        result = overview.generate(req.message, history=history, history_titles=history_titles)
        categories = [{
            "header": cat["header"],
            "items": [{**rec_service.card(it["book_id"]), "reason": it["reason"]}
                      for it in cat["items"]],
        } for cat in result["categories"]]
        return {"intro": result["intro"], "categories": categories}

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
    book_ids = catalog["book_id"].tolist()
    feed_service = FeedService(hybrid, emb, book_ids)

    # LLM chat (RAG overview) — optional: disabled gracefully if the encoder/Ollama are absent.
    overview = None
    try:
        from sentence_transformers import SentenceTransformer

        from book_recsys.features.document import build_documents
        from book_recsys.llm.clients import LiteLLMClient
        from book_recsys.llm.overview import OverviewGenerator
        from book_recsys.llm.retrieve import Retriever

        encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")   # matches the 384-d catalog
        retriever = Retriever(book_ids, emb, encoder=encoder)
        id_to_doc = dict(zip(book_ids, (d[:220] for d in build_documents(catalog))))
        client = LiteLLMClient("ollama/qwen2.5:7b", api_base="http://localhost:11434")
        overview = OverviewGenerator(retriever, id_to_doc, client, n=40)
    except Exception as exc:   # noqa: BLE001 — chat is optional; keep the rest of the UI working
        print("LLM chat disabled:", exc)

    app = create_app(rec_service, feed_service, SessionStore(), overview=overview)
    web = os.path.join(os.path.dirname(__file__), "..", "ui", "web")
    app.mount("/", StaticFiles(directory=web, html=True), name="web")  # serves the SPA
    return app
