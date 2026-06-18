"""FastAPI swipe API. `create_app` is injectable (tested with fakes); `get_app`
wires the real artifact-backed services for uvicorn."""
import logging
from dataclasses import asdict
from typing import Union

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from book_recsys.api.sessions import SessionStore

# logs to the uvicorn console (its "uvicorn.error" logger is configured at INFO by default)
log = logging.getLogger("uvicorn.error")


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
    use_history: bool = False  # blend the session's liked books into the query (UC3)


class SteerReq(BaseModel):
    message: str
    session_id: Union[str, None] = None
    k: int = 10


def create_app(rec_service, feed_service, session_store, overview=None,
               steerer=None, ranker=None) -> FastAPI:
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
            raise HTTPException(status_code=503,
                                detail="LLM chat unavailable (is Ollama running?)")
        history, history_titles = [], []
        if req.use_history and req.session_id:
            try:
                liked = session_store.get(req.session_id).liked
            except KeyError:
                liked = []
            history = list(liked)
            history_titles = [rec_service.card(b)["title"] for b in history]
        try:
            log.info("chat: %r (use_history=%s, history=%d)", req.message, req.use_history,
                     len(history))
            result = overview.generate(req.message, history=history, history_titles=history_titles)
        except Exception:  # noqa: BLE001 — Ollama down / model load OOM -> graceful 503
            log.exception("chat failed -> 503")  # surface the real traceback in the console
            raise HTTPException(status_code=503,
                                detail="LLM chat unavailable (is Ollama running?)")
        categories = [{
            "header":
            cat["header"],
            "items": [{
                **rec_service.card(it["book_id"]), "reason": it["reason"]
            } for it in cat["items"]],
        } for cat in result["categories"]]
        return {"intro": result["intro"], "categories": categories}

    @app.post("/steer")
    def steer(req: SteerReq):
        if steerer is None or ranker is None:
            raise HTTPException(status_code=503,
                                detail="LLM steering unavailable (is Ollama running?)")
        sid = session_store.ensure(req.session_id)
        session = session_store.get(sid)
        session_store.append_message(sid, "user", req.message)
        anchor_titles = [rec_service.card(b)["title"] for b in session.liked][:15]
        try:
            state = steerer.update(session.messages[-6:], session.steering, anchor_titles)
        except Exception:  # noqa: BLE001 — Ollama down / model load -> graceful 503
            log.exception("steer failed -> 503")
            raise HTTPException(status_code=503,
                                detail="LLM steering unavailable (is Ollama running?)")
        session_store.set_steering(sid, state)
        session_store.append_message(sid, "assistant", state.reply)
        anchor_id = None
        if state.anchor_book:
            hits = rec_service.search(state.anchor_book, 1)
            anchor_id = hits[0] if hits else None
        book_ids = ranker.rank(state, session.liked, session.seen, k=req.k, anchor_id=anchor_id)
        return {"session_id": sid, "reply": state.reply, "state": asdict(state),
                "cards": [rec_service.card(b) for b in book_ids]}

    return app


def _find(name: str) -> str:  # pragma: no cover
    import glob
    for pat in (f"artifacts/{name}", f"../artifacts/{name}", f"../../artifacts/{name}"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under artifacts/ (run notebooks/07_models.ipynb)")


class _LazyOverview:  # pragma: no cover
    """Builds the heavy LLM/encoder stack on first use, then reuses it."""

    def __init__(self, build):
        self._build = build
        self._gen = None

    def generate(self, *args, **kwargs):
        if self._gen is None:
            self._gen = self._build()
        return self._gen.generate(*args, **kwargs)


def _build_overview(catalog, emb, book_ids):  # pragma: no cover
    from sentence_transformers import SentenceTransformer

    from book_recsys import config
    from book_recsys.features.document import build_documents
    from book_recsys.llm.clients import LiteLLMClient
    from book_recsys.llm.overview import OverviewGenerator
    from book_recsys.llm.retrieve import Retriever

    log.info("LLM stack: building (first /chat call) ...")
    # Pin faiss to one OpenMP thread. This process already holds two OpenMP runtimes
    # (scikit-learn/MKL via models.joblib + torch via the encoder); letting faiss spin up
    # a third thread team segfaults on macOS/Anaconda at index-build time. Single-threaded
    # flat search over ~468k vectors is sub-millisecond, so this costs nothing here.
    import faiss
    faiss.omp_set_num_threads(1)
    log.info("LLM stack: loading encoder %s", config.EMBED_MODEL)
    encoder = SentenceTransformer(config.EMBED_MODEL)  # must match the 384-d catalog
    log.info("LLM stack: building FAISS index over %d x %d embeddings", *emb.shape)
    retriever = Retriever(book_ids, emb, encoder=encoder)
    log.info("LLM stack: building catalog documents")
    id_to_doc = dict(zip(book_ids, (d[:220] for d in build_documents(catalog))))
    client = LiteLLMClient(config.LLM_MODEL, api_base=config.LLM_API_BASE)
    log.info("LLM stack: ready (LLM=%s @ %s)", config.LLM_MODEL, config.LLM_API_BASE)
    return OverviewGenerator(retriever, id_to_doc, client, n=config.OVERVIEW_N)


def _build_steer(models, catalog, emb, book_ids):  # pragma: no cover
    import faiss
    from sentence_transformers import SentenceTransformer

    from book_recsys import config
    from book_recsys.llm.clients import LiteLLMClient
    from book_recsys.llm.rank import SteeredRanker
    from book_recsys.llm.retrieve import Retriever
    from book_recsys.llm.steer import Steerer

    faiss.omp_set_num_threads(1)  # avoid the sklearn+torch+faiss OpenMP segfault
    encoder = SentenceTransformer(config.EMBED_MODEL)
    retriever = Retriever(book_ids, emb, encoder=encoder)
    genre = (dict(zip(catalog["book_id"], catalog["genre"]))
             if "genre" in catalog.columns else None)
    ranker = SteeredRanker(models["hybrid_cf_content"], retriever, models["similar"], emb,
                           book_ids, encoder, catalog_genre=genre)
    steerer = Steerer(LiteLLMClient(config.LLM_MODEL, api_base=config.LLM_API_BASE))
    return steerer, ranker


class _LazySteer:  # pragma: no cover
    """Builds the steer stack (encoder + FAISS + ranker) on first use, then reuses it."""

    def __init__(self, build):
        self._build = build
        self._pair = None

    def _ensure(self):
        if self._pair is None:
            self._pair = self._build()
        return self._pair

    def update(self, *args, **kwargs):
        return self._ensure()[0].update(*args, **kwargs)

    def rank(self, *args, **kwargs):
        return self._ensure()[1].rank(*args, **kwargs)


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

    # LLM chat (RAG overview) is built LAZILY on the first /chat call — it loads a
    # sentence encoder + a FAISS index that would otherwise bloat startup memory (the
    # hybrid model alone is ~3GB). The swipe UI never waits on it, and if the encoder /
    # Ollama can't load, /chat just returns 503 while everything else keeps working.
    overview = _LazyOverview(lambda: _build_overview(catalog, emb, book_ids))
    steer = _LazySteer(lambda: _build_steer(models, catalog, emb, book_ids))
    app = create_app(rec_service, feed_service, SessionStore(), overview=overview,
                     steerer=steer, ranker=steer)
    web = os.path.join(os.path.dirname(__file__), "..", "ui", "web")
    app.mount("/", StaticFiles(directory=web, html=True), name="web")  # serves the SPA
    return app
