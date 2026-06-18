"""Streamlit book-recommender demo. Run: `streamlit run book_recsys/ui/app.py`.

Logic lives in book_recsys (RecommenderService + the recommenders); this file is the
thin UI shell. Needs artifacts/catalog.parquet, embeddings.npy, and models.joblib —
the last is built by notebooks/07_models.ipynb, so the UI loads instantly instead of
refitting models on launch.
"""
import glob

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from book_recsys.models.content.maxsim import MaxSimRecommender
from book_recsys.ui.service import RecommenderService

# UC1 method label -> key in models.joblib (only those present are shown)
_HIST_METHODS = {
    "Max-similarity — nearest to each liked book (most input-sensitive)": "maxsim",
    "SVD — collaborative filtering": "svd",
    "Hybrid — CF + content": "hybrid_cf_content",
    "Hybrid — CF + content (de-popularized)": "hybrid_cf_content_popneg",
    "Popularity": "popularity",
}
# methods whose recommend() accepts recency weights (escape the mean-pool / popularity centroid)
_WEIGHT_AWARE = {"maxsim", "svd"}


def _find(name: str) -> str:
    for pat in (f"artifacts/{name}", f"../artifacts/{name}", f"../../artifacts/{name}"):
        hits = glob.glob(pat)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"{name} not found under artifacts/ (run notebooks/07_models.ipynb)")


@st.cache_resource(show_spinner="Loading pre-built models…")
def load_service() -> RecommenderService:
    models = joblib.load(_find("models.joblib"))  # built by 07_models.ipynb
    catalog = pd.read_parquet(_find("catalog.parquet"))
    # max-sim is built here from the cached embeddings (aligned to catalog order), no retrain
    models["maxsim"] = MaxSimRecommender(list(catalog["book_id"]),
                                         np.load(_find("embeddings.npy")))
    hist = {label: models[key] for label, key in _HIST_METHODS.items() if key in models}
    return RecommenderService(catalog, hist, models["similar"])  # UC4 item-item


st.set_page_config(page_title="Book Recommender", page_icon="📚")
st.title("📚 Book Recommender")
service = load_service()

tab1, tab2 = st.tabs(["Recommend from my favorites (UC1)", "Books similar to one (UC4)"])

with tab1:
    st.caption("Pick a few books you love → recommendations from the model you choose.")
    method = st.selectbox("Recommendation method", service.methods(), key="uc1_method")
    recency = st.checkbox("Weight recent picks more (recency)", key="uc1_recency")
    weight_aware = _HIST_METHODS.get(method) in _WEIGHT_AWARE
    if recency and not weight_aware:
        st.caption("Recency applies to Max-similarity / SVD only — ignored for this method.")
    if "picks" not in st.session_state:
        st.session_state.picks = []
    query = st.text_input("Search a book to add", key="uc1_query")
    if query:
        match = st.selectbox("Matches",
                             service.search(query, limit=20),
                             format_func=service.label,
                             key="uc1_match")
        if st.button("➕ Add", key="uc1_add") and match:
            st.session_state.picks.append(match)
    st.write("**Your favorites:**")
    for book_id in st.session_state.picks:
        st.write("•", service.label(book_id))
    col_a, col_b = st.columns(2)
    if col_a.button("🎯 Recommend", key="uc1_go") and st.session_state.picks:
        st.subheader("Recommended for you")
        recs = service.recommend_by_history(st.session_state.picks,
                                            method,
                                            k=10,
                                            recency=recency and weight_aware)
        for rec in recs:
            st.write("•", rec)
    if col_b.button("Clear", key="uc1_clear"):
        st.session_state.picks = []
        st.rerun()

with tab2:
    st.caption("Pick a book → most similar books by content embedding.")
    query2 = st.text_input("Search a book", key="uc4_query")
    if query2:
        anchor = st.selectbox("Book",
                              service.search(query2, limit=20),
                              format_func=service.label,
                              key="uc4_anchor")
        if st.button("🔎 Find similar", key="uc4_go") and anchor:
            st.subheader(f"Books similar to *{service.label(anchor)}*")
            for rec in service.similar_to(anchor, k=10):
                st.write("•", rec)
