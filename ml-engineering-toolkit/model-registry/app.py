from __future__ import annotations

"""Streamlit UI for the Mini Model Registry."""

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import streamlit as st
from registry import Registry

REGISTRY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry")
MODEL_NAME = "churn_clf"

STAGE_COLORS = {
    "production": "#1a7f37",  # green
    "staging": "#bf8700",     # amber
    "archived": "#8b949e",    # grey
}


def seed_registry(reg: Registry) -> None:
    """Populate a fresh registry with a few demo versions."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=600, n_features=20, n_informative=8, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    for i, C in enumerate([0.01, 1.0, 10.0], start=1):
        model = LogisticRegression(C=C, max_iter=1000).fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        preds = model.predict(X_te)
        metrics = {
            "auc": round(float(roc_auc_score(y_te, proba)), 4),
            "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        }
        reg.register(
            MODEL_NAME,
            model,
            metrics=metrics,
            params={"C": C},
            created_at=f"2026-07-17T09:0{i}:00",
        )


@st.cache_resource
def get_registry() -> Registry:
    reg = Registry(REGISTRY_DIR)
    if not reg.list_models(MODEL_NAME):
        seed_registry(reg)
    return reg


def color_stage(val: str) -> str:
    color = STAGE_COLORS.get(val, "#8b949e")
    return f"background-color: {color}; color: white; font-weight: 600;"


def main() -> None:
    st.set_page_config(page_title="Mini Model Registry", page_icon="🗂️", layout="wide")
    st.title("🗂️ Mini Model Registry")
    st.caption(
        "Pain point: *\"We lose track of model versions.\"*  "
        "A versioned local store + JSON metadata index + stage promotion, "
        "so \"which model is in production?\" has a one-word answer."
    )

    reg = get_registry()
    records: List[Dict] = reg.list_models(MODEL_NAME)
    frame = reg.to_frame()

    # ---- metric cards --------------------------------------------------
    prod = [r for r in records if r["stage"] == "production"]
    prod_version = prod[0]["version"] if prod else None
    n_models = frame["name"].nunique() if not frame.empty else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Models", n_models)
    c2.metric("Versions", len(records))
    c3.metric(
        "Production version",
        f"v{prod_version}" if prod_version is not None else "none",
    )

    # ---- registry table ------------------------------------------------
    st.subheader("Registry")
    styled = frame.style.map(color_stage, subset=["stage"])
    st.dataframe(styled, use_container_width=True)

    # ---- metric comparison chart --------------------------------------
    st.subheader("Compare versions")
    metric_cols = [c for c in frame.columns if c in ("auc", "accuracy")]
    if metric_cols:
        metric = st.selectbox("Metric", metric_cols, index=0)
        sub = frame.dropna(subset=[metric]).sort_values("version")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = [
            STAGE_COLORS["production"] if s == "production" else "#4c78a8"
            for s in sub["stage"]
        ]
        bars = ax.bar([f"v{v}" for v in sub["version"]], sub[metric], color=colors)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} per version (production highlighted green)")
        ax.set_ylim(0, max(1.0, float(sub[metric].max()) * 1.15))
        for bar, val in zip(bars, sub[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        st.pyplot(fig)

    # ---- promotion controls -------------------------------------------
    st.subheader("Promote to production")
    versions = sorted(r["version"] for r in records)
    if versions:
        sel = st.selectbox(
            "Select a version to promote",
            versions,
            format_func=lambda v: f"v{v}",
        )
        if st.button("Promote to production", type="primary"):
            reg.promote(MODEL_NAME, int(sel), "production")
            st.success(
                f"v{sel} promoted to production. Any previous production version "
                "was auto-archived (single-production invariant)."
            )
            st.rerun()

    # ---- best helper ---------------------------------------------------
    if metric_cols:
        best = reg.best(MODEL_NAME, metric_cols[0], higher_is_better=True)
        st.info(
            f"Best by {metric_cols[0]}: v{best['version']} "
            f"({metric_cols[0]}={best['metrics'][metric_cols[0]]})."
        )


if __name__ == "__main__":
    main()
