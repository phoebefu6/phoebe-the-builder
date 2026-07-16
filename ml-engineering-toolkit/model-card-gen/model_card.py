"""Model Card Generator — core logic.

Turns a trained scikit-learn model + a held-out test set into a
Google-style Model Card in Markdown: intended use, training details,
quantitative performance (overall + per-slice), and ethical/limitation
sections. The point Phoebe's team keeps hitting: models ship with zero
documentation, so nobody downstream knows what the model is for, what it
was trained on, or where it breaks.

Feed it what you have; it introspects the estimator for the rest.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _classification_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            m["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass
    return m


def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _slice_table(df_eval: pd.DataFrame, slice_name: str, task: str) -> str:
    """Per-group metric breakdown for a sensitive/segment column."""
    header = (
        f"| {slice_name} | n | accuracy | f1 |"
        if task == "classification"
        else f"| {slice_name} | n | rmse | r2 |"
    )
    lines = [header, "|---|---|---|---|"]
    for group, g in df_eval.groupby("slice"):
        n = len(g)
        if task == "classification":
            a = accuracy_score(g["y_true"], g["y_pred"])
            f = f1_score(g["y_true"], g["y_pred"], average="weighted", zero_division=0)
            lines.append(f"| {group} | {n} | {a:.4f} | {f:.4f} |")
        else:
            r = root_mean_squared_error(g["y_true"], g["y_pred"])
            r2 = r2_score(g["y_true"], g["y_pred"]) if n > 1 else float("nan")
            lines.append(f"| {group} | {n} | {r:.4f} | {r2:.4f} |")
    return "\n".join(lines)


def compute_metrics(
    model,
    X_test: pd.DataFrame,
    y_test,
    slices: Optional[pd.Series] = None,
    slice_name: str = "slice",
) -> Dict[str, object]:
    """Score the model and (optionally) break metrics down by a slice series.

    ``slices`` is an array aligned to X_test (e.g. a sensitive attribute held
    out of the feature matrix). Edge case: if its length doesn't match the
    test set, slicing is skipped rather than raising — the card still
    generates with overall metrics.
    """
    y_test = np.asarray(y_test)
    y_pred = model.predict(X_test)
    task = "classification" if is_classifier(model) else "regression"

    y_prob = None
    if task == "classification" and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.shape[1] == 2:
            y_prob = proba[:, 1]

    overall = (
        _classification_metrics(y_test, y_pred, y_prob)
        if task == "classification"
        else _regression_metrics(y_test, y_pred)
    )

    slice_md = None
    if slices is not None and len(slices) == len(y_test):
        df_eval = pd.DataFrame(
            {"y_true": y_test, "y_pred": y_pred, "slice": np.asarray(slices)}
        )
        slice_md = _slice_table(df_eval, slice_name, task)

    return {"task": task, "overall": overall, "slice_md": slice_md, "n_test": len(y_test)}


def generate_model_card(
    model,
    X_test: pd.DataFrame,
    y_test,
    *,
    model_name: str,
    intended_use: str,
    owners: str = "Data Science Team",
    version: str = "1.0.0",
    training_data: str = "Not documented",
    slices: Optional[pd.Series] = None,
    slice_name: str = "slice",
    limitations: Optional[List[str]] = None,
    ethical_considerations: Optional[List[str]] = None,
) -> str:
    """Produce a Markdown model card. Everything the model can tell us is
    introspected; everything it can't (intent, data provenance, ethics) is
    prompted for via arguments so the card is never silently blank."""
    res = compute_metrics(model, X_test, y_test, slices=slices, slice_name=slice_name)
    est = type(model).__name__
    params = getattr(model, "get_params", lambda: {})()
    key_params = {k: v for k, v in params.items() if "__" not in k}

    limitations = limitations or [
        "Performance is only validated on the held-out test distribution above.",
        "No guarantee of fairness across groups not evaluated in the slice table.",
    ]
    ethical_considerations = ethical_considerations or [
        "Review for disparate impact before use in decisions affecting people.",
        "Do not use as the sole basis for high-stakes automated decisions.",
    ]

    metrics_rows = "\n".join(f"| {k} | {_fmt(v)} |" for k, v in res["overall"].items())
    params_rows = "\n".join(f"| `{k}` | {_fmt(v)} |" for k, v in key_params.items())

    card = f"""# Model Card: {model_name}

> {intended_use}

| Field | Value |
|---|---|
| Model name | {model_name} |
| Version | {version} |
| Owners | {owners} |
| Task | {res['task']} |
| Algorithm | `{est}` |
| Test set size | {res['n_test']:,} rows |

## Intended Use
{intended_use}

**Out of scope:** any use not described above. Re-validate before applying to a new
population, time period, or decision context.

## Training Data
{training_data}

## Model Details
| Hyperparameter | Value |
|---|---|
{params_rows}

## Quantitative Performance
Evaluated on a held-out test set of {res['n_test']:,} rows.

| Metric | Value |
|---|---|
{metrics_rows}
"""

    if res["slice_md"]:
        card += f"""
### Performance by Slice (`{slice_name}`)
Watch for large gaps between groups — they signal fairness or robustness risk.

{res['slice_md']}
"""

    card += "\n## Limitations\n" + "\n".join(f"- {x}" for x in limitations)
    card += "\n\n## Ethical Considerations\n" + "\n".join(
        f"- {x}" for x in ethical_considerations
    )
    card += "\n\n---\n*Generated by model-card-gen. Human review required before publishing.*\n"
    return card


def demo_train():
    """Train a churn classifier on synthetic data; return (model, X_te, y_te, region_te)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    n = 1500
    df = pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 72, n),
            "monthly_spend": rng.normal(60, 20, n).round(2),
            "support_tickets": rng.poisson(1.5, n),
            "region": rng.choice(["North", "South", "East", "West"], n),
        }
    )
    logit = -2.0 - 0.03 * df["tenure_months"] + 0.4 * df["support_tickets"]
    df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    features = ["tenure_months", "monthly_spend", "support_tickets"]
    X, y, region = df[features], df["churned"], df["region"]
    X_tr, X_te, y_tr, y_te, _, region_te = train_test_split(
        X, y, region, test_size=0.3, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=120, max_depth=6, random_state=42)
    model.fit(X_tr, y_tr)
    return model, X_te, y_te, region_te


if __name__ == "__main__":
    model, X_te, y_te, region_te = demo_train()
    card = generate_model_card(
        model,
        X_te,
        y_te,
        model_name="Customer Churn Classifier",
        intended_use="Flag customers at risk of churning for proactive retention outreach.",
        training_data="1,500 synthetic customer records (tenure, spend, support tickets, region).",
        slices=region_te,
        slice_name="region",
    )
    print(card)
