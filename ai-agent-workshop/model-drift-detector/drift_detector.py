from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# PSI interpretation bands (industry-standard credit-risk convention)
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25


def population_stability_index(
    reference: np.ndarray,
    production: np.ndarray,
    bins: int = 10,
) -> float:
    """PSI between a reference (training) sample and a production sample.

    Bin edges are quantiles of the reference so each reference bin holds ~equal mass;
    a small epsilon avoids log(0) when a production bin is empty."""
    reference = np.asarray(reference, dtype=float)
    production = np.asarray(production, dtype=float)
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]
    if reference.size == 0 or production.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:  # constant reference feature — nothing to compare
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    prod_counts, _ = np.histogram(production, bins=edges)

    eps = 1e-6
    ref_pct = ref_counts / ref_counts.sum() + eps
    prod_pct = prod_counts / prod_counts.sum() + eps

    return float(np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct)))


def _severity(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "none"


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    severity: str
    ref_mean: float
    prod_mean: float


@dataclass
class DriftReport:
    features: List[FeatureDrift]
    prediction_drift: Optional[FeatureDrift]
    alert: bool
    reasons: List[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "feature": f.feature,
                "psi": round(f.psi, 4),
                "severity": f.severity,
                "ref_mean": round(f.ref_mean, 3),
                "prod_mean": round(f.prod_mean, 3),
            }
            for f in self.features
        ]
        if self.prediction_drift is not None:
            p = self.prediction_drift
            rows.append({
                "feature": "prediction",
                "psi": round(p.psi, 4),
                "severity": p.severity,
                "ref_mean": round(p.ref_mean, 3),
                "prod_mean": round(p.prod_mean, 3),
            })
        return pd.DataFrame(rows)


def _feature_drift(name: str, ref: np.ndarray, prod: np.ndarray, bins: int) -> FeatureDrift:
    psi = population_stability_index(ref, prod, bins=bins)
    return FeatureDrift(
        feature=name,
        psi=psi,
        severity=_severity(psi),
        ref_mean=float(np.nanmean(ref)) if len(ref) else 0.0,
        prod_mean=float(np.nanmean(prod)) if len(prod) else 0.0,
    )


def detect_drift(
    reference: pd.DataFrame,
    production: pd.DataFrame,
    features: Optional[List[str]] = None,
    prediction_col: Optional[str] = None,
    bins: int = 10,
) -> DriftReport:
    """Compare a reference window to a production window and flag drift.

    Alerts when any monitored feature crosses the 'significant' PSI band, or when the
    model's own prediction distribution drifts significantly (concept-drift proxy)."""
    features = features or [
        c for c in reference.columns
        if c != prediction_col and pd.api.types.is_numeric_dtype(reference[c])
    ]

    drifts: List[FeatureDrift] = []
    reasons: List[str] = []
    for feat in features:
        fd = _feature_drift(feat, reference[feat].to_numpy(), production[feat].to_numpy(), bins)
        drifts.append(fd)
        if fd.severity == "significant":
            reasons.append(f"Feature '{feat}' drifted significantly (PSI={fd.psi:.3f}).")

    pred_drift: Optional[FeatureDrift] = None
    if prediction_col and prediction_col in reference and prediction_col in production:
        pred_drift = _feature_drift(
            "prediction", reference[prediction_col].to_numpy(),
            production[prediction_col].to_numpy(), bins,
        )
        if pred_drift.severity == "significant":
            reasons.append(f"Prediction distribution drifted significantly (PSI={pred_drift.psi:.3f}).")

    return DriftReport(features=drifts, prediction_drift=pred_drift, alert=bool(reasons), reasons=reasons)


def emit_alert(report: DriftReport, model_name: str = "model") -> str:
    """Send a drift alert to ALERT_WEBHOOK_URL if set (Slack-style JSON), else dry-run."""
    if not report.alert:
        return "no drift — no alert sent"
    text = f":rotating_light: Drift detected in *{model_name}*\n" + "\n".join(f"- {r}" for r in report.reasons)
    url = os.environ.get("ALERT_WEBHOOK_URL")
    if not url:
        return f"[dry-run] would alert:\n{text}"
    try:
        import json
        import urllib.request

        req = urllib.request.Request(
            url, data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return f"alert sent (HTTP {resp.status})"
    except Exception as exc:  # noqa: BLE001
        return f"alert failed: {exc}"


def make_sample_data(drift_strength: float = 1.5, n: int = 2000, seed: int = 42):
    """Reference vs. production frames. drift_strength shifts 'income' and degrades the
    score to simulate a silently degrading model. Returns (reference, production)."""
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(60_000, 15_000, n),
        "tenure_months": rng.gamma(2.0, 12, n),
        "score": np.clip(rng.beta(2, 5, n), 0, 1),
    })
    prod = pd.DataFrame({
        "age": rng.normal(40, 10, n),  # stable
        "income": rng.normal(60_000 + drift_strength * 12_000, 15_000, n),  # drifts with strength
        "tenure_months": rng.gamma(2.0, 12, n),  # stable
        "score": np.clip(rng.beta(2, 5 - min(drift_strength, 3.5), n), 0, 1),  # prediction drifts
    })
    return ref, prod
