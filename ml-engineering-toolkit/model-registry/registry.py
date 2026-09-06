from __future__ import annotations

"""Mini Model Registry.

A tiny, file-backed model registry that solves one pain point:
"We lose track of model versions."

- Versioned local artifact store (joblib) under root/artifacts/<name>/v<N>.joblib
- Metadata index in root/index.json (a list of version records)
- Content hashing (sha256, first 12 chars) of the artifact bytes
- Stage promotion with a single-production-per-model invariant

No wall-clock timestamps are stamped automatically: pass created_at explicitly
if you want one, otherwise it is stored as null (keeps demos reproducible).
"""

import hashlib
import io
import json
import os
from typing import Dict, List, Optional

import joblib

VALID_STAGES = ("staging", "production", "archived")


class Registry:
    """A minimal file-backed model registry."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.artifacts_dir = os.path.join(root, "artifacts")
        self.index_path = os.path.join(root, "index.json")
        os.makedirs(self.artifacts_dir, exist_ok=True)
        if not os.path.exists(self.index_path):
            self._write_index([])

    # ---- index helpers -------------------------------------------------
    def _read_index(self) -> List[Dict]:
        with open(self.index_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write_index(self, records: List[Dict]) -> None:
        with open(self.index_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)

    def _artifact_path(self, name: str, version: int) -> str:
        return os.path.join(self.artifacts_dir, name, f"v{version}.joblib")

    # ---- core API ------------------------------------------------------
    def register(
        self,
        name: str,
        model: object,
        metrics: Dict,
        params: Optional[Dict] = None,
        created_at: Optional[str] = None,
    ) -> Dict:
        """Serialize and store a model as the next version of `name`."""
        records = self._read_index()
        existing = [r for r in records if r["name"] == name]
        version = max((r["version"] for r in existing), default=0) + 1

        # Serialize to bytes so we can hash the exact artifact content.
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        payload = buffer.getvalue()
        content_hash = hashlib.sha256(payload).hexdigest()[:12]

        artifact_path = self._artifact_path(name, version)
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
        with open(artifact_path, "wb") as fh:
            fh.write(payload)

        record: Dict = {
            "name": name,
            "version": version,
            "stage": "staging",
            "hash": content_hash,
            "metrics": dict(metrics),
            "params": dict(params) if params else {},
            "created_at": created_at,
        }
        records.append(record)
        self._write_index(records)
        return record

    def list_models(self, name: Optional[str] = None) -> List[Dict]:
        """Return all records, optionally filtered by model name."""
        records = self._read_index()
        if name is not None:
            records = [r for r in records if r["name"] == name]
        return records

    def get(self, name: str, version: Optional[int] = None) -> Dict:
        """Return a single record (latest version if `version` is None)."""
        records = [r for r in self._read_index() if r["name"] == name]
        if not records:
            raise KeyError(f"No model named {name!r} in registry")
        if version is None:
            return max(records, key=lambda r: r["version"])
        for r in records:
            if r["version"] == version:
                return r
        raise KeyError(f"No version {version} for model {name!r}")

    def load_model(self, name: str, version: Optional[int] = None) -> object:
        """Deserialize and return the stored model object."""
        record = self.get(name, version)
        return joblib.load(self._artifact_path(name, record["version"]))

    def promote(self, name: str, version: int, stage: str) -> Dict:
        """Set the stage of a version, enforcing one production per model."""
        if stage not in VALID_STAGES:
            raise ValueError(f"stage must be one of {VALID_STAGES}, got {stage!r}")
        records = self._read_index()

        target: Optional[Dict] = None
        for r in records:
            if r["name"] == name and r["version"] == version:
                target = r
                break
        if target is None:
            raise KeyError(f"No version {version} for model {name!r}")

        if stage == "production":
            # Demote any currently-production version of the same model.
            for r in records:
                if (
                    r["name"] == name
                    and r["stage"] == "production"
                    and r["version"] != version
                ):
                    r["stage"] = "archived"

        target["stage"] = stage
        self._write_index(records)
        return target

    def best(self, name: str, metric: str, higher_is_better: bool = True) -> Dict:
        """Return the version with the best value for `metric`."""
        records = [
            r for r in self._read_index()
            if r["name"] == name and metric in r["metrics"]
        ]
        if not records:
            raise KeyError(f"No versions of {name!r} carry metric {metric!r}")
        return (max if higher_is_better else min)(
            records, key=lambda r: r["metrics"][metric]
        )

    def to_frame(self):
        """Return a pandas DataFrame of all records for display."""
        import pandas as pd

        records = self._read_index()
        rows: List[Dict] = []
        for r in records:
            row: Dict = {
                "name": r["name"],
                "version": r["version"],
                "stage": r["stage"],
                "hash": r["hash"],
                "created_at": r["created_at"],
            }
            for k, v in r["metrics"].items():
                row[k] = v
            rows.append(row)
        return pd.DataFrame(rows)


def _demo() -> None:
    import tempfile

    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    root = tempfile.mkdtemp(prefix="model_registry_demo_")
    reg = Registry(root)
    print(f"Registry root: {root}\n")

    X, y = make_classification(
        n_samples=600, n_features=20, n_informative=8, random_state=42
    )
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)

    # Three versions with different regularization -> different metrics.
    for i, C in enumerate([0.01, 1.0, 10.0], start=1):
        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X_te)[:, 1]
        preds = model.predict(X_te)
        metrics = {
            "auc": round(float(roc_auc_score(y_te, proba)), 4),
            "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        }
        rec = reg.register(
            "churn_clf",
            model,
            metrics=metrics,
            params={"C": C},
            created_at=f"2026-07-17T09:0{i}:00",
        )
        print(f"registered v{rec['version']}  hash={rec['hash']}  metrics={metrics}")

    print("\nAll versions:")
    print(reg.to_frame().to_string(index=False))

    winner = reg.best("churn_clf", "auc", higher_is_better=True)
    print(f"\nBest by auc -> v{winner['version']} (auc={winner['metrics']['auc']})")
    reg.promote("churn_clf", winner["version"], "production")

    # Register a 4th, promote it too, to prove the previous prod is archived.
    model = LogisticRegression(C=5.0, max_iter=1000).fit(X_tr, y_tr)
    rec4 = reg.register(
        "churn_clf",
        model,
        metrics={"auc": 0.5, "accuracy": 0.5},
        params={"C": 5.0},
    )
    reg.promote("churn_clf", rec4["version"], "production")

    print("\nAfter promotions:")
    frame = reg.to_frame()
    print(frame.to_string(index=False))

    prod = [r for r in reg.list_models("churn_clf") if r["stage"] == "production"]
    assert len(prod) == 1, f"invariant broken: {len(prod)} production versions"
    print(
        f"\nSingle-production invariant holds: "
        f"exactly 1 production version (v{prod[0]['version']})."
    )


if __name__ == "__main__":
    _demo()
