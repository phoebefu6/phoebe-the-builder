from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Recommendation:
    item: str
    score: float


class ItemItemCF:
    """Item-item collaborative filtering: 'users who liked X also liked Y'.

    Builds a similarity matrix between items from the user-item interaction matrix, then recommends
    items similar to what a user already engaged with. Simple, explainable, no training loop.
    """

    def __init__(self) -> None:
        self.item_sim: pd.DataFrame | None = None
        self.matrix: pd.DataFrame | None = None

    def fit(self, interactions: pd.DataFrame) -> "ItemItemCF":
        """interactions: rows=users, columns=items, values=rating/engagement (0 = none)."""
        self.matrix = interactions.fillna(0.0)
        M = self.matrix.values.astype(float)
        # cosine similarity between item columns
        norms = np.linalg.norm(M, axis=0)
        norms[norms == 0] = 1e-9
        Mn = M / norms
        sim = Mn.T @ Mn
        np.fill_diagonal(sim, 0.0)  # an item is not its own recommendation
        self.item_sim = pd.DataFrame(sim, index=self.matrix.columns, columns=self.matrix.columns)
        return self

    def similar_items(self, item: str, n: int = 5) -> list:
        """Items most similar to a given item - the 'related products' rail."""
        if self.item_sim is None or item not in self.item_sim.index:
            return []
        s = self.item_sim[item].sort_values(ascending=False)
        return [Recommendation(i, round(float(v), 3)) for i, v in s.head(n).items() if v > 0]

    def recommend_for_user(self, user: str, n: int = 5) -> list:
        """Score every unseen item by summed similarity to what the user engaged with."""
        if self.matrix is None or user not in self.matrix.index:
            return []
        user_vec = self.matrix.loc[user]
        seen = set(user_vec[user_vec > 0].index)
        # weighted sum of item similarities, weighted by the user's own ratings
        scores = self.item_sim.mul(user_vec, axis=0).sum(axis=0)
        scores = scores.drop(index=[i for i in seen if i in scores.index], errors="ignore")
        scores = scores[scores > 0].sort_values(ascending=False)
        return [Recommendation(i, round(float(v), 3)) for i, v in scores.head(n).items()]

    def explain(self, user: str, item: str, n: int = 3) -> list:
        """Why this item was recommended: the user's items most similar to it."""
        if self.matrix is None or self.item_sim is None:
            return []
        user_vec = self.matrix.loc[user]
        seen = user_vec[user_vec > 0].index
        contribs = {s: float(self.item_sim.loc[s, item]) * float(user_vec[s]) for s in seen}
        top = sorted(contribs.items(), key=lambda x: x[1], reverse=True)[:n]
        return [s for s, v in top if v > 0]


def sample_interactions() -> pd.DataFrame:
    """A small users × products rating matrix with clear taste clusters."""
    data = {
        # sci-fi / tech books cluster and cooking cluster
        "Dune": [5, 4, 5, 0, 0, 1, 0],
        "Foundation": [5, 5, 4, 0, 0, 0, 0],
        "Neuromancer": [4, 5, 0, 0, 1, 0, 0],
        "The Martian": [4, 0, 5, 0, 0, 0, 1],
        "Salt Fat Acid Heat": [0, 0, 0, 5, 4, 5, 0],
        "The Food Lab": [0, 1, 0, 5, 5, 4, 0],
        "Mastering Pasta": [0, 0, 0, 4, 5, 0, 5],
        "Sapiens": [3, 4, 0, 0, 0, 4, 5],
    }
    users = ["Ada", "Ben", "Cy", "Dee", "Eli", "Fin", "Gus"]
    return pd.DataFrame(data, index=users)
