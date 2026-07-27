from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


@dataclass
class TrainedClassifier:
    pipeline: Pipeline
    labels: list
    accuracy: float
    report: dict
    confusion: np.ndarray
    test_texts: list = field(default_factory=list)
    test_true: list = field(default_factory=list)
    test_pred: list = field(default_factory=list)


def train_classifier(texts: list, labels: list, test_size: float = 0.3, seed: int = 0) -> TrainedClassifier:
    """Train a TF-IDF + logistic-regression text classifier and evaluate on a held-out split.

    The workhorse for auto-tagging tickets: interpretable, fast, strong on small labeled sets.
    """
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000, C=4.0)),
    ])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    classes = sorted(set(labels))
    acc = float((np.array(pred) == np.array(y_te)).mean())
    rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, pred, labels=classes)
    return TrainedClassifier(pipe, classes, acc, rep, cm, list(X_te), list(y_te), list(pred))


def predict(model: TrainedClassifier, text: str) -> dict:
    """Predict a label for new text with class probabilities - the auto-tag output."""
    proba = model.pipeline.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    classes = model.pipeline.classes_
    return {
        "label": classes[order[0]],
        "confidence": round(float(proba[order[0]]), 3),
        "scores": {classes[i]: round(float(proba[i]), 3) for i in order},
    }


def top_features(model: TrainedClassifier, n: int = 6) -> dict:
    """The words that most push a document toward each class - why the model tags what it tags."""
    vec = model.pipeline.named_steps["tfidf"]
    clf = model.pipeline.named_steps["clf"]
    vocab = np.array(vec.get_feature_names_out())
    out: dict = {}
    coefs = clf.coef_
    classes = clf.classes_
    if len(classes) == 2:
        # binary: coef row applies to the positive class
        top_pos = vocab[np.argsort(coefs[0])[::-1][:n]]
        top_neg = vocab[np.argsort(coefs[0])[:n]]
        out[classes[1]] = list(top_pos)
        out[classes[0]] = list(top_neg)
    else:
        for i, cls in enumerate(classes):
            out[cls] = list(vocab[np.argsort(coefs[i])[::-1][:n]])
    return out


def sample_data() -> tuple:
    """Support tickets across three categories - the classic auto-tagging use case."""
    data = [
        ("I was charged twice for my subscription this month", "billing"),
        ("My invoice shows the wrong amount, please refund the difference", "billing"),
        ("How do I update the credit card on file for payments", "billing"),
        ("The annual plan renewed but I wanted to cancel before the charge", "billing"),
        ("Why was my payment declined when my card is valid", "billing"),
        ("I need a receipt for last month's payment for expenses", "billing"),
        ("Please refund the duplicate charge on my statement", "billing"),
        ("My subscription price went up without any notice", "billing"),
        ("Can I switch from monthly billing to an annual invoice", "billing"),
        ("I was billed after I already cancelled my plan", "billing"),
        ("The app crashes every time I open the reports page", "technical"),
        ("I'm getting a 500 error when I try to export my data", "technical"),
        ("The dashboard won't load and just shows a blank screen", "technical"),
        ("API requests are timing out after the latest update", "technical"),
        ("The mobile app freezes on the login screen on Android", "technical"),
        ("Charts are rendering incorrectly with overlapping labels", "technical"),
        ("The page throws an error when I click the export button", "technical"),
        ("Data sync is broken and my numbers are out of date", "technical"),
        ("The website is very slow and requests keep failing", "technical"),
        ("I see a bug where the filter resets every refresh", "technical"),
        ("I can't log in, it says my password is incorrect", "account"),
        ("How do I add a teammate or invite a coworker to my workspace", "account"),
        ("I want to delete my account and remove all my data", "account"),
        ("Can I change the email address associated with my login", "account"),
        ("I need to reset two-factor authentication on my account", "account"),
        ("How do I upgrade a user from viewer to admin role", "account"),
        ("Please remove a former colleague from our team members", "account"),
        ("I forgot my password and the reset email never arrives", "account"),
        ("How do I transfer ownership of the workspace to someone else", "account"),
        ("Add a new user seat and set their permissions to editor", "account"),
    ]
    texts = [t for t, _ in data]
    labels = [lab for _, lab in data]
    return texts, labels
