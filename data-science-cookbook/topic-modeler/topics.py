from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class Topic:
    id: int
    top_words: list = field(default_factory=list)
    label: str = ""


@dataclass
class TopicModel:
    topics: list                  # list[Topic]
    doc_topic: np.ndarray         # (n_docs, n_topics) weights
    vectorizer: object
    model: object

    def dominant_topic(self, doc_idx: int) -> int:
        return int(np.argmax(self.doc_topic[doc_idx]))


def fit_topics(docs: list, n_topics: int = 3, n_top_words: int = 8,
               max_df: float = 0.95, min_df: int = 1) -> TopicModel:
    """Discover topics in a document collection with NMF over TF-IDF. 'What are these docs about?'.

    NMF on TF-IDF separates SHORT-text corpora far better than LDA (whose bag-of-words generative
    assumption needs long documents). Same interface: top words per topic + a doc-topic matrix.
    """
    vec = TfidfVectorizer(stop_words="english", max_df=max_df, min_df=min_df)
    X = vec.fit_transform(docs)
    model = NMF(n_components=n_topics, random_state=0, init="nndsvda", max_iter=400)
    doc_topic = model.fit_transform(X)

    vocab = np.array(vec.get_feature_names_out())
    topics = []
    for k, comp in enumerate(model.components_):
        top_idx = comp.argsort()[::-1][:n_top_words]
        words = vocab[top_idx].tolist()
        topics.append(Topic(id=k, top_words=words, label=" / ".join(words[:3])))
    return TopicModel(topics=topics, doc_topic=doc_topic, vectorizer=vec, model=model)


def topic_sizes(model: TopicModel) -> dict:
    """How many docs each topic dominates - the topic distribution across the corpus."""
    dom = model.doc_topic.argmax(axis=1)
    counts = {t.id: 0 for t in model.topics}
    for d in dom:
        counts[int(d)] += 1
    return counts


def label_documents(model: TopicModel, docs: list) -> list:
    """Assign each doc its dominant topic + confidence - the practical output."""
    out = []
    for i, doc in enumerate(docs):
        k = model.dominant_topic(i)
        out.append({
            "doc": doc[:60] + ("..." if len(doc) > 60 else ""),
            "topic": k,
            "label": model.topics[k].label,
            "confidence": round(float(model.doc_topic[i, k]), 3),
        })
    return out


SAMPLE_DOCS = [
    # AI / ML
    "The new GPU delivers massive speedups for training deep neural networks.",
    "Our model reached state of the art accuracy on the image classification benchmark.",
    "Transformers and attention mechanisms power modern natural language processing.",
    "Researchers fine-tuned a large language model for medical question answering.",
    "The neural network overfit the training data so we added dropout and regularization.",
    "Gradient descent optimizes the model weights to minimize the loss function.",
    "The dataset was labeled and split into training and validation sets for the classifier.",
    # Finance / markets
    "The central bank raised interest rates to combat rising inflation.",
    "Stock markets fell as bond yields climbed and the dollar strengthened.",
    "Investors worry about a recession as GDP growth slows and unemployment rises.",
    "Quarterly earnings beat expectations, sending the company's shares higher.",
    "The hedge fund shorted the stock ahead of the disappointing revenue report.",
    "Bond prices rose as investors sought safe assets amid market volatility.",
    "The IPO priced above range as demand from institutional investors surged.",
    # Sports
    "The team scored in the final minute to win the championship game.",
    "The striker's hat-trick led his club to a decisive victory this weekend.",
    "Fans celebrated as the home team clinched a playoff spot with a late goal.",
    "The coach praised the defense after a hard-fought win over their rivals.",
    "The quarterback threw three touchdowns as the team dominated the second half.",
    "An injury to the star player worries fans ahead of the crucial derby match.",
    "The tennis champion won the tournament final in straight sets on Sunday.",
]
