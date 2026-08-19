"""
kb.py
-------
Loads the support knowledge base and lets the agent search it.

We use TF-IDF (a classic, free, no-API-needed text search technique)
instead of a paid embedding API. This keeps the whole project 100% free
to run and deploy.
"""

import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = os.path.join(os.path.dirname(__file__), "kb_data", "articles.json")


class KnowledgeBase:
    def __init__(self, path: str = KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.articles = json.load(f)

        self.corpus = [
            f"{a['title']} {a['category']} {a['content']}" for a in self.articles
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int = 2):
        """Return the top_k most relevant KB articles for a query."""
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]

        results = []
        for idx in ranked_idx:
            if scores[idx] <= 0:
                continue
            article = self.articles[idx]
            results.append(
                {
                    "id": article["id"],
                    "title": article["title"],
                    "category": article["category"],
                    "content": article["content"],
                    "relevance_score": round(float(scores[idx]), 3),
                }
            )
        return results


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(kb.search("I can't reset my password"))
