"""
rag_pipeline.py
---------------
RAG (Retrieval-Augmented Generation) pipeline for the symptom chatbot.

Medical documents are now loaded from data/medical_docs.csv instead of
being hardcoded. The TF-IDF retrieval engine is unchanged.

Flow:
  1. Load medical documents from CSV at startup
  2. Build TF-IDF index
  3. At query time, embed the user's symptom description
  4. Retrieve top-k most relevant document chunks
  5. Return chunks as context to inject into the LLM prompt
"""

import csv
import math
import os
import re
from collections import defaultdict


# ---------------------------------------------------------------------------
# 1. CSV Loader
# ---------------------------------------------------------------------------

def load_documents_from_csv(csv_path: str) -> list[dict]:
    """
    Read medical_docs.csv and return a list of document dicts:
      {id, condition, title, content}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Medical docs CSV not found: {csv_path}")

    documents: list[dict] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            documents.append({
                "id":        f"doc_{row['condition'].strip()}",
                "condition": row["condition"].strip(),
                "title":     row["title"].strip(),
                "content":   row["content"].strip(),
            })

    print(f"[RAG] Loaded {len(documents)} medical documents from CSV")
    return documents


# ---------------------------------------------------------------------------
# 2. TF-IDF vectoriser
# ---------------------------------------------------------------------------

class TFIDFRetriever:
    def __init__(self):
        self.documents   = []
        self.vocab       = {}
        self.idf         = {}
        self.tfidf_matrix = []

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        stopwords = {
            "the","a","an","is","are","was","were","be","been","being",
            "have","has","had","do","does","did","will","would","could",
            "should","may","might","shall","can","need","dare","ought",
            "used","to","of","in","on","at","by","for","with","about",
            "against","between","through","during","before","after",
            "above","below","from","up","down","out","off","over","under",
            "again","then","once","and","but","or","nor","so","yet",
            "both","either","neither","not","no","nor","only","own",
            "same","than","too","very","it","its","this","that","these",
            "those","which","who","whom","what","when","where","why","how"
        }
        return [t for t in tokens if t not in stopwords and len(t) > 2]

    def _tf(self, tokens: list[str]) -> dict:
        tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        total = len(tokens) if tokens else 1
        return {k: v / total for k, v in tf.items()}

    def index(self, documents: list[dict]):
        self.documents = documents
        tokenized = [self._tokenize(d["content"] + " " + d["title"]) for d in documents]

        all_terms = set(t for doc in tokenized for t in doc)
        self.vocab = {term: i for i, term in enumerate(sorted(all_terms))}

        N = len(documents)
        df: dict[str, int] = defaultdict(int)
        for doc_tokens in tokenized:
            for term in set(doc_tokens):
                df[term] += 1
        self.idf = {
            term: math.log((N + 1) / (df[term] + 1)) + 1
            for term in self.vocab
        }

        self.tfidf_matrix = []
        for doc_tokens in tokenized:
            tf = self._tf(doc_tokens)
            vec = {
                term: tf.get(term, 0) * self.idf.get(term, 0)
                for term in self.vocab
            }
            self.tfidf_matrix.append(vec)

        print(f"[RAG] Indexed {len(documents)} documents, vocab size: {len(self.vocab)}")

    def _cosine_similarity(self, vec_a: dict, vec_b: dict) -> float:
        common = set(vec_a.keys()) & set(vec_b.keys())
        dot    = sum(vec_a[k] * vec_b[k] for k in common)
        norm_a = math.sqrt(sum(v**2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v**2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        query_tokens = self._tokenize(query)
        query_tf     = self._tf(query_tokens)
        query_vec    = {
            term: query_tf.get(term, 0) * self.idf.get(term, 0)
            for term in self.vocab
        }

        scores = []
        for i, doc_vec in enumerate(self.tfidf_matrix):
            score = self._cosine_similarity(query_vec, doc_vec)
            scores.append((score, i))

        scores.sort(reverse=True)
        results = []
        for score, idx in scores[:top_k]:
            if score > 0:
                doc = self.documents[idx].copy()
                doc["relevance_score"] = round(score, 4)
                results.append(doc)
        return results


# ---------------------------------------------------------------------------
# 3. RAG Pipeline class
# ---------------------------------------------------------------------------

class RAGPipeline:
    def __init__(self, csv_path: str | None = None):
        if csv_path and os.path.exists(csv_path):
            documents = load_documents_from_csv(csv_path)
        else:
            # Fallback: empty (should not normally reach here)
            documents = []
            print("[RAG] WARNING: no medical_docs.csv found; RAG context will be empty.")

        self.retriever = TFIDFRetriever()
        self.retriever.index(documents)

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        """Return formatted context string for the LLM prompt."""
        docs = self.retriever.retrieve(query, top_k=top_k)
        if not docs:
            return ""
        parts = [f"[{doc['title']}]\n{doc['content']}" for doc in docs]
        return "\n\n---\n\n".join(parts)

    def retrieve_raw(self, query: str, top_k: int = 3) -> list[dict]:
        """Return raw document list with scores — useful for debugging."""
        return self.retriever.retrieve(query, top_k=top_k)


# ---------------------------------------------------------------------------
# 4. Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    _here = pathlib.Path(__file__).parent.parent.parent
    csv_p = str(_here / "data" / "medical_docs.csv")

    rag = RAGPipeline(csv_path=csv_p)
    for q in [
        "I have a headache that throbs on one side with light sensitivity",
        "burning when I urinate and need to go frequently",
        "stomach cramps and vomiting after eating out",
    ]:
        print(f"\nQuery: '{q}'")
        for r in rag.retrieve_raw(q, top_k=2):
            print(f"  → {r['title']} (score: {r['relevance_score']})")
