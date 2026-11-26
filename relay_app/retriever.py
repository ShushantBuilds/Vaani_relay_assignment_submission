import re
import time
import httpx
import numpy as np
from pathlib import Path
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.conf import settings

ARXIV_PAPERS = {
    "1706.03762v7": {
        "title": "Attention Is All You Need",
        "url": "https://arxiv.org/pdf/1706.03762v7.pdf",
    },
    "2410.00037v1": {
        "title": "Moshi: a speech-text foundation model for real-time dialogue",
        "url": "https://arxiv.org/pdf/2410.00037v1.pdf",
    },
    "2507.07955v2": {
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "url": "https://arxiv.org/pdf/2507.07955v2.pdf",
    },
}

RETRIEVAL_SETTINGS = {
    "250_3_no": {"chunk_tokens": 250, "k": 3, "rerank": False},
    "500_5_yes": {"chunk_tokens": 500, "k": 5, "rerank": True},
    "1000_8_yes": {"chunk_tokens": 1000, "k": 8, "rerank": True},
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())


class PaperLibrary:
    _instance = None

    def __init__(self):
        self.papers_dir: Path = settings.PAPERS_DIR
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.pages_by_paper = {}
        self.indices = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance.ensure_papers_downloaded()
            cls._instance.build_all_indices()
        return cls._instance

    def ensure_papers_downloaded(self):
        headers = {"User-Agent": "VaaniRelayAssignment/1.0 (Academic Research)"}
        with httpx.Client(timeout=60.0, follow_redirects=True, headers=headers) as client:
            for arxiv_id, meta in ARXIV_PAPERS.items():
                pdf_path = self.papers_dir / f"{arxiv_id}.pdf"
                if not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
                    print(f"[Library] Downloading {arxiv_id} ({meta['title']})...")
                    resp = client.get(meta["url"])
                    resp.raise_for_status()
                    pdf_path.write_bytes(resp.content)
                    print(f"[Library] Saved {pdf_path.name} ({len(resp.content) // 1024} KB)")

                reader = PdfReader(str(pdf_path))
                pages = []
                for idx, page in enumerate(reader.pages):
                    raw_text = page.extract_text() or ""
                    clean_text = re.sub(r"\s+", " ", raw_text).strip()
                    if clean_text:
                        pages.append({"page": idx + 1, "text": clean_text})
                self.pages_by_paper[arxiv_id] = pages

    def build_all_indices(self):
        for setting_name, cfg in RETRIEVAL_SETTINGS.items():
            chunk_size = cfg["chunk_tokens"]
            chunks = []
            tokenized_corpus = []

            for arxiv_id, pages in self.pages_by_paper.items():
                title = ARXIV_PAPERS[arxiv_id]["title"]
                for p in pages:
                    words = p["text"].split()
                    # Approximate 1 token ~= 0.75 words -> chunk_words = int(chunk_size * 0.75)
                    step = max(50, int(chunk_size * 0.75))
                    overlap = max(20, int(step * 0.15))
                    for start in range(0, len(words), max(1, step - overlap)):
                        slice_words = words[start : start + step]
                        if len(slice_words) < 25:
                            continue
                        chunk_str = " ".join(slice_words)
                        citation_id = f"arXiv:{arxiv_id}, p.{p['page']}"
                        chunks.append({
                            "arxiv_id": arxiv_id,
                            "title": title,
                            "page": p["page"],
                            "citation": f"[{citation_id}]",
                            "text": chunk_str,
                            "est_tokens": int(len(slice_words) * 1.33),
                        })
                        tokenized_corpus.append(tokenize(f"{title} {arxiv_id} {chunk_str}"))

            bm25 = BM25Okapi(tokenized_corpus)
            tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, stop_words="english")
            tfidf_matrix = tfidf.fit_transform([c["text"] for c in chunks])

            self.indices[setting_name] = {
                "chunks": chunks,
                "bm25": bm25,
                "tfidf": tfidf,
                "tfidf_matrix": tfidf_matrix,
                "cfg": cfg,
            }
        print(f"[Library] Built {len(self.indices)} retrieval indices across 3 papers.")

    def retrieve(self, query: str, setting_name: str = "500_5_yes") -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        if setting_name not in self.indices:
            setting_name = "500_5_yes"

        idx_data = self.indices[setting_name]
        cfg = idx_data["cfg"]
        k = cfg["k"]
        chunks = idx_data["chunks"]

        q_tokens = tokenize(query)
        bm25_scores = idx_data["bm25"].get_scores(q_tokens)

        # Fetch candidate pool for optional re-ranking
        pool_size = k * 4 if cfg["rerank"] else k
        top_candidates = np.argsort(bm25_scores)[::-1][:pool_size]

        if cfg["rerank"] and len(top_candidates) > 0:
            q_vec = idx_data["tfidf"].transform([query])
            cand_matrix = idx_data["tfidf_matrix"][top_candidates]
            cos_scores = cosine_similarity(q_vec, cand_matrix).flatten()
            # Normalize BM25 and combine with bi-gram cosine similarity
            cand_bm25 = bm25_scores[top_candidates]
            max_bm25 = np.max(cand_bm25) if np.max(cand_bm25) > 0 else 1.0
            combined = 0.55 * (cand_bm25 / max_bm25) + 0.45 * cos_scores
            reranked_order = np.argsort(combined)[::-1][:k]
            selected_indices = [top_candidates[i] for i in reranked_order]
        else:
            selected_indices = top_candidates[:k]

        results = [chunks[i] for i in selected_indices]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return results, elapsed_ms