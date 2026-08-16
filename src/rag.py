"""
RAG retrieval over the training set: given a new dialogue, retrieve the
k most similar (dialogue, summary) exemplars to ground the drafter agent.
"""
import argparse
import json
import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from config import CFG
from data import load_split

INDEX_PATH = os.path.join(CFG.index_dir, "train.faiss")
META_PATH = os.path.join(CFG.index_dir, "train_meta.jsonl")


class Retriever:
    def __init__(self):
        self.embedder = SentenceTransformer(CFG.embed_model_name)
        self.index = None
        self.meta = []

    def build(self):
        train = load_split("train")
        texts = [ex["dialogue"] for ex in train]
        print(f"Embedding {len(texts)} training dialogues...")
        embs = self.embedder.encode(texts, batch_size=32, show_progress_bar=True,
                                     normalize_embeddings=True)
        embs = np.asarray(embs, dtype="float32")

        index = faiss.IndexFlatIP(embs.shape[1])  # cosine sim via normalized IP
        index.add(embs)
        faiss.write_index(index, INDEX_PATH)

        with open(META_PATH, "w") as f:
            for ex in train:
                f.write(json.dumps(ex) + "\n")

        print(f"Wrote FAISS index ({index.ntotal} vectors) to {INDEX_PATH}")
        self.index = index
        self.meta = train

    def load(self):
        if self.index is None:
            self.index = faiss.read_index(INDEX_PATH)
            with open(META_PATH) as f:
                self.meta = [json.loads(line) for line in f]
        return self

    def retrieve(self, dialogue: str, k: int = None):
        k = k or CFG.rag_top_k
        self.load()
        q = self.embedder.encode([dialogue], normalize_embeddings=True)
        q = np.asarray(q, dtype="float32")
        scores, idxs = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            ex = dict(self.meta[idx])
            ex["similarity"] = float(score)
            results.append(ex)
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-index", action="store_true")
    parser.add_argument("--query", type=str, default=None,
                         help="Test retrieval with a sample dialogue string")
    args = parser.parse_args()

    r = Retriever()
    if args.build_index:
        r.build()
    if args.query:
        for res in r.retrieve(args.query):
            print(f"[{res['similarity']:.3f}] {res['summary'][:120]}...")
