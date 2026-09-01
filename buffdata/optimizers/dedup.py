import hashlib
from typing import List, Optional, Set, Tuple
import numpy as np
from buffdata.engine.client import GeminiClient
from buffdata.models.schemas import DatasetItem

class Deduplicator:
    """Performs exact, lexical (MinHash n-gram), and semantic embedding deduplication."""

    def __init__(self, client: Optional[GeminiClient] = None):
        self.client = client or GeminiClient()

    def deduplicate_exact(self, items: List[DatasetItem]) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Fast exact deduplication based on prompt+response SHA256 hashes."""
        seen_hashes: Set[str] = set()
        kept: List[DatasetItem] = []
        dropped: List[DatasetItem] = []

        for it in items:
            p, r = it.get_prompt_and_response()
            combined = f"{p.strip()}|{r.strip()}".encode("utf-8")
            h = hashlib.sha256(combined).hexdigest()
            if h in seen_hashes:
                it.metadata["dedup_reason"] = "exact_duplicate"
                dropped.append(it)
            else:
                seen_hashes.add(h)
                kept.append(it)
        return kept, dropped

    def deduplicate_minhash(
        self,
        items: List[DatasetItem],
        threshold: float = 0.85,
        shingle_size: int = 3,
    ) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Lexical near-duplicate removal using Jaccard similarity of character/word shingles."""
        def get_shingles(text: str) -> Set[str]:
            words = text.lower().split()
            if len(words) < shingle_size:
                return set(words)
            return set(" ".join(words[i:i+shingle_size]) for i in range(len(words)-shingle_size+1))

        shingle_sets: List[Set[str]] = []
        kept: List[DatasetItem] = []
        dropped: List[DatasetItem] = []

        for it in items:
            p, r = it.get_prompt_and_response()
            curr_shingles = get_shingles(f"{p} {r}")
            
            is_dup = False
            for prev_shingles in shingle_sets:
                union = len(curr_shingles.union(prev_shingles))
                if union > 0:
                    sim = len(curr_shingles.intersection(prev_shingles)) / union
                    if sim >= threshold:
                        is_dup = True
                        break
            if is_dup:
                it.metadata["dedup_reason"] = f"minhash_near_dup (threshold {threshold})"
                dropped.append(it)
            else:
                shingle_sets.append(curr_shingles)
                kept.append(it)
        return kept, dropped

    def deduplicate_semantic(
        self,
        items: List[DatasetItem],
        threshold: float = 0.88,
    ) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Semantic vector deduplication using Gemini text embeddings."""
        if not items:
            return [], []

        texts = [f"{it.get_prompt_and_response()[0]} {it.get_prompt_and_response()[1]}" for it in items]
        embeddings = self.client.embed_texts(texts)
        vecs = np.array(embeddings, dtype=np.float32)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vecs / norms

        kept_indices = []
        dropped_indices = []

        for i in range(len(items)):
            if not kept_indices:
                kept_indices.append(i)
                continue

            # Compute similarity against already kept vectors
            sims = np.dot(normed[kept_indices], normed[i])
            if np.max(sims) >= threshold:
                items[i].metadata["dedup_reason"] = f"semantic_dup (sim: {np.max(sims):.3f} >= {threshold})"
                dropped_indices.append(i)
            else:
                kept_indices.append(i)

        kept = [items[i] for i in kept_indices]
        dropped = [items[i] for i in dropped_indices]
        return kept, dropped
