import hashlib
from typing import List, Optional, Set, Tuple
import numpy as np
from buffdata.engine.client import GeminiClient, LLMClient
from buffdata.models.schemas import DatasetItem

class Deduplicator:
    """Performs exact, lexical (MinHash n-gram), and semantic embedding deduplication."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or GeminiClient()

    @staticmethod
    def _content(item: DatasetItem) -> str:
        prompt, response = item.get_prompt_and_response()
        combined = f"{prompt} {response}".strip()
        return combined or item.get_classification_text()

    def deduplicate_exact(self, items: List[DatasetItem]) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Fast exact deduplication based on prompt+response SHA256 hashes."""
        seen_hashes: Set[str] = set()
        kept: List[DatasetItem] = []
        dropped: List[DatasetItem] = []

        import xxhash
        for it in items:
            combined = self._content(it).strip().encode("utf-8")
            h = xxhash.xxh64(combined).hexdigest()
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
        """Lexical near-duplicate removal using Jaccard similarity of character/word shingles.

        Each kept item's shingle set is stored as fixed-size 64-bit hash fingerprints
        (xxhash, already used for exact dedup above), not the raw shingle strings -- for
        long texts this accumulator (one growing set per kept item, held for the entire
        file/window) is the dominant memory cost of this stage, and a fixed-size int is far
        cheaper than a variable-length string built from the original text. The Jaccard
        computation itself (intersection/union cardinality against the threshold) is
        unchanged, so decisions are identical except for the same astronomically small
        hash-collision risk this module already accepts for exact-dedup fingerprints.
        """
        import xxhash

        def get_shingles(text: str) -> Set[int]:
            words = text.lower().split()
            if len(words) < shingle_size:
                return {xxhash.xxh64_intdigest(word.encode("utf-8")) for word in words}
            return {xxhash.xxh64_intdigest(" ".join(words[i:i+shingle_size]).encode("utf-8"))
                    for i in range(len(words) - shingle_size + 1)}

        shingle_sets: List[Set[int]] = []
        kept: List[DatasetItem] = []
        dropped: List[DatasetItem] = []

        for it in items:
            curr_shingles = get_shingles(self._content(it))

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


    def deduplicate_semantic_local(
        self,
        items: List[DatasetItem],
        threshold: float = 0.85,
        batch_size: int = 32,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Deduplicate items semantically using a local PyTorch SentenceTransformer model."""
        import torch
        from sentence_transformers import SentenceTransformer, util

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load a small, fast model
        model = SentenceTransformer(model_name, device=device)

        texts = []
        for item in items:
            texts.append(self._content(item))

        print(f"Encoding {len(texts)} items locally using PyTorch on {device}...")
        embeddings = model.encode(texts, batch_size=batch_size, convert_to_tensor=True)

        kept_indices: List[int] = []
        dropped: List[DatasetItem] = []
        for index, item in enumerate(items):
            if not kept_indices:
                kept_indices.append(index)
                continue
            similarities = util.cos_sim(embeddings[index], embeddings[kept_indices])[0]
            best = float(torch.max(similarities).item())
            if best >= threshold:
                item.metadata["dedup_reason"] = f"semantic_local_dup (sim: {best:.3f})"
                dropped.append(item)
            else:
                kept_indices.append(index)
        return [items[index] for index in kept_indices], dropped

    def deduplicate_semantic(
        self,
        items: List[DatasetItem],
        threshold: float = 0.88,
    ) -> Tuple[List[DatasetItem], List[DatasetItem]]:
        """Semantic vector deduplication using Gemini text embeddings."""
        if not items:
            return [], []

        texts = [self._content(item) for item in items]
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
