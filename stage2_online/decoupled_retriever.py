import sys
import os
import re
import numpy as np

# Add parent directory to path so we can import from stage1_offline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage1_offline.vector_indexer import VectorIndexer
from config import KA_DEFAULT, KV_DEFAULT

class DecoupledRetriever:
    def __init__(self, indexer: VectorIndexer = None):
        self.indexer = indexer if indexer else VectorIndexer()
        
    def retrieve_audio(self, question: str, k: int = KA_DEFAULT):
        print(f"Retrieving top {k} audio candidates for: '{question}'")
        q_embed = self.indexer.embed_audio_text(question).tolist()
        
        # Check if collection is empty
        if self.indexer.audio_collection.count() == 0:
            return []
            
        results = self.indexer.audio_collection.query(
            query_embeddings=[q_embed],
            n_results=k
        )
        
        candidates = []
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                candidates.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "score": results['distances'][0][i] # L2 distance typically in Chroma
                })
        return candidates

    def retrieve_visual(self, question: str, k: int = KV_DEFAULT):
        print(f"Retrieving top {k} visual candidates for: '{question}'")
        q_embed = self.indexer.embed_visual_text(question)
        
        if self.indexer.visual_index.ntotal == 0:
            return []
            
        q_embed_np = np.array([q_embed]).astype('float32')
        search_k = min(self.indexer.visual_index.ntotal, max(k, k * 2))
        distances, indices = self.indexer.visual_index.search(q_embed_np, search_k)
        
        # Extract question keywords for lexical boost
        q_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
        
        candidates = []
        seen_indices = set()
        
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.indexer.visual_metadata) and idx not in seen_indices:
                seen_indices.add(idx)
                metadata = self.indexer.visual_metadata[idx]
                text_lower = metadata["text"].lower()
                
                # Lexical overlap boost
                lexical_overlap = sum(1 for w in q_words if w in text_lower)
                lexical_boost = min(0.3, lexical_overlap * 0.05)
                
                # Boost counting facts for counting queries
                is_counting_q = any(w in question.lower() for w in ["how many", "count", "number of", "total", "unique"])
                counting_boost = 0.0
                if is_counting_q:
                    if "video-level unique entity counts" in text_lower or "frame-level counting" in text_lower:
                        counting_boost = 0.35
                
                base_score = float(distances[0][i])
                combined_score = base_score + lexical_boost + counting_boost
                
                candidates.append({
                    "id": f"visual_{idx}",
                    "text": metadata["text"],
                    "metadata": metadata,
                    "score": combined_score,
                    "_orig_index": idx
                })
        
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:k]
        
        # Temporal sequence expansion: for questions mentioning before/after/then/order
        is_temporal_q = any(tw in question.lower() for tw in ["before", "after", "then", "order", "sequence", "earlier", "later"])
        if is_temporal_q and top_candidates:
            expanded_indices = set()
            for cand in top_candidates[:3]:
                c_idx = cand.get("_orig_index")
                if c_idx is not None:
                    if c_idx > 0:
                        expanded_indices.add(c_idx - 1)
                    if c_idx + 1 < len(self.indexer.visual_metadata):
                        expanded_indices.add(c_idx + 1)
            
            for exp_idx in expanded_indices:
                if exp_idx not in seen_indices and exp_idx < len(self.indexer.visual_metadata):
                    seen_indices.add(exp_idx)
                    metadata = self.indexer.visual_metadata[exp_idx]
                    top_candidates.append({
                        "id": f"visual_{exp_idx}",
                        "text": metadata["text"],
                        "metadata": metadata,
                        "score": 0.5,
                        "_orig_index": exp_idx
                    })
                    
        return top_candidates

    def retrieve(self, question: str, k_a: int = KA_DEFAULT, k_v: int = KV_DEFAULT):
        """Perform decoupled retrieval."""
        audio_candidates = self.retrieve_audio(question, k=k_a)
        visual_candidates = self.retrieve_visual(question, k=k_v)
        
        return audio_candidates, visual_candidates

if __name__ == "__main__":
    pass
