import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from config import RERANKER_MODEL, DEVICE, AUDIO_BONUS, TEMPORAL_AGREEMENT_BONUS

class ReRanker:
    def __init__(self):
        print(f"Loading Re-Ranker: {RERANKER_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(DEVICE)
        self.model.eval()

    def compute_temporal_agreement(self, candidate, all_candidates):
        """
        Adds extra weight to evidence sharing timestamp windows with other top candidates.
        """
        c_start = candidate["metadata"]["start_time"]
        c_end = candidate["metadata"]["end_time"]
        
        agreement_score = 0.0
        for other in all_candidates:
            if other["id"] == candidate["id"]:
                continue
            
            o_start = other["metadata"]["start_time"]
            o_end = other["metadata"]["end_time"]
            
            # Check for overlap
            overlap_start = max(c_start, o_start)
            overlap_end = min(c_end, o_end)
            if overlap_start < overlap_end:
                agreement_score += TEMPORAL_AGREEMENT_BONUS
                
        return agreement_score

    def score_candidates(self, question: str, candidates: list, beta_q: float):
        if not candidates:
            return []
            
        pairs = [[question, cand["text"]] for cand in candidates]
        scores = []
        
        # Mini-batched inference with max_length=192 (optimized for fast CPU/GPU throughput)
        batch_size = 6
        with torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                b_pairs = pairs[i : i + batch_size]
                inputs = self.tokenizer(
                    b_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=192
                ).to(DEVICE)
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
                if logits.ndim == 0:
                    scores.append(float(logits))
                else:
                    scores.extend(logits.tolist())
            
        # Apply formula: Score = Relevance + (beta(q) * AudioBonus) + TemporalAgreement
        for i, cand in enumerate(candidates):
            relevance = scores[i] if i < len(scores) else 0.0
            
            is_audio = 1.0 if cand["metadata"]["type"] in ["speech", "sound"] else 0.0
            audio_bonus_term = beta_q * (AUDIO_BONUS if is_audio else 0.0)
            
            temporal_agreement = self.compute_temporal_agreement(cand, candidates)
            
            final_score = relevance + audio_bonus_term + temporal_agreement
            
            cand["relevance_score"] = float(relevance)
            cand["final_score"] = float(final_score)
            
        return sorted(candidates, key=lambda x: x["final_score"], reverse=True)
