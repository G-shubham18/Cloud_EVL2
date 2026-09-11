from config import MODALITY_THRESHOLD, MIN_AUDIO_EVIDENCE_THRESHOLD

class SufficiencyGate:
    def __init__(self, min_relevance_score: float = -10.0):
        self.min_relevance_score = min_relevance_score

    def check_sufficiency(self, beta_q: float, ranked_candidates: list) -> bool:
        """
        Check if the gathered evidence is sufficient based on beta(q) and candidate relevance scores.
        """
        if not ranked_candidates:
            print("SufficiencyGate: Candidate set is empty. Evidence insufficient.")
            return False

        # Filter candidates meeting minimum relevance score quality
        valid_candidates = [
            c for c in ranked_candidates 
            if c.get("relevance_score", c.get("score", 0.0)) >= self.min_relevance_score
        ]

        if not valid_candidates:
            print(f"SufficiencyGate: No candidates meet the minimum relevance threshold ({self.min_relevance_score}).")
            return False

        if beta_q > MODALITY_THRESHOLD:
            # Audio-heavy question: verify audio evidence exists and is relevant
            audio_evidence_count = sum(
                1 for c in valid_candidates 
                if c["metadata"]["type"] in ["speech", "sound"]
            )
            if audio_evidence_count < MIN_AUDIO_EVIDENCE_THRESHOLD:
                # If audio facts are sparse, fallback to visual evidence so the LLM can deduce audio events from visual cues
                visual_count = sum(1 for c in valid_candidates if c["metadata"]["type"] == "visual")
                if visual_count > 0:
                    print(f"SufficiencyGate: Audio evidence sparse ({audio_evidence_count}), falling back to {visual_count} visual candidates.")
                    return True
                print(f"SufficiencyGate: Insufficient audio evidence ({audio_evidence_count} < {MIN_AUDIO_EVIDENCE_THRESHOLD}) for beta={beta_q:.2f}. Triggering loopback.")
                return False
        else:
            # Visual-heavy question: verify visual evidence exists and is relevant
            visual_evidence_count = sum(
                1 for c in valid_candidates 
                if c["metadata"]["type"] == "visual"
            )
            if visual_evidence_count < 1:
                print(f"SufficiencyGate: Insufficient visual evidence (0 visual chunks) for visual-heavy question (beta={beta_q:.2f}). Triggering loopback.")
                return False

        return True

