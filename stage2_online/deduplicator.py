from config import NMS_OVERLAP_THRESHOLD

def calculate_iou(start1, end1, start2, end2):
    """Calculates temporal Intersection over Union (IoU)."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    intersection = max(0, intersection_end - intersection_start)
    union = max(end1, end2) - min(start1, start2)
    
    if union == 0:
        return 0.0
    return intersection / union

def calculate_overlap_ratio(start1, end1, start2, end2):
    """Calculates the ratio of overlap relative to the smaller segment."""
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    
    intersection = max(0, intersection_end - intersection_start)
    duration1 = end1 - start1
    duration2 = end2 - start2
    min_duration = min(duration1, duration2)
    
    if min_duration == 0:
        return 0.0
    return intersection / min_duration

class Deduplicator:
    def __init__(self, threshold=NMS_OVERLAP_THRESHOLD):
        self.threshold = threshold

    def apply_nms(self, candidates, score_key="final_score"):
        """
        Applies Temporal Non-Maximum Suppression (NMS).
        Candidates must be sorted by score (descending) before calling this.
        """
        if not candidates:
            return []
            
        # Ensure candidates are sorted by score
        candidates = sorted(candidates, key=lambda x: x.get(score_key, 0.0), reverse=True)
        
        keep = []
        for i, cand in enumerate(candidates):
            overlap = False
            for kept_cand in keep:
                # Calculate temporal overlap ratio
                start1 = cand["metadata"]["start_time"]
                end1 = cand["metadata"]["end_time"]
                start2 = kept_cand["metadata"]["start_time"]
                end2 = kept_cand["metadata"]["end_time"]
                
                # Check if they are of the same modality. If different modalities, we might want to keep both.
                # The prompt implies NMS drops redundant lower-scoring evidence.
                if cand["metadata"]["type"] == kept_cand["metadata"]["type"]:
                    overlap_ratio = calculate_overlap_ratio(start1, end1, start2, end2)
                    if overlap_ratio >= self.threshold:
                        overlap = True
                        break
                        
            if not overlap:
                keep.append(cand)
                
        return keep
