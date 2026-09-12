"""
================================================================================
ECHOVISION STAGE 1: VISUAL EXTRACTION & ADAPTIVE KEYFRAME PIPELINE
WITH MULTI-FEATURE PERSON IDENTITY TRACKING & UNIQUE COUNTING
================================================================================

Architecture Pipeline (Exact 16-Step Sequential Execution):
  1. INPUT VIDEO: Open video stream and inspect stream metadata.
  2. READ VIDEO FRAMES: Read video frames consecutively at native frame rate.
  3. CALCULATE VISUAL CHANGE SCORE: Fast HSV pixel difference between frame (t-1) and frame (t).
  4. COMPARE CHANGE SCORE WITH THRESHOLD = 18.0: Strict fixed boundary detection.
  5. SCENE CHANGE DETECTED: Collect scene-change candidate frames (anchor frame 0 + scene cuts).
  6. RUN GROUNDING DINO ON SCENE-CHANGE FRAMES: Zero-shot open-vocabulary detection of
     persons, faces, furniture, instruments, vehicles, and key objects.
  7. RESOLVE OVERLAPPING DETECTIONS & COUNT: Merge person/man/woman/child boxes (IoU > 0.45)
     into single physical person detections; associate face boxes; compute frame-level counts.
  8. MULTI-METRIC IMAGE QUALITY EVALUATION: Sharpness (Laplacian variance), brightness,
     contrast, and resolution assessment with automated degradation rejection.
  9. GENERATE CLIP EMBEDDINGS: Compute normalized image embeddings via CLIP ViT.
 10. COSINE SIMILARITY CALCULATION: Compute dot-product cosine similarity against retained frames.
 11. REMOVE DUPLICATE / HIGHLY SIMILAR FRAMES: Prune frames where similarity > 0.90.
 12. KEEP ALL REMAINING UNIQUE FRAMES AS ADAPTIVE KEYFRAMES: No temporal bins, no MAX_KEYFRAMES cap.
 13. STRUCTURED VLM SCENE CAPTIONING: Generate structured scene descriptions (Sections A to G)
     using Qwen2.5-VL vision model.
 14. MULTI-FEATURE PERSON ASSOCIATION: Compare person crops against existing person tracks
     using: (1) Face embedding, (2) CLIP body visual embedding, (3) Clothing/appearance colors,
     (4) Position/movement continuity, (5) Temporal consistency & occlusion buffer.
 15. TEMPORAL SCENE NOTES & ACTION ANALYSIS: Track observable actions, quadrant shifts,
     and count dynamics between consecutive adaptive keyframes.
 16. SAVE FINAL KEYFRAMES & COMPLETE METADATA: Export high-quality JPEG images and metadata.json.
================================================================================
"""

import os
import json
import re
import requests
from typing import Tuple, List, Dict, Optional, Any, Set
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None
from transformers import AutoProcessor
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# SEGMENT 1: SYSTEM CONFIGURATION & HARDWARE CONSTANTS
# ------------------------------------------------------------------------------
import config
from config import (
    CAPTIONING_MODEL,
    QWEN_VL_MODEL,
    HF_TOKEN,
    resolve_captioning_model,
    CLIP_MODEL,
    SIMILARITY_THRESHOLD,
    DEVICE,
    IS_CUDA_AVAILABLE,
    TORCH_DTYPE,
    SCENE_THRESHOLD,
    MAX_SCENE_WINDOW_SEC,
    CLIP_BATCH_SIZE,
    VLM_BATCH_SIZE,
    MIN_KEYFRAMES,
    MAX_KEYFRAMES,
    DYNAMIC_KEYFRAME_INTERVAL_SEC,
    MIN_VISION_PIXELS,
    MAX_VISION_PIXELS,
    VLM_MAX_NEW_TOKENS,
    VLM_IMAGE_RESIZE_MAX,
    IS_GPU,
    BLUR_THRESHOLD,
    DARK_THRESHOLD,
    BRIGHTNESS_MAX_THRESHOLD,
    LOW_CONTRAST_THRESHOLD,
    MIN_FRAME_WIDTH,
    MIN_FRAME_HEIGHT,
    CANDIDATE_SAMPLES_PER_SCENE,
    KEYFRAMES_SAVE_DIR,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    empty_gpu_cache,
)

# Optional Grounding DINO model identifier from config
try:
    from config import GROUNDING_DINO_MODEL
except ImportError:
    GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"

# Consecutive-frame visual change score threshold (default 18.0, imported from config)
SCENE_THRESHOLD = getattr(config, "SCENE_THRESHOLD", 18.0)

# Dynamic imports for Vision-Language and Zero-Shot Object Detection models
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None

try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

try:
    from transformers import AutoModelForZeroShotObjectDetection
except ImportError:
    AutoModelForZeroShotObjectDetection = None


# ------------------------------------------------------------------------------
# SEGMENT 2: STRUCTURED VLM PROMPT DEFINITIONS (SECTIONS A THROUGH G)
# ------------------------------------------------------------------------------
STRUCTURED_VLM_PROMPT = """Analyze this video frame and provide a factual, concise visual breakdown:

A. Scene / Environment:
- Location type: indoor/outdoor, room/site/setting (e.g. kitchen, living room, office, stage, church, street, park).
- Overall scene description: concise factual summary of the environment.

B. People:
- Number of visible people: <count>
- Demographic identification: identify man/woman/child when visually identifiable.
- Clothing & colors: specific garments and colors (e.g. blue jacket, red shirt, black pants, white dress).
- Position & quadrant: 2D quadrant (Top-Left, Top-Right, Bottom-Left, Bottom-Right, or Center).

C. Objects:
- Chairs, tables, instruments, tools, vehicles, and other prominent objects visible in the frame.

D. Spatial Relationships:
- Relative spatial positions: left/right, foreground/midground/background, beside, behind, in front of, on top of, under (e.g. chair behind table, tool beside person).

E. Actions:
- What people are visibly doing: observable physical actions (e.g. sitting, standing, walking, holding object, talking).
- Hand movements: visible hand positions and gestures.
- Object interactions: picking up, putting down, touching, or using objects/tools.
- Movement direction: observable direction of motion (left-to-right, right-to-left, towards camera, stationary).

F. Visible Text:
- Signs, labels, brand names, or readable on-screen text.

G. Confidence:
- Overall visual observation confidence: HIGH / MEDIUM / LOW

Frame-Level Counting:
People:
- People: <count>
- Men: <count>
- Women: <count>
- Children: <count if clearly identifiable, else 0>
Objects:
- Chairs: <count>
- Tables: <count>
- Laptops: <count>
- Computers: <count>
- Phones: <count>
- Cups: <count>
- Bottles: <count>
- Musical Instruments: <count>
- Vehicles: <count>
- Bags: <count>
- Books: <count>
(Count any other prominent visible objects)
Counting Confidence: HIGH / MEDIUM / LOW

Entity Descriptions:
List each visible person or persistent object with category, specific type, quadrant/position, and appearance:
- [person] <man/woman/child/person description, clothing colors, appearance> | Position: <quadrant>
- [object] <category/specific type description, material, state> | Position: <quadrant>

Be strictly factual. Do not infer intentions or hallucinate objects/actions that cannot be visually confirmed."""


# ------------------------------------------------------------------------------
# SEGMENT 3: CLOTHING & APPEARANCE FEATURE EXTRACTION
# ------------------------------------------------------------------------------
def extract_clothing_features(crop_bgr: np.ndarray) -> dict:
    """
    Extracts simple clothing appearance and color features from an actual person crop:
    - Upper body (torso: 15% to 55% vertical height) -> upper_color & upper_hist
    - Lower body (legs/pants: 55% to 95% vertical height) -> lower_color & lower_hist
    - Full clothing region (15% to 95% vertical height) -> dominant_color & full_hist
    - Computes 2D color histograms in HSV space (8 H bins, 4 S bins = 32 dimensions)
    - Determines named dominant colors (red, blue, black, white, gray, green, yellow, etc.)
    - Flags clothing_available=True when reliable non-occluded clothing pixels exist.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 12 or crop_bgr.shape[1] < 12:
        return {
            "upper_color": "unknown",
            "lower_color": "unknown",
            "dominant_color": "unknown",
            "clothing_available": False,
            "upper_hist": np.zeros(32, dtype=np.float32),
            "lower_hist": np.zeros(32, dtype=np.float32),
            "full_hist": np.zeros(32, dtype=np.float32),
            "clothing_type": "clothing"
        }

    h, w = crop_bgr.shape[:2]
    # Upper torso: 15% to 55% of height
    upper = crop_bgr[int(0.15 * h):int(0.55 * h), :]
    # Lower body: 55% to 95% of height
    lower = crop_bgr[int(0.55 * h):int(0.95 * h), :]
    # Full clothing region
    full = crop_bgr[int(0.15 * h):int(0.95 * h), :]

    def _get_hist_and_color(region: np.ndarray):
        if region.size == 0 or region.shape[0] < 4 or region.shape[1] < 4:
            return np.zeros(32, dtype=np.float32), "unknown"
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        hist_flat = hist.flatten().astype(np.float32)

        mean_v = float(np.mean(hsv[:, :, 2]))
        mean_s = float(np.mean(hsv[:, :, 1]))
        mean_h = float(np.mean(hsv[:, :, 0]))

        if mean_v < 45:
            color = "black"
        elif mean_s < 35 and mean_v > 185:
            color = "white"
        elif mean_s < 40:
            color = "gray"
        elif mean_h < 10 or mean_h >= 170:
            color = "red"
        elif 10 <= mean_h < 25:
            color = "brown" if mean_v < 130 else "orange"
        elif 25 <= mean_h < 35:
            color = "yellow"
        elif 35 <= mean_h < 85:
            color = "green"
        elif 85 <= mean_h < 130:
            color = "blue"
        elif 130 <= mean_h < 160:
            color = "purple"
        else:
            color = "pink"
        return hist_flat, color

    u_hist, u_color = _get_hist_and_color(upper)
    l_hist, l_color = _get_hist_and_color(lower)
    f_hist, d_color = _get_hist_and_color(full)

    clothing_valid = (u_color != "unknown" or l_color != "unknown" or d_color != "unknown")

    return {
        "upper_color": u_color,
        "lower_color": l_color,
        "dominant_color": d_color,
        "clothing_available": clothing_valid,
        "upper_hist": u_hist,
        "lower_hist": l_hist,
        "full_hist": f_hist,
        "clothing_type": "clothing"
    }


def compute_clothing_similarity(c1: dict, c2: dict) -> Tuple[float, bool]:
    """
    Computes simple clothing color and appearance similarity in [0.0, 1.0].
    Returns: (clothing_similarity_score, clothing_available_flag)
    """
    if not c1 or not c2:
        return 0.50, False

    avail1 = c1.get("clothing_available", True)
    avail2 = c2.get("clothing_available", True)
    if not avail1 or not avail2:
        return 0.50, False

    u_hist1 = c1.get("upper_hist")
    u_hist2 = c2.get("upper_hist")
    l_hist1 = c1.get("lower_hist")
    l_hist2 = c2.get("lower_hist")

    hist_sim = 0.50
    if u_hist1 is not None and u_hist2 is not None and l_hist1 is not None and l_hist2 is not None:
        u_sim = float(cv2.compareHist(u_hist1, u_hist2, cv2.HISTCMP_INTERSECT))
        l_sim = float(cv2.compareHist(l_hist1, l_hist2, cv2.HISTCMP_INTERSECT))
        hist_sim = 0.50 * u_sim + 0.50 * l_sim

    bonus = 0.0
    u1, u2 = c1.get("upper_color", "unknown"), c2.get("upper_color", "unknown")
    l1, l2 = c1.get("lower_color", "unknown"), c2.get("lower_color", "unknown")
    d1, d2 = c1.get("dominant_color", "unknown"), c2.get("dominant_color", "unknown")

    # Upper body clothing color match/mismatch
    if u1 != "unknown" and u2 != "unknown":
        if u1 == u2:
            bonus += 0.15
        elif (u1 in ["red", "blue", "green", "yellow", "orange", "purple", "pink"]) and \
             (u2 in ["red", "blue", "green", "yellow", "orange", "purple", "pink"]):
            bonus -= 0.35
        else:
            bonus -= 0.15

    # Lower body clothing color match/mismatch
    if l1 != "unknown" and l2 != "unknown":
        if l1 == l2:
            bonus += 0.15
        elif (l1 in ["red", "blue", "green", "yellow", "orange", "purple", "pink"]) and \
             (l2 in ["red", "blue", "green", "yellow", "orange", "purple", "pink"]):
            bonus -= 0.30
        else:
            bonus -= 0.15

    # Overall dominant clothing color match/mismatch
    if d1 != "unknown" and d2 != "unknown":
        if d1 == d2:
            bonus += 0.10
        else:
            bonus -= 0.15

    return float(np.clip(hist_sim + bonus, 0.0, 1.0)), True


def compute_position_continuity(box1: Optional[list], box2: Optional[list], dt: float) -> float:
    """
    Computes position and movement continuity score [0.0, 1.0] between bounding boxes.
    Supports natural movement across the scene by expanding the permissible radius over time dt.
    """
    if not box1 or not box2 or len(box1) != 4 or len(box2) != 4:
        return 0.50

    c1 = ((box1[0] + box1[2]) / 2.0, (box1[1] + box1[3]) / 2.0)
    c2 = ((box2[0] + box2[2]) / 2.0, (box2[1] + box2[3]) / 2.0)
    dist = float(np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2))

    # Permissible movement radius expands with time elapsed:
    # At dt=0s: ~450px radius. At dt=3s: ~650px radius.
    max_dist = 450.0 + min(400.0, dt * 65.0)
    prox_score = max(0.0, 1.0 - (dist / max_dist))

    # Compute bounding-box overlap (IoU)
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2], box2[2])
    inter_y2 = min(box1[3], box2[3])
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    union = area1 + area2 - inter
    iou = inter / union if union > 0 else 0.0

    if iou > 0.15:
        pos_sim = max(prox_score, 0.65 + 0.35 * iou)
    else:
        pos_sim = prox_score

    # Size consistency: persons maintain similar vertical scale unless moving far in depth
    scale_ratio = min(area1, area2) / max(area1, area2)
    pos_sim *= (0.75 + 0.25 * scale_ratio)

    return float(np.clip(pos_sim, 0.0, 1.0))


def compute_temporal_score(dt: float) -> float:
    """Computes temporal consistency factor with occlusion tolerance."""
    if dt <= 2.5:
        return 1.0
    elif dt <= 8.0:
        return 0.85
    elif dt <= 25.0:
        return 0.70
    elif dt <= 60.0:
        return 0.55
    else:
        return 0.40


# ------------------------------------------------------------------------------
# SEGMENT 4: ENTITY TAXONOMY, CATEGORY NORMALIZATION & CROSS-FRAME TRACKING
# ------------------------------------------------------------------------------
PEOPLE_SUPER_CATEGORY = "people"
PERSON_SINGULAR = "person"

PEOPLE_SUBCATEGORIES = {
    "man": [
        "man", "men", "gentleman", "gentlemen", "guy", "guys", 
        "male", "males", "father", "son", "husband", "priest", "monk"
    ],
    "woman": [
        "woman", "women", "lady", "ladies", "female", "females", 
        "mother", "daughter", "wife", "nun"
    ],
    "child": [
        "child", "children", "kid", "kids", "boy", "boys", 
        "girl", "girls", "baby", "babies", "toddler", "toddlers"
    ],
}

OBJECT_CATEGORIES = {
    "chairs": ["chair", "chairs", "armchair", "armchairs", "stool", "stools", "seat", "seats", "bench", "benches"],
    "tables": ["table", "tables", "desk", "desks", "counter", "bar counter", "dining table"],
    "laptops": ["laptop", "laptops", "notebook", "notebooks"],
    "computers": ["computer", "computers", "pc", "desktop", "monitor", "monitors", "screen", "screens"],
    "phones": ["phone", "phones", "cellphone", "cellphones", "smartphone", "smartphones", "mobile", "telephone", "telephones"],
    "cups": ["cup", "cups", "mug", "mugs", "glass", "glasses", "goblet", "chalice", "teacup", "coffee cup"],
    "bottles": ["bottle", "bottles", "flask", "jar", "wine bottle", "beer bottle"],
    "musical_instruments": [
        "musical instrument", "musical instruments", "instrument", "instruments",
        "guitar", "guitars", "violin", "violins", "piano", "pianos", "keyboard", "keyboards",
        "drum", "drums", "flute", "flutes", "cello", "cellos", "trumpet", "trumpets",
        "saxophone", "saxophones", "accordion", "accordions", "harp", "harps", "clarinet"
    ],
    "vehicles": [
        "vehicle", "vehicles", "car", "cars", "automobile", "automobiles", "truck", "trucks",
        "bus", "buses", "bicycle", "bicycles", "bike", "bikes", "motorcycle", "motorcycles", "van", "vans"
    ],
    "bags": ["bag", "bags", "backpack", "backpacks", "handbag", "handbags", "purse", "purses", "suitcase", "suitcases"],
    "books": ["book", "books", "textbook", "textbooks"],
    "beds": ["bed", "beds"],
    "couches": ["couch", "couches", "sofa", "sofas"],
    "mirrors": ["mirror", "mirrors"],
    "doors": ["door", "doors"],
    "windows": ["window", "windows"],
}

SPECIFIC_INSTRUMENT_TYPES = {
    "guitar": ["guitar", "acoustic guitar", "electric guitar", "bass guitar"],
    "violin": ["violin", "fiddle"],
    "piano": ["piano", "grand piano", "upright piano", "keyboard"],
    "drum": ["drum", "drums", "drum set", "drum kit"],
    "flute": ["flute"],
    "cello": ["cello"],
    "trumpet": ["trumpet"],
    "saxophone": ["saxophone", "sax"],
}

SPECIFIC_VEHICLE_TYPES = {
    "car": ["car", "sedan", "suv", "automobile"],
    "truck": ["truck", "pickup"],
    "bus": ["bus"],
    "bicycle": ["bicycle", "bike", "cycle"],
    "motorcycle": ["motorcycle", "motorbike"],
}

SPECIFIC_BAG_TYPES = {
    "backpack": ["backpack", "rucksack"],
    "handbag": ["handbag", "purse"],
    "suitcase": ["suitcase", "luggage"],
}

ID_PREFIXES = {
    "person": "person",
    "people": "person",
    "man": "person",
    "woman": "person",
    "child": "person",
    "chair": "chair",
    "table": "table",
    "laptop": "laptop",
    "computer": "computer",
    "phone": "phone",
    "cup": "cup",
    "bottle": "bottle",
    "musical_instrument": "instrument",
    "vehicle": "vehicle",
    "bag": "bag",
    "book": "book",
    "bed": "bed",
    "couch": "couch",
    "mirror": "mirror",
    "door": "door",
    "window": "window",
}

POSITION_ROIS = {
    "top-left": (0.0, 0.0, 0.55, 0.55),
    "top-right": (0.45, 0.0, 1.0, 0.55),
    "top-middle": (0.25, 0.0, 0.75, 0.55),
    "top": (0.1, 0.0, 0.9, 0.55),
    "bottom-left": (0.0, 0.45, 0.55, 1.0),
    "bottom-right": (0.45, 0.45, 1.0, 1.0),
    "bottom-middle": (0.25, 0.45, 0.75, 1.0),
    "bottom": (0.1, 0.45, 0.9, 1.0),
    "center": (0.2, 0.2, 0.8, 0.8),
    "center foreground": (0.2, 0.35, 0.8, 0.95),
    "left": (0.0, 0.1, 0.55, 0.9),
    "right": (0.45, 0.1, 1.0, 0.9),
    "foreground": (0.1, 0.35, 0.9, 1.0),
    "midground": (0.15, 0.2, 0.85, 0.75),
    "background": (0.1, 0.0, 0.9, 0.55),
}

COLOR_KEYWORDS = {
    "blue", "red", "white", "black", "green", "yellow", "gray", "grey", 
    "brown", "pink", "purple", "orange", "dark", "light", "gold", "silver"
}

CLOTHING_KEYWORDS = {
    "shirt", "coat", "jacket", "dress", "suit", "vest", "pants", "hat", 
    "blouse", "robe", "headscarf", "tie", "skirt", "sweater", "top", "collar", "veil"
}


class CategoryNormalizer:
    """Normalizes raw category strings into canonical categories and subcategories."""
    @staticmethod
    def normalize(raw_name: str) -> Dict[str, Any]:
        cleaned = raw_name.strip().lower().replace("-", " ").replace("_", " ")
        cleaned = re.sub(r'\s+', ' ', cleaned)

        for subcat, terms in PEOPLE_SUBCATEGORIES.items():
            for t in terms:
                pattern = rf"\b{re.escape(t)}\b"
                if re.search(pattern, cleaned):
                    return {
                        "category": PERSON_SINGULAR,
                        "subcategory": subcat,
                        "specific_type": None,
                        "plural_key": "men" if subcat == "man" else ("women" if subcat == "woman" else "children"),
                        "parent_plural_key": PEOPLE_SUPER_CATEGORY,
                    }

        if any(w in cleaned for w in ["people", "person", "human", "individual", "someone", "subject"]):
            return {
                "category": PERSON_SINGULAR,
                "subcategory": None,
                "specific_type": None,
                "plural_key": PEOPLE_SUPER_CATEGORY,
                "parent_plural_key": None,
            }

        for inst_type, aliases in SPECIFIC_INSTRUMENT_TYPES.items():
            for a in aliases:
                if re.search(rf"\b{re.escape(a)}\b", cleaned):
                    return {
                        "category": "musical_instrument",
                        "subcategory": None,
                        "specific_type": inst_type,
                        "plural_key": "musical_instruments",
                        "parent_plural_key": None,
                    }

        for veh_type, aliases in SPECIFIC_VEHICLE_TYPES.items():
            for a in aliases:
                if re.search(rf"\b{re.escape(a)}\b", cleaned):
                    return {
                        "category": "vehicle",
                        "subcategory": None,
                        "specific_type": veh_type,
                        "plural_key": "vehicles",
                        "parent_plural_key": None,
                    }

        for bag_type, aliases in SPECIFIC_BAG_TYPES.items():
            for a in aliases:
                if re.search(rf"\b{re.escape(a)}\b", cleaned):
                    return {
                        "category": "bag",
                        "subcategory": None,
                        "specific_type": bag_type,
                        "plural_key": "bags",
                        "parent_plural_key": None,
                    }

        for plural_key, aliases in OBJECT_CATEGORIES.items():
            for a in aliases:
                if re.search(rf"\b{re.escape(a)}\b", cleaned):
                    singular = a
                    if plural_key.endswith("s") and plural_key != "glasses":
                        singular = plural_key[:-1]
                    if plural_key == "musical_instruments":
                        singular = "musical_instrument"
                    parent = "computers" if plural_key == "laptops" else None
                    return {
                        "category": singular,
                        "subcategory": None,
                        "specific_type": None,
                        "plural_key": plural_key,
                        "parent_plural_key": parent,
                    }

        singular = cleaned.split()[0] if cleaned else "object"
        plural = singular + "s" if not singular.endswith("s") else singular
        return {
            "category": singular,
            "subcategory": None,
            "specific_type": None,
            "plural_key": plural,
            "parent_plural_key": None,
        }


class TrackedEntity:
    """
    Represents an ongoing cross-frame track for a persistent physical entity (Person or Object).
    Tracks simple clothing features, spatial coordinates, movement continuity, and history.
    Does NOT use CLIP body embeddings or face embeddings.
    """
    def __init__(
        self,
        persistent_id: str,
        category: str,
        subcategory: Optional[str] = None,
        specific_type: Optional[str] = None,
        initial_frame_id: int = 0,
        initial_timestamp: float = 0.0,
        initial_desc: str = "",
        initial_pos: str = "Center",
        initial_bbox: Optional[list] = None,
        clothing_features: Optional[dict] = None,
        **kwargs
    ):
        self.persistent_id = persistent_id
        self.category = category
        self.subcategory = subcategory
        self.specific_type = specific_type
        self.first_seen_time = initial_timestamp
        self.last_seen_time = initial_timestamp
        self.frames_seen = [initial_frame_id]
        self.timestamps = [initial_timestamp]
        self.positions = [initial_pos]
        self.bounding_boxes = [initial_bbox] if initial_bbox else []
        self.descriptions = [initial_desc]
        self.association_confidences = [1.0]
        self.clothing_features = clothing_features or {}
        self.clothing_similarity_history = []

    def update(
        self,
        frame_id: int,
        timestamp: float,
        description: str,
        position: str,
        confidence: float,
        clothing_features: Optional[dict] = None,
        bounding_box: Optional[list] = None,
        subcategory: Optional[str] = None,
        specific_type: Optional[str] = None,
        clothing_similarity: Optional[float] = None,
        **kwargs
    ):
        self.frames_seen.append(frame_id)
        self.timestamps.append(timestamp)
        self.positions.append(position)
        self.descriptions.append(description)
        self.association_confidences.append(confidence)
        self.last_seen_time = timestamp

        if bounding_box:
            self.bounding_boxes.append(bounding_box)

        if clothing_features and clothing_features.get("clothing_available", False):
            self.clothing_features = clothing_features

        if not self.subcategory and subcategory:
            self.subcategory = subcategory
        if not self.specific_type and specific_type:
            self.specific_type = specific_type

        if clothing_similarity is not None:
            self.clothing_similarity_history.append(clothing_similarity)

    def to_dict(self) -> Dict[str, Any]:
        """Exports tracked person metadata matching the required identity metadata schema."""
        avg_app = (
            round(float(np.mean(self.clothing_similarity_history)), 4)
            if self.clothing_similarity_history
            else 1.0
        )
        return {
            "persistent_id": self.persistent_id,
            "category": self.category,
            "subcategory": self.subcategory,
            "first_seen_time": round(float(self.first_seen_time), 2),
            "last_seen_time": round(float(self.last_seen_time), 2),
            "frames_seen": list(self.frames_seen),
            "positions": list(self.positions),
            "bounding_boxes": [b for b in self.bounding_boxes if b],
            "appearance_similarity": avg_app,
            "clothing_features": {
                "upper_color": self.clothing_features.get("upper_color", "unknown"),
                "lower_color": self.clothing_features.get("lower_color", "unknown"),
                "dominant_color": self.clothing_features.get("dominant_color", "unknown"),
                "clothing_type": self.clothing_features.get("clothing_type", "clothing"),
            },
            "face_available": False,
            "association_confidence": round(float(np.mean(self.association_confidences)), 4),
        }


class CrossFrameEntityTracker:
    """
    Multi-Feature Cross-Frame Entity Tracking and Unique Counting Engine.
    Identifies the same person across frames using:
      1. Simple clothing features only (dominant color, upper/lower colors, HSV histograms).
      2. Bounding-box position and movement continuity as a supporting feature.
      3. Temporal consistency & occlusion buffer.
    Strictly adheres to constraints:
      - Does NOT use CLIP/body appearance embeddings.
      - Does NOT use face embeddings.
      - Clothing color is the main identity feature, falling back to position/movement continuity
        when clothing is unavailable or heavily occluded.
      - Maintains persistent IDs (person_001, person_002, etc.).
      - Video-level unique count counts distinct persistent IDs, NEVER summing frame counts.
    """
    def __init__(self, clip_model=None, text_embedder=None, similarity_threshold: float = 0.60):
        self.clip_model = clip_model
        self.text_embedder = text_embedder
        self.similarity_threshold = similarity_threshold

        self.tracked_entities: Dict[str, TrackedEntity] = {}
        self.category_counters: Dict[str, int] = {}
        self.frame_counts_history: Dict[int, Dict[str, int]] = {}
        self.frame_entities_history: Dict[int, List[Dict[str, Any]]] = {}
        self.temporal_transitions: List[Dict[str, Any]] = []

    def _next_persistent_id(self, category: str) -> str:
        prefix = ID_PREFIXES.get(category.lower(), category.lower())
        curr = self.category_counters.get(prefix, 0) + 1
        self.category_counters[prefix] = curr
        return f"{prefix}_{curr:03d}"

    def compute_person_identity_score(
        self,
        ent: Dict[str, Any],
        track: TrackedEntity,
        timestamp: float
    ) -> Tuple[float, float]:
        """
        Computes person identity match score using simple clothing features only,
        supported by bounding-box position & movement continuity and temporal buffer.
        """
        # 1. Demographic consistency check (Hard Rejection)
        e_sub = ent.get("subcategory")
        t_sub = track.subcategory
        if e_sub and t_sub:
            if (e_sub == "man" and t_sub == "woman") or (e_sub == "woman" and t_sub == "man"):
                return 0.0, 0.0
            if (e_sub == "child" and t_sub in ["man", "woman"]) or (t_sub == "child" and e_sub in ["man", "woman"]):
                return 0.0, 0.0

        dt = max(0.0, timestamp - track.last_seen_time)
        temp_sim = compute_temporal_score(dt)

        e_box = ent.get("bounding_box")
        t_box = track.bounding_boxes[-1] if track.bounding_boxes else None
        pos_sim = compute_position_continuity(e_box, t_box, dt)

        e_cloth = ent.get("clothing_features", {})
        t_cloth = track.clothing_features or {}
        cloth_sim, cloth_avail = compute_clothing_similarity(e_cloth, t_cloth)

        if cloth_avail:
            # Main identity feature is clothing
            w_cloth = 0.65
            w_pos = 0.25
            w_temp = 0.10
            identity_score = w_cloth * cloth_sim + w_pos * pos_sim + w_temp * temp_sim

            # If clothing matches strongly (same person moving across the scene), maintain high score
            if cloth_sim >= 0.80:
                identity_score = max(identity_score, 0.72)

            # If clothing colors clearly differ, guard against matching
            if cloth_sim < 0.40:
                identity_score = min(identity_score, 0.45)

            # If clothing is similar but spatial jump is implausibly large in short dt
            # (two people wearing similar clothes in different locations)
            if cloth_sim >= 0.70 and dt < 2.0 and pos_sim < 0.30:
                identity_score = min(identity_score, 0.55)
        else:
            # Clothing unavailable or heavily occluded: rely on position & movement continuity
            w_pos = 0.75
            w_temp = 0.25
            identity_score = w_pos * pos_sim + w_temp * temp_sim

        return float(np.clip(identity_score, 0.0, 1.0)), cloth_sim

    def associate_frame(
        self,
        frame_entities: List[Dict[str, Any]],
        frame_id: int,
        timestamp: float,
        pil_image: Optional[Image.Image] = None,
    ) -> List[Dict[str, Any]]:
        """
        Associates detected entities across frames using simple clothing features only
        for persons, supported by position and movement continuity.
        Guarantees mutual exclusivity within the frame (one physical detection = one track).
        """
        if not frame_entities:
            return []

        W, H = pil_image.size if pil_image else (640, 480)

        # Feature extraction for each detected entity (simple clothing features only)
        for ent in frame_entities:
            box = ent.get("bounding_box")
            if pil_image and box and len(box) == 4 and ent.get("category") == PERSON_SINGULAR:
                x1 = max(0, min(W - 2, int(box[0])))
                y1 = max(0, min(H - 2, int(box[1])))
                x2 = max(x1 + 4, min(W, int(box[2])))
                y2 = max(y1 + 4, min(H, int(box[3])))
                crop_pil = pil_image.crop((x1, y1, x2, y2))
                crop_bgr = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
                ent["clothing_features"] = extract_clothing_features(crop_bgr)
            elif ent.get("category") == PERSON_SINGULAR:
                ent["clothing_features"] = {
                    "upper_color": "unknown",
                    "lower_color": "unknown",
                    "dominant_color": "unknown",
                    "clothing_available": False,
                    "clothing_type": "clothing"
                }

        # Group entities into people vs non-people
        people_entities = [e for e in frame_entities if e.get("category") == PERSON_SINGULAR]
        other_entities = [e for e in frame_entities if e.get("category") != PERSON_SINGULAR]

        # ----------------------------------------------------------------------
        # A. Multi-Feature Association for Persons (Clothing + Position Continuity)
        # ----------------------------------------------------------------------
        existing_person_tracks = [t for t in self.tracked_entities.values() if t.category == PERSON_SINGULAR]

        if not existing_person_tracks:
            for ent in people_entities:
                new_id = self._next_persistent_id(PERSON_SINGULAR)
                ent["persistent_id"] = new_id
                ent["association_confidence"] = 1.0
                ent["association_status"] = "new"
                self.tracked_entities[new_id] = TrackedEntity(
                    persistent_id=new_id,
                    category=PERSON_SINGULAR,
                    subcategory=ent.get("subcategory"),
                    specific_type=ent.get("specific_type"),
                    initial_frame_id=frame_id,
                    initial_timestamp=timestamp,
                    initial_desc=ent.get("description", ""),
                    initial_pos=ent.get("position", "Center"),
                    initial_bbox=ent.get("bounding_box"),
                    clothing_features=ent.get("clothing_features"),
                )
        else:
            # Pairwise identity scoring
            pairwise_scores = []
            for e_idx, ent in enumerate(people_entities):
                for t_idx, track in enumerate(existing_person_tracks):
                    score, cloth_sim = self.compute_person_identity_score(ent, track, timestamp)
                    if score > 0.0:
                        pairwise_scores.append((score, cloth_sim, e_idx, t_idx))

            pairwise_scores.sort(key=lambda x: x[0], reverse=True)
            assigned_entities: Set[int] = set()
            assigned_tracks: Set[int] = set()

            for score, cloth_sim, e_idx, t_idx in pairwise_scores:
                if e_idx in assigned_entities or t_idx in assigned_tracks:
                    continue

                track = existing_person_tracks[t_idx]
                ent = people_entities[e_idx]

                if score >= self.similarity_threshold:
                    ent["persistent_id"] = track.persistent_id
                    ent["association_confidence"] = round(float(score), 4)
                    ent["association_status"] = "confirmed"
                    track.update(
                        frame_id=frame_id,
                        timestamp=timestamp,
                        description=ent.get("description", ""),
                        position=ent.get("position", "Center"),
                        confidence=score,
                        clothing_features=ent.get("clothing_features"),
                        bounding_box=ent.get("bounding_box"),
                        subcategory=ent.get("subcategory"),
                        specific_type=ent.get("specific_type"),
                        clothing_similarity=cloth_sim,
                    )
                    assigned_entities.add(e_idx)
                    assigned_tracks.add(t_idx)

            for e_idx, ent in enumerate(people_entities):
                if e_idx not in assigned_entities:
                    new_id = self._next_persistent_id(PERSON_SINGULAR)
                    ent["persistent_id"] = new_id
                    ent["association_confidence"] = 1.0
                    ent["association_status"] = "new"
                    self.tracked_entities[new_id] = TrackedEntity(
                        persistent_id=new_id,
                        category=PERSON_SINGULAR,
                        subcategory=ent.get("subcategory"),
                        specific_type=ent.get("specific_type"),
                        initial_frame_id=frame_id,
                        initial_timestamp=timestamp,
                        initial_desc=ent.get("description", ""),
                        initial_pos=ent.get("position", "Center"),
                        initial_bbox=ent.get("bounding_box"),
                        clothing_features=ent.get("clothing_features"),
                    )

        # ----------------------------------------------------------------------
        # B. Association for Non-Person Objects (Chairs, Tables, Instruments, etc.)
        # ----------------------------------------------------------------------
        for ent in other_entities:
            cat = ent.get("category", "object")
            existing_cat_tracks = [t for t in self.tracked_entities.values() if t.category == cat]
            best_track = None
            best_sim = 0.0

            for track in existing_cat_tracks:
                dt = max(0.0, timestamp - track.last_seen_time)
                t_box = track.bounding_boxes[-1] if track.bounding_boxes else None
                e_box = ent.get("bounding_box")
                pos_sim = compute_position_continuity(e_box, t_box, dt)
                temp_sim = compute_temporal_score(dt)
                score = 0.80 * pos_sim + 0.20 * temp_sim
                if score > best_sim:
                    best_sim = score
                    best_track = track

            if best_track and best_sim >= 0.55:
                ent["persistent_id"] = best_track.persistent_id
                ent["association_confidence"] = round(float(best_sim), 4)
                ent["association_status"] = "confirmed"
                best_track.update(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    description=ent.get("description", ""),
                    position=ent.get("position", "Center"),
                    confidence=best_sim,
                    bounding_box=ent.get("bounding_box"),
                )
            else:
                new_id = self._next_persistent_id(cat)
                ent["persistent_id"] = new_id
                ent["association_confidence"] = 1.0
                ent["association_status"] = "new"
                self.tracked_entities[new_id] = TrackedEntity(
                    persistent_id=new_id,
                    category=cat,
                    subcategory=ent.get("subcategory"),
                    specific_type=ent.get("specific_type"),
                    initial_frame_id=frame_id,
                    initial_timestamp=timestamp,
                    initial_desc=ent.get("description", ""),
                    initial_pos=ent.get("position", "Center"),
                    initial_bbox=ent.get("bounding_box"),
                )

        self.frame_entities_history[frame_id] = frame_entities
        return frame_entities

    def compute_unique_video_counts(self) -> Dict[str, int]:
        """
        Computes the video-level unique counts by counting distinct persistent IDs.
        NEVER calculates unique people by summing frame counts!
        """
        unique_counts: Dict[str, int] = {}
        unique_person_ids: Set[str] = set()
        unique_men_ids: Set[str] = set()
        unique_women_ids: Set[str] = set()
        unique_children_ids: Set[str] = set()
        object_ids_by_plural: Dict[str, Set[str]] = {}

        for p_id, track in self.tracked_entities.items():
            cat = track.category
            if cat == PERSON_SINGULAR:
                unique_person_ids.add(p_id)
                if track.subcategory == "man":
                    unique_men_ids.add(p_id)
                elif track.subcategory == "woman":
                    unique_women_ids.add(p_id)
                elif track.subcategory == "child":
                    unique_children_ids.add(p_id)
            else:
                norm = CategoryNormalizer.normalize(cat)
                p_key = norm["plural_key"]
                object_ids_by_plural.setdefault(p_key, set()).add(p_id)

        unique_counts[PEOPLE_SUPER_CATEGORY] = len(unique_person_ids)
        unique_counts["men"] = len(unique_men_ids)
        unique_counts["women"] = len(unique_women_ids)
        unique_counts["children"] = len(unique_children_ids)

        for p_key, ids in object_ids_by_plural.items():
            unique_counts[p_key] = len(ids)

        return unique_counts

    def compute_temporal_count_changes(self, frame_sequence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preserves temporal counting dynamics across consecutive keyframes."""
        if len(frame_sequence) < 2:
            return []

        transitions = []
        for i in range(len(frame_sequence) - 1):
            curr_kf = frame_sequence[i]
            next_kf = frame_sequence[i + 1]

            f_curr_id = curr_kf["frame_id"]
            f_next_id = next_kf["frame_id"]
            t_curr = curr_kf["mid_time"]
            t_next = next_kf["mid_time"]

            curr_ents = self.frame_entities_history.get(f_curr_id, [])
            next_ents = self.frame_entities_history.get(f_next_id, [])

            curr_ids = {e["persistent_id"] for e in curr_ents if e.get("persistent_id")}
            next_ids = {e["persistent_id"] for e in next_ents if e.get("persistent_id")}

            persisted = curr_ids & next_ids
            disappeared = curr_ids - next_ids
            appeared = next_ids - curr_ids

            new_entered = []
            reappeared = []
            for p_id in appeared:
                track = self.tracked_entities.get(p_id)
                if track and track.frames_seen[0] == f_next_id:
                    new_entered.append(p_id)
                else:
                    reappeared.append(p_id)

            c_people = sum(1 for e in curr_ents if e["category"] == PERSON_SINGULAR)
            n_people = sum(1 for e in next_ents if e["category"] == PERSON_SINGULAR)

            parts = []
            if c_people != n_people:
                action_word = "increased" if n_people > c_people else "decreased"
                parts.append(f"The visible people count {action_word} from {c_people} to {n_people}.")
            else:
                parts.append(f"The visible people count remained constant at {n_people}.")

            if new_entered:
                parts.append(f"New entity ({', '.join(new_entered)}) entered the scene.")
            if reappeared:
                parts.append(f"Previously tracked entity ({', '.join(reappeared)}) became visible again.")
            if disappeared:
                parts.append(f"Entity ({', '.join(disappeared)}) became temporarily not visible or occluded.")

            summary_text = " ".join(parts)
            transitions.append({
                "from_frame": f_curr_id,
                "to_frame": f_next_id,
                "from_time": round(float(t_curr), 2),
                "to_time": round(float(t_next), 2),
                "summary": summary_text,
                "persisted_entities": sorted(list(persisted)),
                "new_entered_entities": sorted(new_entered),
                "reappeared_entities": sorted(reappeared),
                "occluded_or_left_entities": sorted(list(disappeared)),
            })

        self.temporal_transitions = transitions
        return transitions

    def get_metadata(self) -> Dict[str, Any]:
        """Exports final tracking metadata separating all required entity sections."""
        unique_counts = self.compute_unique_video_counts()

        tracked_entities_list = []
        entity_tracks_dict = {}
        for p_id, track in sorted(self.tracked_entities.items(), key=lambda x: x[1].first_seen_time):
            t_dict = track.to_dict()
            tracked_entities_list.append(t_dict)
            entity_tracks_dict[p_id] = t_dict

        peak_frame_counts: Dict[str, int] = {}
        for f_counts in self.frame_counts_history.values():
            for cat, cnt in f_counts.items():
                peak_frame_counts[cat] = max(peak_frame_counts.get(cat, 0), cnt)

        return {
            "frame_counts": peak_frame_counts,
            "frame_counts_by_frame": self.frame_counts_history,
            "tracked_entities": tracked_entities_list,
            "entity_tracks": entity_tracks_dict,
            "unique_video_counts": unique_counts,
            "temporal_count_changes": self.temporal_transitions,
        }


# ------------------------------------------------------------------------------
# SEGMENT 5: VISUAL EXTRACTOR CLASS INITIALIZATION & HARDWARE FALLBACK
# ------------------------------------------------------------------------------
class VisualExtractor:
    """
    Main Visual Extractor implementing the exact 16-step architecture:
    Consecutive frame difference -> Scene cut threshold = 18 -> Grounding DINO ->
    Multi-Feature Person Tracking -> Quality Check -> CLIP Deduplication ->
    Adaptive Keyframe Retention -> Qwen VLM Captioning -> Temporal Notes -> Save Output.
    """
    def __init__(self, load_vlm: bool = True, captioning_model: Optional[str] = None):
        self.load_vlm = load_vlm
        self.captioning_model_name = captioning_model or CAPTIONING_MODEL
        self.captioner = None
        self.model = None
        self.processor = None

        if load_vlm:
            from stage1_offline.captioning import VisualCaptioner
            self.captioner = VisualCaptioner(
                model_name=self.captioning_model_name,
                device=DEVICE,
                hf_token=HF_TOKEN
            )
            self.model = getattr(self.captioner, "model", None)
            self.processor = getattr(self.captioner, "processor", None)
            # Share with generator if it's a Qwen-VL model
            if self.model is not None and self.processor is not None and "qwen" in self.captioning_model_name.lower():
                try:
                    from stage3_generator.generator import Generator
                    Generator.set_shared_model(self.model, self.processor, DEVICE)
                except Exception:
                    pass

        clip_device = DEVICE if IS_CUDA_AVAILABLE else "cpu"
        print(f"Loading CLIP Embedding Model: {CLIP_MODEL} on device={clip_device}")
        self.clip_model = SentenceTransformer(CLIP_MODEL, device=clip_device)

        self.grounding_dino_model_id = GROUNDING_DINO_MODEL
        self.dino_processor = None
        self.dino_model = None
        self.dino_device = "cuda" if DEVICE == "cuda" else "cpu"
        try:
            print(f"Loading Grounding DINO Model: {self.grounding_dino_model_id} on device={self.dino_device}")
            self.dino_processor = AutoProcessor.from_pretrained(self.grounding_dino_model_id)
            if AutoModelForZeroShotObjectDetection is not None:
                self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    self.grounding_dino_model_id
                ).to(self.dino_device)
                self.dino_model.eval()
                print("Grounding DINO Model loaded successfully.")
            else:
                print("AutoModelForZeroShotObjectDetection is not available in transformers.")
        except Exception as e:
            print(f"Warning: Failed to load Grounding DINO model ({self.grounding_dino_model_id}): {e}")

        self.scene_threshold = SCENE_THRESHOLD
        self.similarity_threshold = SIMILARITY_THRESHOLD
        self.blur_threshold = BLUR_THRESHOLD
        self.dark_threshold = DARK_THRESHOLD
        self.brightness_max_threshold = BRIGHTNESS_MAX_THRESHOLD
        self.low_contrast_threshold = LOW_CONTRAST_THRESHOLD
        self.min_frame_width = MIN_FRAME_WIDTH
        self.min_frame_height = MIN_FRAME_HEIGHT
        self.candidate_samples_per_scene = CANDIDATE_SAMPLES_PER_SCENE
        self.clip_batch_size = CLIP_BATCH_SIZE
        self.keyframes_save_dir = KEYFRAMES_SAVE_DIR
        self.latest_candidate_metadata = []
        self.latest_scene_detection = {}
        self.latest_tracking_metadata = {}
        self.latest_temporal_notes = []


    # --------------------------------------------------------------------------
    # SEGMENT 6: MULTI-METRIC IMAGE QUALITY EVALUATION
    # --------------------------------------------------------------------------
    def evaluate_image_quality(self, frame_bgr: np.ndarray) -> dict:
        """
        Step 8: Multi-Metric Image Quality Evaluation.
        Evaluates sharpness (Laplacian variance), brightness, contrast, and resolution.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return {
                "sharpness": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "resolution": [0, 0],
                "quality_score": 0.0,
                "status": "LOW_QUALITY",
                "rejection_reason": "EMPTY_FRAME"
            }

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        resolution = [w, h]

        status = "CLEAR"
        rejection_reason = None

        if brightness < self.dark_threshold:
            status = "TOO_DARK"
            rejection_reason = "TOO_DARK"
        elif brightness > self.brightness_max_threshold:
            status = "LOW_QUALITY"
            rejection_reason = "OVEREXPOSED"
        elif sharpness < self.blur_threshold:
            status = "BLURRY"
            rejection_reason = "BLURRY"
        elif contrast < self.low_contrast_threshold:
            status = "LOW_CONTRAST"
            rejection_reason = "LOW_CONTRAST"
        elif w < self.min_frame_width or h < self.min_frame_height:
            status = "LOW_QUALITY"
            rejection_reason = "LOW_RESOLUTION"

        sharpness_norm = min(1.0, sharpness / 500.0)
        contrast_norm = min(1.0, contrast / 70.0)
        brightness_dist = abs(brightness - 128.0) / 128.0
        brightness_norm = max(0.0, 1.0 - brightness_dist)
        res_norm = min(1.0, (w * h) / (1280.0 * 720.0))

        continuous_score = (
            0.40 * sharpness_norm +
            0.25 * contrast_norm +
            0.20 * brightness_norm +
            0.15 * res_norm
        ) * 100.0

        if status != "CLEAR":
            continuous_score = min(continuous_score, 45.0)

        return {
            "sharpness": round(sharpness, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "resolution": resolution,
            "quality_score": round(continuous_score, 2),
            "status": status,
            "rejection_reason": rejection_reason,
        }


    # --------------------------------------------------------------------------
    # SEGMENT 7: CONSECUTIVE-FRAME VISUAL CHANGE SCORING & SCENE CUT DETECTION
    # --------------------------------------------------------------------------
    def calculate_visual_change_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Step 3: Fast HSV pixel difference between consecutive frames."""
        if prev_frame is None or curr_frame is None:
            return 0.0

        w_fast, h_fast = 256, 144
        prev_small = cv2.resize(prev_frame, (w_fast, h_fast), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_frame, (w_fast, h_fast), interpolation=cv2.INTER_AREA)

        prev_hsv = cv2.cvtColor(prev_small, cv2.COLOR_BGR2HSV).astype(np.float32)
        curr_hsv = cv2.cvtColor(curr_small, cv2.COLOR_BGR2HSV).astype(np.float32)

        diff = np.abs(prev_hsv - curr_hsv)
        score = float(np.mean(diff) * (100.0 / 255.0))
        return score

    def detect_scene_changes(self, video_path: str, threshold: float = SCENE_THRESHOLD):
        """Steps 1, 2, 4, 5: Consecutive-frame scene cut detection using fixed threshold = 18.0."""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps > 0 else 0.0

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        print(f"\n[Scene Detection] Processing '{video_stem}' (Total Frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s)")
        print(f"[Scene Detection] Consecutive-Frame Pipeline active (Threshold: {threshold:.1f})")

        candidates = []
        scene_changes = []

        ret, prev_frame = cap.read()
        if not ret or prev_frame is None:
            cap.release()
            return [], [], duration, fps, total_frames

        h, w = prev_frame.shape[:2]
        anchor_pil = Image.fromarray(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB))
        candidates.append({
            "video_id": video_stem,
            "frame_id": 0,
            "timestamp": 0.0,
            "scene_id": 0,
            "image": anchor_pil,
            "_frame_bgr": prev_frame,
            "visual_change_score": 0.0,
            "scene_change": True,
            "resolution": [w, h],
        })

        frame_count = 1
        scene_id = 0

        while True:
            ret, curr_frame = cap.read()
            if not ret or curr_frame is None:
                break

            timestamp = frame_count / fps
            score = self.calculate_visual_change_score(prev_frame, curr_frame)

            if score > threshold:
                status_str = "SCENE CHANGE"
                print(
                    f"[Scene Detection] Frame {frame_count - 1} -> {frame_count} | "
                    f"Change Score: {score:.2f} | Threshold: {threshold:.1f} | {status_str}"
                )
                scene_id += 1
                scene_changes.append({
                    "previous_frame_id": frame_count - 1,
                    "current_frame_id": frame_count,
                    "previous_timestamp": round((frame_count - 1) / fps, 2),
                    "current_timestamp": round(timestamp, 2),
                    "visual_change_score": round(score, 2),
                    "threshold": threshold,
                    "scene_change": True
                })

                curr_h, curr_w = curr_frame.shape[:2]
                curr_pil = Image.fromarray(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB))
                candidates.append({
                    "video_id": video_stem,
                    "frame_id": frame_count,
                    "timestamp": round(timestamp, 2),
                    "scene_id": scene_id,
                    "image": curr_pil,
                    "_frame_bgr": curr_frame,
                    "visual_change_score": round(score, 2),
                    "scene_change": True,
                    "resolution": [curr_w, curr_h],
                })

            prev_frame = curr_frame
            frame_count += 1

        cap.release()
        print(f"[Scene Detection] Completed: {len(candidates)} candidate frame(s) collected ({len(scene_changes)} scene cut(s) detected).")
        return candidates, scene_changes, duration, fps, total_frames


    # --------------------------------------------------------------------------
    # SEGMENT 8: GROUNDING DINO ZERO-SHOT DETECTION & DUPLICATE RESOLUTION
    # --------------------------------------------------------------------------
    def detect_objects_grounding_dino(
        self,
        pil_image: Image.Image,
        frame_id: int = 0,
        timestamp: float = 0.0,
        confidence_threshold: float = 0.25,
    ) -> Tuple[list, dict]:
        """
        Steps 6 & 7: Zero-Shot Grounding DINO Detection, Overlapping Duplicate Resolution,
        Face Association, and Accurate Frame-Level Counting.
        Resolves overlapping person+man or person+child detections into single physical persons.
        """
        if self.dino_model is None or self.dino_processor is None:
            return [], {}

        W, H = pil_image.size
        prompt = (
            "person. man. woman. child. chair. table. laptop. computer. phone. "
            "cup. bottle. musical instrument. vehicle. bag. book."
        )

        try:
            inputs = self.dino_processor(images=pil_image, text=prompt, return_tensors="pt")
            dino_dev = getattr(self, "dino_device", "cpu")
            inputs = {k: v.to(dino_dev) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.dino_model(**inputs)

            post_results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=confidence_threshold,
                text_threshold=0.20,
                target_sizes=[(H, W)],
            )
            raw_detections = post_results[0]
        except Exception as e:
            print(f"  [Grounding DINO Warning] Inference failed for frame {frame_id}: {e}")
            return [], {}

        def _parse_dino_label(label_str: str) -> Tuple[str, Optional[str], Optional[str], str]:
            lbl = label_str.lower().strip()
            if "child" in lbl or "baby" in lbl or "kid" in lbl or "toddler" in lbl:
                return "person", "child", None, "children"
            elif "man" in lbl or "men" in lbl or "gentleman" in lbl or "male" in lbl:
                return "person", "man", None, "men"
            elif "woman" in lbl or "women" in lbl or "lady" in lbl or "female" in lbl:
                return "person", "woman", None, "women"
            elif "person" in lbl or "people" in lbl or "human" in lbl:
                return "person", None, None, "people"
            elif "musical instrument" in lbl or "instrument" in lbl:
                return "musical_instrument", None, None, "musical_instruments"
            elif "chair" in lbl or "armchair" in lbl or "stool" in lbl or "seat" in lbl:
                return "chair", None, None, "chairs"
            elif "table" in lbl or "desk" in lbl or "counter" in lbl:
                return "table", None, None, "tables"
            elif "laptop" in lbl:
                return "laptop", None, None, "laptops"
            elif "computer" in lbl or "pc" in lbl or "monitor" in lbl:
                return "computer", None, None, "computers"
            elif "phone" in lbl or "cellphone" in lbl or "smartphone" in lbl:
                return "phone", None, None, "phones"
            elif "cup" in lbl or "mug" in lbl or "glass" in lbl:
                return "cup", None, None, "cups"
            elif "bottle" in lbl or "flask" in lbl:
                return "bottle", None, None, "bottles"
            elif "vehicle" in lbl or "car" in lbl or "truck" in lbl or "bus" in lbl or "bicycle" in lbl or "motorcycle" in lbl:
                return "vehicle", None, None, "vehicles"
            elif "bag" in lbl or "backpack" in lbl or "handbag" in lbl or "suitcase" in lbl:
                return "bag", None, None, "bags"
            elif "book" in lbl or "textbook" in lbl:
                return "book", None, None, "books"
            else:
                norm = CategoryNormalizer.normalize(lbl)
                return norm["category"], norm["subcategory"], norm["specific_type"], norm["plural_key"]

        scores = raw_detections["scores"].cpu().tolist()
        boxes = raw_detections["boxes"].cpu().tolist()
        text_labels = raw_detections["text_labels"]

        raw_person_dets = []
        raw_other_dets = []

        for score, box, label in zip(scores, boxes, text_labels):
            cat, subcat, spec, p_key = _parse_dino_label(label)
            x1 = max(0.0, min(float(W), float(box[0])))
            y1 = max(0.0, min(float(H), float(box[1])))
            x2 = max(0.0, min(float(W), float(box[2])))
            y2 = max(0.0, min(float(H), float(box[3])))
            if (x2 - x1) < 6 or (y2 - y1) < 6:
                continue

            det_dict = {
                "category": cat,
                "subcategory": subcat,
                "specific_type": spec,
                "plural_key": p_key,
                "bounding_box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "confidence": round(float(score), 4),
                "frame_id": frame_id,
                "timestamp": round(float(timestamp), 2)
            }

            if cat == "person":
                raw_person_dets.append(det_dict)
            else:
                raw_other_dets.append(det_dict)

        # ----------------------------------------------------------------------
        # 1. Resolve Overlapping Person Detections (person + man / woman / child)
        # ----------------------------------------------------------------------
        raw_person_dets.sort(key=lambda d: d["confidence"], reverse=True)
        resolved_persons = []

        for p_det in raw_person_dets:
            b1 = p_det["bounding_box"]
            merged = False

            for kept in resolved_persons:
                b2 = kept["bounding_box"]
                inter_x1 = max(b1[0], b2[0])
                inter_y1 = max(b1[1], b2[1])
                inter_x2 = min(b1[2], b2[2])
                inter_y2 = min(b1[3], b2[3])
                inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                union = area1 + area2 - inter
                iou = inter / union if union > 0 else 0.0
                containment = inter / min(area1, area2) if min(area1, area2) > 0 else 0.0

                if iou > 0.45 or containment > 0.65:
                    merged = True
                    # Preserve specific demographic classification if found
                    if not kept.get("subcategory") and p_det.get("subcategory"):
                        kept["subcategory"] = p_det["subcategory"]
                        kept["plural_key"] = p_det["plural_key"]
                    # Expand bounding box if candidate is more comprehensive
                    if area1 > area2 and p_det["confidence"] > 0.50:
                        kept["bounding_box"] = p_det["bounding_box"]
                    break

            if not merged:
                resolved_persons.append(p_det)

        # ----------------------------------------------------------------------
        # 2. NMS for Non-Person Objects
        # ----------------------------------------------------------------------
        raw_other_dets.sort(key=lambda d: d["confidence"], reverse=True)
        resolved_others = []
        for o_det in raw_other_dets:
            b1 = o_det["bounding_box"]
            overlap = False
            for kept in resolved_others:
                if kept["category"] == o_det["category"]:
                    b2 = kept["bounding_box"]
                    inter_x1 = max(b1[0], b2[0])
                    inter_y1 = max(b1[1], b2[1])
                    inter_x2 = min(b1[2], b2[2])
                    inter_y2 = min(b1[3], b2[3])
                    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
                    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
                    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou > 0.55:
                        overlap = True
                        break
            if not overlap:
                resolved_others.append(o_det)

        all_resolved_detections = resolved_persons + resolved_others

        # ----------------------------------------------------------------------
        # 4. Frame-Level Visible Counting (Physical Persons & Objects)
        # ----------------------------------------------------------------------
        object_counts = {
            "people": len(resolved_persons),
            "men": sum(1 for p in resolved_persons if p.get("subcategory") == "man"),
            "women": sum(1 for p in resolved_persons if p.get("subcategory") == "woman"),
            "children": sum(1 for p in resolved_persons if p.get("subcategory") == "child"),
            "chairs": 0,
            "tables": 0,
            "laptops": 0,
            "computers": 0,
            "phones": 0,
            "cups": 0,
            "bottles": 0,
            "musical_instruments": 0,
            "vehicles": 0,
            "bags": 0,
            "books": 0,
        }

        for o in resolved_others:
            pk = o["plural_key"]
            object_counts[pk] = object_counts.get(pk, 0) + 1

        # Calculate 2D quadrant position for spatial layout
        for det in all_resolved_detections:
            box = det["bounding_box"]
            cx = (box[0] + box[2]) / 2.0 / W
            cy = (box[1] + box[3]) / 2.0 / H
            v_pos = "Top" if cy < 0.40 else ("Bottom" if cy > 0.60 else "Center")
            h_pos = "Left" if cx < 0.40 else ("Right" if cx > 0.60 else "Center")
            if v_pos == "Center" and h_pos == "Center":
                quad = "Center"
            elif v_pos == "Center":
                quad = h_pos
            elif h_pos == "Center":
                quad = v_pos
            else:
                quad = f"{v_pos}-{h_pos}"
            det["position"] = quad

        return all_resolved_detections, object_counts


    # --------------------------------------------------------------------------
    # SEGMENT 9: KEYFRAME SAVING & COMPLETE METADATA JSON SERIALIZATION
    # --------------------------------------------------------------------------
    def save_keyframes_to_disk(
        self,
        video_path: str,
        keyframes: list,
        output_dir: str = None,
        candidate_metadata: list = None,
        tracking_metadata: dict = None,
        scene_detection_metadata: dict = None,
        temporal_scene_notes: list = None,
    ) -> list:
        """Step 16: Saves accepted keyframes to disk as JPG files and exports metadata.json."""
        if output_dir is None:
            output_dir = self.keyframes_save_dir

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_folder = os.path.join(output_dir, video_name)
        os.makedirs(save_folder, exist_ok=True)

        print(f"Saving {len(keyframes)} accepted keyframes to disk in '{save_folder}'...")
        for kf in keyframes:
            ts = kf.get("timestamp", kf.get("mid_time", 0.0))
            frame_filename = f"frame_{kf['frame_id']:04d}_{ts:.2f}s.jpg"
            save_path = os.path.join(save_folder, frame_filename)
            if "image" in kf and kf["image"] is not None:
                kf["image"].save(save_path, quality=95)
            kf["image_path"] = save_path

        source_metadata = candidate_metadata if candidate_metadata is not None else getattr(self, "latest_candidate_metadata", keyframes)

        all_metadata_records = []
        for record in source_metadata:
            rec_copy = {
                "video_id": record.get("video_id", video_name),
                "frame_id": record.get("frame_id"),
                "scene_id": record.get("scene_id"),
                "timestamp": record.get("timestamp", record.get("mid_time")),
                "start_time": record.get("start_time"),
                "end_time": record.get("end_time"),
                "mid_time": record.get("mid_time"),
                "image_path": record.get("image_path"),
                "visual_change_score": record.get("visual_change_score", 0.0),
                "scene_change": record.get("scene_change", True),
                "sharpness_score": record.get("sharpness_score"),
                "brightness_score": record.get("brightness_score"),
                "contrast_score": record.get("contrast_score"),
                "resolution": record.get("resolution"),
                "quality_score": record.get("quality_score"),
                "quality_status": record.get("quality_status"),
                "clip_similarity": record.get("clip_similarity"),
                "novelty_score": record.get("novelty_score"),
                "accepted": record.get("accepted", False),
                "rejection_reason": record.get("rejection_reason"),
                "grounding_dino_detections": record.get("grounding_dino_detections", []),
                "object_counts": record.get("object_counts", {}),
                "vlm_caption": record.get("vlm_caption", ""),
            }
            all_metadata_records.append(rec_copy)

        scene_det_block = scene_detection_metadata if scene_detection_metadata is not None else getattr(self, "latest_scene_detection", {
            "method": "consecutive_frame_visual_change",
            "threshold": self.scene_threshold,
            "scene_changes_detected": []
        })

        kf_map = {k.get("frame_id"): k for k in keyframes}
        adaptive_kfs_records = []
        for kf in source_metadata:
            f_id = kf.get("frame_id")
            if f_id in kf_map:
                kf["image_path"] = kf_map[f_id].get("image_path", kf.get("image_path"))
                if "vlm_caption" in kf_map[f_id]:
                    kf["vlm_caption"] = kf_map[f_id]["vlm_caption"]
                kf["accepted"] = True
                kf["rejection_reason"] = None

            ts = kf.get("timestamp", kf.get("mid_time", 0.0))
            k_rec = {
                "frame_id": kf.get("frame_id"),
                "timestamp": round(float(ts), 2),
                "scene_id": kf.get("scene_id", 0),
                "image_path": kf.get("image_path"),
                "visual_change_score": kf.get("visual_change_score", 0.0),
                "scene_change": kf.get("scene_change", True),
                "quality_metrics": kf.get("quality_metrics", {
                    "sharpness": kf.get("sharpness_score", 0.0),
                    "brightness": kf.get("brightness_score", 0.0),
                    "contrast": kf.get("contrast_score", 0.0),
                    "resolution": kf.get("resolution", [0, 0]),
                }),
                "quality_score": kf.get("quality_score", 0.0),
                "quality_status": kf.get("quality_status", "CLEAR"),
                "clip_similarity": kf.get("clip_similarity"),
                "novelty_score": kf.get("novelty_score"),
                "accepted": kf.get("accepted", False),
                "rejection_reason": kf.get("rejection_reason"),
                "grounding_dino_detections": kf.get("grounding_dino_detections", []),
                "object_counts": kf.get("object_counts", {}),
                "vlm_caption": kf.get("vlm_caption", ""),
            }
            adaptive_kfs_records.append(k_rec)

        metadata_path = os.path.join(save_folder, "metadata.json")
        notes = temporal_scene_notes if temporal_scene_notes is not None else getattr(self, "latest_temporal_notes", [])
        v_facts = tracking_metadata.get("visual_facts", []) if tracking_metadata else []

        metadata_wrapper = {
            "video_id": video_name,
            "video_path": os.path.abspath(video_path),
            "scene_detection": scene_det_block,
            "adaptive_keyframes": adaptive_kfs_records,
            "frame_counts": tracking_metadata.get("frame_counts", {}) if tracking_metadata else {},
            "tracked_entities": tracking_metadata.get("tracked_entities", []) if tracking_metadata else [],
            "unique_video_counts": tracking_metadata.get("unique_video_counts", {}) if tracking_metadata else {},
            "temporal_scene_notes": notes,
            "visual_facts": v_facts,
            "keyframes": all_metadata_records,
            "total_candidates_evaluated": len(all_metadata_records),
            "accepted_keyframes_count": len(keyframes),
            "frame_counts_by_frame": tracking_metadata.get("frame_counts_by_frame", {}) if tracking_metadata else {},
            "entity_tracks": tracking_metadata.get("entity_tracks", {}) if tracking_metadata else {},
            "temporal_count_changes": tracking_metadata.get("temporal_count_changes", []) if tracking_metadata else [],
        }

        if tracking_metadata:
            tracking_json_path = os.path.join(save_folder, "entity_tracking_metadata.json")
            with open(tracking_json_path, "w", encoding="utf-8") as tf:
                json.dump(tracking_metadata, tf, indent=4, ensure_ascii=False)
            print(f"Saved entity tracking metadata to '{tracking_json_path}'.")

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_wrapper, f, indent=4, ensure_ascii=False)

        print(f"Saved keyframe metadata to '{metadata_path}'.")
        return keyframes


    # --------------------------------------------------------------------------
    # SEGMENT 10: FAST FRAME DIVERSITY SCORING
    # --------------------------------------------------------------------------
    def compute_frame_diversity_score(self, frame_bgr_1: np.ndarray, frame_bgr_2: np.ndarray) -> float:
        """Computes a fast visual diversity score in [0.0, 1.0] between two frames."""
        if frame_bgr_1 is None or frame_bgr_2 is None:
            return 1.0
        try:
            h1 = cv2.calcHist([frame_bgr_1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            h2 = cv2.calcHist([frame_bgr_2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            hist_sim = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
            hist_div = max(0.0, min(1.0, (1.0 - hist_sim) / 2.0))

            g1 = cv2.cvtColor(cv2.resize(frame_bgr_1, (160, 90)), cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(cv2.resize(frame_bgr_2, (160, 90)), cv2.COLOR_BGR2GRAY)
            e1 = cv2.Canny(g1, 50, 150)
            e2 = cv2.Canny(g2, 50, 150)
            edge_diff = float(np.mean(e1 != e2))

            diversity = 0.65 * hist_div + 0.35 * edge_diff
            return float(np.clip(diversity, 0.0, 1.0))
        except Exception:
            return 0.5


    # --------------------------------------------------------------------------
    # SEGMENT 11: CLIP COSINE SIMILARITY DEDUPLICATION & ADAPTIVE RETENTION
    # --------------------------------------------------------------------------
    def filter_keyframes(
        self,
        keyframes: list,
        clip_batch_size: int = CLIP_BATCH_SIZE,
        **kwargs,
    ) -> list:
        """Steps 9 to 12: CLIP Cosine Similarity Duplicate Removal & Adaptive Keyframe Retention."""
        if not keyframes:
            return []

        if len(keyframes) == 1:
            kf = keyframes[0]
            kf["clip_similarity"] = None
            kf["novelty_score"] = 1.0
            kf["accepted"] = True
            kf["rejection_reason"] = None
            return keyframes

        images = [kf["image"] for kf in keyframes]
        print(f"Generating batched CLIP embeddings for {len(images)} candidate frames (batch_size={clip_batch_size})...")

        with torch.inference_mode():
            embeddings = self.clip_model.encode(
                images,
                batch_size=clip_batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

        for i, kf in enumerate(keyframes):
            kf["_clip_emb"] = embeddings[i]

        retained_candidates = []
        retained_embeddings = []

        for kf in keyframes:
            if not kf.get("accepted", False):
                kf["clip_similarity"] = None
                kf["novelty_score"] = None
                continue

            curr_emb = kf["_clip_emb"]

            if len(retained_candidates) == 0:
                kf["clip_similarity"] = None
                kf["novelty_score"] = 1.0
                kf["accepted"] = True
                kf["rejection_reason"] = None
                retained_candidates.append(kf)
                retained_embeddings.append(curr_emb)
                print(f"  [CLIP Deduplication] Frame {kf['frame_id']} @ {kf.get('timestamp', kf.get('mid_time', 0.0)):.2f}s retained as initial reference anchor.")
            else:
                sims = [float(np.dot(curr_emb, r_emb)) for r_emb in retained_embeddings]
                max_sim = max(sims)
                kf["clip_similarity"] = round(max_sim, 4)
                kf["novelty_score"] = round(max(0.0, min(1.0, 1.0 - max_sim)), 4)

                if max_sim > self.similarity_threshold:
                    kf["accepted"] = False
                    kf["rejection_reason"] = "CLIP_DUPLICATE"
                    print(f"  [CLIP Deduplication] Frame {kf['frame_id']} @ {kf.get('timestamp', kf.get('mid_time', 0.0)):.2f}s rejected as DUPLICATE (Max CLIP Sim: {max_sim:.4f} > {self.similarity_threshold})")
                else:
                    kf["accepted"] = True
                    kf["rejection_reason"] = None
                    retained_candidates.append(kf)
                    retained_embeddings.append(curr_emb)
                    print(f"  [CLIP Deduplication] Frame {kf['frame_id']} @ {kf.get('timestamp', kf.get('mid_time', 0.0)):.2f}s retained as UNIQUE (Max CLIP Sim: {max_sim:.4f} <= {self.similarity_threshold})")

        for kf in keyframes:
            kf.pop("_clip_emb", None)

        if not retained_candidates and keyframes:
            best_kf = max(keyframes, key=lambda x: x.get("quality_score", 0.0))
            best_kf["accepted"] = True
            best_kf["rejection_reason"] = None
            retained_candidates.append(best_kf)

        print(f"Adaptive Scene Retention: Kept ALL {len(retained_candidates)} unique adaptive keyframe(s) out of {len(keyframes)} candidates.")
        return retained_candidates


    # --------------------------------------------------------------------------
    # SEGMENT 12: END-TO-END KEYFRAME EXTRACTION PIPELINE (STEPS 1 TO 12)
    # --------------------------------------------------------------------------
    def extract_keyframes(
        self,
        video_path: str,
        scene_threshold: float = SCENE_THRESHOLD,
        clip_batch_size: int = CLIP_BATCH_SIZE,
        **kwargs,
    ) -> list:
        """Executes Steps 1 to 12 of the pipeline."""
        candidates, scene_changes, duration, fps, total_frames = self.detect_scene_changes(
            video_path, threshold=scene_threshold
        )

        self.latest_scene_detection = {
            "method": "consecutive_frame_visual_change",
            "threshold": scene_threshold,
            "scene_changes_detected": scene_changes,
        }

        if not candidates:
            self.latest_candidate_metadata = []
            return []

        # Step 6 & 7: Zero-shot Grounding DINO detection, duplicate resolution, and counting
        print(f"\n[Grounding DINO] Running zero-shot object detection on {len(candidates)} scene-change candidate frame(s)...")
        for cand in candidates:
            dets, counts = self.detect_objects_grounding_dino(
                cand["image"],
                cand["frame_id"],
                cand["timestamp"]
            )
            cand["grounding_dino_detections"] = dets
            cand["object_counts"] = counts
            print(f"  [Frame {cand['frame_id']} @ {cand['timestamp']:.2f}s]: Detected {len(dets)} objects (Counts: {counts})")

        # Step 8: Multi-metric image quality evaluation
        print(f"\n[Quality Evaluation] Evaluating image quality on {len(candidates)} candidate frame(s)...")
        for cand in candidates:
            q_eval = self.evaluate_image_quality(cand.get("_frame_bgr"))
            cand["sharpness_score"] = q_eval["sharpness"]
            cand["brightness_score"] = q_eval["brightness"]
            cand["contrast_score"] = q_eval["contrast"]
            cand["resolution"] = q_eval["resolution"]
            cand["quality_score"] = q_eval["quality_score"]
            cand["quality_status"] = q_eval["status"]
            cand["quality_metrics"] = {
                "sharpness": q_eval["sharpness"],
                "brightness": q_eval["brightness"],
                "contrast": q_eval["contrast"],
                "resolution": q_eval["resolution"],
            }
            if q_eval["status"] == "CLEAR":
                cand["accepted"] = True
                cand["rejection_reason"] = None
            else:
                cand["accepted"] = False
                cand["rejection_reason"] = q_eval["rejection_reason"]

        # Steps 9 to 12: CLIP Cosine Similarity deduplication and adaptive keyframe retention
        accepted_keyframes = self.filter_keyframes(
            candidates, clip_batch_size=clip_batch_size
        )

        for c in candidates:
            c.pop("_frame_bgr", None)

        num_kfs = len(accepted_keyframes)
        for idx, kf in enumerate(accepted_keyframes):
            t_curr = kf["timestamp"]
            if num_kfs == 1:
                kf["start_time"] = 0.0
                kf["end_time"] = round(duration, 2)
            elif idx == 0:
                next_t = accepted_keyframes[1]["timestamp"]
                kf["start_time"] = 0.0
                kf["end_time"] = round((t_curr + next_t) / 2.0, 2)
            elif idx == num_kfs - 1:
                prev_t = accepted_keyframes[idx - 1]["timestamp"]
                kf["start_time"] = round((prev_t + t_curr) / 2.0, 2)
                kf["end_time"] = round(duration, 2)
            else:
                prev_t = accepted_keyframes[idx - 1]["timestamp"]
                next_t = accepted_keyframes[idx + 1]["timestamp"]
                kf["start_time"] = round((prev_t + t_curr) / 2.0, 2)
                kf["end_time"] = round((t_curr + next_t) / 2.0, 2)
            kf["mid_time"] = round(t_curr, 2)

        self.latest_candidate_metadata = candidates
        return accepted_keyframes


    # --------------------------------------------------------------------------
    # SEGMENT 13: STRUCTURED QWEN VLM SCENE CAPTIONING
    # --------------------------------------------------------------------------
    def generate_descriptions(
        self, keyframes: list, vlm_batch_size: int = VLM_BATCH_SIZE
    ):
        """Step 13: Passes accepted keyframes to VisualCaptioner in batches."""
        if not keyframes:
            return []

        results = []
        num_keyframes = len(keyframes)

        if self.captioner is None:
            from stage1_offline.captioning import VisualCaptioner
            self.captioner = VisualCaptioner(
                model_name=self.captioning_model_name,
                device=DEVICE,
                hf_token=HF_TOKEN
            )
            self.model = getattr(self.captioner, "model", None)
            self.processor = getattr(self.captioner, "processor", None)

        for i in range(0, num_keyframes, vlm_batch_size):
            batch_kfs = keyframes[i : i + vlm_batch_size]
            batch_end = min(i + vlm_batch_size, num_keyframes)
            print(
                f"Generating visual captions for keyframes {i+1}-{batch_end}/{num_keyframes} using '{self.captioner.model_id}'..."
            )

            batch_images = [kf["image"] for kf in batch_kfs]
            batch_captions = self.captioner.caption_batch(batch_images)

            for kf, desc in zip(batch_kfs, batch_captions):
                desc_clean = desc.strip()
                kf["vlm_caption"] = desc_clean
                print(f"  [Frame {kf['frame_id']} @ {kf['mid_time']:.2f}s]: \"{desc_clean[:120]}...\"")
                results.append(
                    {
                        "type": "visual",
                        "frame_id": kf["frame_id"],
                        "start_time": kf["start_time"],
                        "end_time": kf["end_time"],
                        "text": desc_clean,
                    }
                )

            empty_gpu_cache()

        return results

    def _extract_structured_field(self, text: str, field_name: str) -> str:
        """Extracts content of a specific section from structured visual descriptions."""
        if not text:
            return ""
        pattern = rf"(?:^|\n)[-•*]?\s*{re.escape(field_name)}[\s:]+([^\n]+(?:\n(?![-•*]|\d+\.)[^\n]+)*)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _is_ollama_available(self) -> bool:
        """Checks if local Ollama server is responsive."""
        try:
            res = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=0.8)
            return res.status_code == 200
        except Exception:
            return False


    # --------------------------------------------------------------------------
    # SEGMENT 14: TEMPORAL SCENE NOTES & ACTION TRANSITION ANALYSIS
    # --------------------------------------------------------------------------
    def generate_temporal_scene_notes(
        self,
        visual_facts: list,
        temporal_count_changes: list = None,
        keyframes: list = None,
    ) -> Tuple[list, list]:
        """Steps 14 & 15: Chronological Temporal Scene Notes & Action Analysis."""
        if len(visual_facts) < 2:
            return visual_facts, []

        num_pairs = len(visual_facts) - 1
        print(f"Generating temporal scene notes & action analysis for {num_pairs} keyframe transition pair(s)...")

        transition_facts = []
        temporal_notes_list = []
        use_ollama = False  # Completely local, deterministic structured semantic diffing (No Ollama required)
        print("  [Temporal Reasoning] Using high-density structured semantic diffing (No Ollama required).")

        count_change_map = {}
        if temporal_count_changes:
            for idx, tc in enumerate(temporal_count_changes):
                count_change_map[(tc.get("from_frame"), tc.get("to_frame"))] = tc
                count_change_map[idx] = tc

        for i in range(num_pairs):
            curr_fact = visual_facts[i]
            next_fact = visual_facts[i + 1]

            t_start = curr_fact["start_time"]
            t_end = next_fact["end_time"]
            f_curr_id = curr_fact.get("frame_id")
            f_next_id = next_fact.get("frame_id")

            c_action = self._extract_structured_field(curr_fact["text"], "Actions") or self._extract_structured_field(curr_fact["text"], "Actions, Hand Movements & Trajectories")
            n_action = self._extract_structured_field(next_fact["text"], "Actions") or self._extract_structured_field(next_fact["text"], "Actions, Hand Movements & Trajectories")
            c_obj = self._extract_structured_field(curr_fact["text"], "Objects") or self._extract_structured_field(curr_fact["text"], "Objects, Furniture & Spatial Positions")
            n_obj = self._extract_structured_field(next_fact["text"], "Objects") or self._extract_structured_field(next_fact["text"], "Objects, Furniture & Spatial Positions")
            c_env = self._extract_structured_field(curr_fact["text"], "Scene / Environment") or self._extract_structured_field(curr_fact["text"], "Environment & Setting")
            n_env = self._extract_structured_field(next_fact["text"], "Scene / Environment") or self._extract_structured_field(next_fact["text"], "Environment & Setting")

            note_text = None
            diff_summary = ""

            if use_ollama:
                try:
                    prompt = (
                        f"You are a Video Action, Spatial & Temporal Transition Analyzer.\n"
                        f"Compare these two consecutive video keyframes:\n\n"
                        f"[Frame {f_curr_id} at {t_start:.2f}s]:\n{curr_fact['text'][:250]}\n\n"
                        f"[Frame {f_next_id} at {t_end:.2f}s]:\n{next_fact['text'][:250]}\n\n"
                        f"Analyze observable actions and transitions between these frames:\n"
                        f"- What actions started, ended, or continued (e.g. person walks, sits, stands, picks up/puts down object, interacts with table/chair/tool).\n"
                        f"- Observable person movement or quadrant shift (left-to-right, foreground-to-background).\n"
                        f"- Object movement or positional change.\n"
                        f"- People entering or leaving, count changes.\n"
                        f"Report ONLY observable actions supported by visual evidence. Do not infer intentions.\n"
                        f"In 1 or 2 concise sentences, describe the transition."
                    )
                    payload = {
                        "model": OLLAMA_MODEL,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 70,
                        }
                    }
                    res = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=4.0)
                    if res.status_code == 200:
                        raw_desc = res.json().get("response", "").strip()
                        if raw_desc:
                            diff_summary = raw_desc
                            note_text = f"Temporal Transition [{t_start:.2f}s to {t_end:.2f}s]:\n{raw_desc}"
                except Exception:
                    pass

            if not note_text:
                parts = []
                if c_action and n_action:
                    if c_action.strip().lower() == n_action.strip().lower():
                        parts.append(f"Ongoing action: {c_action}.")
                    else:
                        parts.append(f"Action sequence: After [{c_action}], the action transitions to [{n_action}].")
                elif n_action:
                    parts.append(f"Action begins: {n_action}.")
                elif c_action:
                    parts.append(f"Previous action [{c_action}] concludes.")

                if c_obj and n_obj and c_obj.strip().lower() != n_obj.strip().lower():
                    parts.append(f"Spatial & object state changes from [{c_obj}] to [{n_obj}].")

                if c_env and n_env and c_env.strip().lower() != n_env.strip().lower():
                    parts.append(f"Setting changes from [{c_env}] to [{n_env}].")

                if not parts:
                    c_clean = curr_fact["text"].replace("\n", " ").strip()
                    n_clean = next_fact["text"].replace("\n", " ").strip()
                    parts.append(f"State evolves from [{c_clean[:90]}] into [{n_clean[:90]}].")

                diff_summary = " ".join(parts)
                note_text = f"Temporal Transition [{t_start:.2f}s to {t_end:.2f}s]:\n{diff_summary}"

            matched_tc = count_change_map.get((f_curr_id, f_next_id), count_change_map.get(i))
            count_change_summary = ""
            if matched_tc and matched_tc.get("summary"):
                count_change_summary = matched_tc["summary"]
                note_text += f"\nVisible Count & Entity Dynamics: {count_change_summary}"

            print(f"  [Temporal Transition {i}->{i+1}]: {note_text.splitlines()[-1][:90]}...")
            transition_facts.append({
                "type": "visual",
                "frame_id": f"trans_{i}_{i+1}",
                "start_time": t_start,
                "end_time": t_end,
                "text": note_text,
            })

            temporal_notes_list.append({
                "from_frame": f_curr_id,
                "to_frame": f_next_id,
                "from_time": round(t_start, 2),
                "to_time": round(t_end, 2),
                "temporal_note": note_text,
                "action_analysis": diff_summary,
                "count_changes": count_change_summary,
            })

        all_visual_facts = visual_facts + transition_facts
        all_visual_facts.sort(key=lambda x: x["start_time"])
        self.latest_temporal_notes = temporal_notes_list
        return all_visual_facts, temporal_notes_list


    # --------------------------------------------------------------------------
    # SEGMENT 15: MULTI-FEATURE PERSON TRACKING & UNIQUE VIDEO COUNTING
    # --------------------------------------------------------------------------
    def track_entities_and_counts(
        self, keyframes: list, visual_facts: list
    ) -> Tuple[list, CrossFrameEntityTracker, dict]:
        """
        Cross-Frame Multi-Feature Person Tracking & Unique Category Counting.
        Integrates resolved Grounding DINO detections (person crops, face embeddings,
        clothing colors, spatial position) with VLM contextual captions.
        Assigns persistent IDs (PERSON_001, CHAIR_001) and computes video-level unique counts.
        """
        print(f"Tracking person identities and computing video-level unique counts across {len(visual_facts)} keyframe(s)...")
        tracker = CrossFrameEntityTracker(
            similarity_threshold=0.60
        )

        kf_by_id = {k["frame_id"]: k for k in keyframes}

        for fact in visual_facts:
            f_id = fact["frame_id"]
            if str(f_id).startswith("trans_") or f_id == "video_unique_counts":
                continue

            mid_t = (fact["start_time"] + fact["end_time"]) / 2.0
            kf_obj = kf_by_id.get(f_id)
            pil_img = kf_obj.get("image") if kf_obj else None

            dino_dets = kf_obj.get("grounding_dino_detections", []) if kf_obj else []
            dino_counts = kf_obj.get("object_counts", {}) if kf_obj else {}

            # Construct detection entity records directly from Grounding DINO resolved boxes
            frame_entities = []
            for det in dino_dets:
                cat = det["category"]
                pos = det.get("position", "Center")
                box = det.get("bounding_box")
                norm = CategoryNormalizer.normalize(cat)
                canonical_cat = norm["category"]

                desc = f"{cat} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"
                if det.get("subcategory"):
                    desc = f"{det['subcategory']} at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}]"

                frame_entities.append({
                    "category": canonical_cat,
                    "subcategory": det.get("subcategory") or norm["subcategory"],
                    "specific_type": det.get("specific_type") or norm["specific_type"],
                    "frame_id": f_id,
                    "timestamp": mid_t,
                    "description": desc,
                    "position": pos,
                    "bounding_box": box,
                    "persistent_id": None,
                    "association_confidence": det.get("confidence", 1.0),
                    "association_status": "unassigned"
                })

            # Record frame visible counts (physical persons + objects)
            tracker.frame_counts_history[f_id] = dict(dino_counts)

            # Associate persons and objects via multi-feature matching
            associated_entities = tracker.associate_frame(
                frame_entities=frame_entities,
                frame_id=f_id,
                timestamp=mid_t,
                pil_image=pil_img
            )

            fact["frame_counts"] = dict(dino_counts)
            fact["tracked_entities"] = [
                {
                    "persistent_id": e["persistent_id"],
                    "category": e["category"],
                    "subcategory": e.get("subcategory"),
                    "specific_type": e.get("specific_type"),
                    "description": e["description"],
                    "position": e["position"],
                    "association_confidence": e.get("association_confidence", 1.0)
                }
                for e in associated_entities
            ]

            tracked_summary = []
            for e in associated_entities:
                tracked_summary.append(
                    f"- {e['persistent_id']} [{e['category']}]: {e['description']} (Position: {e['position']})"
                )
            if tracked_summary:
                fact["text"] += "\n\nTracked Physical Entities in Frame:\n" + "\n".join(tracked_summary)

        unique_video_counts = tracker.compute_unique_video_counts()
        print(f"Unique Video-Level Counts: {unique_video_counts}")

        temporal_count_changes = tracker.compute_temporal_count_changes(keyframes)
        tracking_metadata = tracker.get_metadata()

        return visual_facts, tracker, tracking_metadata


    # --------------------------------------------------------------------------
    # SEGMENT 16: COMPLETE 16-STEP PIPELINE ORCHESTRATION (`process_video`)
    # --------------------------------------------------------------------------
    def process_video(self, video_path: str, save_keyframes: bool = True):
        """Executes the exact 16-step visual extraction pipeline in sequential order."""
        print(f"\n================================================================================")
        print(f"Processing Video through Exact 16-Step Pipeline: {video_path}")
        print(f"================================================================================")

        adaptive_keyframes = self.extract_keyframes(video_path, scene_threshold=self.scene_threshold)

        if not adaptive_keyframes:
            print("[Pipeline Warning] No adaptive keyframes extracted for video.")
            return []

        print(f"\n[Step 13] Generating structured Qwen VLM scene captions for {len(adaptive_keyframes)} keyframe(s)...")
        visual_facts = self.generate_descriptions(adaptive_keyframes)
        for kf, fact in zip(adaptive_keyframes, visual_facts):
            kf["vlm_caption"] = fact.get("text", "")

        print(f"\n[Person Tracking] Performing multi-feature person identity tracking and persistent ID assignment...")
        visual_facts, tracker, tracking_metadata = self.track_entities_and_counts(adaptive_keyframes, visual_facts)

        print(f"\n[Steps 14 & 15] Generating temporal scene notes and action analysis...")
        temporal_count_changes = tracking_metadata.get("temporal_count_changes", [])
        visual_facts_complete, temporal_scene_notes = self.generate_temporal_scene_notes(
            visual_facts=visual_facts,
            temporal_count_changes=temporal_count_changes,
            keyframes=adaptive_keyframes,
        )

        vid_duration = adaptive_keyframes[-1]["end_time"] if adaptive_keyframes else 0.0
        unique_video_counts = tracking_metadata.get("unique_video_counts", {})
        unique_counts_text_parts = ["Video-Level Unique Entity Counts & Persistent Tracking:"]
        for cat, cnt in unique_video_counts.items():
            if cat == "people":
                people_ids = [p_id for p_id, t in tracker.tracked_entities.items() if t.category == "person"]
                unique_counts_text_parts.append(f"- Unique People in Video: {cnt} (Persistent IDs: {', '.join(sorted(people_ids))})")
            elif cat == "men":
                men_ids = [p_id for p_id, t in tracker.tracked_entities.items() if t.category == "person" and t.subcategory == "man"]
                unique_counts_text_parts.append(f"- Unique Men: {cnt} (Persistent IDs: {', '.join(sorted(men_ids))})")
            elif cat == "women":
                women_ids = [p_id for p_id, t in tracker.tracked_entities.items() if t.category == "person" and t.subcategory == "woman"]
                unique_counts_text_parts.append(f"- Unique Women: {cnt} (Persistent IDs: {', '.join(sorted(women_ids))})")
            elif cat == "children":
                child_ids = [p_id for p_id, t in tracker.tracked_entities.items() if t.category == "person" and t.subcategory == "child"]
                unique_counts_text_parts.append(f"- Unique Children: {cnt} (Persistent IDs: {', '.join(sorted(child_ids))})")
            else:
                c_norm = CategoryNormalizer.normalize(cat)
                cat_ids = [p_id for p_id, t in tracker.tracked_entities.items() if t.category == c_norm["category"]]
                unique_counts_text_parts.append(f"- Unique {cat.replace('_', ' ').title()} in Video: {cnt} (Persistent IDs: {', '.join(sorted(cat_ids))})")

        unique_counts_text_parts.append(
            "\nTracking Summary: Persistent IDs are assigned to uniquely distinguish entities across frames. "
            "Same entities appearing across multiple frames are counted only once in video-level unique counts."
        )
        unique_counts_fact_text = "\n".join(unique_counts_text_parts)

        video_counting_fact = {
            "type": "visual",
            "frame_id": "video_unique_counts",
            "start_time": 0.0,
            "end_time": vid_duration,
            "text": unique_counts_fact_text,
            "unique_video_counts": unique_video_counts
        }
        visual_facts_complete.append(video_counting_fact)

        tracking_metadata["visual_facts"] = visual_facts_complete
        tracking_metadata["temporal_scene_notes"] = temporal_scene_notes
        self.latest_tracking_metadata = tracking_metadata

        if save_keyframes:
            print(f"\n[Step 16] Saving keyframes and metadata to disk...")
            self.save_keyframes_to_disk(
                video_path=video_path,
                keyframes=adaptive_keyframes,
                candidate_metadata=self.latest_candidate_metadata,
                tracking_metadata=tracking_metadata,
                scene_detection_metadata=self.latest_scene_detection,
                temporal_scene_notes=temporal_scene_notes,
            )

        print(f"================================================================================")
        print(f"Visual Extraction Pipeline Finished: {len(adaptive_keyframes)} adaptive keyframes, {len(visual_facts_complete)} visual facts.")
        print(f"================================================================================\n")
        return visual_facts_complete


if __name__ == "__main__":
    pass
