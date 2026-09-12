import os
from typing import Optional, Dict, Any, List

# Set Intel Level Zero device selector before torch import to target dedicated GPU and avoid oneDNN conflicts
if "ONEAPI_DEVICE_SELECTOR" not in os.environ:
    os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"

import torch

# Base Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_stores")

# Create data directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

# Hardware Configuration & Dual Mode (GPU if available, else optimized CPU fallback)
FORCE_DEVICE = os.environ.get("FORCE_DEVICE", os.environ.get("DEVICE", "")).lower().strip()

IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_XPU_AVAILABLE = hasattr(torch, "xpu") and torch.xpu.is_available() if hasattr(torch, "xpu") else False

# Safety verification for Intel XPU / Level Zero:
# Qwen2.5-VL-3B requires >= 8-10 GB dedicated VRAM in FP16/BF16.
# Entry-level Intel Arc GPUs (e.g. Arc A380 6GB) or integrated Xe graphics cannot fit the weights without paging over PCIe,
# which triggers Windows TDR timeouts (UR_RESULT_ERROR_DEVICE_LOST) and Level Zero driver crashes (UR_RESULT_ERROR_UNKNOWN: 2147483646).
if IS_XPU_AVAILABLE and FORCE_DEVICE != "xpu":
    try:
        xpu_mem_bytes = torch.xpu.get_device_properties(0).total_memory
        xpu_mem_gb = xpu_mem_bytes / (1024**3)
        if xpu_mem_gb < 10.0:
            gpu_name = torch.xpu.get_device_name(0)
            print(f"[Hardware Setup] Intel Arc/XPU GPU detected: '{gpu_name}' with {xpu_mem_gb:.2f} GB VRAM.")
            print(f"[Hardware Setup] Vision Model (Qwen2.5-VL) requires >= 8-10 GB VRAM. To avoid Level Zero driver crashes (UR_RESULT_ERROR_UNKNOWN) and Windows TDR resets, automatically routing pipeline to Optimized CPU Mode.")
            print(f"[Hardware Setup] (Set environment variable FORCE_DEVICE=xpu if you explicitly wish to override this safety check).")
            IS_XPU_AVAILABLE = False
    except Exception as e:
        print(f"[Hardware Setup Warning] Could not inspect XPU memory: {e}")

IS_DML_AVAILABLE = False
try:
    import torch_directml
    if torch_directml.is_available():
        IS_DML_AVAILABLE = True
except ImportError:
    IS_DML_AVAILABLE = False

if FORCE_DEVICE in ["cpu", "cuda", "xpu", "directml"]:
    if FORCE_DEVICE == "cuda" and IS_CUDA_AVAILABLE:
        DEVICE = "cuda"
    elif FORCE_DEVICE == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
        DEVICE = "xpu"
    elif FORCE_DEVICE == "directml" and IS_DML_AVAILABLE:
        import torch_directml
        DEVICE = torch_directml.device()
    else:
        DEVICE = "cpu"
elif IS_CUDA_AVAILABLE:
    DEVICE = "cuda"
elif IS_DML_AVAILABLE:
    import torch_directml
    DEVICE = torch_directml.device()
elif IS_XPU_AVAILABLE:
    DEVICE = "xpu"
else:
    DEVICE = "cpu"

IS_GPU = DEVICE != "cpu"

# Precision Configuration
if DEVICE == "cuda":
    TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
elif DEVICE == "xpu":
    TORCH_DTYPE = torch.bfloat16 if (hasattr(torch.xpu, "is_bf16_supported") and torch.xpu.is_bf16_supported()) else torch.float16
elif IS_GPU:
    TORCH_DTYPE = torch.float16
else:
    TORCH_DTYPE = torch.float32

def empty_gpu_cache():
    """Safely frees accelerator memory on CUDA or Intel XPU devices."""
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif DEVICE == "xpu" and hasattr(torch, "xpu") and hasattr(torch.xpu, "empty_cache"):
        torch.xpu.empty_cache()

print(f"[Hardware Setup] Pipeline Running in Dual Mode: {'GPU (' + str(DEVICE) + ')' if IS_GPU else 'CPU (Optimized Fallback Mode)'}")
print(f"[Hardware Setup] Selected Data Type: {TORCH_DTYPE}")

# Automatically load .env file from BASE_DIR if present
ENV_FILE = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_FILE):
    try:
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip().strip("\"'")
                    if k not in os.environ and v:
                        os.environ[k] = v
    except Exception:
        pass

# The Hugging Face access token is loaded from environment variables, .env, or HF cache.
# Supported env vars: HF_TOKEN, HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_TOKEN
HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
)
if not HF_TOKEN:
    try:
        from huggingface_hub import get_token
        HF_TOKEN = get_token()
    except Exception:
        pass
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()

def get_hf_token() -> str:
    """Returns the Hugging Face access token securely (or None if not set)."""
    return HF_TOKEN

def has_hf_token() -> bool:
    """Returns True if a valid Hugging Face access token is present in the environment."""
    return bool(HF_TOKEN)

# ------------------------------------------------------------------------------
# EXPERIMENTAL MODEL REGISTRY & CONFIGURATION
# ------------------------------------------------------------------------------
# Supported LLM Generators for experimentation:
#   - qwen2.5-v1-72b-instruct (DEFAULT): Qwen 2.5 72B Instruct via HF API or multi-GPU
#   - gemma-4-31b: Google Gemma 2 27B/31B Instruct
#   - phi-3.5-vision-instruct: Microsoft Phi-3.5 Vision Instruct
#   (Any arbitrary Hugging Face model repository ID is also accepted)
SUPPORTED_LLM_GENERATORS = {
    "qwen2.5-v1-72b-instruct": {
        "primary_hf_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "fallback_hf_id": "Qwen/Qwen2.5-72B-Instruct",
        "type": "vision_llm",
        "description": "Qwen2.5-VL 72B Instruct (Vision-Language & Text)",
        "aliases": ["qwen2.5-v1-72b-instruct", "qwen2.5-vl-72b-instruct", "qwen2.5-72b-instruct", "qwen2.5-72b", "qwen-72b"]
    },
    "gemma-4-31b": {
        "primary_hf_id": "google/gemma-2-27b-it",
        "fallback_hf_id": "google/gemma-2-9b-it",
        "type": "text_llm",
        "description": "Google Gemma 2 27B/31B Instruct",
        "aliases": ["gemma-4-31b", "gemma-2-27b-it", "gemma-2-27b", "gemma-31b", "gemma"]
    },
    "phi-3.5-vision-instruct": {
        "primary_hf_id": "microsoft/Phi-3.5-vision-instruct",
        "fallback_hf_id": "microsoft/Phi-3-vision-128k-instruct",
        "type": "vision_llm",
        "description": "Microsoft Phi-3.5 Vision Instruct",
        "aliases": ["phi-3.5-vision-instruct", "phi-3.5-vision", "phi-3.5", "phi3.5"]
    },
    "qwen/qwen2.5-vl-3b-instruct": {
        "primary_hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "fallback_hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "type": "vision_llm",
        "description": "Qwen2.5-VL 3B Instruct (Lightweight Local VLM)",
        "aliases": ["qwen/qwen2.5-vl-3b-instruct", "qwen2.5-vl-3b-instruct", "qwen2.5-vl-3b", "qwen-3b"]
    }
}

# Supported Visual Captioning Models for experimentation:
#   - Salesforce/blip-image-captioning-large (DEFAULT): SOTA dense image captioning
#   - Salesforce/blip-image-captioning-base: Fast lightweight BLIP captioning
#   - HuggingFaceTB/SmolVLM-256M-Instruct: Compact 256M parameter multimodal VLM
#   - wraps/moondream-caption (or vikhyatk/moondream2): Efficient edge vision model
#   (Any arbitrary Hugging Face model repository ID is also accepted)
SUPPORTED_CAPTIONING_MODELS = {
    "salesforce/blip-image-captioning-large": {
        "hf_id": "Salesforce/blip-image-captioning-large",
        "fallback_hf_id": "Salesforce/blip-image-captioning-base",
        "architecture": "blip",
        "description": "Salesforce BLIP Large Image Captioning",
        "aliases": ["salesforce/blip-image-captioning-large", "blip-large", "blip"]
    },
    "salesforce/blip-image-captioning-base": {
        "hf_id": "Salesforce/blip-image-captioning-base",
        "fallback_hf_id": "Salesforce/blip-image-captioning-large",
        "architecture": "blip",
        "description": "Salesforce BLIP Base Image Captioning",
        "aliases": ["salesforce/blip-image-captioning-base", "blip-base"]
    },
    "huggingfacetb/smolvlm-256m-instruct": {
        "hf_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "fallback_hf_id": "Salesforce/blip-image-captioning-base",
        "architecture": "smolvlm",
        "description": "HuggingFaceTB SmolVLM 256M Instruct",
        "aliases": ["huggingfacetb/smolvlm-256m-instruct", "smolvlm", "smolvlm-256m"]
    },
    "wraps/moondream-caption": {
        "hf_id": "wraps/moondream-caption",
        "fallback_hf_id": "vikhyatk/moondream2",
        "architecture": "moondream",
        "description": "Moondream Vision Captioning",
        "aliases": ["wraps/moondream-caption", "moondream-caption", "moondream", "vikhyatk/moondream2"]
    },
    "qwen/qwen2.5-vl-3b-instruct": {
        "hf_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "fallback_hf_id": "Salesforce/blip-image-captioning-large",
        "architecture": "qwen_vl",
        "description": "Qwen2.5-VL 3B Instruct Structured Captioning",
        "aliases": ["qwen/qwen2.5-vl-3b-instruct", "qwen2.5-vl-3b", "qwen-vl"]
    }
}

def resolve_generator_model(name: Optional[str] = None) -> dict:
    """Resolves model name/alias to canonical HF repo ID and configuration."""
    raw = (name or GENERATOR_MODEL).strip()
    low = raw.lower()
    for key, spec in SUPPORTED_LLM_GENERATORS.items():
        if low == key.lower() or low in [a.lower() for a in spec.get("aliases", [])]:
            return {
                "name": key,
                "hf_id": spec["primary_hf_id"],
                "fallback_hf_id": spec.get("fallback_hf_id", spec["primary_hf_id"]),
                "type": spec.get("type", "vision_llm"),
                "description": spec.get("description", raw)
            }
    # Custom user-specified repository
    return {
        "name": raw,
        "hf_id": raw,
        "fallback_hf_id": raw,
        "type": "auto",
        "description": f"Custom model ({raw})"
    }

def resolve_captioning_model(name: Optional[str] = None) -> dict:
    """Resolves captioning model name/alias to canonical HF repo ID and architecture."""
    raw = (name or CAPTIONING_MODEL).strip()
    low = raw.lower()
    for key, spec in SUPPORTED_CAPTIONING_MODELS.items():
        if low == key.lower() or low in [a.lower() for a in spec.get("aliases", [])]:
            return {
                "name": key,
                "hf_id": spec["hf_id"],
                "fallback_hf_id": spec.get("fallback_hf_id", spec["hf_id"]),
                "architecture": spec.get("architecture", "blip"),
                "description": spec.get("description", raw)
            }
    # Custom captioning model
    arch = "blip" if "blip" in low else ("smolvlm" if "smol" in low else ("moondream" if "moondream" in low else "auto"))
    return {
        "name": raw,
        "hf_id": raw,
        "fallback_hf_id": raw,
        "architecture": arch,
        "description": f"Custom captioning model ({raw})"
    }

# Active Default Model Identifiers (Overridable via environment variables)
# Default LLM Generator: qwen2.5-v1-72b-instruct
GENERATOR_MODEL = os.environ.get(
    "GENERATOR_MODEL",
    os.environ.get("LLM_MODEL", "qwen2.5-v1-72b-instruct")
)

# Default Visual Captioning Model: Salesforce/blip-image-captioning-large
CAPTIONING_MODEL = os.environ.get(
    "CAPTIONING_MODEL",
    os.environ.get("VLM_MODEL", "Salesforce/blip-image-captioning-large")
)
# Preserved for backward compatibility across pipeline
QWEN_VL_MODEL = CAPTIONING_MODEL

# Generator Execution Backend:
#   - 'auto' (default): Automatically selects HF Serverless API for large models (>=20B)
#                       or when local VRAM is limited; falls back to local PyTorch if offline.
#   - 'api' or 'serverless': Uses Hugging Face Inference API with HF_TOKEN.
#   - 'local': Uses local PyTorch / Transformers execution.
GENERATOR_BACKEND = os.environ.get(
    "GENERATOR_BACKEND",
    os.environ.get("LLM_BACKEND", "auto")
).lower().strip()

# Stage 1: Extraction & Indexing Models
WHISPER_MODEL = "large-v3"
CLAP_MODEL = "laion/clap-htsat-unfused"
CLIP_MODEL = "openai/clip-vit-base-patch32"
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast dense semantic embedder for FAISS & ChromaDB
SIMILARITY_THRESHOLD = 0.90

# Stage 2: Retrieval & Re-ranking
MODALITY_ESTIMATOR_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "BAAI/bge-reranker-large"

# Stage 3: Generation Hyperparameters
GENERATOR_MAX_NEW_TOKENS = 32
GENERATOR_DEVICE = DEVICE

# Legacy / Optional Ollama Configurations (preserved for backwards compatibility)
OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_HOST = "http://localhost:11434"

# Retrieval & Threshold Configurations
CHROMA_AUDIO_COLLECTION = "audio_collection"
FAISS_VISUAL_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "visual_index.faiss")
VISUAL_METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "visual_metadata.json")

# Hyperparameters
KA_DEFAULT = 5 # Default number of audio chunks to retrieve
KV_DEFAULT = 10 # Default number of visual chunks to retrieve
NMS_OVERLAP_THRESHOLD = 0.8 # 80% time overlap drops redundant lower-scoring evidence
AUDIO_BONUS = 1.0 # Added to reranker score if candidate is audio and beta(q) is high
TEMPORAL_AGREEMENT_BONUS = 0.5
MIN_AUDIO_EVIDENCE_THRESHOLD = 1
MODALITY_THRESHOLD = 0.6 # If beta(q) > 0.6, it's considered an audio-heavy question

# Audio Extraction Configurations
AUDIO_CHUNK_LENGTH = 3.0 # Duration of each audio chunk in seconds
AUDIO_OVERLAP = 1.0 # Overlap between consecutive audio chunks in seconds
AUDIO_THRESHOLD = 0.3 # Confidence threshold for CLAP detection
AUDIO_BATCH_SIZE = 16 if IS_GPU else 4 # Adaptive batch size for CLAP inference
MIN_EVENT_DURATION = 0.5 # Minimum duration for sound events in seconds
AUDIO_RMS_THRESHOLD = 0.001 # RMS energy threshold to skip silent/low-energy chunks
STEREO_BALANCE_THRESHOLD = 0.15 # Energy ratio threshold for Left vs Right channel sound source localization

# Visual Extraction Configurations
SCENE_THRESHOLD = 18.0 # Consecutive-frame visual change score threshold
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
MAX_SCENE_WINDOW_SEC = 4.0 # Maximum time window per keyframe segment to capture intra-scene actions
CLIP_BATCH_SIZE = 32 if IS_GPU else 8 # Batch size for CLIP image embedding generation
VLM_BATCH_SIZE = 1 # Batch size for Vision Model inference
MIN_KEYFRAMES = 8 # Minimum keyframe floor for short videos
MAX_KEYFRAMES = 60 if IS_GPU else 30 # Maximum keyframe cap for long videos (optimized for latency)
DYNAMIC_KEYFRAME_INTERVAL_SEC = 2.5 # 1 keyframe target per 2.5s of video

# Dual Mode Vision Optimization Settings
# Capping vision resolution and max output tokens on CPU prevents 50-minute delays
MIN_VISION_PIXELS = 256 * 14 * 14 if not IS_GPU else 256 * 28 * 28
MAX_VISION_PIXELS = 384 * 14 * 14 if not IS_GPU else 512 * 28 * 28
VLM_MAX_NEW_TOKENS = 128 if not IS_GPU else 256
VLM_IMAGE_RESIZE_MAX = 512 if not IS_GPU else 768
OLLAMA_MAX_TOKENS = 30 # Generation max new tokens for concise QA answers

# Visual Extraction Image Quality Check Configurations
BLUR_THRESHOLD = 100.0 # Laplacian variance threshold below which image is deemed blurry
DARK_THRESHOLD = 15.0 # Mean pixel brightness threshold below which image is deemed dark/black
BRIGHTNESS_MAX_THRESHOLD = 240.0 # Upper brightness threshold above which frame is deemed overexposed
LOW_CONTRAST_THRESHOLD = 20.0 # Standard deviation of pixel intensities threshold below which frame is low contrast
MIN_FRAME_WIDTH = 128 # Minimum frame width requirement in pixels
MIN_FRAME_HEIGHT = 128 # Minimum frame height requirement in pixels
CANDIDATE_SAMPLES_PER_SCENE = 5 # Number of candidate frames sampled per detected scene

KEYFRAMES_SAVE_DIR = os.path.join(DATA_DIR, "keyframes")
os.makedirs(KEYFRAMES_SAVE_DIR, exist_ok=True)







