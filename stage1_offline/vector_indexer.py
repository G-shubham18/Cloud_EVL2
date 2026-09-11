import os
import json
import numpy as np
import chromadb
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False
    faiss = None

class NumpyIndexFlatIP:
    """Pure-NumPy fallback for faiss.IndexFlatIP when faiss DLL is blocked by OS policy."""
    def __init__(self, d):
        self.d = d
        self.vectors = np.empty((0, d), dtype=np.float32)
        self.ntotal = 0

    def add(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.vectors.size == 0:
            self.vectors = x
        else:
            self.vectors = np.vstack([self.vectors, x])
        self.ntotal = len(self.vectors)

    def search(self, x, k):
        if self.ntotal == 0:
            return np.empty((len(x), 0), dtype=np.float32), np.empty((len(x), 0), dtype=np.int64)
        scores = np.dot(x, self.vectors.T)
        k = min(k, self.ntotal)
        indices = np.argsort(-scores, axis=1)[:, :k]
        top_scores = np.take_along_axis(scores, indices, axis=1)
        return top_scores, indices

def _create_flat_ip_index(dim):
    if HAS_FAISS and faiss is not None:
        try:
            return faiss.IndexFlatIP(dim)
        except Exception:
            pass
    return NumpyIndexFlatIP(dim)

def _read_faiss_index(path):
    if HAS_FAISS and faiss is not None:
        try:
            return faiss.read_index(path)
        except Exception:
            pass
    npy_path = path + ".npy"
    if os.path.exists(npy_path):
        arr = np.load(npy_path)
        idx = NumpyIndexFlatIP(arr.shape[1])
        idx.add(arr)
        return idx
    return NumpyIndexFlatIP(384)

def _write_faiss_index(index, path):
    if HAS_FAISS and faiss is not None:
        try:
            faiss.write_index(index, path)
        except Exception:
            pass
    if hasattr(index, "vectors"):
        np.save(path + ".npy", index.vectors)
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor
import torch

from config import (
    VECTOR_STORE_DIR, CHROMA_AUDIO_COLLECTION, 
    FAISS_VISUAL_INDEX_PATH, VISUAL_METADATA_PATH,
    CLAP_MODEL, CLIP_MODEL, TEXT_EMBEDDING_MODEL, DEVICE
)

def is_video_indexed_on_disk(video_path: str, store_dir: str) -> bool:
    """Fast check on disk without loading heavy models or ChromaDB."""
    if not os.path.exists(store_dir):
        return False
    info_path = os.path.join(store_dir, "indexed_video.json")
    if not os.path.exists(info_path):
        return False
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            indexed_data = json.load(f)
            indexed_vid = indexed_data.get("video_path")
            if not indexed_vid:
                return False
            norm_target = os.path.normpath(os.path.abspath(video_path)).lower()
            norm_indexed = os.path.normpath(os.path.abspath(indexed_vid)).lower()
            if norm_target == norm_indexed or os.path.basename(norm_target) == os.path.basename(norm_indexed):
                faiss_path = os.path.join(store_dir, "visual_index.faiss")
                chroma_dir = os.path.join(store_dir, "chroma")
                if os.path.exists(faiss_path) or os.path.exists(chroma_dir):
                    return True
    except Exception:
        return False
    return False

class VectorIndexer:
    _shared_clap_model = None
    _shared_clap_processor = None
    _shared_audio_embedder = None
    _shared_visual_embedder = None
    _models_initialized = False

    @classmethod
    def _init_shared_models(cls):
        if cls._models_initialized:
            return
        
        # Audio Embedding Model (CLAP)
        print(f"Loading CLAP Model for Audio Store: {CLAP_MODEL}")
        from config import IS_CUDA_AVAILABLE
        embed_dev = DEVICE if IS_CUDA_AVAILABLE else "cpu"
        cls._shared_clap_device = embed_dev
        try:
            from transformers import ClapModel, ClapProcessor
            cls._shared_clap_model = ClapModel.from_pretrained(CLAP_MODEL).to(embed_dev)
            cls._shared_clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL)
            print(f"[VectorIndexer] Successfully loaded CLAP text encoder on {embed_dev} for audio store.")
        except Exception as e:
            print(f"[VectorIndexer Warning] Could not load CLAP via transformers: {e}. Falling back to sentence-transformers.")
            cls._shared_audio_embedder = SentenceTransformer(TEXT_EMBEDDING_MODEL, device="cpu")

        # High-Speed SOTA Visual Text Embedding Model
        print(f"Loading Dense Semantic Text Embedder for Visual Store: {TEXT_EMBEDDING_MODEL} on device={embed_dev}")
        cls._shared_visual_embedder = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=embed_dev)
        cls._models_initialized = True

    def __init__(self, store_dir: str = None):
        self.store_dir = os.path.abspath(store_dir) if store_dir else VECTOR_STORE_DIR
        os.makedirs(self.store_dir, exist_ok=True)
        
        self.faiss_path = os.path.join(self.store_dir, "visual_index.faiss")
        self.metadata_path = os.path.join(self.store_dir, "visual_metadata.json")
        self.video_info_path = os.path.join(self.store_dir, "indexed_video.json")
        self.chroma_dir = os.path.join(self.store_dir, "chroma")

        self._init_shared_models()
        self.clap_model = self._shared_clap_model
        self.clap_processor = self._shared_clap_processor
        self.audio_embedder = self._shared_audio_embedder
        self.visual_embedder = self._shared_visual_embedder

        # Initialize ChromaDB for Audio
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_dir)
        self.audio_collection = self.chroma_client.get_or_create_collection(name=CHROMA_AUDIO_COLLECTION)

        # Initialize FAISS for Visual
        get_dim_fn = getattr(self.visual_embedder, "get_embedding_dimension", getattr(self.visual_embedder, "get_sentence_embedding_dimension", None))
        self.visual_dim = get_dim_fn() if get_dim_fn else 384
        if os.path.exists(self.faiss_path) and os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.visual_metadata = json.load(f)
                self.visual_index = _read_faiss_index(self.faiss_path)
                
                # If index dimension changed, rebuild FAISS index in milliseconds from cached visual_metadata
                index_d = getattr(self.visual_index, "d", getattr(self.visual_index, "dim", self.visual_dim))
                if index_d != self.visual_dim:
                    print(f"[VectorIndexer] Migrating FAISS visual index dimension ({index_d} -> {self.visual_dim}) at '{os.path.basename(self.store_dir)}'...")
                    self.visual_index = _create_flat_ip_index(self.visual_dim)
                    if self.visual_metadata:
                        texts = [m["text"] for m in self.visual_metadata]
                        embeddings = self.visual_embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                        self.visual_index.add(embeddings.astype("float32"))
                        _write_faiss_index(self.visual_index, self.faiss_path)
            except Exception as e:
                print(f"[VectorIndexer Warning] Re-initializing FAISS index due to: {e}")
                self.visual_index = _create_flat_ip_index(self.visual_dim)
                self.visual_metadata = []
        else:
            self.visual_index = _create_flat_ip_index(self.visual_dim) # Inner product for cosine sim
            self.visual_metadata = []

    def clear_index(self):
        """Clears all stored audio and visual vector indices and metadata."""
        print(f"[VectorIndexer] Clearing vector store at '{self.store_dir}'...")
        try:
            self.chroma_client.delete_collection(name=CHROMA_AUDIO_COLLECTION)
        except Exception:
            pass
        self.audio_collection = self.chroma_client.get_or_create_collection(name=CHROMA_AUDIO_COLLECTION)

        self.visual_index = _create_flat_ip_index(self.visual_dim)
        self.visual_metadata = []
        if os.path.exists(self.faiss_path):
            os.remove(self.faiss_path)
        if os.path.exists(self.metadata_path):
            os.remove(self.metadata_path)
        if os.path.exists(self.video_info_path):
            os.remove(self.video_info_path)

    def get_indexed_video(self):
        """Returns the absolute path of the video currently indexed, or None if empty."""
        if os.path.exists(self.video_info_path):
            try:
                with open(self.video_info_path, "r") as f:
                    return json.load(f).get("video_path")
            except Exception:
                return None
        return None

    def set_indexed_video(self, video_path: str):
        """Saves the absolute path of the newly indexed video."""
        with open(self.video_info_path, "w") as f:
            json.dump({"video_path": os.path.abspath(video_path)}, f)

    def is_indexed(self, video_path: str) -> bool:
        """Returns True if the specified video is already indexed in this store."""
        indexed_vid = self.get_indexed_video()
        if indexed_vid:
            norm_target = os.path.normpath(os.path.abspath(video_path)).lower()
            norm_indexed = os.path.normpath(os.path.abspath(indexed_vid)).lower()
            if norm_target == norm_indexed or os.path.basename(norm_target) == os.path.basename(norm_indexed):
                if os.path.exists(self.video_info_path) and (os.path.exists(self.faiss_path) or os.path.exists(self.chroma_dir)):
                    return True
        return False

    def embed_audio_text(self, text: str):
        """Embeds audio text facts using CLAP's text encoder (or SentenceTransformer fallback)."""
        if getattr(self, "clap_model", None) is not None and getattr(self, "clap_processor", None) is not None:
            c_dev = getattr(self, "_shared_clap_device", getattr(self.__class__, "_shared_clap_device", "cpu"))
            inputs = self.clap_processor(text=text, return_tensors="pt", padding=True, truncation=True).to(c_dev)
            with torch.no_grad():
                outputs = self.clap_model.get_text_features(**inputs)
                if isinstance(outputs, torch.Tensor):
                    tensor = outputs
                elif hasattr(outputs, "text_embeds"):
                    tensor = outputs.text_embeds
                elif hasattr(outputs, "pooler_output"):
                    tensor = outputs.pooler_output
                else:
                    tensor = outputs[0]
            embed = tensor.cpu().numpy()[0]
        else:
            embed = self.audio_embedder.encode(text, convert_to_numpy=True)
        norm = np.linalg.norm(embed)
        if norm > 0:
            embed = embed / norm
        return embed

    def embed_visual_text(self, text: str):
        """Embeds text descriptions using dense semantic text embedder with L2 normalization."""
        embed = self.visual_embedder.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return embed.astype("float32")

    def index_audio_facts(self, audio_facts: list):
        """Indexes audio transcripts and sound events into ChromaDB."""
        if not audio_facts:
            return

        # Normalize dict returned by AudioExtractor into flat fact list
        if isinstance(audio_facts, dict):
            flat_facts = []
            for t in audio_facts.get("transcript", []):
                t_text = t.get("text", "").strip()
                if t_text:
                    flat_facts.append({
                        "type": "speech",
                        "start_time": t.get("start_time", 0.0),
                        "end_time": t.get("end_time", 0.0),
                        "text": t_text
                    })
            for e in audio_facts.get("events", []):
                label = e.get("label") or e.get("event") or e.get("text") or ""
                source = e.get("source_side", "")
                source_tag = f" [Audio Source: {source}]" if source else ""
                e_text = f"{label}{source_tag}".strip()
                if e_text:
                    flat_facts.append({
                        "type": "sound",
                        "start_time": e.get("start_time", 0.0),
                        "end_time": e.get("end_time", 0.0),
                        "text": e_text,
                        "score": e.get("score", 1.0)
                    })
            audio_facts = flat_facts

        if not audio_facts:
            print("No valid audio facts to index.")
            return

        print(f"Indexing {len(audio_facts)} audio facts into ChromaDB...")
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []
        
        for i, fact in enumerate(audio_facts):
            if not isinstance(fact, dict):
                continue
            text = fact.get("text", "")
            if not text:
                continue
            fact_id = f"audio_{i}"
            
            ids.append(fact_id)
            documents.append(text)
            embeddings.append(self.embed_audio_text(text).tolist())
            
            metadata = {
                "type": fact.get("type", "audio"),
                "start_time": fact.get("start_time", 0.0),
                "end_time": fact.get("end_time", 0.0)
            }
            if "score" in fact:
                metadata["score"] = fact["score"]
            metadatas.append(metadata)
            
        if ids:
            self.audio_collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )

    def index_visual_facts(self, visual_facts: list, tracking_metadata: dict = None):
        """Indexes visual descriptions and tracking/counting facts into FAISS and stores metadata."""
        if not visual_facts:
            return
            
        print(f"Indexing {len(visual_facts)} visual facts into FAISS...")
        
        embeddings = []
        
        for fact in visual_facts:
            text = fact["text"]
            embed = self.embed_visual_text(text)
            embeddings.append(embed)
            
            meta_record = {
                "type": "visual",
                "frame_id": fact["frame_id"],
                "start_time": fact["start_time"],
                "end_time": fact["end_time"],
                "text": text
            }
            if "frame_counts" in fact:
                meta_record["frame_counts"] = fact["frame_counts"]
            if "unique_video_counts" in fact:
                meta_record["unique_video_counts"] = fact["unique_video_counts"]
            if "tracked_entities" in fact:
                meta_record["tracked_entities"] = fact["tracked_entities"]

            self.visual_metadata.append(meta_record)
            
        # Add to FAISS
        embeddings_np = np.vstack(embeddings).astype('float32')
        self.visual_index.add(embeddings_np)
        
        # Save to disk
        _write_faiss_index(self.visual_index, self.faiss_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.visual_metadata, f, indent=4, ensure_ascii=False)

        if tracking_metadata:
            tracking_path = os.path.join(self.store_dir, "tracking_metadata.json")
            with open(tracking_path, "w", encoding="utf-8") as tf:
                json.dump(tracking_metadata, tf, indent=4, ensure_ascii=False)
            print(f"Saved tracking metadata to '{tracking_path}'.")

if __name__ == "__main__":
    pass
