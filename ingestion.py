import os
import json
import time
import hashlib
import traceback
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from config import VECTOR_STORE_DIR
from stage1_offline.vector_indexer import VectorIndexer, is_video_indexed_on_disk

# Supported video file extensions for recursive scanning
SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".webm", ".m4v", ".3gp", ".ts"
}


def format_duration(seconds: Optional[float]) -> str:
    """
    Formats duration in seconds into a human-readable string:
    e.g., '12.34s', '3m 45.2s', '1h 12m 57s'.
    """
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes = int(seconds // 60)
    rem_sec = seconds % 60
    if minutes < 60:
        return f"{minutes}m {rem_sec:.1f}s"
    hours = int(minutes // 60)
    rem_min = minutes % 60
    return f"{hours}h {rem_min}m {rem_sec:.0f}s"


def discover_videos(videos_dir: str) -> List[str]:
    """
    Recursively scans the given videos directory and returns a sorted list of absolute paths
    for all supported video files.
    """
    videos_path = Path(videos_dir)
    if not videos_path.exists():
        print(f"[Ingestion] Warning: Videos directory '{videos_dir}' does not exist.")
        return []

    discovered = []
    for root, _, files in os.walk(videos_path):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in SUPPORTED_VIDEO_EXTENSIONS:
                abs_path = os.path.abspath(os.path.join(root, file))
                discovered.append(abs_path)

    discovered.sort()
    return discovered


def get_video_store_dir(video_path: str, base_store_dir: Optional[str] = None) -> str:
    """
    Generates a unique, isolated vector store directory for a specific video file.
    Uses video filename stem and an MD5 path hash to guarantee isolation.
    """
    base_dir = base_store_dir if base_store_dir else VECTOR_STORE_DIR
    abs_path = os.path.abspath(video_path)
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()[:8]
    stem = Path(abs_path).stem
    # Sanitize stem for directory name
    safe_stem = "".join([c if c.isalnum() or c in ('-', '_') else '_' for c in stem])
    folder_name = f"{safe_stem}_{path_hash}"
    return os.path.join(base_dir, folder_name)


class Stage1Ingestor:
    """
    Handles Stage 1 (Offline Extraction & Indexing) independently for every video in the dataset.
    Stores Stage 1 output/index for each video separately so that data from one video
    never overwrites or mixes with another video.
    """
    def __init__(self, base_store_dir: Optional[str] = None, captioning_model: Optional[str] = None):
        self.base_store_dir = base_store_dir if base_store_dir else VECTOR_STORE_DIR
        self.captioning_model = captioning_model
        self.audio_extractor = None
        self.visual_extractor = None
        self.video_latencies: Dict[str, Dict[str, Any]] = {}
        self.latest_latency: Optional[Dict[str, Any]] = None

    def _lazy_init_extractors(self):
        """Lazily instantiates extractor models once to reuse model weights across videos."""
        if self.audio_extractor is None:
            print("[Stage 1 Ingestion] Initializing AudioExtractor model (Whisper & CLAP)...")
            from stage1_offline.audio_extractor import AudioExtractor
            self.audio_extractor = AudioExtractor()
        if self.visual_extractor is None:
            print("[Stage 1 Ingestion] Initializing VisualExtractor model...")
            from stage1_offline.visual_extractor import VisualExtractor
            self.visual_extractor = VisualExtractor(captioning_model=self.captioning_model)

    def _read_stored_latency(self, video_store_dir: str) -> Optional[Dict[str, Any]]:
        """Reads previously stored latency.json for a video if it exists on disk."""
        lat_file = os.path.join(video_store_dir, "latency.json")
        if os.path.exists(lat_file):
            try:
                with open(lat_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _save_stored_latency(self, video_store_dir: str, latency_record: Dict[str, Any]):
        """Persists video-specific latency record into its isolated vector store directory."""
        lat_file = os.path.join(video_store_dir, "latency.json")
        try:
            os.makedirs(video_store_dir, exist_ok=True)
            with open(lat_file, "w", encoding="utf-8") as f:
                json.dump(latency_record, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Stage 1 Ingestion Warning] Failed to save latency to {lat_file}: {e}")

    def save_latency_report(self, output_path: str = "output/ingestion_latency.json") -> str:
        """
        Saves the recorded video ingestion latencies to a JSON file.
        Also generates a secondary mapping file (video_name -> latency_seconds) for convenience.

        Returns:
            The absolute path of the primary saved JSON file.
        """
        abs_output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output_path), exist_ok=True)
        records = list(self.video_latencies.values())

        # Primary JSON format: list of objects with video_name and latency
        with open(abs_output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4, ensure_ascii=False)

        # Secondary JSON format: direct key-value mapping (video_name -> latency_seconds)
        map_path = os.path.splitext(abs_output_path)[0] + "_map.json"
        try:
            mapping = {
                rec["video_name"]: (
                    rec.get("latency_seconds") if rec.get("latency_seconds") is not None else rec.get("latency")
                )
                for rec in records
            }
            with open(map_path, "w", encoding="utf-8") as mf:
                json.dump(mapping, mf, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[Stage 1 Ingestion Warning] Could not save secondary latency map: {e}")

        return abs_output_path

    def process_single_video(
        self,
        video_path: str,
        force_reindex: bool = False
    ) -> Tuple[Optional[VectorIndexer], bool, Optional[str]]:
        """
        Runs Stage 1 (Offline Extraction & Indexing) independently for a single video.
        Accurately measures and logs latency during the ingestion phase.

        Returns:
            (indexer: Optional[VectorIndexer], was_reused: bool, error_message: Optional[str])
        """
        t_start = time.perf_counter()
        abs_video_path = os.path.abspath(video_path)
        video_store_dir = get_video_store_dir(abs_video_path, self.base_store_dir)
        v_name = os.path.basename(abs_video_path)

        print(f"\n--------------------------------------------------")
        print(f"[Stage 1 Ingestion] Video File: {abs_video_path}")
        print(f"[Stage 1 Ingestion] Isolated Vector Store: {video_store_dir}")

        # Fast check on disk: if index already exists and force_reindex is False, skip extraction
        if not force_reindex and is_video_indexed_on_disk(abs_video_path, video_store_dir):
            print(f"[Stage 1 Ingestion] [SKIP] Stage 1 index already exists for '{v_name}'. Reusing existing vector store.")
            temp_indexer = VectorIndexer(store_dir=video_store_dir)

            stored_lat = self._read_stored_latency(video_store_dir)
            if stored_lat:
                lat_record = dict(stored_lat)
                lat_record["reused"] = True
                lat_record["status"] = "success"
            else:
                elapsed = time.perf_counter() - t_start
                lat_record = {
                    "video_name": v_name,
                    "video_id": v_name,
                    "video_path": abs_video_path,
                    "latency": round(elapsed, 4),
                    "latency_seconds": round(elapsed, 4),
                    "latency_formatted": format_duration(elapsed),
                    "status": "success",
                    "reused": True,
                    "note": "Reused existing vector store (cache check latency)"
                }
            self.video_latencies[abs_video_path] = lat_record
            self.latest_latency = lat_record
            return temp_indexer, True, None

        temp_indexer = VectorIndexer(store_dir=video_store_dir)
        if not force_reindex and temp_indexer.is_indexed(abs_video_path):
            print(f"[Stage 1 Ingestion] [SKIP] Stage 1 index already exists for '{v_name}'. Reusing existing vector store.")
            stored_lat = self._read_stored_latency(video_store_dir)
            if stored_lat:
                lat_record = dict(stored_lat)
                lat_record["reused"] = True
                lat_record["status"] = "success"
            else:
                elapsed = time.perf_counter() - t_start
                lat_record = {
                    "video_name": v_name,
                    "video_id": v_name,
                    "video_path": abs_video_path,
                    "latency": round(elapsed, 4),
                    "latency_seconds": round(elapsed, 4),
                    "latency_formatted": format_duration(elapsed),
                    "status": "success",
                    "reused": True,
                    "note": "Reused existing vector store (cache check latency)"
                }
            self.video_latencies[abs_video_path] = lat_record
            self.latest_latency = lat_record
            return temp_indexer, True, None

        print(f"[Stage 1 Ingestion] [PROCESSING] Starting Stage 1 processing for '{v_name}'...")
        try:
            self._lazy_init_extractors()

            print(f" -> Stage 1: Concurrently extracting audio facts and visual keyframe descriptions...")

            def run_audio():
                ta0 = time.perf_counter()
                res = self.audio_extractor.process_video(abs_video_path)
                return res, time.perf_counter() - ta0

            def run_visual():
                tv0 = time.perf_counter()
                res = self.visual_extractor.process_video(abs_video_path)
                return res, time.perf_counter() - tv0

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                audio_future = executor.submit(run_audio)
                visual_future = executor.submit(run_visual)
                audio_facts, audio_time = audio_future.result()
                visual_facts, visual_time = visual_future.result()

            print(f" -> Stage 1: Building isolated vector stores (ChromaDB & FAISS)...")
            t_idx0 = time.perf_counter()
            temp_indexer.clear_index()
            temp_indexer.index_audio_facts(audio_facts)
            tracking_meta = getattr(self.visual_extractor, "latest_tracking_metadata", None)
            temp_indexer.index_visual_facts(visual_facts, tracking_metadata=tracking_meta)
            temp_indexer.set_indexed_video(abs_video_path)
            indexing_time = time.perf_counter() - t_idx0

            total_latency = time.perf_counter() - t_start
            lat_record = {
                "video_name": v_name,
                "video_id": v_name,
                "video_path": abs_video_path,
                "latency": round(total_latency, 3),
                "latency_seconds": round(total_latency, 3),
                "latency_formatted": format_duration(total_latency),
                "status": "success",
                "reused": False,
                "breakdown": {
                    "audio_extraction_seconds": round(audio_time, 3),
                    "visual_extraction_seconds": round(visual_time, 3),
                    "indexing_seconds": round(indexing_time, 3)
                },
                "timestamp": datetime.now().isoformat()
            }
            self._save_stored_latency(video_store_dir, lat_record)
            self.video_latencies[abs_video_path] = lat_record
            self.latest_latency = lat_record

            print(f"[Stage 1 Ingestion] Stage 1 COMPLETED successfully for '{v_name}' in {format_duration(total_latency)}.")
            return temp_indexer, False, None

        except Exception as e:
            elapsed = time.perf_counter() - t_start
            lat_record = {
                "video_name": v_name,
                "video_id": v_name,
                "video_path": abs_video_path,
                "latency": round(elapsed, 3),
                "latency_seconds": round(elapsed, 3),
                "latency_formatted": format_duration(elapsed),
                "status": "failed",
                "reused": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.video_latencies[abs_video_path] = lat_record
            self.latest_latency = lat_record
            err_msg = f"Failed Stage 1 processing for video '{abs_video_path}': {str(e)}\n{traceback.format_exc()}"
            print(f"[Stage 1 Ingestion ERROR] {err_msg}")
            return None, False, str(e)

    def process_dataset(
        self,
        videos_dir: str,
        force_reindex: bool = False,
        latency_file: Optional[str] = "output/ingestion_latency.json"
    ) -> Dict[str, Dict]:
        """
        Scans videos_dir recursively and runs Stage 1 for all discovered videos.
        Automatically skips videos that are already ingested.
        Records and incrementally saves the ingestion latency of each video to a JSON file.

        Returns a dictionary mapping video_path -> {
            'indexer': VectorIndexer,
            'store_dir': str,
            'status': 'success' | 'failed',
            'reused': bool,
            'latency_seconds': Optional[float],
            'latency': Optional[float],
            'breakdown': Optional[Dict],
            'error': Optional[str]
        }
        """
        discovered_videos = discover_videos(videos_dir)
        total_videos = len(discovered_videos)
        print(f"\n==================================================")
        print(f"[Stage 1 Dataset Ingestion] Discovered {total_videos} video(s) in '{videos_dir}'")
        if latency_file:
            print(f"[Stage 1 Dataset Ingestion] Latency JSON report output: '{os.path.abspath(latency_file)}'")
        print(f"==================================================")

        results = {}
        for idx, video_path in enumerate(discovered_videos, start=1):
            print(f"\n[Overall Progress: Video {idx}/{total_videos}] Current Video: {os.path.basename(video_path)}")
            indexer, reused, err = self.process_single_video(video_path, force_reindex=force_reindex)

            store_dir = get_video_store_dir(video_path, self.base_store_dir)
            lat_info = self.video_latencies.get(video_path, {})
            lat_sec = lat_info.get("latency_seconds") if lat_info.get("latency_seconds") is not None else lat_info.get("latency")

            if indexer is not None:
                results[video_path] = {
                    'indexer': indexer,
                    'store_dir': store_dir,
                    'status': 'success',
                    'reused': reused,
                    'latency_seconds': lat_sec,
                    'latency': lat_sec,
                    'breakdown': lat_info.get("breakdown"),
                    'error': None
                }
            else:
                results[video_path] = {
                    'indexer': None,
                    'store_dir': store_dir,
                    'status': 'failed',
                    'reused': False,
                    'latency_seconds': lat_sec,
                    'latency': lat_sec,
                    'breakdown': None,
                    'error': err
                }

            # Incrementally update latency JSON file after each video is processed
            if latency_file:
                self.save_latency_report(latency_file)

        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Ingestion: Extract & Index Video Features with Latency Profiling")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Root dataset directory (containing videos/)")
    parser.add_argument("--videos_dir", type=str, default=None, help="Directory containing videos (defaults to dataset_dir/videos)")
    parser.add_argument("--latency_file", type=str, default="output/ingestion_latency.json", help="Path to save ingestion latency JSON report")
    parser.add_argument("--captioning_model", type=str, default=None, help="Visual captioning model (default: from config)")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing even if already indexed")
    args = parser.parse_args()

    v_dir = args.videos_dir
    if not v_dir:
        candidate_v_dir = os.path.join(args.dataset_dir, "videos")
        if os.path.exists(candidate_v_dir):
            v_dir = candidate_v_dir
        else:
            v_dir = args.dataset_dir

    ingestor = Stage1Ingestor(captioning_model=args.captioning_model)
    results = ingestor.process_dataset(v_dir, force_reindex=args.force_reindex, latency_file=args.latency_file)

    total_videos = len(results)
    successful_videos = sum(1 for res in results.values() if res['status'] == 'success')
    reused_videos = sum(1 for res in results.values() if res.get('reused', False))
    newly_indexed_videos = successful_videos - reused_videos
    failed_videos = total_videos - successful_videos

    print("\n" + "=" * 50)
    print("      STAGE 1 INGESTION COMPLETE                 ")
    print("=" * 50)
    print(f"Total Videos Discovered    : {total_videos}")
    print(f"Already Indexed (Skipped)  : {reused_videos}")
    print(f"Newly Indexed (Processed)  : {newly_indexed_videos}")
    print(f"Failed                     : {failed_videos}")
    if args.latency_file:
        print(f"Latency JSON Report Saved  : {os.path.abspath(args.latency_file)}")
        map_f = os.path.splitext(os.path.abspath(args.latency_file))[0] + "_map.json"
        if os.path.exists(map_f):
            print(f"Latency Map Saved          : {map_f}")
    print("=" * 50)
    print("Stage 1 vector stores are ready! Next step:")
    print("Run: python main.py")
    print("=" * 50 + "\n")

