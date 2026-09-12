import argparse
import cv2
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from config import KA_DEFAULT, KV_DEFAULT, VECTOR_STORE_DIR
from ingestion import Stage1Ingestor, discover_videos, get_video_store_dir
from stage1_offline.vector_indexer import VectorIndexer
from stage2_online.question_classifier import QuestionClassifier
from stage2_online.decoupled_retriever import DecoupledRetriever
from stage2_online.deduplicator import Deduplicator
from stage2_online.reranker import ReRanker
from stage2_online.sufficiency_gate import SufficiencyGate
from stage3_generator.generator import Generator


def get_video_duration_sec(video_path: str) -> float:
    """Extracts duration of a video file in seconds using OpenCV."""
    if not video_path or not os.path.exists(video_path):
        return 0.0
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps and fps > 0 and total_frames > 0:
            return round(float(total_frames / fps), 6)
    except Exception as e:
        print(f"[Video Duration Warning] Could not read duration for {video_path}: {e}")
    return 0.0


def normalize_video_id_key(vid_str: str) -> str:
    """Strips paths, extensions, and redundant leading prefixes like v_, video_, v_v_."""
    s = Path(str(vid_str).replace('\\', '/').strip()).stem.lower()
    # Repeatedly strip leading prefixes
    changed = True
    while changed:
        changed = False
        for prefix in ["v_", "video_", "vid_"]:
            if s.startswith(prefix):
                s = s[len(prefix):]
                changed = True
    return s.strip("_- ")


def match_identifier_to_video_path(video_id_str: str, discovered_video_paths: List[str]) -> Optional[str]:
    """
    Robustly matches a video identifier from JSON to one of the discovered video paths.
    Supports matching by exact path, filename with extension, stem without extension,
    normalized prefix stripping (handles typos like 'v_v_...'), and substring/fuzzy matching.
    """
    if not video_id_str or not discovered_video_paths:
        return None

    clean_id = str(video_id_str).replace('\\', '/').strip()
    id_name = Path(clean_id).name.lower()
    id_stem = Path(clean_id).stem.lower()
    norm_id = normalize_video_id_key(clean_id)

    # 1. Exact absolute or relative path match
    for vpath in discovered_video_paths:
        norm_vpath = vpath.replace('\\', '/')
        if clean_id.lower() == norm_vpath.lower():
            return vpath

    # 2. Match by filename with extension (e.g. video1.mp4)
    for vpath in discovered_video_paths:
        norm_vpath = vpath.replace('\\', '/')
        if id_name == Path(norm_vpath).name.lower():
            return vpath

    # 3. Match by stem without extension (e.g. video1)
    for vpath in discovered_video_paths:
        norm_vpath = vpath.replace('\\', '/')
        if id_stem == Path(norm_vpath).stem.lower():
            return vpath

    # 4. Normalized key match (stripping v_, v_v_, video_ prefixes)
    if norm_id:
        for vpath in discovered_video_paths:
            v_stem = Path(vpath.replace('\\', '/')).stem
            if norm_id == normalize_video_id_key(v_stem):
                return vpath

    # 5. Path ends with identifier or contains normalized key
    for vpath in discovered_video_paths:
        norm_vpath = vpath.replace('\\', '/').lower()
        if norm_vpath.endswith(clean_id.lower()):
            return vpath
        if norm_id and norm_id in Path(norm_vpath).stem.lower():
            return vpath

    return None


def extract_questions_from_json(json_file_path: str, discovered_video_paths: List[str]) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Reads a JSON file, preserves its original data structure, and extracts a list of question entries.
    Each extracted item contains:
      - 'item_dict': reference to the dictionary representing the question entry
      - 'video_identifier': raw video ID from JSON
      - 'matched_video_path': matched absolute video path or None
      - 'question_text': extracted question string
      - 'ground_truth_answer': ground truth answer string
      - 'predicted_answer': initialized string
      - 'video_duration_sec': float initialized to 0.0
    """
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    extracted_items = []
    
    def process_dict_item(item_dict: dict, fallback_vid_id: Optional[str] = None):
        if not isinstance(item_dict, dict):
            return
        
        # Check candidate keys for video identifier
        vid_id = None
        for key in ["video_name", "video_id", "video", "video_path", "video_file", 
                    "video_filename", "filename", "file_name", "vid", "movie", "video_file_name"]:
            if key in item_dict and item_dict[key]:
                vid_id = str(item_dict[key])
                break
        
        if not vid_id and fallback_vid_id:
            vid_id = fallback_vid_id

        # Check candidate keys for question string
        q_text = None
        for key in ["question", "q", "query", "question_text", "text", "prompt"]:
            if key in item_dict and isinstance(item_dict[key], str):
                q_text = item_dict[key]
                break

        # Check candidate keys for ground truth answer
        gt_answer = ""
        for key in ["answer", "ground_truth_answer", "ground_truth", "gt_answer", "label", "target"]:
            if key in item_dict and item_dict[key] is not None:
                gt_answer = str(item_dict[key])
                break

        if q_text and vid_id:
            matched_vpath = match_identifier_to_video_path(vid_id, discovered_video_paths)
            extracted_items.append({
                "item_dict": item_dict,
                "video_identifier": vid_id,
                "matched_video_path": matched_vpath,
                "question_text": q_text,
                "ground_truth_answer": gt_answer,
                "predicted_answer": "",
                "video_duration_sec": 0.0
            })

    if isinstance(raw_data, list):
        for entry in raw_data:
            process_dict_item(entry)
    elif isinstance(raw_data, dict):
        # Check if root dict wraps a list under a common key
        found_wrapper = False
        for wrapper_key in ["questions", "data", "samples", "entries", "items"]:
            if wrapper_key in raw_data and isinstance(raw_data[wrapper_key], list):
                for entry in raw_data[wrapper_key]:
                    process_dict_item(entry)
                found_wrapper = True
                break

        if not found_wrapper:
            # Map of video_name -> list of questions or question dicts
            for key, val in raw_data.items():
                if isinstance(val, list):
                    for entry in val:
                        process_dict_item(entry, fallback_vid_id=key)
                elif isinstance(val, dict):
                    process_dict_item(val, fallback_vid_id=key)

    return raw_data, extracted_items


def answer_question_for_video(
    indexer: VectorIndexer, 
    question: str, 
    shared_components: dict
) -> str:
    """
    Executes Stage 2 (Retrieval & Re-ranking) and Stage 3 (Generation) for a given question
    using the provided video's VectorIndexer.
    """
    qc = shared_components['qc']
    dedup = shared_components['dedup']
    reranker = shared_components['reranker']
    gate = shared_components['gate']
    generator = shared_components['generator']

    retriever = DecoupledRetriever(indexer=indexer)

    # Step 1: Modality Estimator
    beta_q = qc.estimate_beta(question)

    # Dynamic initial top-k allocation continuously scaled by beta_q
    total_k = KA_DEFAULT + KV_DEFAULT
    k_a = max(2, int(round(total_k * beta_q)))
    k_v = max(2, int(round(total_k * (1.0 - beta_q))))

    # Loopback mechanism
    max_loops = 1
    final_candidates = []
    for loop in range(max_loops + 1):
        audio_candidates, visual_candidates = retriever.retrieve(question, k_a=k_a, k_v=k_v)
        combined_candidates = audio_candidates + visual_candidates

        if not combined_candidates:
            final_candidates = []
            break

        scored_candidates = reranker.score_candidates(question, combined_candidates, beta_q)
        dedup_candidates = dedup.apply_nms(scored_candidates, score_key="final_score")

        top_k = KA_DEFAULT + KV_DEFAULT
        final_candidates = dedup_candidates[:top_k]

        is_sufficient = gate.check_sufficiency(beta_q, final_candidates)
        if is_sufficient:
            break
        elif loop < max_loops:
            print("[Stage 2] Loopback triggered! Doubling retrieval depths.")
            k_a *= 2
            k_v *= 2
        else:
            print("[Stage 2] Sufficiency check failed after maximum loopbacks.")

    if not final_candidates:
        fallback_vis = retriever.retrieve_visual(question, k=10)
        if fallback_vis:
            final_candidates = fallback_vis

    # Stage 3: Generation
    context = generator.format_context(final_candidates)
    answer = generator.generate_answer(question, context)
    return answer


def run_single_pipeline(
    video_path: str,
    question: str,
    force_reindex: bool = False,
    latency_file: Optional[str] = "output/ingestion_latency.json",
    llm_model: Optional[str] = None,
    captioning_model: Optional[str] = None,
    llm_backend: Optional[str] = None
):
    """Legacy single-video execution entry point."""
    abs_video_path = os.path.abspath(video_path)
    print(f"--- Starting Single EchoVision Pipeline ---")
    print(f"Video: {abs_video_path}")
    print(f"Question: {question}")

    ingestor = Stage1Ingestor(captioning_model=captioning_model)
    indexer, reused, err = ingestor.process_single_video(abs_video_path, force_reindex=force_reindex)
    if indexer is None:
        raise RuntimeError(f"Stage 1 failed for video {abs_video_path}: {err}")
    if latency_file:
        ingestor.save_latency_report(latency_file)

    shared_components = {
        'qc': QuestionClassifier(),
        'dedup': Deduplicator(),
        'reranker': ReRanker(),
        'gate': SufficiencyGate(),
        'generator': Generator(model_name=llm_model, backend=llm_backend)
    }

    ans = answer_question_for_video(indexer, question, shared_components)
    print("\n--- Final Answer ---")
    print(ans)
    print("--------------------")
    return ans


def load_stage1_indices(
    videos_dir: str,
    base_store_dir: Optional[str] = None,
    allow_auto_ingest: bool = False,
    force_reindex: bool = False,
    latency_file: Optional[str] = None,
    captioning_model: Optional[str] = None
) -> Dict[str, Dict]:
    """
    Scans videos_dir and loads existing Stage 1 vector stores from disk.
    Does NOT instantiate heavy Stage 1 feature extraction models unless allow_auto_ingest=True.
    """
    discovered_videos = discover_videos(videos_dir)
    total_videos = len(discovered_videos)
    print(f"\n==================================================")
    print(f"[Stage 1 Index Loader] Checking pre-indexed stores for {total_videos} video(s) in '{videos_dir}'...")
    print(f"==================================================")

    results = {}
    missing_videos = []

    for video_path in discovered_videos:
        store_dir = get_video_store_dir(video_path, base_store_dir)
        indexer = VectorIndexer(store_dir=store_dir)
        if indexer.is_indexed(video_path) and not force_reindex:
            print(f"[Stage 1 Index Loader] Loaded Stage 1 store for '{os.path.basename(video_path)}'.")
            results[video_path] = {
                'indexer': indexer,
                'store_dir': store_dir,
                'status': 'success',
                'reused': True,
                'error': None
            }
        else:
            missing_videos.append(video_path)
            results[video_path] = {
                'indexer': None,
                'store_dir': store_dir,
                'status': 'failed',
                'reused': False,
                'error': 'Stage 1 index not found. Run python ingestion.py first.'
            }

    if missing_videos:
        if allow_auto_ingest:
            print(f"\n[Stage 1 Index Loader] Missing Stage 1 indices for {len(missing_videos)} video(s). Running Stage 1 Ingestion automatically...")
            ingestor = Stage1Ingestor(base_store_dir=base_store_dir, captioning_model=captioning_model)
            for m_vpath in missing_videos:
                idxer, reused, err = ingestor.process_single_video(m_vpath, force_reindex=force_reindex)
                m_store_dir = get_video_store_dir(m_vpath, base_store_dir)
                if idxer is not None:
                    results[m_vpath] = {
                        'indexer': idxer,
                        'store_dir': m_store_dir,
                        'status': 'success',
                        'reused': reused,
                        'error': None
                    }
                else:
                    results[m_vpath] = {
                        'indexer': None,
                        'store_dir': m_store_dir,
                        'status': 'failed',
                        'reused': False,
                        'error': err
                    }
            if latency_file:
                ingestor.save_latency_report(latency_file)
        else:
            print(f"\n[Stage 1 Index Loader WARNING] Stage 1 index is missing for {len(missing_videos)} video(s):")
            for m_vpath in missing_videos:
                print(f"  - {os.path.basename(m_vpath)}")
            print("To build vector stores for these videos, please run: python ingestion.py\n")

    return results


def process_dataset_pipeline(
    videos_dir: str = "dataset/videos",
    json_dir: str = "dataset/json",
    output_dir: str = "output",
    force_reindex: bool = False,
    auto_ingest: bool = False,
    latency_file: Optional[str] = None,
    llm_model: Optional[str] = None,
    captioning_model: Optional[str] = None,
    llm_backend: Optional[str] = None
):
    """
    Main batch processing workflow for EchoVision:
    1. Loads pre-built Stage 1 vector indices from disk (created by running `python ingestion.py`).
    2. Reads all JSON files from json_dir.
    3. Matches & associates questions by video and calculates video durations.
    4. Runs Stage 2 (Retrieval & Re-ranking) and Stage 3 (Generation).
    5. Saves formatted output JSON files containing video_id, question, ground_truth_answer, predicted_answer, and video_duration_sec.
    6. Prints comprehensive summary report.
    """
    print("\n==================================================")
    print("      ECHOVISION BATCH DATASET PROCESSING         ")
    print("==================================================")
    print(f"Videos Directory : {os.path.abspath(videos_dir)}")
    print(f"JSON Directory   : {os.path.abspath(json_dir)}")
    print(f"Output Directory : {os.path.abspath(output_dir)}")
    print("==================================================\n")

    lat_report_path = latency_file if latency_file else os.path.join(output_dir, "ingestion_latency.json")

    # Step 1: Load pre-built Stage 1 stores (created via `python ingestion.py`)
    stage1_results = load_stage1_indices(
        videos_dir=videos_dir,
        allow_auto_ingest=auto_ingest,
        force_reindex=force_reindex,
        latency_file=lat_report_path,
        captioning_model=captioning_model
    )

    discovered_videos = list(stage1_results.keys())
    total_videos = len(discovered_videos)
    successful_videos = sum(1 for res in stage1_results.values() if res['status'] == 'success')
    failed_videos = total_videos - successful_videos

    # Pre-calculate durations for all discovered videos
    video_durations = {vpath: get_video_duration_sec(vpath) for vpath in discovered_videos}

    # Step 2: Read all JSON files from json_dir
    json_path_obj = Path(json_dir)
    json_files = []
    if json_path_obj.exists():
        if json_path_obj.is_file():
            json_files.append(str(json_path_obj))
        else:
            for root, _, files in os.walk(json_path_obj):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in [".json", ",json"]:
                        json_files.append(os.path.abspath(os.path.join(root, f)))
    json_files.sort()

    if not json_files:
        print(f"[Dataset Pipeline Warning] No JSON files found in '{json_dir}'.")

    # Step 3: Parse JSON files & group questions by matched video path
    all_json_tasks = []  # List of dicts: {'file_path', 'raw_data', 'questions': [...]}
    video_to_questions_map: Dict[str, List[Dict[str, Any]]] = {}

    total_questions = 0

    for jf in json_files:
        raw_data, extracted_q_list = extract_questions_from_json(jf, discovered_videos)
        all_json_tasks.append({
            "file_path": jf,
            "raw_data": raw_data,
            "questions": extracted_q_list
        })

        for q_info in extracted_q_list:
            total_questions += 1
            matched_v = q_info["matched_video_path"]
            if matched_v:
                q_info["video_duration_sec"] = video_durations.get(matched_v, 0.0)
                if matched_v not in video_to_questions_map:
                    video_to_questions_map[matched_v] = []
                video_to_questions_map[matched_v].append(q_info)
            else:
                print(f"[Dataset Pipeline WARNING] Question '{q_info['question_text']}' in file '{os.path.basename(jf)}' could not be matched to any video (identifier: '{q_info['video_identifier']}').")

    # Step 4: Initialize Stage 2 & 3 Shared Models for Question Answering
    print("\n[QA Pipeline] Pre-loading Stage 2 & Stage 3 models for batch inference...")
    shared_components = {
        'qc': QuestionClassifier(),
        'dedup': Deduplicator(),
        'reranker': ReRanker(),
        'gate': SufficiencyGate(),
        'generator': Generator(model_name=llm_model, backend=llm_backend)
    }

    successful_questions = 0
    failed_questions = 0

    # Step 5: Process questions grouped by video (Stage 1 runs ONCE per video)
    print(f"\n==================================================")
    print(f"[QA Pipeline] Processing Questions for {len(video_to_questions_map)} matched video(s)...")
    print(f"==================================================")

    for vid_idx, (vpath, q_list) in enumerate(video_to_questions_map.items(), start=1):
        v_name = os.path.basename(vpath)
        v_stage1_info = stage1_results.get(vpath)

        print(f"\n--------------------------------------------------")
        print(f"[Overall Progress: Video {vid_idx}/{total_videos}] Current Video: {v_name}")
        print(f"Video Path                     : {vpath}")
        print(f"Number of associated questions : {len(q_list)}")

        if not v_stage1_info or v_stage1_info['status'] != 'success' or v_stage1_info['indexer'] is None:
            err_reason = v_stage1_info['error'] if v_stage1_info else "Stage 1 index missing"
            print(f"[QA Pipeline ERROR] Skipping questions for video '{v_name}' because Stage 1 store is unavailable: {err_reason}")
            for q_info in q_list:
                err_msg = f"Error: Stage 1 index missing ({err_reason})"
                q_info['predicted_answer'] = err_msg
                q_info['item_dict']['generated_answer'] = err_msg
                failed_questions += 1
            continue

        video_indexer = v_stage1_info['indexer']
        print(f"[QA Pipeline] Stage 1 index ready. Reusing index from '{v_stage1_info['store_dir']}' for all {len(q_list)} question(s).")

        for q_idx, q_info in enumerate(q_list, start=1):
            q_text = q_info['question_text']
            print(f"\n -> Video {vid_idx}/{total_videos} | Question {q_idx}/{len(q_list)}: '{q_text}'")
            try:
                answer = answer_question_for_video(video_indexer, q_text, shared_components)
                q_info['predicted_answer'] = answer
                q_info['item_dict']['generated_answer'] = answer
                successful_questions += 1
                print(f" -> Generated Answer: {answer}")
            except Exception as e:
                err_str = f"Error generating answer: {str(e)}"
                print(f" -> [QA Pipeline ERROR] {err_str}")
                print(traceback.format_exc())
                q_info['predicted_answer'] = f"Error: {str(e)}"
                q_info['item_dict']['generated_answer'] = f"Error: {str(e)}"
                failed_questions += 1

    # Handle unmatched questions
    for task in all_json_tasks:
        for q_info in task["questions"]:
            if not q_info["matched_video_path"]:
                q_info['predicted_answer'] = f"Error: Could not match video identifier '{q_info['video_identifier']}' to any discovered video file."
                failed_questions += 1

    # Step 6: Save output JSON files in output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n==================================================")
    print(f"[Output Saver] Writing output JSON files to '{os.path.abspath(output_dir)}'...")
    print(f"==================================================")

    for task in all_json_tasks:
        orig_file = task["file_path"]
        rel_name = os.path.basename(orig_file)
        if rel_name.lower().endswith(",json"):
            rel_name = rel_name[:-5] + ".json"
        out_file_path = os.path.join(output_dir, rel_name)

        output_list = []
        for q_info in task["questions"]:
            output_list.append({
                "video_id": q_info["video_identifier"],
                "question": q_info["question_text"],
                "ground_truth_answer": q_info["ground_truth_answer"],
                "predicted_answer": q_info["predicted_answer"],
                "video_duration_sec": q_info["video_duration_sec"]
            })

        with open(out_file_path, "w", encoding="utf-8") as out_f:
            json.dump(output_list, out_f, indent=4, ensure_ascii=False)

        print(f"Saved: {out_file_path}")

    # Step 7: Print Final Summary
    print("\n" + "=" * 50)
    print("           ECHOVISION PIPELINE SUMMARY            ")
    print("=" * 50)
    print(f"Total Number of Videos          : {total_videos}")
    print(f"Successfully Processed Videos  : {successful_videos}")
    print(f"Failed Videos                   : {failed_videos}")
    print("-" * 50)
    print(f"Total Number of Questions       : {total_questions}")
    print(f"Successfully Answered Questions : {successful_questions}")
    print(f"Failed Questions                : {failed_questions}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EchoVision: Decoupled Audio-Visual RAG Pipeline & Batch Video QA")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory containing videos/ and json/")
    parser.add_argument("--videos_dir", type=str, default=None, help="Path to videos directory (defaults to dataset_dir/videos)")
    parser.add_argument("--json_dir", type=str, default=None, help="Path to json directory (defaults to dataset_dir/json)")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to save output JSON files")
    parser.add_argument("--video", type=str, default=None, help="Path to a single video file (for legacy single-video mode)")
    parser.add_argument("--question", type=str, default=None, help="Question for single video mode")
    parser.add_argument("--llm_model", type=str, default=None, help="LLM generator model or HF ID (default: from config, qwen2.5-v1-72b-instruct)")
    parser.add_argument("--captioning_model", type=str, default=None, help="Visual captioning model or HF ID (default: from config, Salesforce/blip-image-captioning-large)")
    parser.add_argument("--llm_backend", type=str, default=None, choices=["auto", "api", "serverless", "local"], help="LLM inference backend ('auto', 'api', or 'local')")
    parser.add_argument("--force-reindex", action="store_true", help="Force re-indexing even if Stage 1 output exists")
    parser.add_argument("--auto-ingest", action="store_true", help="Automatically run Stage 1 ingestion if index is missing")
    parser.add_argument("--latency_file", type=str, default=None, help="Path to save ingestion latency JSON report (defaults to <output_dir>/ingestion_latency.json)")

    args = parser.parse_args()

    # Legacy single-video mode if --video and --question are provided
    if args.video and args.question:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        lat_f = args.latency_file if args.latency_file else os.path.join(args.output_dir, "ingestion_latency.json")
        run_single_pipeline(
            args.video,
            args.question,
            force_reindex=args.force_reindex,
            latency_file=lat_f,
            llm_model=args.llm_model,
            captioning_model=args.captioning_model,
            llm_backend=args.llm_backend
        )
    else:
        # Default batch dataset mode
        v_dir = args.videos_dir if args.videos_dir else os.path.join(args.dataset_dir, "videos")
        j_dir = args.json_dir if args.json_dir else os.path.join(args.dataset_dir, "json")
        process_dataset_pipeline(
            videos_dir=v_dir,
            json_dir=j_dir,
            output_dir=args.output_dir,
            force_reindex=args.force_reindex,
            auto_ingest=args.auto_ingest,
            latency_file=args.latency_file,
            llm_model=args.llm_model,
            captioning_model=args.captioning_model,
            llm_backend=args.llm_backend
        )

