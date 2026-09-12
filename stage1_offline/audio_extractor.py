import os
import subprocess
import tempfile
import wave
import numpy as np
import torch
from faster_whisper import WhisperModel
from transformers import pipeline
import shutil

from config import (
    WHISPER_MODEL,
    DEVICE,
    IS_CUDA_AVAILABLE,
    IS_GPU,
    TORCH_DTYPE,
    empty_gpu_cache,
    CLAP_MODEL,
    AUDIO_CHUNK_LENGTH,
    AUDIO_OVERLAP,
    AUDIO_THRESHOLD,
    AUDIO_BATCH_SIZE,
    MIN_EVENT_DURATION,
    AUDIO_RMS_THRESHOLD,
)


class AudioExtractor:
    def __init__(self):
        whisper_dev = "cuda" if IS_CUDA_AVAILABLE else "cpu"
        whisper_compute = "float16" if IS_CUDA_AVAILABLE else "int8"
        print(f"[AudioExtractor] Initializing Whisper model on device={whisper_dev} ({whisper_compute})...")

        try:
            from faster_whisper import WhisperModel
            self.whisper_mode = "faster_whisper"
            self.whisper_model = WhisperModel(
                WHISPER_MODEL,
                device=whisper_dev,
                compute_type=whisper_compute,
            )
        except Exception as e:
            print(f"[AudioExtractor Warning] faster-whisper on {whisper_dev} failed ({e}), falling back to HF Whisper pipeline...")
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self.whisper_mode = "hf_pipeline"
            hf_whisper_id = "distil-whisper/distil-large-v3"
            hf_dev = DEVICE if (IS_GPU and DEVICE != "cpu") else "cpu"
            hf_dtype = TORCH_DTYPE if (IS_GPU and DEVICE != "cpu") else torch.float32
            self.hf_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                hf_whisper_id,
                torch_dtype=hf_dtype,
                low_cpu_mem_usage=True,
            ).to(hf_dev)
            self.hf_processor = AutoProcessor.from_pretrained(hf_whisper_id)
            self.whisper_pipe = pipeline(
                "automatic-speech-recognition",
                model=self.hf_model,
                tokenizer=self.hf_processor.tokenizer,
                feature_extractor=self.hf_processor.feature_extractor,
                torch_dtype=hf_dtype,
                device=hf_dev,
            )

        clap_device = DEVICE if (IS_GPU and DEVICE != "cpu") else -1
        print(f"[AudioExtractor] Loading Audio Classification model (Zero-Shot CLAP) on device={clap_device}: {CLAP_MODEL}")
        self.audio_classifier = pipeline(
            task="zero-shot-audio-classification",
            model=CLAP_MODEL,
            device=clap_device,
        )

        # Define comprehensive sound event taxonomy to cover everyday, mechanical, musical, and contact acoustics
        self.sound_event_labels = [
            "violin or fiddle playing",
            "acoustic guitar strumming",
            "electric guitar",
            "piano or keyboard",
            "cello playing",
            "flute or recorder",
            "clarinet or woodwind",
            "trumpet or brass horn",
            "drums or percussion",
            "accordion",
            "musical instrument performance",
            "singing or vocal performance",
            "faint plastic click",
            "sharp plastic thud",
            "muffled wooden thud",
            "metallic scraping sound",
            "metallic squeak",
            "metallic clinking",
            "ceramic clinking",
            "clank or metal impact",
            "glass clinking or breaking",
            "footsteps or walking",
            "door opening",
            "door closing or slamming",
            "cabinet door opening or closing",
            "laptop lid opening or closing",
            "paper folding or crinkling",
            "running water",
            "water dripping or faucet squeak",
            "liquid pouring",
            "water splashing",
            "tattoo machine buzzing",
            "electric motor buzzing or humming",
            "hair dryer blowing air",
            "steady white noise from a fan",
            "appliances humming",
            "computer keyboard typing",
            "mouse clicking",
            "car engine",
            "siren or alarm",
            "phone ringing",
            "applause and clapping",
            "laughter and cheering",
            "intermittent voices or talking",
            "dialogue or conversation",
            "breathing or sighing",
            "whispering",
            "wind or background breeze",
            "rain or thunder",
            "silence or quiet room",
        ]

    @staticmethod
    def _is_working_ffmpeg(bin_path: str) -> bool:
        if not bin_path:
            return False
        try:
            res = subprocess.run(
                [bin_path, "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
            )
            return res.returncode == 0
        except Exception:
            return False

    def _get_ffmpeg_cmd(self):
        if hasattr(self, "_cached_ffmpeg") and self._cached_ffmpeg:
            return self._cached_ffmpeg

        # 1. Try imageio_ffmpeg (robust precompiled static binary)
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if self._is_working_ffmpeg(exe):
                self._cached_ffmpeg = exe
                return exe
        except Exception:
            pass

        # 2. Check local workspace directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_ffmpeg = os.path.join(base_dir, "ffmpeg.exe")
        if self._is_working_ffmpeg(local_ffmpeg):
            self._cached_ffmpeg = local_ffmpeg
            return local_ffmpeg

        # 3. Check system PATH
        ffmpeg_bin = shutil.which("ffmpeg")
        if self._is_working_ffmpeg(ffmpeg_bin):
            self._cached_ffmpeg = ffmpeg_bin
            return ffmpeg_bin

        # 4. Auto-install imageio-ffmpeg into active Python environment
        try:
            import sys

            print("Working FFmpeg executable not found. Auto-installing imageio-ffmpeg...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "imageio-ffmpeg"], check=True
            )
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if self._is_working_ffmpeg(exe):
                self._cached_ffmpeg = exe
                return exe
        except Exception as e:
            print(f"Warning: Failed to auto-install imageio-ffmpeg: {e}")

        # 5. Direct download of standalone static ffmpeg.exe for Windows
        try:
            import urllib.request
            import zipfile

            print("Downloading standalone FFmpeg binary for Windows...")
            url = "https://github.com/ffbinaries/ffbinaries-prebuilt/releases/download/v4.4.1/ffmpeg-4.4.1-win-64.zip"
            zip_path = local_ffmpeg + ".zip"
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(base_dir)
            if os.path.exists(zip_path):
                os.remove(zip_path)
            if self._is_working_ffmpeg(local_ffmpeg):
                self._cached_ffmpeg = local_ffmpeg
                return local_ffmpeg
        except Exception as e:
            print(f"Warning: Failed to download standalone FFmpeg: {e}")

        self._cached_ffmpeg = "ffmpeg"
        return "ffmpeg"

    def extract_audio_from_video(self, video_path: str, output_wav_path: str):
        """Extracts the audio track from a video file as a 16kHz mono .wav file."""
        print(f"Extracting audio from {video_path} to {output_wav_path}")
        ffmpeg_bin = self._get_ffmpeg_cmd()
        command = [
            ffmpeg_bin,
            "-y",
            "-i",
            video_path,
            "-vn",  # Disable video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM
            "-ar",
            "16000",  # 16kHz sampling rate
            "-ac",
            "1",  # 1 channel (Mono)
            output_wav_path,
        ]

        try:
            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() if e.stderr else str(e)
            raise RuntimeError(
                f"FFmpeg audio extraction failed for '{video_path}' using '{ffmpeg_bin}' (exit {e.returncode}): {err_msg}"
            ) from e
        return output_wav_path

    def transcribe_speech(self, audio_path: str) -> list:
        """Transcribes speech from an audio file using Whisper with torch.inference_mode()."""
        print(f"Transcribing speech from {audio_path}")
        results = []
        with torch.inference_mode():
            if getattr(self, "whisper_mode", "faster_whisper") == "faster_whisper":
                segments, info = self.whisper_model.transcribe(audio_path, beam_size=1)
                for segment in segments:
                    txt = segment.text.strip()
                    if txt and txt.lower().strip(" .!?,") not in ["thank you", "thanks", "you", "subtitles by", "bye"]:
                        results.append(
                            {
                                "type": "speech",
                                "start_time": round(float(segment.start), 2),
                                "end_time": round(float(segment.end), 2),
                                "text": txt,
                            }
                        )
            else:
                with wave.open(audio_path, "rb") as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    sampwidth = wf.getsampwidth()
                    n_channels = wf.getnchannels()
                    raw_bytes = wf.readframes(n_frames)

                if sampwidth == 2:
                    y = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    y = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
                elif sampwidth == 1:
                    y = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
                else:
                    raise ValueError(f"Unsupported sample width: {sampwidth}")

                if n_channels > 1:
                    y = y.reshape(-1, n_channels).mean(axis=1)

                duration = len(y) / float(sr) if sr > 0 else 0.0
                chunk_sec = 10.0
                t = 0.0
                while t < duration:
                    t_end = min(t + chunk_sec, duration)
                    chunk_samples = y[int(t * sr) : int(t_end * sr)]
                    if len(chunk_samples) > 0 and np.sqrt(np.mean(chunk_samples**2)) >= 0.005:
                        chunk_out = self.whisper_pipe(chunk_samples, return_timestamps=False)
                        txt = chunk_out.get("text", "").strip()
                        if txt and txt.lower().strip(" .!?,") not in ["thank you", "thanks", "you", "subtitles by", "bye"]:
                            results.append(
                                {
                                    "type": "speech",
                                    "start_time": round(t, 2),
                                    "end_time": round(t_end, 2),
                                    "text": txt,
                                }
                            )
                    t += chunk_sec
        empty_gpu_cache()
        return results

    def _transcribe_segment(
        self,
        segment_audio: np.ndarray,
        segment_start: float,
        segment_end: float,
    ) -> list:
        """Step E: Transcribes speech within a 3s segment using Faster-Whisper."""
        if len(segment_audio) == 0:
            return []

        results = []
        try:
            with torch.inference_mode():
                if getattr(self, "whisper_mode", "faster_whisper") == "faster_whisper":
                    segments, _ = self.whisper_model.transcribe(segment_audio, beam_size=1)
                    for seg in segments:
                        txt = seg.text.strip()
                        if txt and txt.lower().strip(" .!?,") not in ["thank you", "thanks", "you", "subtitles by", "bye"]:
                            sp_start = round(segment_start + float(seg.start), 2)
                            sp_end = min(round(segment_end, 2), round(segment_start + float(seg.end), 2))
                            results.append({
                                "type": "speech",
                                "start_time": sp_start,
                                "end_time": sp_end,
                                "text": txt,
                            })
                else:
                    chunk_out = self.whisper_pipe(segment_audio, return_timestamps=False)
                    txt = chunk_out.get("text", "").strip()
                    if txt and txt.lower().strip(" .!?,") not in ["thank you", "thanks", "you", "subtitles by", "bye"]:
                        results.append({
                            "type": "speech",
                            "start_time": round(segment_start, 2),
                            "end_time": round(segment_end, 2),
                            "text": txt,
                        })
        except Exception:
            pass

        return results

    def _remove_speech_from_segment(
        self,
        segment_audio: np.ndarray,
        segment_start: float,
        segment_end: float,
        speech_segments: list,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Step G: Mute/zero-out portions of an audio segment that contain detected speech."""
        if not speech_segments or len(segment_audio) == 0:
            return segment_audio

        cleaned = np.array(segment_audio, dtype=np.float32, copy=True)
        for speech in speech_segments:
            sp_start = float(speech["start_time"])
            sp_end = float(speech["end_time"])

            overlap_start = max(segment_start, sp_start)
            overlap_end = min(segment_end, sp_end)

            if overlap_start < overlap_end:
                local_start = max(
                    0, int(round((overlap_start - segment_start) * sample_rate))
                )
                local_end = min(
                    len(cleaned),
                    int(round((overlap_end - segment_start) * sample_rate))
                )
                if local_start < local_end:
                    cleaned[local_start:local_end] = 0.0

        return cleaned

    def _merge_overlapping_speech(self, speech_facts: list) -> list:
        """Step K (helper): Merges and deduplicates overlapping speech facts from sliding windows."""
        if not speech_facts:
            return []

        speech_facts.sort(key=lambda x: (x["start_time"], x["end_time"]))
        merged = []

        for sp in speech_facts:
            if not merged:
                merged.append(dict(sp))
                continue

            prev = merged[-1]
            prev_txt = prev["text"].lower().strip()
            curr_txt = sp["text"].lower().strip()

            # Check temporal overlap between adjacent window detections
            if sp["start_time"] < prev["end_time"] or abs(sp["start_time"] - prev["end_time"]) < 0.3:
                if curr_txt == prev_txt or curr_txt in prev_txt:
                    prev["end_time"] = max(prev["end_time"], sp["end_time"])
                    continue
                elif prev_txt in curr_txt:
                    prev["text"] = sp["text"]
                    prev["end_time"] = max(prev["end_time"], sp["end_time"])
                    continue

            merged.append(dict(sp))

        return merged

    def _merge_adjacent_sound_events(
        self, events: list, min_event_duration: float = MIN_EVENT_DURATION
    ):
        """Step I (helper): Merges identical sound events and filters out events shorter than min_event_duration."""
        if not events:
            return []

        events.sort(key=lambda x: x["start_time"])
        merged = []
        current = None

        for ev in events:
            if current is None:
                current = dict(ev)
            else:
                if (
                    ev["text"] == current["text"]
                    and ev["start_time"] <= current["end_time"]
                ):
                    current["end_time"] = max(current["end_time"], ev["end_time"])
                    current["score"] = max(current["score"], ev["score"])
                else:
                    if (
                        current["end_time"] - current["start_time"]
                    ) >= min_event_duration:
                        merged.append(current)
                    current = dict(ev)

        if current is not None:
            if (current["end_time"] - current["start_time"]) >= min_event_duration:
                merged.append(current)

        return merged

    def detect_sound_events(
        self,
        audio_path: str,
        chunk_length_s: float = AUDIO_CHUNK_LENGTH,
        overlap_s: float = AUDIO_OVERLAP,
        threshold: float = AUDIO_THRESHOLD,
        batch_size: int = AUDIO_BATCH_SIZE,
        min_event_duration: float = MIN_EVENT_DURATION,
        rms_threshold: float = AUDIO_RMS_THRESHOLD,
    ) -> list:
        """Standalone helper: runs CLAP background sound detection over sliding audio windows."""
        all_facts = self.process_audio(
            audio_path=audio_path,
            chunk_length_s=chunk_length_s,
            overlap_s=overlap_s,
            threshold=threshold,
            batch_size=batch_size,
            min_event_duration=min_event_duration,
            rms_threshold=rms_threshold,
        )
        return [f for f in all_facts if f.get("type") == "sound"]

    def process_audio(
        self,
        audio_path: str,
        chunk_length_s: float = AUDIO_CHUNK_LENGTH,
        overlap_s: float = AUDIO_OVERLAP,
        threshold: float = AUDIO_THRESHOLD,
        batch_size: int = AUDIO_BATCH_SIZE,
        min_event_duration: float = MIN_EVENT_DURATION,
        rms_threshold: float = AUDIO_RMS_THRESHOLD,
    ) -> list:
        """
        Executes the exact requested Audio Pipeline:
          B: 16 kHz Mono Audio Input
          C: Segment Audio (3s Window + 1s Overlap)
          D: Non-Silent Segment Check (RMS >= threshold; Skip if silent)
          E: Faster-Whisper Speech Detection + Timestamps
          F: Speech Facts (Text + Start/End Time)
          G: Remove Speech Intervals (Zero-out speech in segment)
          H: LAION-CLAP Background Sound Detection on speech-removed audio
          I: Sound Confidence + Minimum Duration Check
          J: Sound Facts (Label + Start/End Time)
          K: Merge + Sort All Audio Facts
          L: Final Audio Timeline
        """
        print(f"[AudioExtractor] Processing audio pipeline on: {audio_path}")

        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            y = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            y = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sampwidth == 1:
            y = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        if n_channels > 1:
            y = y.reshape(-1, n_channels).mean(axis=1)

        duration = float(len(y)) / float(sr) if sr > 0 else 0.0
        step_s = chunk_length_s - overlap_s

        speech_facts = []
        bg_chunks_to_classify = []
        total_segments = 0
        skipped_silent = 0

        # Steps C, D, E, F, G: Segment, check silence, transcribe speech, mute speech intervals
        current_time = 0.0
        while current_time < duration:
            end_time = min(current_time + chunk_length_s, duration)
            if end_time - current_time < 0.5:
                break

            total_segments += 1
            start_sample = int(current_time * sr)
            end_sample = int(end_time * sr)
            chunk = y[start_sample:end_sample]

            # Step D: Non-Silent Segment?
            rms = float(np.sqrt(np.mean(chunk**2))) if len(chunk) > 0 else 0.0
            if rms < rms_threshold:
                # Step D -- No --> Skip
                skipped_silent += 1
                current_time += step_s
                continue

            # Step D -- Yes --> Step E: Faster-Whisper Speech Detection + Timestamps
            seg_speech = self._transcribe_segment(chunk, current_time, end_time)

            # Step F: Speech Facts (Text + Start/End Time)
            for sp in seg_speech:
                speech_facts.append(sp)

            # Step G: Remove Speech Intervals
            bg_audio = self._remove_speech_from_segment(
                chunk, current_time, end_time, seg_speech, sr
            )

            # Check residual background audio energy for Step H
            bg_rms = float(np.sqrt(np.mean(bg_audio**2))) if len(bg_audio) > 0 else 0.0
            if bg_rms >= rms_threshold:
                bg_chunks_to_classify.append(
                    {
                        "start_time": current_time,
                        "end_time": end_time,
                        "audio": bg_audio,
                    }
                )

            current_time += step_s

        if skipped_silent > 0:
            print(f"[AudioExtractor] Skipped {skipped_silent}/{total_segments} silent segments (RMS < {rms_threshold}).")

        # Step H: LAION-CLAP Background Sound Detection (Batched)
        raw_sound_events = []
        num_bg_chunks = len(bg_chunks_to_classify)

        for i in range(0, num_bg_chunks, batch_size):
            batch_items = bg_chunks_to_classify[i : i + batch_size]
            batch_audios = [item["audio"] for item in batch_items]

            chunk_idx_display = min(i + batch_size, num_bg_chunks)
            print(f"[AudioExtractor] Classifying background sound for segment {chunk_idx_display}/{num_bg_chunks}")

            with torch.inference_mode():
                batch_classifications = self.audio_classifier(
                    batch_audios, candidate_labels=self.sound_event_labels
                )

                if (
                    len(batch_audios) == 1
                    and isinstance(batch_classifications, list)
                    and len(batch_classifications) > 0
                    and isinstance(batch_classifications[0], dict)
                ):
                    batch_classifications = [batch_classifications]

                for item, classifications in zip(batch_items, batch_classifications):
                    # Step I: Sound Confidence Check
                    for top_class in classifications[:2]:
                        if (
                            top_class["score"] >= threshold
                            and "silence" not in top_class["label"].lower()
                        ):
                            # Step J: Sound Facts (Label + Start/End Time)
                            raw_sound_events.append(
                                {
                                    "type": "sound",
                                    "start_time": item["start_time"],
                                    "end_time": item["end_time"],
                                    "text": f"Sound of {top_class['label']}",
                                    "label": top_class["label"],
                                    "score": top_class["score"],
                                }
                            )

            empty_gpu_cache()

        # Step I: Minimum Duration Check & Merge Adjacent Identical Sound Events
        merged_sound_events = self._merge_adjacent_sound_events(
            raw_sound_events, min_event_duration=min_event_duration
        )

        # Step K: Merge + Sort All Audio Facts
        merged_speech = self._merge_overlapping_speech(speech_facts)
        all_audio_facts = merged_speech + merged_sound_events
        all_audio_facts.sort(key=lambda x: (x["start_time"], x["end_time"]))

        # Step L: Final Audio Timeline
        print(f"[AudioExtractor] Extracted {len(merged_speech)} speech fact(s) and {len(merged_sound_events)} sound fact(s). Total: {len(all_audio_facts)}.")
        return all_audio_facts

    def process_audio_segments(self, *args, **kwargs):
        """Backward-compatible alias for process_audio."""
        return self.process_audio(*args, **kwargs)

    def process_video(self, video_path: str):
        """
        Step A: Input Video -> Step B: Extract Audio 16 kHz Mono -> Steps C-L: Process Audio Facts -> Step M: ChromaDB.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name

        try:
            self.extract_audio_from_video(video_path, wav_path)
            all_audio_facts = self.process_audio(wav_path)
            return all_audio_facts
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


if __name__ == "__main__":
    # Simple test
    pass
