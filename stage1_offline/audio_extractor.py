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
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
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

    def transcribe_speech(self, audio_path: str):
        """Transcribes speech on GPU using torch.inference_mode() and returns timestamped segments."""
        print(f"Transcribing speech on GPU from {audio_path}")
        results = []
        with torch.inference_mode():
            if getattr(self, "whisper_mode", "faster_whisper") == "faster_whisper":
                segments, info = self.whisper_model.transcribe(audio_path, beam_size=1)
                for segment in segments:
                    results.append(
                        {
                            "type": "speech",
                            "start_time": segment.start,
                            "end_time": segment.end,
                            "text": segment.text.strip(),
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
                else:
                    y = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0

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

    def _merge_adjacent_sound_events(
        self, events: list, min_event_duration: float = MIN_EVENT_DURATION
    ):
        """Merges adjacent identical sound events and filters out events shorter than min_event_duration."""
        if not events:
            return []

        # Sort by start time
        events.sort(key=lambda x: x["start_time"])

        merged = []
        current = None

        for ev in events:
            if current is None:
                current = dict(ev)
            else:
                # Merge if it's the same sound event label/text and adjacent/overlapping
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
    ):
        """Detects sound events by sliding a window over the audio."""
        print(f"Detecting sound events in {audio_path}")

        # Load audio using standard wave module & numpy
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

        chunks_to_process = []
        current_time = 0.0
        total_chunks = 0

        # Extract audio chunks
        while current_time < duration:
            end_time = min(current_time + chunk_length_s, duration)
            if end_time - current_time < 0.5:  # Skip very short final chunks
                break

            total_chunks += 1
            start_sample = int(current_time * sr)
            end_sample = int(end_time * sr)
            chunk = y[start_sample:end_sample]

            # RMS energy threshold check to skip low-energy (silent) chunks
            rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0.0

            if rms >= rms_threshold:
                chunks_to_process.append(
                    {
                        "start_time": current_time,
                        "end_time": end_time,
                        "audio": chunk,
                    }
                )

            current_time += step_s

        num_valid_chunks = len(chunks_to_process)
        if total_chunks > num_valid_chunks:
            print(
                f"Skipped {total_chunks - num_valid_chunks}/{total_chunks} low-energy (silent) chunks using RMS threshold ({rms_threshold})."
            )

        raw_events = []

        # Batch CLAP inference
        for i in range(0, num_valid_chunks, batch_size):
            batch_items = chunks_to_process[i : i + batch_size]
            batch_audios = [item["audio"] for item in batch_items]

            chunk_idx_display = min(i + batch_size, num_valid_chunks)
            print(f"Processing audio chunk {chunk_idx_display}/{num_valid_chunks}")

            with torch.inference_mode():
                # HF pipeline batch inference
                batch_classifications = self.audio_classifier(
                    batch_audios, candidate_labels=self.sound_event_labels
                )

                # Ensure output structure is a list of results per chunk
                if (
                    len(batch_audios) == 1
                    and isinstance(batch_classifications, list)
                    and len(batch_classifications) > 0
                    and isinstance(batch_classifications[0], dict)
                ):
                    batch_classifications = [batch_classifications]

                for item, classifications in zip(
                    batch_items, batch_classifications
                ):
                    # Check top 2 candidate sound labels above threshold
                    for top_class in classifications[:2]:
                        if (
                            top_class["score"] > threshold
                            and "silence" not in top_class["label"].lower()
                        ):
                            raw_events.append(
                                {
                                    "type": "sound",
                                    "start_time": item["start_time"],
                                    "end_time": item["end_time"],
                                    "text": f"Sound of {top_class['label']}",
                                    "score": top_class["score"],
                                }
                            )

            # Free GPU memory after each inference batch across CUDA and XPU
            empty_gpu_cache()

        # Merge adjacent identical sound events
        merged_events = self._merge_adjacent_sound_events(
            raw_events, min_event_duration=min_event_duration
        )
        return merged_events

    def process_video(self, video_path: str):
        """Runs the full audio extraction pipeline."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            wav_path = temp_wav.name

        try:
            self.extract_audio_from_video(video_path, wav_path)

            speech_segments = self.transcribe_speech(wav_path)
            sound_events = self.detect_sound_events(wav_path)

            # Combine and sort by start time
            all_audio_facts = speech_segments + sound_events
            all_audio_facts.sort(key=lambda x: x["start_time"])

            return all_audio_facts
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


if __name__ == "__main__":
    # Simple test
    pass

