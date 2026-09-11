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
    CLAP_MODEL,
    AUDIO_CHUNK_LENGTH,
    AUDIO_OVERLAP,
    AUDIO_THRESHOLD,
    AUDIO_BATCH_SIZE,
    MIN_EVENT_DURATION,
    AUDIO_RMS_THRESHOLD,
    STEREO_BALANCE_THRESHOLD,
    empty_gpu_cache,
)


class AudioExtractor:
    def __init__(self):
        print(f"Loading Whisper model: {WHISPER_MODEL}")
        whisper_dev = "cuda" if IS_CUDA_AVAILABLE else "cpu"
        whisper_compute = "float16" if IS_CUDA_AVAILABLE else "int8"
        self.whisper_model = WhisperModel(
            WHISPER_MODEL,
            device=whisper_dev,
            compute_type=whisper_compute,
        )

        print(f"Loading Audio Classification model (Zero-Shot CLAP): {CLAP_MODEL}")
        # Use CUDA if available; otherwise run CLAP on CPU to avoid Level Zero / XPU multi-threading
        # contention with visual extraction and eliminate GPU VRAM overhead.
        clap_dev = 0 if IS_CUDA_AVAILABLE else -1
        print(f"CLAP Audio Classifier Device: {'cuda:0' if clap_dev == 0 else 'cpu'}")
        self.audio_classifier = pipeline(
            task="zero-shot-audio-classification",
            model=CLAP_MODEL,
            device=clap_dev,
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
        """Extract the audio track from a video file as a 16kHz mono .wav file."""
        print(f"Extracting mono audio from {video_path} to {output_wav_path}")
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
            "1",  # Mono audio required by the requested pipeline
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
        """Transcribes speech using faster-whisper with torch.inference_mode() and returns timestamped segments."""
        print(f"Transcribing speech from {audio_path}")
        with torch.inference_mode():
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=1)

            results = []
            for segment in segments:
                results.append(
                    {
                        "type": "speech",
                        "start_time": segment.start,
                        "end_time": segment.end,
                        "text": segment.text.strip(),
                    }
                )
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

    def _remove_speech_from_segment(
        self,
        segment_audio,
        segment_start,
        segment_end,
        speech_segments,
        sample_rate,
    ):
        """Mute portions of a fixed audio segment that contain speech."""
        cleaned = np.array(segment_audio, dtype=np.float32, copy=True)

        for speech in speech_segments:
            speech_start = float(speech["start_time"])
            speech_end = float(speech["end_time"])

            overlap_start = max(segment_start, speech_start)
            overlap_end = min(segment_end, speech_end)

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

    def _classify_background_audio(
        self,
        audio_segment,
        start_time,
        end_time,
        threshold=AUDIO_THRESHOLD,
    ):
        """Run the existing CLAP model on speech-removed audio."""
        if len(audio_segment) == 0:
            return []

        try:
            with torch.inference_mode():
                classifications = self.audio_classifier(
                    audio_segment,
                    candidate_labels=self.sound_event_labels,
                )
        except Exception as clf_err:
            print(
                f"[AudioExtractor Warning] CLAP failed ({clf_err}). "
                "Retrying on CPU..."
            )
            try:
                if not hasattr(self, "_cpu_fallback_classifier") or                         self._cpu_fallback_classifier is None:
                    self._cpu_fallback_classifier = pipeline(
                        task="zero-shot-audio-classification",
                        model=CLAP_MODEL,
                        device=-1,
                    )

                classifications = self._cpu_fallback_classifier(
                    audio_segment,
                    candidate_labels=self.sound_event_labels,
                )
            except Exception as fb_err:
                print(
                    f"[AudioExtractor Error] CPU fallback failed ({fb_err}). "
                    "Skipping event."
                )
                return []

        events = []

        for top_class in classifications[:2]:
            label = top_class["label"]
            score = float(top_class["score"])

            if score >= threshold and "silence" not in label.lower():
                events.append(
                    {
                        "type": "sound",
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "text": f"Sound of {label}",
                        "score": score,
                    }
                )

        empty_gpu_cache()
        return events

    def process_audio_segments(
        self,
        audio_path: str,
        segment_duration_s: float = AUDIO_CHUNK_LENGTH,
        threshold: float = AUDIO_THRESHOLD,
    ):
        """Process audio using the requested fixed-segment pipeline.

        Flow:
            mono WAV
              -> fixed-duration segments
              -> Faster-Whisper
              -> mute detected speech
              -> existing CLAP classifier
              -> transcript + segments + events
        """
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw_bytes = wf.readframes(n_frames)

        if n_channels != 1:
            raise ValueError(
                f"Expected mono audio, but received {n_channels} channels."
            )

        if sampwidth == 2:
            y = np.frombuffer(raw_bytes, dtype=np.int16).astype(
                np.float32
            ) / 32768.0
        elif sampwidth == 4:
            y = np.frombuffer(raw_bytes, dtype=np.int32).astype(
                np.float32
            ) / 2147483648.0
        elif sampwidth == 1:
            y = (
                np.frombuffer(raw_bytes, dtype=np.uint8).astype(
                    np.float32
                ) - 128.0
            ) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        segment_samples = max(1, int(round(segment_duration_s * sr)))
        transcript = []
        segment_records = []
        events = []

        # ---------------------------------------------------------------
        # Split the complete mono audio into fixed-duration segments.
        # ---------------------------------------------------------------
        for start_sample in range(0, len(y), segment_samples):
            end_sample = min(
                start_sample + segment_samples,
                len(y),
            )

            segment_start = start_sample / float(sr)
            segment_end = end_sample / float(sr)
            segment_audio = y[start_sample:end_sample]

            if len(segment_audio) == 0:
                continue

            # -----------------------------------------------------------
            # Faster-Whisper: speech stream
            # -----------------------------------------------------------
            try:
                with torch.inference_mode():
                    whisper_segments, _ = self.whisper_model.transcribe(
                        segment_audio,
                        beam_size=1,
                    )
                    whisper_segments = list(whisper_segments)
            except Exception as whisper_err:
                print(
                    f"[AudioExtractor Warning] Whisper failed for "
                    f"{segment_start:.2f}-{segment_end:.2f}s: {whisper_err}"
                )
                whisper_segments = []

            segment_speech = []

            for speech in whisper_segments:
                speech_text = speech.text.strip()

                if not speech_text:
                    continue

                # Whisper timestamps are local to this fixed segment.
                # Convert them to the original video/audio timeline.
                speech_start = segment_start + float(speech.start)
                speech_end = min(
                    segment_end,
                    segment_start + float(speech.end),
                )

                item = {
                    "type": "speech",
                    "start_time": speech_start,
                    "end_time": speech_end,
                    "text": speech_text,
                }

                transcript.append(item)
                segment_speech.append(item)

            # -----------------------------------------------------------
            # Remove/mute detected speech before event classification.
            # -----------------------------------------------------------
            background_audio = self._remove_speech_from_segment(
                segment_audio,
                segment_start,
                segment_end,
                segment_speech,
                sr,
            )

            # -----------------------------------------------------------
            # Existing CLAP model: background audio event stream
            # -----------------------------------------------------------
            rms = (
                float(np.sqrt(np.mean(background_audio ** 2)))
                if len(background_audio)
                else 0.0
            )

            segment_events = []

            if rms >= AUDIO_RMS_THRESHOLD:
                segment_events = self._classify_background_audio(
                    background_audio,
                    segment_start,
                    segment_end,
                    threshold=threshold,
                )

            events.extend(segment_events)

            segment_records.append(
                {
                    "start_time": segment_start,
                    "end_time": segment_end,
                    "speech": segment_speech,
                    "speech_removed": bool(segment_speech),
                    "events": segment_events,
                }
            )

        # Remove repeated detections caused by adjacent fixed segments.
        events = self._merge_adjacent_sound_events(
            events,
            min_event_duration=MIN_EVENT_DURATION,
        )

        transcript.sort(
            key=lambda x: (x["start_time"], x["end_time"])
        )
        events.sort(
            key=lambda x: (x["start_time"], x["end_time"])
        )

        return {
            "transcript": transcript,
            "segments": segment_records,
            "events": events,
        }

    def process_video(self, video_path: str):
        """Run the complete requested audio pipeline.

        Video
          -> FFmpeg
          -> 16 kHz mono
          -> fixed-duration segments
          -> Faster-Whisper
          -> mute detected speech
          -> existing CLAP classifier
          -> transcript + segments + events
        """
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False,
        ) as temp_wav:
            wav_path = temp_wav.name

        try:
            self.extract_audio_from_video(video_path, wav_path)

            return self.process_audio_segments(
                wav_path,
                segment_duration_s=AUDIO_CHUNK_LENGTH,
                threshold=AUDIO_THRESHOLD,
            )
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)



if __name__ == "__main__":
    # Simple test
    pass

