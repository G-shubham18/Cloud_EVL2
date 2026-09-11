import os
import re
import json
from typing import Optional, List, Dict, Any
import torch
from transformers import AutoProcessor

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

from config import (
    QWEN_VL_MODEL,
    DEVICE,
    TORCH_DTYPE,
    GENERATOR_MODEL,
    GENERATOR_MAX_NEW_TOKENS,
    GENERATOR_DEVICE,
)


class Generator:
    """
    Stage 3: Grounded Answer Generator using native Qwen2.5-VL-3B-Instruct.
    Operates 100% locally in Python via PyTorch/Transformers with zero Ollama dependency.
    Shares model weights with VisualExtractor to eliminate duplicate RAM usage.
    """
    _shared_model = None
    _shared_processor = None
    _shared_device = None

    @classmethod
    def set_shared_model(cls, model, processor, device=None):
        """Allows VisualExtractor or main pipeline to share already-loaded VLM weights."""
        cls._shared_model = model
        cls._shared_processor = processor
        if device is not None:
            cls._shared_device = device

    @classmethod
    def unload_model(cls):
        """Frees cached model weights from memory if needed."""
        cls._shared_model = None
        cls._shared_processor = None
        cls._shared_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __init__(
        self,
        model=None,
        processor=None,
        device=None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        lazy_load: bool = False,
    ):
        self.device = device or GENERATOR_DEVICE
        self.max_tokens = max_tokens or GENERATOR_MAX_NEW_TOKENS
        self.model_name = model_name or GENERATOR_MODEL

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            Generator.set_shared_model(model, processor, self.device)
        elif Generator._shared_model is not None and Generator._shared_processor is not None:
            self.model = Generator._shared_model
            self.processor = Generator._shared_processor
        else:
            self.model = None
            self.processor = None
            if not lazy_load:
                self._ensure_model_loaded()

    def _ensure_model_loaded(self):
        """Ensures Qwen2.5-VL model and processor are initialized and cached."""
        if self.model is not None and self.processor is not None:
            return

        if Generator._shared_model is not None and Generator._shared_processor is not None:
            self.model = Generator._shared_model
            self.processor = Generator._shared_processor
            return

        print(f"[Generator] Loading native model: {self.model_name} on device={self.device}...")
        kwargs = {
            "torch_dtype": TORCH_DTYPE,
            "device_map": "auto" if self.device == "cuda" else None,
            "low_cpu_mem_usage": True,
        }

        loaded_model = None
        if "2.5" in self.model_name and Qwen2_5_VLForConditionalGeneration is not None:
            try:
                loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name, **kwargs
                )
            except Exception as e:
                print(f"[Generator Warning] Could not load via Qwen2_5_VLForConditionalGeneration: {e}")

        if loaded_model is None and AutoModelForImageTextToText is not None:
            try:
                loaded_model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, **kwargs
                )
            except Exception as e:
                print(f"[Generator Warning] Could not load via AutoModelForImageTextToText: {e}")

        if loaded_model is None and Qwen2VLForConditionalGeneration is not None:
            try:
                loaded_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    self.model_name, **kwargs
                )
            except Exception as e:
                print(f"[Generator Warning] Could not load via Qwen2VLForConditionalGeneration: {e}")

        if loaded_model is None:
            raise RuntimeError(f"Could not load generator model '{self.model_name}'.")

        if self.device != "cuda":
            loaded_model = loaded_model.to(self.device)

        loaded_model.eval()
        loaded_processor = AutoProcessor.from_pretrained(self.model_name)

        self.model = loaded_model
        self.processor = loaded_processor
        Generator.set_shared_model(self.model, self.processor, self.device)
        print(f"[Generator] Successfully loaded {self.model_name} on {self.device} (No Ollama required).")

    def format_context(self, candidates: list) -> str:
        """
        Formats the context from audio and visual candidates.
        """
        context_lines = []
        for cand in sorted(candidates, key=lambda x: x["metadata"]["start_time"]):
            md = cand["metadata"]
            start = md["start_time"]
            end = md["end_time"]
            fact_type = md["type"]
            text = cand["text"].strip()
            
            if fact_type == "visual":
                context_lines.append(f"[Visual - {start:.2f}s to {end:.2f}s] {text}")
            elif fact_type == "speech":
                context_lines.append(f"[Speech - {start:.2f}s to {end:.2f}s] {text}")
            elif fact_type == "sound":
                context_lines.append(f"[Sound - {start:.2f}s to {end:.2f}s] {text}")
                
        return "\n".join(context_lines)

    def clean_answer(self, raw_answer: str, question: str = "") -> str:
        """
        Strips conversational prefixes, filler explanations, markdown artifacts,
        and applies benchmark-aligned normalization.
        """
        if not raw_answer:
            return ""
        
        ans = raw_answer.strip().replace("**", "").replace("*", "").replace("`", "").replace('"', '').replace("'", "")
        
        # Split into lines and take the first informative line
        lines = [line.strip() for line in ans.split("\n") if line.strip()]
        if lines:
            ans = lines[0]
            
        # Strip common verbose lead-ins iteratively
        verbose_patterns = [
            r"^short answer\s*:\s*",
            r"^answer\s*:\s*",
            r"^based on (?:the )?(?:provided )?(?:visual |video |audio |speech )*(?:context|evidence|scenes?|timestamps?|descriptions?)[,\s:]*",
            r"^in the (?:provided )?(?:visual |video |audio )*(?:scenes?|video)[,\s:]*",
            r"^from the (?:provided )?(?:visual |video |audio )*(?:evidence|video)[,\s:]*",
            r"^the answer is\s*:\s*",
            r"^the answer is\s*",
            r"^it takes place in\s*(?:a|an)?\s*",
            r"^the performance is (?:located )?(?:in|at)\s*(?:a|an)?\s*",
            r"^the person (?:is|appears to be|was)?\s*(?:standing|sitting|positioned|seen)?\s*(?:in front of|behind|near|beside|next to|on|under)\s*",
            r"^there is (?:a|an)\s*",
            r"^it is (?:a|an)\s*",
            r"^[-•*]\s+",
            r"^\d+[\.\)]\s+",
            r"^[-•*0-9]+[.)]\s*",
        ]
        
        changed = True
        while changed:
            changed = False
            for pat in verbose_patterns:
                new_ans = re.sub(pat, "", ans, flags=re.IGNORECASE).strip()
                if new_ans != ans:
                    ans = new_ans
                    changed = True
            
        # Strip trailing periods for short answers
        if len(ans.split()) <= 8:
            ans = ans.rstrip(" .;,")

        q_lower = question.lower() if question else ""
        ans_lower = ans.lower().strip()

        # 1. Yes/No binary question routing
        yes_no_starters = ["is there", "is the", "are there", "are the", "did ", "does ", "was there", "were there", "can ", "could ", "would ", "given the"]
        if any(q_lower.startswith(starter) for starter in yes_no_starters):
            if "yes" in ans_lower:
                return "yes"
            elif "no" in ans_lower:
                return "no"

        # 2. Spatial channel localization (MusicAVQA: Which object / left or right)
        if "sound" in q_lower and ("which" in q_lower or "left" in q_lower or "right" in q_lower):
            if "left" in ans_lower and "right" not in ans_lower:
                return "left"
            elif "right" in ans_lower and "left" not in ans_lower:
                return "right"

        # 3. Environment normalizer ("outdoor setting" -> "outdoor")
        if "where is the performance" in q_lower or "where does the performance" in q_lower:
            if any(w in ans_lower for w in ["outdoor", "street", "park", "outside"]):
                return "outdoor"
            elif any(w in ans_lower for w in ["indoor", "inside", "hall", "room", "auditorium"]):
                return "indoor"

        # 4. Number word normalization for counting questions
        word_to_num = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        if q_lower.startswith("how many") or "count" in q_lower or "number of" in q_lower:
            # Extract first number or digit
            num_match = re.search(r'\b(\d+)\b', ans)
            if num_match:
                return num_match.group(1)
            for w, n in word_to_num.items():
                if re.search(rf'\b{w}\b', ans_lower):
                    return n

        return ans

    def generate_answer(
        self,
        question: str,
        context: str,
        image: Optional[Any] = None
    ) -> str:
        """
        Generates a direct, concise answer based on multi-modal evidence using native Qwen2.5-VL.
        No Ollama server is required.
        """
        self._ensure_model_loaded()

        system_prompt = (
            "You are an expert Video Question Answering model.\n"
            "Answer the question accurately, directly, and concisely using the provided video evidence "
            "(visual descriptions, temporal transitions, speech, and sound events).\n\n"
            "Task Instructions & Reasoning Rules:\n"
            "1. Short Direct Answer: Output ONLY a short, direct answer (typically 1 to 5 words, "
            "e.g. \"mirror\", \"shelf\", \"table\", \"tie up hair\", \"hair styling\", \"dance\", "
            "\"moved from right to left\", \"yes\", \"no\", \"1\", \"2\", \"outdoor\", \"left\", \"right\").\n"
            "2. Spatial & Spatiotemporal Questions:\n"
            "   - For positions (behind, in front of, under, on): identify the primary object or furniture "
            "directly adjacent to or behind the subject (e.g. table, shelf, cabinet, stage, mirror, sink, bread).\n"
            "   - For left vs right sound source questions (e.g. \"Which <Object> makes the sound?\"): "
            "check for \"[Audio Source: Left Side]\" or \"[Audio Source: Right Side]\" in the sound tags and output \"left\" or \"right\".\n"
            "   - For performance location: if outdoor/street/park, answer \"outdoor\"; if inside a building/room, "
            "answer the specific room (e.g. \"dining room\", \"church\", \"indoor\").\n"
            "3. Yes/No Verification Questions:\n"
            "   - For questions starting with \"Is there...\", \"Is the...\", \"Did...\", \"Does...\", \"Was...\", output strictly \"yes\" or \"no\".\n"
            "4. Temporal Sequence Questions (what happened before / after / when...):\n"
            "   - Track chronological transitions across timestamps to find the immediate previous or next action in the sequence.\n"
            "5. Counting Questions (how many people / chairs / instruments):\n"
            "   - Output ONLY the exact single number (e.g. \"1\", \"2\", \"3\", \"4\").\n"
            "   - Distinguish between frame-level visible count (e.g. \"visible in the scene / at 12s / on the table\") "
            "and video-level unique entity count (e.g. \"how many people in the video / unique people / in total\").\n"
            "   - For whole-video counting questions, use the Video-Level Unique Entity Counts evidence. "
            "Persistent entities appearing repeatedly across multiple frames are counted only once.\n"
            "   - For scene/frame-level counting questions, use the Frame-Level Counting from the corresponding timestamp.\n"
            "   - Never calculate video-level unique counts by summing frame counts across multiple frames.\n"
            "6. Formatting & Reliability:\n"
            "   - Do NOT write explanations, reasoning steps, conversational filler, or timestamps. Output ONLY the concise final answer.\n"
            "   - NEVER output \"Unknown\", \"I don't know\", or \"Unclear\". Always make a best-effort prediction using the most relevant visual/audio scene evidence."
        )

        evidence_str = context.strip() if context and context.strip() else "[No specific evidence retrieved; predict best probable answer from scene context]"
        user_prompt = (
            f"Evidence:\n{evidence_str}\n\n"
            f"Question: {question}\n"
            f"Short Answer:"
        )

        user_content = []
        if image is not None:
            user_content.append({
                "type": "image",
                "image": image,
                "min_pixels": 256 * 14 * 14,
                "max_pixels": 384 * 14 * 14,
            })
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content}
        ]

        try:
            prompt_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            if image is not None:
                from qwen_vl_utils import process_vision_info
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[prompt_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    text=[prompt_text],
                    images=None,
                    videos=None,
                    padding=True,
                    return_tensors="pt"
                )

            model_device = getattr(self.model, "device", self.device)
            inputs = inputs.to(model_device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=False,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                raw_ans = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

            cleaned = self.clean_answer(raw_ans, question=question)

            # Robust fallback if model still returned Unknown
            if cleaned.lower() in ["unknown", "unknown.", "n/a", "none", "i don't know", "unclear"]:
                fallback_prompt = (
                    f"Given the video evidence below, what is the single most likely object, entity, or action for the question?\n"
                    f"Evidence:\n{evidence_str}\n"
                    f"Question: {question}\n"
                    f"Output only the object name or action (1 to 3 words, e.g. mirror, shelf, table, styling hair):"
                )
                fb_messages = [
                    {"role": "user", "content": [{"type": "text", "text": fallback_prompt}]}
                ]
                fb_text = self.processor.apply_chat_template(
                    fb_messages, tokenize=False, add_generation_prompt=True
                )
                fb_inputs = self.processor(
                    text=[fb_text], images=None, videos=None, padding=True, return_tensors="pt"
                ).to(model_device)

                with torch.inference_mode():
                    fb_ids = self.model.generate(
                        **fb_inputs, max_new_tokens=15, do_sample=False
                    )
                    fb_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(fb_inputs.input_ids, fb_ids)
                    ]
                    fb_raw = self.processor.batch_decode(
                        fb_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0].strip()
                    fb_cleaned = self.clean_answer(fb_raw, question=question)
                    if fb_cleaned and fb_cleaned.lower() not in ["unknown", "unknown."]:
                        return fb_cleaned

            return cleaned
        except Exception as e:
            print(f"[Generator Error] Generation failed: {e}")
            return f"Error: Generation failed ({e})"
