"""
Stage 3: Grounded Answer Generator
Supports both Serverless/API-based and Local PyTorch/Transformers LLM Generation:
  1. qwen2.5-v1-72b-instruct (Default): Qwen2.5 72B Instruct via HF API or Local
  2. gemma-4-31b: Google Gemma 2 27B/31B Instruct
  3. phi-3.5-vision-instruct: Microsoft Phi-3.5 Vision Instruct
  (Also supports previous Qwen/Qwen2.5-VL-3B-Instruct or any valid HF repo ID)
"""

import os
import re
import json
import gc
from typing import Optional, List, Dict, Any, Union, Tuple
import torch
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
try:
    from transformers import AutoModelForImageTextToText
except ImportError:
    AutoModelForImageTextToText = None
try:
    from transformers import AutoModelForCausalLM
except ImportError:
    AutoModelForCausalLM = None
try:
    from transformers import Qwen2VLForConditionalGeneration
except ImportError:
    Qwen2VLForConditionalGeneration = None

try:
    from huggingface_hub import InferenceClient
    HAS_HF_CLIENT = True
except ImportError:
    InferenceClient = None
    HAS_HF_CLIENT = False

from config import (
    DEVICE,
    TORCH_DTYPE,
    IS_GPU,
    IS_CUDA_AVAILABLE,
    GENERATOR_MODEL,
    GENERATOR_BACKEND,
    GENERATOR_MAX_NEW_TOKENS,
    GENERATOR_DEVICE,
    resolve_generator_model,
    get_hf_token,
    has_hf_token,
    HF_TOKEN,
)


class Generator:
    """
    Stage 3: Grounded Answer Generator supporting configurable LLMs across
    both Serverless / API-based and local PyTorch inference modes.
    """
    _shared_model = None
    _shared_processor = None
    _shared_device = None

    @classmethod
    def set_shared_model(cls, model, processor, device=None):
        """Allows sharing already-loaded VLM weights across pipeline stages."""
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __init__(
        self,
        model=None,
        processor=None,
        device=None,
        max_tokens: Optional[int] = None,
        model_name: Optional[str] = None,
        backend: Optional[str] = None,
        hf_token: Optional[str] = None,
        lazy_load: bool = False,
    ):
        self.device = device or GENERATOR_DEVICE
        self.max_tokens = max_tokens or GENERATOR_MAX_NEW_TOKENS
        self.requested_name = model_name or GENERATOR_MODEL
        self.model_info = resolve_generator_model(self.requested_name)
        self.model_name = self.model_info["hf_id"]
        self.model_type = self.model_info["type"]

        self.backend = (backend or GENERATOR_BACKEND).lower().strip()
        self.hf_token = hf_token or get_hf_token()

        self.model = None
        self.processor = None
        self.tokenizer = None
        self.client = None

        token_status = "[CONFIGURED]" if bool(self.hf_token) else "[NOT SET]"
        print(f"[Generator] Configured model: '{self.requested_name}' -> HF ID: '{self.model_name}' (backend={self.backend}, HF Token={token_status})")

        # Determine effective backend
        self._effective_backend = self._determine_backend()

        if model is not None and processor is not None:
            self.model = model
            self.processor = processor
            Generator.set_shared_model(model, processor, self.device)
        elif Generator._shared_model is not None and Generator._shared_processor is not None:
            self.model = Generator._shared_model
            self.processor = Generator._shared_processor
        elif self._effective_backend == "api":
            self._init_api_client()
        elif not lazy_load:
            self._ensure_local_model_loaded()

    def _is_large_model(self) -> bool:
        """Returns True if the target model is typically too large for single GPU/CPU local inference (>=20B)."""
        name_lower = self.model_name.lower()
        large_indicators = ["72b", "31b", "27b", "70b", "405b"]
        return any(ind in name_lower for ind in large_indicators)

    def _determine_backend(self) -> str:
        """
        Selects between 'api' and 'local' based on configured backend and hardware capacity.
        """
        if self.backend in ["api", "serverless"]:
            return "api"
        if self.backend == "local":
            return "local"

        # 'auto' mode:
        # If the model is 72B, 31B, or 27B, local inference requires 30-140 GB VRAM.
        # Prefer Serverless API if HF_TOKEN is present or if running on standard single GPU / CPU.
        if self._is_large_model():
            if self.hf_token:
                print(f"[Generator] Large model detected ({self.model_name}). Routing to Hugging Face Serverless API.")
                return "api"
            else:
                # Without token, local attempt is tried but user is notified
                print(f"[Generator Notice] Model '{self.model_name}' is large (>=27B) and HF_TOKEN is not set. Attempting local execution (set HF_TOKEN for fast Serverless API).")
                return "local"
        return "local"

    def _init_api_client(self):
        """Initializes Hugging Face Inference API client."""
        if not HAS_HF_CLIENT:
            print("[Generator Warning] huggingface_hub is not installed. Falling back to local mode.")
            self._effective_backend = "local"
            self._ensure_local_model_loaded()
            return

        try:
            self.client = InferenceClient(token=self.hf_token)
            token_state = "authenticated" if self.hf_token else "unauthenticated"
            print(f"[Generator] HF Serverless InferenceClient initialized ({token_state}) for model '{self.model_name}'.")
        except Exception as e:
            print(f"[Generator Warning] Could not initialize InferenceClient: {e}. Falling back to local mode.")
            self._effective_backend = "local"
            self._ensure_local_model_loaded()

    def _ensure_local_model_loaded(self):
        """Ensures PyTorch / Transformers model and processor are initialized locally."""
        if self.model is not None:
            return

        if Generator._shared_model is not None and Generator._shared_processor is not None:
            self.model = Generator._shared_model
            self.processor = Generator._shared_processor
            return

        token_state = "[CONFIGURED]" if bool(self.hf_token) else "[NOT SET]"
        print(f"[Generator] Loading native local model: {self.model_name} on device={self.device} (HF Token={token_state})...")
        kwargs = {
            "low_cpu_mem_usage": True,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token

        if self.device == "cuda" and IS_CUDA_AVAILABLE:
            kwargs["torch_dtype"] = TORCH_DTYPE
            kwargs["device_map"] = "auto"

        token_arg = {"token": self.hf_token} if self.hf_token else {}
        loaded_model = None

        # 1. Try Qwen2.5-VL Conditional Generation
        if "qwen" in self.model_name.lower() and "vl" in self.model_name.lower() and Qwen2_5_VLForConditionalGeneration is not None:
            try:
                loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_name, **kwargs
                )
            except Exception as e:
                print(f"[Generator Warning] Qwen2_5_VLForConditionalGeneration failed: {e}")

        # 2. Try AutoModelForImageTextToText (Vision-Language models)
        if loaded_model is None and AutoModelForImageTextToText is not None:
            try:
                loaded_model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name, **kwargs
                )
            except Exception as e:
                print(f"[Generator Warning] AutoModelForImageTextToText failed: {e}")

        # 3. Try AutoModelForCausalLM (Text LLMs e.g. Gemma, Qwen-Text, Phi)
        if loaded_model is None and AutoModelForCausalLM is not None:
            try:
                causal_kwargs = dict(kwargs)
                if "phi" in self.model_name.lower():
                    causal_kwargs["trust_remote_code"] = True
                loaded_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, **causal_kwargs
                )
            except Exception as e:
                print(f"[Generator Warning] AutoModelForCausalLM failed: {e}")

        if loaded_model is None:
            # Check for fallback model
            fallback_id = self.model_info.get("fallback_hf_id")
            if fallback_id and fallback_id != self.model_name:
                print(f"[Generator Warning] Local load failed for '{self.model_name}'. Trying fallback '{fallback_id}'...")
                try:
                    self.model_name = fallback_id
                    if AutoModelForCausalLM is not None:
                        loaded_model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
                except Exception as fb_e:
                    print(f"[Generator Error] Fallback model '{fallback_id}' also failed: {fb_e}")

        if loaded_model is None:
            # If local load failed and we can use API, switch dynamically to API
            if HAS_HF_CLIENT:
                print(f"[Generator] Local loading failed. Dynamically routing to Hugging Face Serverless API for '{self.model_name}'...")
                self._effective_backend = "api"
                self._init_api_client()
                return
            raise RuntimeError(f"Could not load generator model '{self.model_name}' locally. Please check VRAM or set HF_TOKEN to use Serverless API.")

        if self.device != "cuda" or not IS_CUDA_AVAILABLE:
            loaded_model = loaded_model.to(self.device)
        loaded_model.eval()

        # Load appropriate processor or tokenizer
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, **token_arg)
        except Exception:
            self.processor = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **token_arg)
        except Exception:
            self.tokenizer = None

        self.model = loaded_model
        print(f"[Generator] Successfully loaded local model '{self.model_name}' on {self.device}.")

    def format_context(self, candidates: list) -> str:
        """
        Formats retrieved audio and visual candidates into a clean, chronological text context.
        """
        context_lines = []
        for cand in sorted(candidates, key=lambda x: x.get("metadata", {}).get("start_time", 0.0)):
            md = cand.get("metadata", {})
            start = md.get("start_time", 0.0)
            end = md.get("end_time", 0.0)
            fact_type = md.get("type", "visual").lower()
            text = cand.get("text", "").strip()

            if fact_type == "visual":
                context_lines.append(f"[Visual - {start:.2f}s to {end:.2f}s] {text}")
            elif fact_type in ["speech", "dialogue"]:
                context_lines.append(f"[Speech - {start:.2f}s to {end:.2f}s] {text}")
            elif fact_type in ["sound", "audio"]:
                context_lines.append(f"[Sound - {start:.2f}s to {end:.2f}s] {text}")
            else:
                context_lines.append(f"[{fact_type.capitalize()} - {start:.2f}s to {end:.2f}s] {text}")

        return "\n".join(context_lines)

    def clean_answer(self, raw_answer: str, question: str = "") -> str:
        """
        Strips conversational prefixes, markdown artifacts, filler explanations,
        and applies benchmark-aligned normalization (1-5 words).
        """
        if not raw_answer:
            return ""

        ans = raw_answer.strip().replace("**", "").replace("*", "").replace("`", "").replace('"', '').replace("'", "")

        # Split into lines and take the first informative line
        lines = [line.strip() for line in ans.split("\n") if line.strip()]
        if lines:
            ans = lines[0]

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
            num_match = re.search(r'\b(\d+)\b', ans)
            if num_match:
                return num_match.group(1)
            for w, n in word_to_num.items():
                if re.search(rf'\b{w}\b', ans_lower):
                    return n

        return ans

    def _build_prompts(self, question: str, context: str) -> Tuple[str, str]:
        """Constructs system and user prompts adhering to strict benchmark QA rules."""
        system_prompt = (
            "You are an expert Video Question Answering model.\n"
            "Answer the question accurately, directly, and concisely using the provided video evidence "
            "(visual descriptions, temporal transitions, speech, and sound events).\n\n"
            "Task Instructions & Rules:\n"
            "1. Output ONLY a short, direct answer (typically 1 to 5 words, e.g. \"mirror\", \"shelf\", "
            "\"table\", \"tie up hair\", \"hair styling\", \"dance\", \"yes\", \"no\", \"1\", \"2\", \"outdoor\", \"left\", \"right\").\n"
            "2. For positions: identify the primary object or furniture (e.g. table, shelf, cabinet, mirror, sink).\n"
            "3. For binary questions (is/did/was/does): output strictly \"yes\" or \"no\".\n"
            "4. For counting questions: output ONLY the single number (e.g. \"1\", \"2\", \"3\").\n"
            "5. Do NOT write explanations, conversational filler, or timestamps. Output ONLY the concise final answer."
        )

        evidence_str = context.strip() if context and context.strip() else "[No specific evidence retrieved; predict best probable answer from context]"
        user_prompt = (
            f"Evidence:\n{evidence_str}\n\n"
            f"Question: {question}\n"
            f"Short Answer:"
        )
        return system_prompt, user_prompt

    def _generate_api(self, question: str, context: str, image: Optional[Any] = None) -> str:
        """
        Executes generation via Hugging Face Serverless Inference API.
        """
        system_prompt, user_prompt = self._build_prompts(question, context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if not self.client:
            self._init_api_client()

        try:
            # 1. Try chat_completion endpoint
            response = self.client.chat_completion(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0.01,
            )
            raw_answer = response.choices[0].message.content.strip()
            return self.clean_answer(raw_answer, question=question)
        except Exception as chat_err:
            chat_err_str = str(chat_err).lower()
            if "permission" in chat_err_str or "inference provider" in chat_err_str:
                return "Error: Hugging Face token is missing 'Make calls to Inference Providers' permission. Please enable this permission in your token settings at https://huggingface.co/settings/tokens."
            if "401" in chat_err_str or "unauthorized" in chat_err_str or "403" in chat_err_str:
                return "Error: Hugging Face API authentication failed. Please provide a valid HF_TOKEN."
            if "503" in chat_err_str or "loading" in chat_err_str:
                print(f"[Generator Notice] Model '{self.model_name}' is warming up on Hugging Face Serverless API. Retrying...")

            # 2. Fallback to text_generation endpoint
            try:
                full_prompt = f"{system_prompt}\n\n{user_prompt}\n"
                response = self.client.text_generation(
                    prompt=full_prompt,
                    model=self.model_name,
                    max_new_tokens=self.max_tokens,
                    temperature=0.01,
                )
                raw_answer = response.strip()
                return self.clean_answer(raw_answer, question=question)
            except Exception as text_err:
                print(f"[Generator Warning] HF API text_generation failed: {text_err}")

                # If the 72B vision model is unreachable on free serverless tier, try fallback text LLM
                fallback_id = self.model_info.get("fallback_hf_id")
                if fallback_id and fallback_id != self.model_name:
                    print(f"[Generator] Trying alternative HF Serverless endpoint: '{fallback_id}'...")
                    try:
                        fb_resp = self.client.chat_completion(
                            model=fallback_id,
                            messages=messages,
                            max_tokens=self.max_tokens,
                            temperature=0.01,
                        )
                        return self.clean_answer(fb_resp.choices[0].message.content.strip(), question=question)
                    except Exception as fb_err:
                        print(f"[Generator Warning] Fallback API endpoint '{fallback_id}' failed: {fb_err}")

                return f"Error: Hugging Face API generation failed ({text_err}). Ensure HF_TOKEN is configured."

    def _generate_local(self, question: str, context: str, image: Optional[Any] = None) -> str:
        """
        Executes generation using local PyTorch / Transformers model weights.
        """
        self._ensure_local_model_loaded()
        if self._effective_backend == "api":
            return self._generate_api(question, context, image=image)

        system_prompt, user_prompt = self._build_prompts(question, context)
        model_device = getattr(self.model, "device", self.device)

        try:
            if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
                user_content = []
                if image is not None:
                    user_content.append({"type": "image", "image": image})
                user_content.append({"type": "text", "text": user_prompt})

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                    {"role": "user", "content": user_content}
                ]
                prompt_text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                if image is not None:
                    try:
                        from qwen_vl_utils import process_vision_info
                        image_inputs, video_inputs = process_vision_info(messages)
                        inputs = self.processor(text=[prompt_text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
                    except Exception:
                        inputs = self.processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
                else:
                    inputs = self.processor(text=[prompt_text], padding=True, return_tensors="pt")

                inputs = inputs.to(model_device)
                with torch.inference_mode():
                    out_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
                    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
                    raw_ans = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

            else:
                # Text LLM fallback with tokenizer
                full_text = f"{system_prompt}\n\n{user_prompt}\n"
                tokenizer = self.tokenizer or getattr(self.processor, "tokenizer", None)
                if tokenizer is None:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
                    self.tokenizer = tokenizer

                inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=2048).to(model_device)
                with torch.inference_mode():
                    out_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
                    trimmed = out_ids[0][inputs.input_ids.shape[1]:]
                    raw_ans = tokenizer.decode(trimmed, skip_special_tokens=True).strip()

            return self.clean_answer(raw_ans, question=question)

        except torch.cuda.OutOfMemoryError as oom_e:
            print(f"[Generator OOM Error] Out of GPU memory while running '{self.model_name}' locally: {oom_e}")
            if HAS_HF_CLIENT and self.hf_token:
                print("[Generator] Falling back to Hugging Face Serverless API...")
                return self._generate_api(question, context, image=image)
            return "Error: Local GPU Out of Memory. Run with GENERATOR_BACKEND=api and set HF_TOKEN."
        except Exception as e:
            print(f"[Generator Error] Local generation failed: {e}")
            if HAS_HF_CLIENT and self.hf_token:
                print("[Generator] Attempting API fallback...")
                return self._generate_api(question, context, image=image)
            return f"Error: Generation failed ({e})"

    def build_chronological_context(self, candidates: list) -> str:
        """Alias for format_context aligning with Stage 3 Pipeline Node C."""
        return self.format_context(candidates)

    def build_qa_prompt(self, question: str, context: str) -> Tuple[str, str]:
        """Alias for _build_prompts aligning with Stage 3 Pipeline Node D."""
        return self._build_prompts(question, context)

    def generate_answer(
        self,
        question: str,
        context: Union[str, List[Dict[str, Any]]],
        image: Optional[Any] = None
    ) -> str:
        """
        Stage 3 Grounded Generation Pipeline:
            A. Retrieved Multimodal Evidence (Visual + Speech + Sound + Counts)
            B. User Question
            C. Build Chronological Context (build_chronological_context / format_context)
            D. Build Strict QA Prompt (build_qa_prompt / _build_prompts)
            E. Hugging Face Inference API (HF_TOKEN) / Local PyTorch Fallback
            F. LLM (Qwen / Gemma / Phi)
            G. Clean & Normalize Answer (clean_answer)
            H. Final Answer (1–5 Words)

        Args:
            question: Natural language question (Node B).
            context: Formatted context string OR retrieved candidate list (Node A).
            image: Optional keyframe PIL Image or path.

        Returns:
            Concise, grounded benchmark answer of 1 to 5 words (Node H).
        """
        # Node A -> Node C: Build Chronological Context if raw candidate list provided
        if isinstance(context, list):
            context_str = self.format_context(context)
        else:
            context_str = str(context) if context is not None else ""

        # Node B + Node C -> Node D -> Node E / F -> Node G -> Node H
        if self._effective_backend == "api":
            ans = self._generate_api(question, context_str, image=image)
            # If API failed and not due to auth, and user has local model, try local
            if ans.startswith("Error:") and self.model is not None:
                return self._generate_local(question, context_str, image=image)
            return ans
        else:
            ans = self._generate_local(question, context_str, image=image)
            return ans

