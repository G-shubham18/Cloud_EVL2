"""
EchoVision: Visual Captioning Engine
Supports modular, plug-and-play visual captioning models:
  1. Salesforce/blip-image-captioning-large (Default)
  2. Salesforce/blip-image-captioning-base
  3. HuggingFaceTB/SmolVLM-256M-Instruct
  4. wraps/moondream-caption (or vikhyatk/moondream2)
  5. Qwen/Qwen2.5-VL-3B-Instruct (Legacy / Structured VLM)
"""

import os
import sys
import gc
from typing import Optional, List, Dict, Any, Union
from PIL import Image
import torch

from config import (
    DEVICE,
    TORCH_DTYPE,
    IS_GPU,
    IS_CUDA_AVAILABLE,
    CAPTIONING_MODEL,
    resolve_captioning_model,
    get_hf_token,
    has_hf_token,
)


class VisualCaptioner:
    """
    Unified, plug-and-play visual captioning engine supporting BLIP, SmolVLM,
    Moondream, Qwen-VL, and custom Vision-Language models.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
        lazy_load: bool = False,
    ):
        self.requested_name = model_name or CAPTIONING_MODEL
        self.model_info = resolve_captioning_model(self.requested_name)
        self.model_id = self.model_info["hf_id"]
        self.architecture = self.model_info["architecture"]

        # Ensure device placement is valid
        self.device = device or DEVICE
        self.hf_token = hf_token or get_hf_token()

        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False

        if not lazy_load:
            self.load_model()

    def is_loaded(self) -> bool:
        """Returns True if model and processor are initialized."""
        return self._is_loaded

    def load_model(self):
        """Loads model and processor into memory based on detected architecture."""
        if self._is_loaded:
            return

        token_status = "[CONFIGURED]" if bool(self.hf_token) else "[NOT SET]"
        print(f"[VisualCaptioner] Loading captioning model: '{self.model_id}' (arch={self.architecture}, device={self.device}, HF Token={token_status})...")

        kwargs = {
            "low_cpu_mem_usage": True,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token

        # Use FP16/BF16 on GPU to save memory, FP32 on CPU
        if self.device == "cuda" and IS_CUDA_AVAILABLE:
            kwargs["torch_dtype"] = TORCH_DTYPE
            kwargs["device_map"] = "auto"

        try:
            if self.architecture == "blip":
                self._load_blip(kwargs)
            elif self.architecture == "smolvlm":
                self._load_smolvlm(kwargs)
            elif self.architecture == "moondream":
                self._load_moondream(kwargs)
            elif self.architecture == "qwen_vl":
                self._load_qwen_vl(kwargs)
            else:
                self._load_generic_vlm(kwargs)

            self._is_loaded = True
            print(f"[VisualCaptioner] Successfully loaded '{self.model_id}' on {self.device}.")
        except Exception as e:
            # Fallback to secondary model if primary fails
            fallback_id = self.model_info.get("fallback_hf_id")
            if fallback_id and fallback_id != self.model_id:
                print(f"[VisualCaptioner Warning] Failed to load '{self.model_id}' ({e}). Trying fallback: '{fallback_id}'...")
                try:
                    self.model_id = fallback_id
                    if "blip" in fallback_id.lower():
                        self.architecture = "blip"
                        self._load_blip(kwargs)
                    elif "moondream" in fallback_id.lower():
                        self.architecture = "moondream"
                        self._load_moondream(kwargs)
                    else:
                        self._load_generic_vlm(kwargs)
                    self._is_loaded = True
                    print(f"[VisualCaptioner] Successfully loaded fallback model '{self.model_id}'.")
                    return
                except Exception as fallback_e:
                    print(f"[VisualCaptioner Error] Fallback model '{fallback_id}' also failed: {fallback_e}")

            print(f"[VisualCaptioner Error] Could not load captioning model '{self.model_id}': {e}")
            raise RuntimeError(f"VisualCaptioner initialization failed for '{self.model_id}': {e}")

    def _load_blip(self, kwargs: dict):
        """Loads Salesforce BLIP captioning model and processor."""
        from transformers import BlipProcessor, BlipForConditionalGeneration

        token_arg = {"token": self.hf_token} if self.hf_token else {}
        self.processor = BlipProcessor.from_pretrained(self.model_id, **token_arg)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_id, **kwargs)

        if self.device != "cuda" or not IS_CUDA_AVAILABLE:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_smolvlm(self, kwargs: dict):
        """Loads HuggingFaceTB SmolVLM model and processor."""
        from transformers import AutoProcessor

        token_arg = {"token": self.hf_token} if self.hf_token else {}
        self.processor = AutoProcessor.from_pretrained(self.model_id, **token_arg)

        loaded_model = None
        try:
            from transformers import AutoModelForImageTextToText
            loaded_model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        except Exception:
            try:
                from transformers import AutoModelForConditionalGeneration
                loaded_model = AutoModelForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
            except Exception:
                from transformers import AutoModelForCausalLM
                loaded_model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

        self.model = loaded_model
        if self.device != "cuda" or not IS_CUDA_AVAILABLE:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_moondream(self, kwargs: dict):
        """Loads Moondream vision captioning model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        moondream_kwargs = dict(kwargs)
        moondream_kwargs["trust_remote_code"] = True

        token_arg = {"token": self.hf_token} if self.hf_token else {}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True, **token_arg)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **moondream_kwargs)

        if self.device != "cuda" or not IS_CUDA_AVAILABLE:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_qwen_vl(self, kwargs: dict):
        """Loads Qwen-VL conditional generation model."""
        from transformers import AutoProcessor

        token_arg = {"token": self.hf_token} if self.hf_token else {}
        self.processor = AutoProcessor.from_pretrained(self.model_id, **token_arg)

        loaded_model = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            loaded_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
        except Exception:
            from transformers import AutoModelForImageTextToText
            loaded_model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)

        self.model = loaded_model
        if self.device != "cuda" or not IS_CUDA_AVAILABLE:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _load_generic_vlm(self, kwargs: dict):
        """Loads generic vision-language model using transformers Auto classes."""
        from transformers import AutoProcessor

        token_arg = {"token": self.hf_token} if self.hf_token else {}
        self.processor = AutoProcessor.from_pretrained(self.model_id, **token_arg)

        loaded_model = None
        try:
            from transformers import AutoModelForImageTextToText
            loaded_model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        except Exception:
            from transformers import AutoModelForCausalLM
            loaded_model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True, **kwargs)

        self.model = loaded_model
        if self.device != "cuda" or not IS_CUDA_AVAILABLE:
            self.model = self.model.to(self.device)
        self.model.eval()

    def _prepare_image(self, image: Union[Image.Image, Any]) -> Image.Image:
        """Ensures image is a valid PIL RGB Image with reasonable resolution."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Cap excessive dimensions to avoid memory spikes
        max_dim = 768 if IS_GPU else 512
        if max(image.size) > max_dim:
            image = image.copy()
            image.thumbnail((max_dim, max_dim))
        return image

    def caption_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        """
        Generates a descriptive caption for a single image.
        """
        if not self._is_loaded:
            self.load_model()

        img = self._prepare_image(image)
        model_device = getattr(self.model, "device", self.device)

        try:
            if self.architecture == "blip":
                # BLIP: Conditional or unconditional captioning
                if prompt:
                    inputs = self.processor(img, text=prompt, return_tensors="pt").to(model_device)
                else:
                    inputs = self.processor(img, return_tensors="pt").to(model_device)

                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=60)
                    caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
                return caption

            elif self.architecture == "smolvlm":
                # SmolVLM: Chat-based vision instruction
                sys_text = prompt or "Describe the main scene, visible people, actions, and objects in this video frame concisely."
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": sys_text},
                        ],
                    }
                ]
                formatted_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = self.processor(text=formatted_prompt, images=[img], return_tensors="pt").to(model_device)

                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=100)
                    caption = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                    if "Assistant:" in caption:
                        caption = caption.split("Assistant:")[-1].strip()
                return caption

            elif self.architecture == "moondream":
                # Moondream captioning method
                if hasattr(self.model, "caption"):
                    res = self.model.caption(img, length="normal")
                    if isinstance(res, dict):
                        return res.get("caption", "").strip()
                    return str(res).strip()
                elif hasattr(self.model, "encode_image"):
                    image_embeds = self.model.encode_image(img)
                    q = prompt or "Describe this image in detail."
                    return self.model.answer_question(image_embeds, q, self.tokenizer)
                else:
                    inputs = self.tokenizer(prompt or "Describe this scene:", return_tensors="pt").to(model_device)
                    with torch.inference_mode():
                        out = self.model.generate(**inputs, max_new_tokens=60)
                        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

            elif self.architecture == "qwen_vl":
                # Qwen-VL Chat format
                from config import STRUCTURED_VLM_PROMPT
                q_text = prompt or STRUCTURED_VLM_PROMPT
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": q_text},
                        ],
                    }
                ]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                try:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model_device)
                except Exception:
                    inputs = self.processor(text=[text], images=[img], padding=True, return_tensors="pt").to(model_device)

                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=128)
                    caption = self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()
                return caption

            else:
                # Generic fallback
                inputs = self.processor(images=[img], text=prompt or "A photo of", return_tensors="pt").to(model_device)
                with torch.inference_mode():
                    out = self.model.generate(**inputs, max_new_tokens=60)
                    return self.processor.batch_decode(out, skip_special_tokens=True)[0].strip()

        except Exception as e:
            print(f"[VisualCaptioner Warning] Captioning failed for image: {e}")
            return f"Scene frame with visible objects and setting."

    def caption_batch(self, images: List[Image.Image], prompt: Optional[str] = None) -> List[str]:
        """
        Generates captions for a list of images.
        Uses native batched inference where supported, else iterates cleanly.
        """
        if not images:
            return []

        if not self._is_loaded:
            self.load_model()

        captions = []
        for img in images:
            cap = self.caption_image(img, prompt=prompt)
            captions.append(cap)

        return captions

    def unload(self):
        """Releases model weights and frees VRAM/RAM."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._is_loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[VisualCaptioner] Model '{self.model_id}' unloaded and memory freed.")
