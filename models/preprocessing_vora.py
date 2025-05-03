from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import io
import os
import random
from typing import Any, List, Union

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin, ProcessingKwargs, Unpack, _validate_images_text_input_order
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from data_module.vision_transform import VisionProcessor
from data_module.vqa_processor import VQAProcessor
from utils import logging
from utils.constants import (
    DATA_TYPE_IMAGE,
    DATA_TYPE_TEXT,
    DATA_TYPE_VIDEO,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

logger = logging.get_logger(__name__)


class VoRAProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {"padding": False},
        "images_kwargs": {},
    }


class VoRAProcessor(ProcessorMixin):
    """
    Custom processor for vision-and-language tasks, supporting
    saving/loading via ProcessorMixin and chat templating.
    """
    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template", "image_token", "image_token_index"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        frames_key: str = "frames",
        frames_ops: Any = None,
        aux_frames_ops: Any = None,
        label_key: Union[str, List, None] = None,
        meta_keys: List[str] = ["id", "source", "question", "gt"],
        padding_side: str = "right",
        tokenizer: str = "",
        trust_remote_code: bool = False,
        eos_token: Union[str, None] = None,
        max_seq_len: int = 512,
        max_prompt_len: int = 512,
        sample_method: str = "global_random",
        dummy_frame_shape: tuple = (0, 3, 448, 448),
        max_batch_frames: int = 16,
        num_segments: int = 8,
        training: bool = True,
        verbose: bool = True,
        task_type: str = "completion",
        truncate_mode: str = "qa",
        vqa_processor_params: dict = {},
        chat_template: Union[str, None] = None,
        image_token: str = DEFAULT_IMAGE_TOKEN,
        image_token_index: int = IMAGE_TOKEN_INDEX,
    ):
        # Initialize vision processor
        if isinstance(frames_ops, str):
            img_proc = AutoImageProcessor.from_pretrained(frames_ops)
        else:
            img_proc = VisionProcessor(frames_ops)
        
        # Initialize tokenizer
        tok = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=False, trust_remote_code=trust_remote_code
        )

        # Initialize ProcessorMixin with core sub-components and valid_kwargs
        super().__init__(
            image_processor=img_proc,
            tokenizer=tok,
            chat_template=chat_template,
            
        )
        self.image_token=image_token,
        self.image_token_index=image_token_index,

        # VoRA-specific settings
        self.frames_key = frames_key
        self.aux_frames_ops = aux_frames_ops
        self.enable_aux_frames = aux_frames_ops is not None
        self.label_key = label_key
        self.meta_keys = meta_keys
        self.padding_side = padding_side
        self.eos_token = eos_token
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.sample_method = sample_method
        self.dummy_frame_shape = dummy_frame_shape
        self.max_batch_frames = max_batch_frames
        self.num_segments = num_segments
        self.training = training
        self.verbose = verbose
        self.task_type = task_type
        self.truncate_mode = truncate_mode

        # Tokenizer special tokens
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.pad_id
        self.eos_id = (
            self.tokenizer.convert_tokens_to_ids(eos_token)
            if eos_token
            else self.tokenizer.eos_token_id
        )
        self.ignore_index = IGNORE_INDEX

        # Vision placeholder settings
        self.vision_placeholder = image_token
        self.vision_placeholder_index = image_token_index

        # Auxiliary vision processors
        if self.enable_aux_frames:
            assert isinstance(aux_frames_ops, dict), "aux_frames_ops must be dict"
            self.aux_frame_keys = list(aux_frames_ops.keys())
            self.aux_video_processor = {}
            for name, ops in aux_frames_ops.items():
                if isinstance(ops, str):
                    self.aux_video_processor[name] = AutoImageProcessor.from_pretrained(ops)
                else:
                    self.aux_video_processor[name] = VisionProcessor(ops)

        # VQA processor
        self.vqa_processor = VQAProcessor(
            self.label_key, self.vision_placeholder, **vqa_processor_params
        )

    def __call__(
        self,
        images = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        **kwargs: Unpack[VoRAProcessorKwargs],
    ):
        # Delegate to ProcessorMixin's multimodal handling
        return super().__call__(images=images, text=text, **kwargs)

    #--- Keep helper methods unchanged below ---
    def expand2square(self, pil_img: Image.Image) -> Image.Image:
        background_color = (0, 0, 0)
        width, height = pil_img.size
        if width == height:
            return pil_img
        size = max(width, height)
        result = Image.new(pil_img.mode, (size, size), background_color)
        paste_coords = ((size - width) // 2, (size - height) // 2)
        result.paste(pil_img, paste_coords)
        return result

    def tokenizer_vision_placeholder(
        self, prompt: str, has_frame: bool, add_bos: bool = False
    ) -> List[int]:
        def join_lists(*lists, sep):
            result = []
            for i, lst in enumerate(lists):
                if i > 0 and sep is not None:
                    result.append(sep)
                result.extend(lst)
            return result

        if has_frame:
            chunks = prompt.split(self.vision_placeholder)
            prompt_ids = [self.tokenizer.encode(c) for c in chunks]
            input_ids = join_lists(*prompt_ids, sep=self.vision_placeholder_index)
        else:
            input_ids = self.tokenizer.encode(prompt)
        if add_bos and self.tokenizer.bos_token_id is not None:
            input_ids = [self.tokenizer.bos_token_id] + input_ids
        return input_ids

    def build_text(self, data_dict: dict) -> dict:
        prompt_list, response_list = self.vqa_processor(data_dict)
        has_frame = data_dict.get("has_frame", False)

        prompt_ids = [
            self.tokenizer_vision_placeholder(p, has_frame) for p in prompt_list
        ]
        response_ids = [self.tokenizer.encode(r) for r in response_list]

        input_ids, label_mask = [], []
        for p_ids, r_ids in zip(prompt_ids, response_ids):
            seq_ids = p_ids + r_ids + [self.eos_id]
            mask = [0] * len(p_ids) + [1] * (len(r_ids) + 1)
            input_ids.extend(seq_ids)
            label_mask.extend(mask)
            if len(input_ids) >= self.max_seq_len:
                break

        input_ids = input_ids[: self.max_seq_len]
        label_mask = label_mask[: self.max_seq_len]
        attention_mask = [1] * len(input_ids)
        labels = torch.tensor(input_ids, dtype=torch.long)
        labels[label_mask != 1] = self.ignore_index

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": labels,
        }

    def build_visual(self, data_dict: dict) -> dict:
        frames = data_dict.get(self.frames_key, [])
        if not frames:
            return {self.frames_key: torch.zeros(self.dummy_frame_shape)}

        if isinstance(self.image_processor, VisionProcessor):
            processed = self.image_processor(frames)
            pixel_values = torch.stack(processed)
        else:
            arrs = [np.asarray(f.convert("RGB")) for f in frames]
            pixel_values = self.image_processor(arrs, return_tensors="pt").pixel_values

        return {self.frames_key: pixel_values}

    def batch_transform(self, batch_data: List[dict]) -> dict:
        # Collate frames
        frames_list = [d.pop(self.frames_key) for d in batch_data]
        collated_frames = torch.cat(frames_list, dim=0)

        # Collate text inputs
        input_ids = [d.pop("input_ids") for d in batch_data]
        attention_mask = [d.pop("attention_mask") for d in batch_data]
