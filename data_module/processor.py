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
from google.cloud import storage # Added for GCS access

from data_module.vision_transform import VisionProcessor # Assuming these are your custom modules
from data_module.vqa_processor import VQAProcessor     # Assuming these are your custom modules
from utils import logging # Assuming this is your custom logging
from utils.constants import ( # Assuming these are your custom constants
    DATA_TYPE_IMAGE,
    DATA_TYPE_TEXT,
    DATA_TYPE_VIDEO,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)


logger = logging.get_logger(__name__)


# load_local_frame is superseded by _load_frame_from_uri if GCS is involved,
# but kept here if directly called or for context from original code.
def load_local_frame_original(frame_path): # Renamed to avoid conflict if used elsewhere
    return Image.open(frame_path).convert("RGB")


def random_index(frames_length, num_segments, average=False):
    if frames_length <= num_segments:
        return [i for i in range(frames_length)] + (num_segments - frames_length) * [frames_length - 1]
    else:
        result = []
        stride = frames_length // num_segments
        s_list = [stride] * num_segments
        for i in range(frames_length - num_segments * stride):
            s_list[i] += 1
        if not average:
            random.shuffle(s_list)
        cursor = 0
        for each_stride in s_list:
            left, right = cursor, cursor + each_stride
            cursor += each_stride
            if not average:
                result.append(random.randint(left, right - 1))
            else:
                result.append(left)
        return result


class VoRAProcessor(object):
    def __init__(self,
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
                 max_workers_frame_load: int = 32, # Added for ThreadPoolExecutor
                 ):
        self.frames_key = frames_key
        self.num_segments = num_segments
        self.max_batch_frames = max_batch_frames
        self.truncate_mode = truncate_mode
        self.dummy_frame_shape = dummy_frame_shape
        self.label_key = label_key
        self.meta_keys = meta_keys
        self.padding_side = padding_side
        self.training = training
        self.max_seq_len = max_seq_len
        self.max_prompt_len = max_prompt_len
        self.sample_method = sample_method
        self.verbose = verbose
        self.aux_frames_ops = aux_frames_ops
        self.enable_aux_frames = False
        self.max_workers_frame_load = max_workers_frame_load
        self.gcs_client = None # Lazy initialization for GCS client

        # vision processors
        if isinstance(frames_ops, str):
            self.video_processor = AutoImageProcessor.from_pretrained(
                frames_ops, trust_remote_code=trust_remote_code) # Added trust_remote_code
        else:
            self.video_processor = VisionProcessor(frames_ops)
        self.enable_aux_frames = (aux_frames_ops is not None)
        if self.enable_aux_frames:
            assert isinstance(aux_frames_ops, dict), "aux_frames_ops must be dict"
            self.aux_frame_keys = list(aux_frames_ops.keys())
            self.aux_video_processor = {}
            for name in self.aux_frame_keys:
                current_aux_frames_ops = aux_frames_ops[name]
                if isinstance(current_aux_frames_ops, str):
                    self.aux_video_processor[name] = AutoImageProcessor.from_pretrained(
                        current_aux_frames_ops, trust_remote_code=trust_remote_code) # Added trust_remote_code
                else:
                    self.aux_video_processor[name] = VisionProcessor(current_aux_frames_ops)

        # load tokenizer
        self.eos_token = eos_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, use_fast=False, trust_remote_code=trust_remote_code)
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        self.eos_id = self.tokenizer.convert_tokens_to_ids(eos_token) \
            if eos_token else self.tokenizer.eos_token_id
        self.ignore_index = IGNORE_INDEX

        self.task_type = task_type
        self.vision_placeholder = DEFAULT_IMAGE_TOKEN
        self.vision_placeholder_index = IMAGE_TOKEN_INDEX

        self.vqa_processor = VQAProcessor(
            self.label_key, self.vision_placeholder, **vqa_processor_params)

    def _get_gcs_client(self):
        """Initializes and returns the GCS client, creating it if it doesn't exist."""
        if self.gcs_client is None:
            logger.info("Initializing GCS client for VoRAProcessor.")
            self.gcs_client = storage.Client()
        return self.gcs_client

    def _load_frame_from_uri(self, uri_or_path: str) -> Image.Image:
        """Loads a frame from a URI (GCS path) or a local file path."""
        try:
            if uri_or_path.startswith("gs://"):
                client = self._get_gcs_client()
                # Parse GCS path (gs://bucket-name/path/to/blob)
                path_parts = uri_or_path[5:].split("/", 1)
                bucket_name = path_parts[0]
                blob_name = path_parts[1] if len(path_parts) > 1 else ""

                if not blob_name:
                    raise ValueError(f"Invalid GCS URI (missing object name): {uri_or_path}")

                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                image_bytes = blob.download_as_bytes()
                # logger.debug(f"Successfully loaded {len(image_bytes)} bytes from {uri_or_path}")
                return Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else: # Local path
                # logger.debug(f"Loading local frame {uri_or_path}")
                return Image.open(uri_or_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load frame from URI/path {uri_or_path}: {e}")
            # Depending on desired behavior, could return a placeholder image or None.
            # Re-raising makes the failure explicit.
            raise

    def preprocess(self, data_dict):
        has_frame = False
        if self.frames_key in data_dict and data_dict[self.frames_key]: # Ensure frames list is not empty
            has_frame = True
            data_dict["data_type"] = DATA_TYPE_IMAGE # Default to image
            
            frames = data_dict[self.frames_key]
            # Determine if it's a video based on the number of frames *before* sampling
            if len(frames) > 1:
                data_dict["data_type"] = DATA_TYPE_VIDEO

            # Sample frames first (operates on list of paths/filenames or bytes)
            self.sample_frames(data_dict) 
            
            # After sampling, frames variable needs to be updated with the sampled list
            frames = data_dict[self.frames_key] # Get the sampled list

            if frames and isinstance(frames[0], (str, os.PathLike)):
                # Construct full URIs (can be local or GCS)
                # This assumes 'frames' contains filenames relative to 'image_folder'
                image_folder = data_dict.get("image_folder", "")
                full_frame_uris = [os.path.join(image_folder, frame_filename) for frame_filename in frames]
                
                # data_dict["original_frame_uris"] = full_frame_uris # Optional: for debugging

                # Use ThreadPoolExecutor to load frames (local or GCS)
                with ThreadPoolExecutor(max_workers=self.max_workers_frame_load) as executor:
                    loaded_frames = list(executor.map(self._load_frame_from_uri, full_frame_uris))
                
                data_dict[self.frames_key] = loaded_frames # Store PIL Images back
            
            elif frames and isinstance(frames[0], bytes):
                # This path handles if the Dataset already provided bytes
                loaded_frames = [Image.open(io.BytesIO(frame_bytes)).convert("RGB") for frame_bytes in frames]
                data_dict[self.frames_key] = loaded_frames
            
            # Update n_frames based on the *loaded and sampled* frames
            num_loaded_frames = len(data_dict[self.frames_key]) if isinstance(data_dict[self.frames_key], list) else 0
            data_dict["n_frames"] = num_loaded_frames
            if num_loaded_frames == 0 and has_frame: # if it was supposed to have frames but loading failed or sampled to zero
                logger.warning(f"Item {data_dict.get('id', 'Unknown_ID')} had frames indicated, but 0 frames loaded/sampled.")
                data_dict[self.frames_key] = torch.zeros(self.dummy_frame_shape) # Fallback
                has_frame = False # No usable frames
                data_dict["data_type"] = DATA_TYPE_TEXT # Fallback to text if frames are gone

        else: # No frames_key or empty frames list initially
            data_dict[self.frames_key] = torch.zeros(self.dummy_frame_shape)
            data_dict["n_frames"] = 0
            data_dict["data_type"] = DATA_TYPE_TEXT

        if self.label_key not in data_dict:
            # logger.error(f"label_key {self.label_key} not in data_dict for item {data_dict.get('id', 'Unknown_ID')}")
            # For robustness, let's allow items without labels during inference/some cases
            # but fill with a placeholder if necessary or let downstream handle it.
            # For now, we assume it might be missing for some items.
            # If it's critical, raise ValueError here.
            pass
        elif data_dict.get(self.label_key) and isinstance(data_dict[self.label_key][0], list):
            # Handle nested list for labels if present
            data_dict[self.label_key] = data_dict[self.label_key][0]

        data_dict["has_frame"] = has_frame
        return data_dict

    def sample_frames(self, data_dict):
        average_draw = not self.training
        frames = data_dict[self.frames_key] # This list can contain paths or bytes at this stage
        
        if not frames: # Handle empty frames list before sampling
            data_dict[self.frames_key] = []
            return

        if self.sample_method == "global_random":
            frames_index = random_index(
                len(frames), self.num_segments, average_draw)
            part_frames = [frames[i] for i in frames_index]
        elif self.sample_method == "global":
            part_frames = frames
            if len(part_frames) > self.max_batch_frames: # Ensure not to exceed max_batch_frames
                frames_index = random_index(
                    len(part_frames), self.max_batch_frames, average_draw)
                part_frames = [part_frames[i] for i in frames_index]
        else:
            # Allow no sampling if sample_method is None or "none"
            if self.sample_method is None or str(self.sample_method).lower() == "none":
                part_frames = frames
            else:
                raise NotImplementedError(f"sample method {self.sample_method} not implemented")
        data_dict[self.frames_key] = part_frames
        return

    def transform(self, data_dict):
        try:
            data_dict = self.preprocess(data_dict)

            output = dict()
            # add meta data
            for key in self.meta_keys:
                output[key] = data_dict.get(key, "unknown")

            # add vision info
            output["n_frames"] = data_dict["n_frames"]
            output["data_type"] = data_dict["data_type"]
            # Check 'has_frame' after preprocess, as it might have changed if loading failed
            if data_dict["has_frame"] and data_dict["n_frames"] > 0:
                 output.update(self.build_visual(data_dict))
            else: # Ensure dummy frames are added if no valid frames
                 output[self.frames_key] = torch.zeros(self.dummy_frame_shape)
                 if self.enable_aux_frames:
                     for aux_frame_key in self.aux_frame_keys:
                         output[aux_frame_key] = torch.zeros(self.dummy_frame_shape)


            output.update(self.build_text(data_dict))
            return output
        except Exception as e:
            item_id = data_dict.get("id", "Unknown_ID") if isinstance(data_dict, dict) else "Unknown_ID"
            logger.warning(f"Collapsible data for item {item_id}!!! {e}", exc_info=True) # Add exc_info for traceback
            return None

    # ... (rest of the methods: batch_process, tokenizer_vision_placeholder, build_text, padding_sequence, build_visual, collate_frames, batch_transform)
    # ... should largely remain the same, as they operate on data after it has been loaded (PIL Images or Tensors)
    # ... or on text data.

    # Make sure build_visual correctly handles PIL images passed in data_dict[self.frames_key]
    def build_visual(self, data_dict):
        frames = data_dict[self.frames_key] # These are now expected to be PIL Images
        if data_dict["n_frames"] > 0 and isinstance(frames, list) and all(isinstance(f, Image.Image) for f in frames):
            ret_dict = {}
            if isinstance(self.video_processor, VisionProcessor):
                # Assuming VisionProcessor takes a list of PIL Images
                ret = self.video_processor(frames)
                ret = torch.stack(ret) if isinstance(ret, list) else ret
            elif isinstance(self.video_processor, BaseImageProcessor):
                # Transformers ImageProcessors usually take list of PIL Images or np arrays
                ret = self.video_processor([np.asarray(frame) for frame in frames], return_tensors="pt").data["pixel_values"]
            else:
                raise NotImplementedError(f"Unsupported video_processor type: {type(self.video_processor)}")
            ret_dict[self.frames_key] = ret

            if self.enable_aux_frames:
                for aux_frame_key in self.aux_frame_keys:
                    aux_video_processor = self.aux_video_processor[aux_frame_key]
                    if isinstance(aux_video_processor, VisionProcessor):
                        aux_ret = aux_video_processor(frames)
                        aux_ret = torch.stack(aux_ret) if isinstance(aux_ret, list) else aux_ret
                    elif isinstance(aux_video_processor, BaseImageProcessor):
                        aux_ret = aux_video_processor([np.asarray(frame) for frame in frames], return_tensors="pt").data["pixel_values"]
                    else:
                        raise NotImplementedError(f"Unsupported aux_video_processor type: {type(aux_video_processor)}")
                    ret_dict[aux_frame_key] = aux_ret
            return ret_dict
        else: # Fallback if frames are not as expected (e.g., already tensors or dummy)
            return {self.frames_key: frames if torch.is_tensor(frames) else torch.zeros(self.dummy_frame_shape) }
    
    # Remaining methods (batch_process, tokenizer_vision_placeholder, build_text, 
    # padding_sequence, collate_frames, batch_transform) are mostly unchanged
    # as they operate on data after initial loading or text. Ensure they are robust.

    def tokenizer_vision_placeholder(self, prompt, has_frame, add_bos=False):
        def join_lists(*lists, sep):
            result = []
            for i, lst in enumerate(lists):
                if i > 0 and sep: # Ensure sep is not None or empty if used
                    if isinstance(sep, list):
                        result.extend(sep)
                    else: # Assuming sep is a single token index
                        result.append(sep)
                result.extend(lst)
            return result

        if has_frame:
            prompt_chunks_str = prompt.split(self.vision_placeholder)
            prompt_chunks_ids = [self.tokenizer.encode(chunk, add_special_tokens=False) for chunk in prompt_chunks_str]
            
            # Vision placeholder index should be a list if it represents multiple tokens,
            # or a single int if it's one token index.
            # self.tokenizer.encode might add bos/eos if not add_special_tokens=False
            vision_placeholder_ids = [self.vision_placeholder_index] if isinstance(self.vision_placeholder_index, int) \
                                     else self.tokenizer.encode(self.vision_placeholder, add_special_tokens=False)

            input_ids = []
            for i, chunk_ids in enumerate(prompt_chunks_ids):
                input_ids.extend(chunk_ids)
                if i < len(prompt_chunks_ids) - 1: # Add placeholder between chunks
                    input_ids.extend(vision_placeholder_ids)
        else:
            input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        if add_bos: # Add BOS token if not already added by tokenizer.encode and if needed
            if not input_ids or input_ids[0] != self.tokenizer.bos_token_id:
                 input_ids = [self.tokenizer.bos_token_id] + input_ids
        
        return input_ids

    def build_text(self, data_dict):
        # Ensure label_key exists or handle its absence gracefully
        label_data = data_dict.get(self.label_key, None)
        if label_data is None and self.training: # Labels might be optional for inference
            logger.warning_rank0(f"Label key '{self.label_key}' not found in data_dict for item {data_dict.get('id', 'N/A')} during training.")
            # Create dummy labels or raise error based on requirements
            # For now, let's assume VQAProcessor handles None label_data or this case is non-critical
        
        prompt_list, response_list = self.vqa_processor(data_dict)
        has_frame = data_dict["has_frame"]
        
        # Ensure prompt_list and response_list are indeed lists of strings
        if not isinstance(prompt_list, list): prompt_list = [str(prompt_list)]
        if not isinstance(response_list, list): response_list = [str(response_list)]

        prompt_token_ids_list = [self.tokenizer_vision_placeholder(str(prompt), has_frame) for prompt in prompt_list]
        response_token_ids_list = [self.tokenizer.encode(str(response), add_special_tokens=False) for response in response_list]
        
        input_ids = []
        label_mask = [] # 0 for prompt, 1 for response

        current_total_len = 0
        for i, (prompt_ids, response_ids) in enumerate(zip(prompt_token_ids_list, response_token_ids_list)):
            # EOS token will be added after each response segment
            segment_len = len(prompt_ids) + len(response_ids) + 1 # +1 for EOS

            if current_total_len + segment_len > self.max_seq_len:
                if self.truncate_mode == "qa" and i > 0: # Truncate whole Q/A turns if not the first one
                    logger.warning_rank0(f"Text length exceeds max_seq_len ({self.max_seq_len}). Truncating QA turns for item {data_dict.get('id', 'N/A')}.")
                    break
                else: # Truncate within the current segment (text mode or first QA)
                    logger.warning_rank0(f"Text length exceeds max_seq_len ({self.max_seq_len}). Truncating text for item {data_dict.get('id', 'N/A')}.")
                    remaining_len = self.max_seq_len - current_total_len -1 # -1 for the final EOS to be added
                    
                    if remaining_len <= 0: break # No space left

                    if len(prompt_ids) > remaining_len: # Not enough space even for prompt
                        prompt_ids = prompt_ids[:remaining_len]
                        response_ids = [] # No space for response
                    elif len(prompt_ids) + len(response_ids) > remaining_len:
                        response_ids = response_ids[:remaining_len - len(prompt_ids)]
                    
                    input_ids.extend(prompt_ids)
                    label_mask.extend([0] * len(prompt_ids))
                    input_ids.extend(response_ids)
                    label_mask.extend([1] * len(response_ids))
                    current_total_len += len(prompt_ids) + len(response_ids)
                    break # Max length reached

            input_ids.extend(prompt_ids)
            label_mask.extend([0] * len(prompt_ids))
            input_ids.extend(response_ids)
            label_mask.extend([1] * len(response_ids))
            input_ids.append(self.eos_id) # Add EOS after each response
            label_mask.append(1) # EOS is part of the label
            current_total_len += segment_len
        
        # If nothing was added (e.g. first turn was too long and truncated to nothing)
        if not input_ids:
            # Fallback to a minimal sequence if possible, or handle as error
            logger.warning_rank0(f"Empty input_ids after truncation for item {data_dict.get('id', 'N/A')}. Using BOS+EOS.")
            input_ids = [self.tokenizer.bos_token_id or self.eos_id, self.eos_id] # Ensure there's something
            label_mask = [0, 1] # Or [1,1] if BOS should be ignored and EOS learned

        attention_mask_list = [1] * len(input_ids)
        
        # For debugging or logging, reconstruct prompt/response strings
        # This might be slightly different from original due to tokenization nuances
        # final_prompt_str = self.tokenizer.decode([id for id, mask_val in zip(input_ids, label_mask) if mask_val == 0])
        # final_response_str = self.tokenizer.decode([id for id, mask_val in zip(input_ids, label_mask) if mask_val == 1 and id != self.eos_id])


        return {
            "input_ids": torch.as_tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.as_tensor(attention_mask_list, dtype=torch.int64),
            "labels": torch.as_tensor(input_ids, dtype=torch.int64).masked_fill(torch.as_tensor(label_mask, dtype=torch.bool) == False, self.ignore_index), # type: ignore
            "prompt": " ".join(prompt_list), # Original prompts for meta
            "gt": " ".join(response_list)    # Original responses for meta
        }

    def padding_sequence(self, inputs: List[torch.Tensor], value: int) -> torch.Tensor:
        """ Pad input sequence(input_ids, attention_mask, label) to `max_length` in the batch,
            fill padding place with `value`
        """
        if not inputs: return torch.empty(0) # Handle empty list case
        # Ensure all inputs are 1D tensors before padding
        inputs = [t.squeeze() if t.ndim > 1 else t for t in inputs]

        max_length = max([len(d) for d in inputs]) if inputs else 0
        if max_length == 0: return torch.empty(len(inputs),0) # Handle if all inputs are empty

        padded_data = []
        for t in inputs:
            if len(t) < max_length:
                pad_len = max_length - len(t)
                # Determine padding based on tokenizer's padding side
                # self.tokenizer.padding_side should be "right" or "left"
                pad_tuple = (0, pad_len) if self.tokenizer.padding_side == "right" else (pad_len, 0)
                t = F.pad(t, pad_tuple, mode='constant', value=value)
            padded_data.append(t)
        return torch.stack(padded_data)

    def collate_frames(self, batch_data: List[dict], collate_data: dict) -> tuple[List[dict], dict]:
        frames_list = []
        frame_len_list = []
        aux_frames_dict_of_list = defaultdict(list) # Use defaultdict

        valid_batch_data = [] # To store items that successfully provide frames or are frameless

        for data_item in batch_data:
            # n_frames should be present after preprocess
            num_item_frames = data_item.get("n_frames", 0) 
            
            if self.frames_key in data_item and num_item_frames > 0:
                frames = data_item.pop(self.frames_key)
                if torch.is_tensor(frames) and frames.nelement() > 0 : # Check if it's a non-empty tensor
                    frames_list.append(frames)
                    frame_len_list.append(num_item_frames) # Use n_frames from data_item
                elif not torch.is_tensor(frames): # Should not happen if preprocess is correct
                     logger.warning(f"Item {data_item.get('id','N/A')} frames_key exists but is not a tensor or is empty. Skipping frames.")
                     frame_len_list.append(0)
                else: # Is a tensor but empty
                    frame_len_list.append(0)
            elif self.frames_key not in data_item and num_item_frames == 0: # Frameless item
                 frame_len_list.append(0)
            # else: item might have n_frames > 0 but no frames_key, indicates an issue or dummy frame was intended
            # This case should be handled by ensuring dummy frames are tensors if preprocess sets them.
            # If frames_key had dummy tensor, it should have been caught by `is_tensor and nelement > 0`

            if self.enable_aux_frames:
                for aux_frame_key in self.aux_frame_keys:
                    if aux_frame_key in data_item and num_item_frames > 0 : # Only add aux if main frames are present
                        aux_frames_tensor = data_item.pop(aux_frame_key)
                        if torch.is_tensor(aux_frames_tensor) and aux_frames_tensor.nelement() > 0:
                             aux_frames_dict_of_list[aux_frame_key].append(aux_frames_tensor)
            
            valid_batch_data.append(data_item) # Add item for further processing

        if frames_list: # Only cat if there are actual frame tensors
            collate_data[self.frames_key] = torch.cat(frames_list, dim=0)
        else: # Handle cases with no valid frames across the batch
            # Create an empty tensor with expected shape if possible, or pass empty list
            # For example, (0, C, H, W) if channel/height/width are known
            # Or just an empty tensor: torch.empty((0,)) 
            # Based on dummy_frame_shape: (0, 3, 448, 448) -> (0, 3, 448, 448) means no batch dim initially
            # If dummy_frame_shape is (B, C, H, W), then (0, C, H, W)
            # The dummy_frame_shape is (0, C, H, W) where first 0 is n_frames for a single item
            # So for a batch, if no frames, it should be (0, C, H, W) still.
            # Example: self.dummy_frame_shape = (0, 3, 224, 224)
            # If frames_list is empty, result should be shape (0, 3, 224, 224)
            if self.dummy_frame_shape[0] == 0 and len(self.dummy_frame_shape) == 4: # (0, C, H, W)
                 collate_data[self.frames_key] = torch.empty((0, *self.dummy_frame_shape[1:]))
            else: # Fallback for other dummy_frame_shape structures or if it's not defined for batch
                 collate_data[self.frames_key] = torch.empty((0,))


        if self.enable_aux_frames:
            for aux_frame_key in self.aux_frame_keys:
                if aux_frames_dict_of_list[aux_frame_key]: # Check if list is not empty
                    collate_data[aux_frame_key] = torch.cat(aux_frames_dict_of_list[aux_frame_key], dim=0)
                else: # Similar empty tensor handling for aux frames
                    if self.dummy_frame_shape[0] == 0 and len(self.dummy_frame_shape) == 4:
                        collate_data[aux_frame_key] = torch.empty((0, *self.dummy_frame_shape[1:]))
                    else:
                        collate_data[aux_frame_key] = torch.empty((0,))


        collate_data["n_frames"] = torch.tensor(frame_len_list, dtype=torch.int60) # Ensure n_frames is a tensor

        return valid_batch_data, collate_data


    def batch_transform(self, batch_data: List[dict]):
        # Filter out None items that might have resulted from failed transform in a dataset's __getitem__
        # Or items that failed in self.transform if called per item before batch_transform
        # However, if self.transform is called *before* batching, this list would be already filtered.
        # This assumes batch_data is a list of successfully preprocessed (by self.transform) dicts.
        
        # If self.transform was called by dataset's __getitem__ and returned None,
        # those Nones need to be handled by the DataLoader's collate_fn or filtered before.
        # For this collate_fn, we assume batch_data contains valid dictionaries.
        
        collate_data = {}
        try:
            # The 'frames' key and aux_frame_keys should contain tensors of stacked frames from build_visual
            # The collate_frames method here is to collect these tensors from each item in the batch
            # and concatenate them along the batch dimension (dim=0 effectively for frames)
            # and also to gather 'n_frames' for each item.
            batch_data_after_frame_collate, collate_data = self.collate_frames(batch_data, collate_data)
        except Exception as e:
            logger.error(f"Error in collate_frames: {e}", exc_info=True)
            return None # Critical error in frame collation

        if not batch_data_after_frame_collate: # All items might have been filtered or had issues
            logger.warning("Batch data became empty after frame collation.")
            # Return minimal structure or None, depending on how downstream handles it
            # For now, let's return None if the batch is empty, as it usually indicates an issue.
            if not collate_data.get(self.frames_key, []): # Check if even dummy frames are not there
                return None


        # Collate text-based and meta features
        # Ensure all items in batch_data_after_frame_collate have the expected keys from text processing
        keys_to_collate_as_list = set(self.meta_keys)
        # Add other string keys that should be lists, not tensors
        if batch_data_after_frame_collate:
            for key in batch_data_after_frame_collate[0].keys():
                if isinstance(batch_data_after_frame_collate[0][key], str):
                    keys_to_collate_as_list.add(key)
        
        for key in list(keys_to_collate_as_list): # Iterate over a copy if modifying set
            if key in batch_data_after_frame_collate[0]: # Check if key exists
                 collate_data[key] = [data.pop(key) for data in batch_data_after_frame_collate]
            else: # Key might be missing if it was optional
                 keys_to_collate_as_list.remove(key)


        # Pad sequences for text data (input_ids, attention_mask, labels)
        if "input_ids" in batch_data_after_frame_collate[0]:
            input_ids = [data.pop("input_ids") for data in batch_data_after_frame_collate]
            input_ids_padded = self.padding_sequence(input_ids, value=self.pad_id)
            collate_data["input_ids"] = input_ids_padded
        
        if "attention_mask" in batch_data_after_frame_collate[0]:
            attention_mask = [data.pop("attention_mask") for data in batch_data_after_frame_collate]
            attention_mask_padded = self.padding_sequence(attention_mask, value=0) # Pad attention mask with 0
            collate_data["attention_mask"] = attention_mask_padded

        if "labels" in batch_data_after_frame_collate[0]:
            labels = [data.pop("labels") for data in batch_data_after_frame_collate]
            labels_padded = self.padding_sequence(labels, value=self.ignore_index)
            collate_data["labels"] = labels_padded
        
        # Collate any remaining data using default_collate (e.g., other numerical tensors)
        # Ensure batch_data_after_frame_collate only contains items that can be default_collate'd
        # Pop keys that were manually collated to avoid errors with default_collate
        remaining_keys_to_pop = set()
        if batch_data_after_frame_collate:
            for key_set in [keys_to_collate_as_list, {"input_ids", "attention_mask", "labels"}]:
                for key_to_check in key_set:
                     if key_to_check in batch_data_after_frame_collate[0]:
                          remaining_keys_to_pop.add(key_to_check)
        
        # This step is tricky: default_collate expects a list of dicts where each dict has same structure.
        # We've been popping keys. We need a clean list of dicts for default_collate.
        # Or, handle all collation manually.
        # For now, assume remaining items are collatable.
        # If batch_data_after_frame_collate items are now empty or have inconsistent keys, default_collate might fail.

        # Let's manually collate other known tensor types or numericals if possible
        # Example: if 'some_other_tensor' key exists
        # if batch_data_after_frame_collate and 'some_other_tensor' in batch_data_after_frame_collate[0]:
        #     other_tensors = [data.pop('some_other_tensor') for data in batch_data_after_frame_collate]
        #     collate_data['some_other_tensor'] = default_collate(other_tensors)

        # After specific collations, if batch_data_after_frame_collate items still have common tensor elements,
        # default_collate can handle them.
        # Filter out empty dicts from batch_data_after_frame_collate if popping made them empty
        cleaned_batch_data_for_default_collate = [d for d in batch_data_after_frame_collate if d]
        if cleaned_batch_data_for_default_collate:
            try:
                collate_data.update(default_collate(cleaned_batch_data_for_default_collate))
            except Exception as e:
                logger.warning(f"Could not default_collate remaining data: {e}. Remaining keys: "
                               f"{cleaned_batch_data_for_default_collate[0].keys() if cleaned_batch_data_for_default_collate else 'N/A'}")
        
        # Add fixed meta info
        collate_data['meta_keys'] = self.meta_keys # List of keys considered meta
        collate_data['vision_placeholder_index'] = self.vision_placeholder_index
        collate_data['vision_placeholder'] = self.vision_placeholder
        return collate_data