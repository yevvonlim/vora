import copy
import json

from torch.utils.data import Dataset

from utils import logging


logger = logging.get_logger(__name__)

import json
import copy
import io
import os # Still useful for some path manipulations even if not primary I/O
from torch.utils.data import Dataset
from google.cloud import storage # For reading annotation files from GCS in __init__
# Assuming 'logger' is a pre-configured logger object
# from transformers.utils import logging
# logger = logging.get_logger(__name__)

class VoRADataset(Dataset):
    def __init__(self, data_paths, processor) -> None:
        self.processor = processor
        self.anns = []
        self._datasets_length = []
        self._modality_group_indices = []
        
        if not hasattr(processor, 'frames_key'):
            # Or define a default if appropriate: self.frame_key = getattr(processor, 'frames_key', 'frames')
            raise ValueError("Processor object must have a 'frames_key' attribute.")
        self.frame_key = processor.frames_key
        
        # GCS client for reading annotation files during initialization.
        # Media file reading will be handled by the processor.
        self.init_gcs_client = None # For __init__ phase GCS access

        prefix_length = 0
        index = 0
        text_indices, image_indices, video_indices = [], [], []

        # TODO: Consider using multithreading/multiprocessing for annotation file reading
        # if it becomes a bottleneck (many small annotation files).
        for data_spec in data_paths:
            image_folder = data_spec["image_folder"] # This is the GCS or local URI for media
            anno_path = data_spec["anno_path"]
            lines_to_process = []

            if anno_path.startswith("gs://"):
                try:
                    # print(f"[INFO] Attempting to read annotation file from GCS: {anno_path}")
                    if self.init_gcs_client is None:
                        self.init_gcs_client = storage.Client()
                    
                    path_parts = anno_path[5:].split("/", 1)
                    bucket_name, blob_name = path_parts[0], (path_parts[1] if len(path_parts) > 1 else "")

                    if not blob_name:
                        print(f"[ERROR] Invalid GCS path for annotation (missing object name): {anno_path}")
                        continue

                    bucket = self.init_gcs_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    content_bytes = blob.download_as_bytes()
                    with io.TextIOWrapper(io.BytesIO(content_bytes), encoding='utf-8') as f_mem:
                        lines_to_process = f_mem.readlines()
                    # print(f"[INFO] Successfully read {len(lines_to_process)} lines from GCS: {anno_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to read GCS annotation file {anno_path}: {e}")
                    continue
            else: # Local path for annotations
                try:
                    # print(f"[INFO] Attempting to read annotation file from local path: {anno_path}")
                    with open(anno_path, "r", encoding='utf-8') as f:
                        lines_to_process = f.readlines()
                    # print(f"[INFO] Successfully read {len(lines_to_process)} lines from local path: {anno_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to read local annotation file {anno_path}: {e}")
                    continue

            for line_number, line in enumerate(lines_to_process):
                try:
                    item = json.loads(line)
                except Exception as e:
                    print(f"[WARNING] Skipping line {line_number+1} in {anno_path} due to JSON parsing error: {e}. Line: '{line.strip()}'")
                    continue
                
                item["image_folder"] = image_folder # Store original image_folder (can be GCS or local)
                self.anns.append(item)
                
                num_frames = len(item.get(self.frame_key, []))
                if num_frames == 0:
                    text_indices.append(index)
                elif num_frames == 1:
                    image_indices.append(index)
                else: # num_frames > 1
                    video_indices.append(index)
                index += 1
            
            current_dataset_length = len(self.anns) - prefix_length
            self._datasets_length.append(current_dataset_length)
            prefix_length = len(self.anns)
            print(f"[INFO] Loaded {current_dataset_length} annotations from {anno_path}.")

        for indices_group in (text_indices, image_indices, video_indices):
            if len(indices_group) > 0:
                self._modality_group_indices.append(indices_group)
        
        print(f"[INFO] VoRADataset initialized. Total annotations: {len(self.anns)}")

    def __len__(self):
        return len(self.anns)

    @property
    def datasets_length(self):
        return self._datasets_length

    @property
    def modality_group_indices(self):
        return self._modality_group_indices

    def __getitem__(self, idx: int):
        # Fetches the annotation metadata. This item contains GCS URIs for media.
        item_annotation = copy.deepcopy(self.anns[idx])
        
        # The processor.transform method is now responsible for:
        # 1. Interpreting item_annotation["image_folder"] and item_annotation[self.frame_key] (list of filenames).
        # 2. If item_annotation["image_folder"] is a GCS URI, constructing full GCS paths.
        # 3. Efficiently reading these files *directly from GCS* (e.g., using tf.io.gfile).
        # 4. Decoding and transforming them into tensors.
        try:
            output = self.processor.transform(item_annotation)
            
            if output is None:
                print(f"[WARNING] Item at index {idx} (image_folder: {item_annotation.get('image_folder', 'N/A')}) "
                                      f"returned None from processor. Trying next item.")
                # Safely get next index to avoid infinite recursion on last item if it always fails
                next_idx = (idx + 1) % len(self) if len(self) > 0 else idx 
                if next_idx == idx and len(self) == 1 : # Avoid infinite loop for single-item dataset failing
                    raise RuntimeError(f"Single item dataset at index {idx} continuously fails processing.")
                return self.__getitem__(next_idx)
            return output
        except Exception as e:
            print(f"[ERROR] Error during processor.transform for item {idx} (image_folder: {item_annotation.get('image_folder', 'N/A')}): {e}. Trying next item.")
            next_idx = (idx + 1) % len(self) if len(self) > 0 else idx
            if next_idx == idx and len(self) == 1:
                 raise RuntimeError(f"Single item dataset at index {idx} continuously fails in transform with error: {e}")
            return self.__getitem__(next_idx)

def get_dataset(data_paths, processor):
    return VoRADataset(data_paths, processor)
