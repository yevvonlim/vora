import math
import sys
from typing import List, Tuple, Union

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import Compose
from transformers import AutoImageProcessor
from google.cloud import storage # For GCS
import tempfile
import os
import shutil # Not using shutil.rmtree, using tempfile.TemporaryDirectory object
import torch # Added for type hinting and 'pt' tensor return

class HFImageTransform:
    def __init__(self, path: str):
        """
        Initializes an image transformer by loading a Hugging Face AutoImageProcessor.
        Supports local paths, Hugging Face Hub model IDs, or GCS paths.

        Args:
            path (str): Path to the model/processor files.
                        - Local directory path (e.g., "./my_processor")
                        - Hugging Face Hub model ID (e.g., "google/vit-base-patch16-224-in21k")
                        - GCS directory path (e.g., "gs://bucket-name/path/to/processor_files/")
        """
        self._temp_dir_obj = None  # Stores the TemporaryDirectory object if a GCS path is used

        if path.startswith("gs://"):
            print(f"Attempting to load image processor from GCS path: {path}")
            # tempfile.TemporaryDirectory can be used as a context manager,
            # or if the object is retained, it automatically cleans up the directory upon object destruction.
            self._temp_dir_obj = tempfile.TemporaryDirectory()
            local_download_path = self._temp_dir_obj.name
            path_to_load = local_download_path

            try:
                # Parse GCS path
                parts = path[5:].split("/", 1)
                bucket_name = parts[0]
                gcs_prefix = ""
                if len(parts) > 1:
                    gcs_prefix = parts[1]
                
                # Ensure gcs_prefix ends with '/' to act like a directory
                if gcs_prefix and not gcs_prefix.endswith('/'):
                    gcs_prefix += '/'

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                
                print(f"Downloading files from GCS bucket '{bucket_name}' prefix '{gcs_prefix}'...")
                blobs = list(bucket.list_blobs(prefix=gcs_prefix)) # Convert to list to check if any blobs were found

                if not blobs:
                    # self._temp_dir_obj.cleanup() # Clean up temporary directory
                    raise FileNotFoundError(f"No files found at GCS path: {path} (or permission issue)")

                downloaded_any_file = False
                for blob in blobs:
                    # Skip empty objects representing GCS "folders" themselves
                    # (name is the same as prefix or it's an empty file ending with '/')
                    if blob.name == gcs_prefix and blob.size == 0:
                        continue
                    if blob.name.endswith('/') and blob.size == 0: # Skip objects representing folders
                        continue

                    # Determine the relative path within the temporary directory
                    # E.g.: gcs_prefix="path/to/processor/", blob.name="path/to/processor/config.json"
                    # relative_blob_path = "config.json"
                    relative_blob_path = blob.name[len(gcs_prefix):]
                    
                    # If blob.name is the same as gcs_prefix itself (i.e., gcs_prefix points to a single file),
                    # relative_blob_path might be empty. In this case, use only the file name.
                    if not relative_blob_path:
                         relative_blob_path = os.path.basename(blob.name)

                    local_file_path = os.path.join(local_download_path, relative_blob_path)

                    # Create necessary subdirectories
                    local_file_dir = os.path.dirname(local_file_path)
                    if not os.path.exists(local_file_dir):
                        os.makedirs(local_file_dir, exist_ok=True)

                    print(f"  '{blob.name}' -> '{local_file_path}'")
                    blob.download_to_filename(local_file_path)
                    downloaded_any_file = True
                
                if not downloaded_any_file:
                    # self._temp_dir_obj.cleanup()
                    raise FileNotFoundError(f"No valid files found at GCS path: {path}. "
                                            "Ensure the path points to a 'directory' containing processor files.")

            except Exception as e:
                # Clean up temporary directory if an error occurred (if self._temp_dir_obj was created)
                if self._temp_dir_obj:
                    self._temp_dir_obj.cleanup()
                raise RuntimeError(f"Error loading processor from GCS path '{path}': {e}")
        else:
            print(f"Loading image processor from local or Hugging Face Hub path: {path}")
            path_to_load = path

        # `trust_remote_code=True` is from the original code and requires caution for security.
        # It's used when loading models/processors with custom code from the Hugging Face Hub.
        self.image_processor = AutoImageProcessor.from_pretrained(path_to_load, trust_remote_code=True)
        print(f"ImageProcessor loaded successfully from '{path_to_load}'.")

        # If loaded from GCS, self._temp_dir_obj will be automatically cleaned up
        # when the HFImageTransform instance is garbage collected. Explicit cleanup() can also be called.

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Takes a PIL image as input and returns a preprocessed PyTorch tensor.
        """
        # Most image processors expect RGB images.
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # The `image_processor` typically returns a dictionary,
        # with the 'pixel_values' key containing a batch-shaped tensor.
        # Since we are processing a single image here, we take the first (and only) item.
        # (batch_size, num_channels, height, width) -> (num_channels, height, width)
        processed_image = self.image_processor(images=image, return_tensors='pt')['pixel_values'][0]
        return processed_image

    def cleanup_temp_dir(self):
        """
        Explicitly cleans up the temporary directory created when loading from GCS.
        Usually, it's cleaned up automatically when the object is destroyed, but can be called if needed.
        """
        if self._temp_dir_obj:
            print(f"Cleaning up temporary directory: {self._temp_dir_obj.name}")
            self._temp_dir_obj.cleanup()
            self._temp_dir_obj = None

    # Using __del__ for automatic temporary directory cleanup upon object destruction
    # is also possible, but the tempfile.TemporaryDirectory object already provides this.
    # def __del__(self):
    #     self.cleanup_temp_dir()

class PILToNdarray:
    def __init__(self):
        pass

    def __call__(self, image: Image.Image):
        image_array = np.array(image)
        return image_array.astype(np.float32)


class Rescale:
    def __init__(self, rescale_factor: float):
        self.scale = rescale_factor

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError("Input must be a numpy array.")
        return image * self.scale


class PILExpand2Square:
    def __init__(self, background_color=(0, 0, 0)):
        self.background_color = background_color

    def expand2square(self, pil_img: Image.Image):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), self.background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), self.background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def __call__(self, image: Image.Image):
        return self.expand2square(image)


class ResizeWithAspectRatio:
    def __init__(self, size: Union[List[int], Tuple[int, int]], padding_value: float = 0, resampling=Image.BILINEAR):
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise ValueError("Size must be a list or tuple with two elements: (height, width).")
        self.target_size = size
        self.padding_value = padding_value
        self.resampling = resampling

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError("Input must be a numpy array.")

        original_height, original_width = image.shape[:2]
        target_height, target_width = self.target_size

        # Calculate the new size while maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image = np.array(Image.fromarray(image.astype(np.uint8)).resize((new_width, new_height), self.resampling))

        # Create a new image with the target size and padding value
        new_image = np.full((target_height, target_width, image.shape[2]), self.padding_value, dtype=np.float32)

        # Place the resized image in the center
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        new_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        return new_image.astype(np.float32)


def smart_resize(
    height: int, width: int, factor: int = 14, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 50 * 80
):
    """Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.

    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class PILSmartResize:
    def __init__(
        self,
        patch_size: int = 14,
        merge_size: int = 1,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 50 * 80,
        resampling: int = Image.Resampling.BICUBIC,
    ):
        self.patch_size = patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.resampling = resampling

    def __call__(self, image: Image.Image) -> Image.Image:
        height, width = image.height, image.width
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=self.patch_size * self.merge_size,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        image = image.resize((resized_width, resized_height), resample=self.resampling)
        return image


def create_transform(ops_cfg):
    current_module = sys.modules[__name__]
    transform_list = []
    for op in ops_cfg:
        kwargs = ops_cfg[op]
        if hasattr(T, op):
            transform_list.append(getattr(T, op)(**kwargs))
        elif hasattr(current_module, op):
            transform_list.append(getattr(current_module, op)(**kwargs))
        else:
            raise RuntimeError(f'no op {op} in torchvision.transforms and data.processors.image_transform')

    return Compose(transform_list)


class VisionProcessor:
    def __init__(self, ops):
        self.transform = create_transform(ops)

    def __call__(self, data):
        if isinstance(data, list):
            return [self.transform(_data) for _data in data]
        return self.transform(data)
