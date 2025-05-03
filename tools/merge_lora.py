import os

import torch

from models.modeling_vora import VoRAForCausalLM, VoRAConfig
from utils import logging
from tqdm.auto import tqdm


logger = logging.get_logger(__name__)


def key_mapping(state_dict, key_mapping_dict):
    new_state_dict = dict()
    for k, v in state_dict.items():
        flag = 0
        for prev_key in key_mapping_dict.keys():
            if prev_key in k:
                new_state_dict[k.replace(prev_key, key_mapping_dict[prev_key])] = v
                flag = 1
                break
        if flag == 0:
            new_state_dict[k] = v
    return new_state_dict


def merge_lora(checkpoint, lora_key="lora_A"):
    new_state_dict = {}
    lora_processed = set()

    for key in tqdm(list(checkpoint.keys()), desc="Merging LoRA weights"):
        if lora_key in key:
            try:
                idx = key.index(lora_key)
            except ValueError:
                continue
            root_key = key[:idx]
            suffix = key[idx + len(lora_key):]
            
            if not suffix.startswith('.'):
                continue

            weight_key = f"{root_key}weight"
            lora_A_key = f"{root_key}lora_A.weight"
            lora_B_key = f"{root_key}lora_B.weight"
            bias_key = f"{root_key}bias"  # 新增：显式处理 bias
            
            if weight_key in lora_processed:
                continue
            lora_processed.update({weight_key, lora_A_key, lora_B_key})
            
            if any(k not in checkpoint for k in [weight_key, lora_A_key, lora_B_key]):
                raise KeyError(f"Missing keys for module {root_key}")
            
            W = checkpoint[weight_key]
            A = checkpoint[lora_A_key]
            B = checkpoint[lora_B_key]
            new_state_dict[weight_key] = W + B @ A
            
            if bias_key in checkpoint:
                new_state_dict[bias_key] = checkpoint[bias_key]
                lora_processed.add(bias_key) 

    for key, value in checkpoint.items():
        if key not in lora_processed:
            new_state_dict[key] = value

    return new_state_dict

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm


def _merge_single(task):
    """하나의 root_key에 대해 W + B @ A (+ bias) 계산"""
    root_key, W, A, B, bias = task
    merged = { f"{root_key}weight": W + B @ A }
    if bias is not None:
        merged[f"{root_key}bias"] = bias
    return merged

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

def merge_lora_parallel_threads(checkpoint, lora_key="lora_A", num_workers=None):
    # 1) 작업 리스트 준비
    tasks = []
    processed = set()
    for k in tqdm(checkpoint.keys(), desc="Preparing tasks for LoRA merging"):
        if lora_key in k:
            idx = k.index(lora_key)
            root = k[:idx]
            w_key = f"{root}weight"
            a_key = f"{root}lora_A.weight"
            b_key = f"{root}lora_B.weight"
            bias_key = f"{root}bias"

            if w_key in processed:
                continue
            if any(x not in checkpoint for x in (w_key, a_key, b_key)):
                raise KeyError(f"Missing key for module {root}")

            W = checkpoint[w_key]
            A = checkpoint[a_key]
            B = checkpoint[b_key]
            bias = checkpoint.get(bias_key)

            processed.update({w_key, a_key, b_key, bias_key})
            tasks.append((root, W, A, B, bias))

    # 2) Thread-based 병렬 병합
    new_state_dict = {}
    max_workers = num_workers or os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_merge_single, t): t[0] for t in tasks}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Merging LoRA weights (threads)"):
            new_state_dict.update(future.result())

    # 3) 나머지 키 복사
    for k, v in checkpoint.items():
        if k not in processed:
            new_state_dict[k] = v

    return new_state_dict


def merge_lora_parallel(checkpoint, lora_key="lora_A", num_workers=None):
    # 1) 작업 리스트 준비
    tasks = []
    processed = set()
    for full_key in tqdm(checkpoint, desc="Preparing tasks for LoRA merging"):
        if lora_key in full_key:
            idx = full_key.index(lora_key)
            root = full_key[:idx]
            w_key = f"{root}weight"
            a_key = f"{root}lora_A.weight"
            b_key = f"{root}lora_B.weight"
            bias_key = f"{root}bias"

            # 이미 처리된 모듈이면 건너뛰기
            if w_key in processed:
                continue

            # 필수 키 존재 확인
            for k in (w_key, a_key, b_key):
                if k not in checkpoint:
                    raise KeyError(f"Missing key {k} for module {root}")

            W, A, B = checkpoint[w_key], checkpoint[a_key], checkpoint[b_key]
            bias = checkpoint.get(bias_key, None)

            processed.update({w_key, a_key, b_key, bias_key})
            tasks.append((root, W, A, B, bias))

    # 2) 병렬 병합
    new_state_dict = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = { executor.submit(_merge_single, t): t[0] for t in tasks }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Merging LoRA weights"):
            result = future.result()
            new_state_dict.update(result)

    # 3) 나머지 키들 복사
    for k, v in checkpoint.items():
        if k not in processed:
            new_state_dict[k] = v

    return new_state_dict

def partial_load_from_checkpoints(
        local_checkpoint_path,
        ckpt_rename_parameters=None,
        map_location="cpu",
        model=None,
        valid_prefix=None,
        lazy_load=False
    ):
    
    ckpt_rename_parameters = ckpt_rename_parameters or dict()
    if os.path.isdir(local_checkpoint_path):
        from safetensors.torch import load
        import multiprocessing
        checkpoint = {}
        files = [file for file in os.listdir(local_checkpoint_path) if file.endswith(".safetensors")]
        if len(files) == 0:
            raise ValueError(f"No safetensors file found in {local_checkpoint_path}")
        file_paths = []
        for file in files:
            file_path = os.path.join(local_checkpoint_path, file)
            if not lazy_load:
                print(f"loading checkpoint from {file_path}")
                with open(file_path, "rb") as f:
                    data = f.read()
                loaded = load(data)
                checkpoint.update(loaded)
            else:
                file_paths.append(file_path)
        if lazy_load:
            return file_paths
    else:
        checkpoint = torch.load(local_checkpoint_path, map_location=map_location)

    if "state_dict" in checkpoint:
        logger.info("partial loading checkpoint")
        state_dict = checkpoint["state_dict"]
    elif "module" in checkpoint:
        # for ds zero2 checkpoint
        logger.info("partial loading deepspeed zero2 checkpoint")
        state_dict = checkpoint["module"]
        ckpt_rename_parameters.update({"module.": ""})
    else:
        state_dict = checkpoint

    if valid_prefix:
        new_state_dict = dict()
        for k, v in state_dict.items():
            for prefix in valid_prefix:
                if k.startswith(prefix):
                    new_state_dict[k] = v
        state_dict = new_state_dict
    state_dict = key_mapping(state_dict, ckpt_rename_parameters)
    return state_dict


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()

    config_path = args.config
    checkpoint_path = args.checkpoint
    save_path = args.save_dir

    with open(config_path, "r") as f:
        vora_config = yaml.safe_load(f)["model"]
    vora_config["lora"]["r"] = -1
    config = VoRAConfig(**vora_config)

    model = VoRAForCausalLM._from_config(config=config)

    state_dict = partial_load_from_checkpoints(checkpoint_path)
    state_dict = merge_lora_parallel_threads(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.save_pretrained(save_path)
