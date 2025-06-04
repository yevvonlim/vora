import argparse # For simple test script CLI arguments
from dataclasses import dataclass, field
import json # If your get_args_dict loads from JSON
import yaml # If your get_args_dict loads from YAML
from easydict import EasyDict as edict # Used in your main script
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Your custom module imports (ensure they are in PYTHONPATH)
from data_module.dataset import get_dataset # Assuming VoRADataset is returned by this
from data_module.processor import VoRAProcessor
from utils import logging # Your logging setup

logger = logging.get_logger("dataset_test")

# Simplified dataclasses for test configuration parsing (mirroring your structure)
@dataclass
class TestDataArguments:
    data: dict = field(default_factory=dict)
    # Add other top-level args your get_args_dict might produce if needed

@dataclass
class TestScriptArguments:
    config_path: str = field(metadata={"help": "Path to the YAML/JSON configuration file."})
    num_batches_to_test: int = field(default=2, metadata={"help": "Number of batches to fetch and inspect."})
    batch_size: int = field(default=4, metadata={"help": "Batch size for the DataLoader."})
    num_workers: int = field(default=0, metadata={"help": "Number of DataLoader workers."}) # Set to 0 for easier debugging

def load_config_from_file(config_path: str) -> dict:
    """Loads configuration from YAML or JSON file."""
    if config_path.endswith(".yaml") or config_path.endswith(".yml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith(".json"):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Configuration file must be YAML or JSON.")
    return config

def run_dataset_test(config: dict, script_args: TestScriptArguments):
    """
    Initializes and tests the dataset and processor.
    'config' is expected to be the dictionary structure your get_args_dict() would return.
    """
    logger.info("Starting Dataset Test Script...")
    logger.info(f"Testing with {script_args.num_batches_to_test} batches of size {script_args.batch_size}")

    # 1. Extract configurations for DataArguments and Processor
    #    This mirrors how your main script might structure them.
    #    The HfArgumentParser is not strictly necessary here if we manually map.
    
    # Assuming 'data' key in your config holds data-related sub-configurations
    data_config_dict = config.get("data", {})
    if not data_config_dict:
        logger.error("'data' section not found in the provided configuration.")
        return

    # Processor config: data.train.data_preprocess
    processor_config_dict = data_config_dict.get("train", {}).get("data_preprocess", {})
    if not processor_config_dict:
        logger.error("'data.train.data_preprocess' section not found for VoRAProcessor config.")
        return
    
    # Dataset paths config: data.train.data_fetch
    dataset_fetch_config_dict = data_config_dict.get("train", {}).get("data_fetch", {})
    if not dataset_fetch_config_dict or "data_paths" not in dataset_fetch_config_dict:
        logger.error("'data.train.data_fetch' section or 'data_paths' not found for VoRADataset config.")
        return

    # Ensure configs are edict if your classes expect them (as in main_training_function)
    processor_config = edict(processor_config_dict)
    dataset_fetch_config = edict(dataset_fetch_config_dict)

    # Add training=False to processor_config if not present, for testing
    if 'training' not in processor_config:
        processor_config.training = False # Typically for testing/eval, sampling might differ

    logger.info(f"VoRAProcessor Config: {processor_config}")
    logger.info(f"Dataset Fetch Config (data_paths): {dataset_fetch_config.data_paths}")

    # 2. Initialize VoRAProcessor
    try:
        processor = VoRAProcessor(**processor_config)
        logger.info("VoRAProcessor initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize VoRAProcessor: {e}", exc_info=True)
        return

    # 3. Initialize VoRADataset
    #    Your get_dataset function is used in the main script.
    try:
        # Ensure data_paths is a list, similar to your main script
        if not isinstance(dataset_fetch_config.data_paths, list):
            dataset_fetch_config.data_paths = [dataset_fetch_config.data_paths]
        
        # Assuming get_dataset returns an instance of VoRADataset or compatible
        dataset = get_dataset(
            data_paths=dataset_fetch_config.data_paths,
            processor=processor,
            # Add any other arguments get_dataset might expect from dataset_fetch_config
            # For example:
            # **{k: v for k, v in dataset_fetch_config.items() if k != 'data_paths'}
        )
        logger.info(f"Dataset initialized successfully. Total samples: {len(dataset)}")
        if len(dataset) == 0:
            logger.warning("Dataset is empty. Check your data_paths and annotation files.")
            return
    except Exception as e:
        logger.error(f"Failed to initialize Dataset: {e}", exc_info=True)
        return

    # 4. Create DataLoader
    # For a simple test, use RandomSampler or SequentialSampler if not distributed.
    # The custom distributed samplers are for XLA multi-process training.
    # If `shuffle` is True, DataLoader with RandomSampler is common.
    # If you want to test specific items, use SequentialSampler and then iterate.
    
    # Using SequentialSampler for predictable testing order if needed, or RandomSampler
    # For this test, let's allow shuffling for a more general test.
    # If you need to test specific known bad items, use SequentialSampler and iterate to that index.
    sampler = RandomSampler(dataset) if script_args.batch_size <= len(dataset) else SequentialSampler(dataset)

    try:
        dataloader = DataLoader(
            dataset,
            batch_size=script_args.batch_size,
            sampler=sampler,
            collate_fn=processor.batch_transform, # This is your custom collate function
            num_workers=script_args.num_workers,
            pin_memory=False # Usually True with GPUs, can be False for CPU/TPU host
        )
        logger.info("DataLoader created successfully.")
    except Exception as e:
        logger.error(f"Failed to create DataLoader: {e}", exc_info=True)
        return

    # 5. Iterate and Test a few batches
    logger.info(f"Fetching and inspecting {script_args.num_batches_to_test} batches...")
    batches_tested = 0
    try:
        for i, batch in enumerate(dataloader):
            if i >= script_args.num_batches_to_test:
                break
            
            logger.info(f"\n--- Batch {i+1} ---")
            if batch is None:
                logger.warning(f"Batch {i+1} is None (processor.batch_transform might have returned None). Skipping.")
                continue

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    logger.info(f"  Key: '{key}', Type: Tensor, Shape: {value.shape}, Dtype: {value.dtype}")
                elif isinstance(value, list):
                    logger.info(f"  Key: '{key}', Type: List, Length: {len(value)}")
                    if value: # If list is not empty, print type of first element
                        logger.info(f"    List item 0 type: {type(value[0])}")
                        if isinstance(value[0], torch.Tensor):
                             logger.info(f"    List item 0 tensor shape: {value[0].shape}")

                else:
                    logger.info(f"  Key: '{key}', Type: {type(value)}")
            
            # Example: Check 'input_ids'
            if "input_ids" in batch and isinstance(batch["input_ids"], torch.Tensor):
                logger.info(f"  Sample input_ids[0]: {batch['input_ids'][0, :20]}...") # Print first 20 tokens of first sample
            if "frames" in batch and isinstance(batch["frames"], torch.Tensor):
                 logger.info(f"  Frames tensor min: {batch['frames'].min()}, max: {batch['frames'].max()}, mean: {batch['frames'].mean()}")

            batches_tested += 1
            logger.info("--- End of Batch ---")

    except Exception as e:
        logger.error(f"Error during DataLoader iteration or batch inspection: {e}", exc_info=True)
        logger.info("This error might indicate an issue in VoRADataset.__getitem__, "
                    "VoRAProcessor.transform, or VoRAProcessor.batch_transform (collate_fn).")
    
    if batches_tested > 0:
        logger.info(f"Successfully fetched and inspected {batches_tested} batches.")
    else:
        logger.warning("No batches were successfully processed. Check logs for errors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test VoRA Dataset and Processor.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the YAML/JSON configuration file.")
    parser.add_argument("--num_batches_to_test", type=int, default=2, help="Number of batches to test.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for testing.") # Smaller default for testing
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader num_workers.") # Default 0 for easier debugging
    
    cli_args = parser.parse_args()
    
    # Load the main configuration file (this would be your equivalent of get_args_dict())
    # For this example, it's loaded from a file specified by --config_path
    try:
        main_config = load_config_from_file(cli_args.config_path)
        if not main_config:
            raise ValueError("Configuration loaded as empty.")
    except Exception as e:
        logger.error(f"Failed to load or parse configuration from {cli_args.config_path}: {e}")
        exit(1)

    # Create TestScriptArguments instance
    test_script_args = TestScriptArguments(
        config_path=cli_args.config_path,
        num_batches_to_test=cli_args.num_batches_to_test,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers
    )

    # Run the test
    run_dataset_test(config=main_config, script_args=test_script_args)