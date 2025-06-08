from dataclasses import dataclass, field
from functools import partial
import os
from typing import Dict, Optional, List # Added List

import datasets
from easydict import EasyDict as edict
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler # Added Sampler

import transformers
from transformers import Trainer, HfArgumentParser # HfArgumentParser was already there
from transformers.trainer import is_datasets_available

# PyTorch XLA imports
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
import torch_xla.distributed.parallel_loader as pl

# Fix for gradient checkpointing with XLA: Register XLA in torch namespace
# This is needed because torch.utils.checkpoint tries to access torch.xla directly
# if not hasattr(torch, 'xla'):
#     torch.xla = torch_xla

# import torch_xla.runtime as xr # Import if you prefer xr.world_size() etc.

# Your custom module imports
from data_module.dataset import get_dataset
from data_module.processor import VoRAProcessor
# Assuming your distributed samplers are in data_module.distributed_sampler
from data_module.distributed_sampler import DistributedGlobalGroupRandomSampler, DistributedGroupRandomSampler
from models.configuration_vora import VoRAConfig
from models.modeling_vora import VoRAForCausalLM
from utils import logging # Assuming this is transformers.utils.logging or compatible
from utils.training_utils import AdditionalState, MultiTaskModuleMixin
from utils.parser_utils import get_args_dict


# Import necessary libraries from transformers and PyTorch
from transformers import TrainerCallback, Trainer
from transformers.utils import is_torch_xla_available

# Conditionally import PyTorch/XLA specific modules.
class XlaOptimizerStepCallback(TrainerCallback):
    """
    A custom TrainerCallback to replace the optimizer's step method
    with the PyTorch/XLA equivalent, xm.optimizer_step.

    This version uses the on_step_begin hook to guarantee the optimizer object
    is available and uses a flag to ensure the patch is applied only once.
    """
    def __init__(self):
        super().__init__()
        # Add a stateful flag to ensure the patch is applied only once.
        self._patched = False

    def on_step_begin(self, args, state, control, **kwargs):
        """
        This method is called at the beginning of every step.
        The optimizer is available in kwargs here.
        """
        # Check if the patch has already been applied. If so, do nothing.
        if self._patched:
            return

        # The optimizer object is passed via the kwargs dictionary in on_step_begin.
        optimizer = kwargs.get("optimizer")
        
        # Check if the optimizer exists and if we are in an XLA environment.
        if optimizer is not None and is_torch_xla_available():
            # Use xm.master_print to log only on the master process (rank 0).
            xm.master_print("Callback invoked (on_step_begin): Applying XLA optimizer step patch...")
            
            # This is the core of the solution: monkey-patching the optimizer's step method.
            # We replace the default step() with a lambda function that calls the XLA-specific optimizer step.
            optimizer.step = lambda: xm.optimizer_step(optimizer)
            
            # Set the flag to True so this logic doesn't run again on subsequent steps.
            self._patched = True
        else:
            # If the optimizer is None or XLA is not available, log a warning.
            xm.master_print("Callback invoked (on_step_begin): No optimizer found or XLA not available. Skipping patch.")

logger = logging.get_logger("trainer")
# Removed global processor variable, it should be instantiated within the training function

# --- Helper Functions and Dataclasses ---
def _patching_module_base(module: nn.Module, additional_state: AdditionalState):
    if isinstance(module, nn.Module) \
            and hasattr(module, 'supports_report_metrics') \
            and module.supports_report_metrics \
            and MultiTaskModuleMixin not in module.__class__.__bases__:
        module.__class__.__bases__ = module.__class__.__bases__ + (MultiTaskModuleMixin,)
        module.report_metrics = partial(module.report_metrics, additional_state)

@dataclass
class ModelArguments:
    model: Optional[dict] = field(default_factory=dict)

@dataclass
class DataArguments:
    data: Optional[dict] = field(default_factory=dict)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    visual_encoder_lr_scale: float = field(default=1.0)
    group_by_data_source: bool = field(default=False)
    group_by_data_modality: bool = field(default=False)
    using_torch_lr: bool = field(default=False)
    lr_type: str = field(default="")
    shuffle: bool = field(default=True, metadata={"help": "Whether to shuffle the training data (used by default samplers)."})
    wandb_project: str = field(default="VoRA")
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "When training on TPUs, the number of XLA cores to use. Defaults to XLA auto-detection on the host."}
    )

# --- Custom Trainer ---
class VoRATrainer(Trainer):
    def __init__(self, model: nn.Module, args: TrainingArguments, **kwargs):
        # It's generally safer to let Trainer initialize Accelerator first
        super().__init__(model=model, args=args, **kwargs)
        # Initialize additional_state after super().__init__ so accelerator is available if needed
        self.additional_state = AdditionalState(args) # args are TrainingArguments
        if model is not None:
            report_patching = partial(_patching_module_base, additional_state=self.additional_state)
            model.apply(report_patching)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # No changes to this method, assuming it's working as intended.
        # Just ensuring epoch handling is robust if it's already in logs.
        current_epoch_in_logs = logs.get("epoch")
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2) # Trainer uses rounded epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        additional_logs = {}
        if hasattr(self, 'additional_state') and self.additional_state is not None:
             additional_logs = self.additional_state.pop_metrics(gather_func=self._nested_gather)

        # Preserve original epoch if it was passed in, otherwise use state's epoch
        original_epoch_key_present = 'epoch' in logs
        passed_epoch_value = logs.pop('epoch', None) # Remove to prevent duplication if additional_logs has it

        logs.update(additional_logs)

        if original_epoch_key_present: # If 'epoch' was in logs or added from state
            logs['epoch'] = passed_epoch_value if passed_epoch_value is not None else current_epoch_in_logs
        elif self.state.epoch is not None: # If not in logs at all, but state has it
             logs['epoch'] = round(self.state.epoch,2)


        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _get_train_sampler(self) -> Optional[Sampler[int]]:
        if not hasattr(self.train_dataset, "__len__"):
            logger.info(f"[Rank {self.accelerator.process_index}] Dataset has no __len__, cannot use samplers. Returning None.")
            return None
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset):
            logger.info(f"[Rank {self.accelerator.process_index}] Dataset is Iterable. Returning None for sampler.")
            return None

        # Use self.accelerator for num_replicas and rank
        num_replicas = self.accelerator.num_processes
        rank = self.accelerator.process_index

        if self.args.group_by_data_source:
            if not hasattr(self.train_dataset, 'datasets_length'):
                logger.warning(
                    f"[Rank {rank}] train_dataset lacks 'datasets_length' for DistributedGroupRandomSampler. "
                    "Falling back to default Trainer sampler."
                )
                return super()._get_train_sampler() # Fallback to default
            
            logger.info(f"[Rank {rank}] Using DistributedGroupRandomSampler.")
            return DistributedGroupRandomSampler(
                data_source=self.train_dataset, # type: ignore
                lengths=self.train_dataset.datasets_length, # type: ignore
                num_replicas=num_replicas,
                rank=rank,
                shuffle=self.args.shuffle,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last
            )
        elif self.args.group_by_data_modality:
            if not hasattr(self.train_dataset, 'modality_group_indices'):
                logger.warning(
                    f"[Rank {rank}] train_dataset lacks 'modality_group_indices' for DistributedGlobalGroupRandomSampler. "
                    "Falling back to default Trainer sampler."
                )
                return super()._get_train_sampler() # Fallback to default

            logger.info(f"[Rank {rank}] Using DistributedGlobalGroupRandomSampler.")
            # per_replica_batch_size is the train_batch_size from TrainingArguments
            # as Trainer already makes it per-device.
            per_replica_bs = self.args.train_batch_size
            return DistributedGlobalGroupRandomSampler(
                per_replica_batch_size=per_replica_bs,
                modality_group_indices=self.train_dataset.modality_group_indices, # type: ignore
                num_replicas=num_replicas,
                rank=rank,
                shuffle=self.args.shuffle,
                seed=self.args.seed,
                drop_last=self.args.dataloader_drop_last
            )
        else:
            # Fallback to default Hugging Face Trainer sampler logic
            # which handles DistributedSampler, RandomSampler, or SequentialSampler correctly.
            logger.info(f"[Rank {rank}] No custom grouping. Using default Trainer sampler logic.")
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        # Get the sampler instance using the overridden _get_train_sampler method
        train_sampler = self._get_train_sampler()

        dataloader_params = {
            "batch_size": self._train_batch_size, # This is per-device batch size
            "sampler": train_sampler,
            "collate_fn": data_collator,
            "drop_last": self.args.dataloader_drop_last,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
             # Shuffle is False if a sampler is provided, True otherwise (if args.shuffle is True)
            "shuffle": (train_sampler is None and self.args.shuffle),
        }
        dataloader = pl.MpDeviceLoader(DataLoader(train_dataset, **dataloader_params), xm.xla_device())
        logger.info(f"[Rank {self.accelerator.process_index}] Preparing DataLoader with Accelerator.")
        return self.accelerator.prepare(dataloader)
import os
import tempfile # For temporary directory
from google.cloud import storage # For GCS upload
# Assuming VoRATrainer and logger are defined as in your context
# from .trainer import VoRATrainer, logger # Adjust import based on your file structure

def safe_save_model_for_hf_trainer(trainer: VoRATrainer, output_dir: str):
    # Check if saving should occur (typically only on the main process)
    if not trainer.args.should_save:
        logger.info(f"[Rank {trainer.accelerator.process_index}] Non-main process. Skipping model saving.")
        return

    logger.info(f"[Rank {trainer.accelerator.process_index}] Main process attempting to save model to: {output_dir}")

    if trainer.is_deepspeed_enabled:
        # DeepSpeed integrated with Hugging Face Accelerate/Trainer is expected
        # to handle GCS paths correctly if gcsfs is installed.
        logger.info(f"[Rank {trainer.accelerator.process_index}] DeepSpeed enabled. Using trainer.save_model() for GCS path: {output_dir}")
        try:
            trainer.save_model(output_dir)
            logger.info(f"[Rank {trainer.accelerator.process_index}] DeepSpeed model saved successfully to {output_dir}")
        except Exception as e:
            logger.error(f"[Rank {trainer.accelerator.process_index}] Error saving DeepSpeed model to {output_dir}: {e}", exc_info=True)
        return

    # --- Non-DeepSpeed Path ---
    # First, get the state_dict on CPU
    logger.info(f"[Rank {trainer.accelerator.process_index}] Preparing state_dict on CPU for non-DeepSpeed save.")
    state_dict = trainer.model.state_dict()
    cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
    del state_dict # Free up memory

    if output_dir.startswith("gs://"):
        # Save to a temporary local directory first, then upload to GCS
        gcs_client = None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                logger.info(f"[Rank {trainer.accelerator.process_index}] Saving model to temporary local directory: {tmpdir}")
                # The trainer._save method saves the state_dict (e.g., pytorch_model.bin)
                # and also other necessary files like config.json, tokenizer files, etc.
                trainer._save(tmpdir, state_dict=cpu_state_dict)
                logger.info(f"[Rank {trainer.accelerator.process_index}] Temporary local save complete. Starting GCS upload to {output_dir}")

                gcs_client = storage.Client()
                # Parse GCS path: gs://bucket-name/path/to/output_dir
                bucket_name, gcs_prefix = output_dir[5:].split("/", 1)
                bucket = gcs_client.bucket(bucket_name)

                for root, _, files in os.walk(tmpdir):
                    for filename in files:
                        local_filepath = os.path.join(root, filename)
                        # Create a relative path to maintain directory structure on GCS
                        relative_path = os.path.relpath(local_filepath, tmpdir)
                        gcs_blob_name = os.path.join(gcs_prefix, relative_path)
                        # Normalize GCS blob name (e.g., replace backslashes if on Windows)
                        gcs_blob_name = gcs_blob_name.replace(os.sep, '/')


                        blob = bucket.blob(gcs_blob_name)
                        blob.upload_from_filename(local_filepath)
                        logger.info(f"[Rank {trainer.accelerator.process_index}] Uploaded {filename} to gs://{bucket_name}/{gcs_blob_name}")
                logger.info(f"[Rank {trainer.accelerator.process_index}] GCS upload to {output_dir} complete.")
        except Exception as e:
            logger.error(f"[Rank {trainer.accelerator.process_index}] Error during GCS save for {output_dir}: {e}", exc_info=True)
        finally:
            # storage.Client() doesn't have an explicit close() in the same way
            # some other clients do; connections are typically managed by the underlying http library.
            # If you were using a specific transport that needed closing, you'd do it here.
            del gcs_client # Allow garbage collection
            del cpu_state_dict # Ensure memory is freed
    else:
        # Local output_dir: save directly
        logger.info(f"[Rank {trainer.accelerator.process_index}] Saving model to local directory: {output_dir}")
        try:
            trainer._save(output_dir, state_dict=cpu_state_dict)
            logger.info(f"[Rank {trainer.accelerator.process_index}] Model saved successfully to local directory: {output_dir}")
        except Exception as e:
            logger.error(f"[Rank {trainer.accelerator.process_index}] Error saving model to local directory {output_dir}: {e}", exc_info=True)
        finally:
            del cpu_state_dict

# --- Main Training Function (executed by each XLA process) ---
def main_training_function(model_args_dict: dict, data_args_dict: dict, training_args_dict: dict):
    # Re-parse arguments inside each process to ensure TrainingArguments is fully initialized
    # This also correctly sets up distributed environment awareness within TrainingArguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_dict(
        {"model": model_args_dict, "data": data_args_dict, **training_args_dict}
    )
    
    # Setup logging verbosity for transformers
    if training_args.should_log: # should_log is true on main process by default
        transformers.utils.logging.set_verbosity_info()
    
    logger.info(f"[Rank {training_args.process_index}/{training_args.world_size}] Training process started.")
    logger.info(f"[Rank {training_args.process_index}] TrainingArguments: {training_args}")


    if training_args.report_to and "wandb" in training_args.report_to:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    training_args.remove_unused_columns = False

    # Initialize processor
    # dp_config needs to be edict if VoRAProcessor expects it
    dp_config = edict(data_args.data.get("train", {}).get("data_preprocess", {}))
    processor = VoRAProcessor(**dp_config)

    # Prepare dataset
    # df_config needs to be edict if get_dataset expects it
    df_train_config = edict(data_args.data.get("train", {}).get("data_fetch", {}))
    if not isinstance(df_train_config.get("data_paths"), list):
        df_train_config.data_paths = [df_train_config.get("data_paths")]

    # Dataset loading can be time-consuming.
    # Consider xm.rendezvous if there are one-time downloads handled by main process.
    # However, HuggingFace datasets often manage this internally.
    if training_args.local_process_index == 0: # Or use xm.is_master_ordinal(local=True) for host-local master
         logger.info("Process with local_process_index 0 preparing dataset (if downloads occur)...")
    # xm.rendezvous("dataset_preparation_ヴォラ") # Use a unique name

    train_dataset = get_dataset(
        data_paths=df_train_config.data_paths,
        processor=processor,
        # Pass any other necessary args from df_train_config
    )
    data_collator = processor.batch_transform

    # Model setup
    # model_args.model should be a dict here
    config = VoRAConfig(**model_args.model)
    model = VoRAForCausalLM(config)

    pretrained_path = model_args.model.get("pretrained", "")
    if pretrained_path:
        # Loading checkpoints should ideally be done before model is wrapped by DDP/FSDP.
        # Trainer's `accelerator.prepare(model)` handles the wrapping.
        # `load_sharded_checkpoint` should be called by all processes.
        logger.info(f"[Rank {training_args.process_index}] Loading pretrained model from {pretrained_path}")
        # This assumes load_sharded_checkpoint handles device mapping or loads to CPU.
        # If it loads to a specific device, ensure it's compatible with XLA flow.
        from transformers.modeling_utils import load_sharded_checkpoint # Moved import here
        load_sharded_checkpoint(model, pretrained_path, strict=False)

    if training_args.should_log: # Typically main process
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Number of trainable parameters: {num_params:,}")

    xla_callback = XlaOptimizerStepCallback()

    trainer = VoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset, # type: ignore
        eval_dataset=None, # Pass eval_dataset if you have one
        data_collator=data_collator,
        callbacks=[xla_callback], # Add the XLA optimizer step callback
    )

    logger.info(f"[Rank {training_args.process_index}] Starting training...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    if training_args.should_save:
        logger.info(f"[Rank {training_args.process_index}] Training complete. Saving final model and state.")
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    xm.rendezvous("training_complete") # Unique rendezvous name
    logger.info(f"[Rank {training_args.process_index}] Training function finished.")


# --- XLA Spawn Function ---
def _mp_fn(index, raw_config_dict_from_spawn: dict): # Accepts index + one raw config dict
    """
    `index` is the global ordinal of the current process.
    `raw_config_dict_from_spawn` is the dictionary passed from the main process.
    """
    # Set default dtype for XLA environment
    # Your config has 'mixed_precision: bf16'. Trainer handles this, but default_dtype can be set.
    # If using bf16, ensure your model and inputs are compatible.
    # torch.set_default_dtype(torch.bfloat16 if training_args_dict.get('mixed_precision') == 'bf16' else torch.float32)
    torch.set_default_dtype(torch.float32) # Or torch.bfloat16 based on your needs

    # Log the rank using xm or xr, now that we are in an XLA process.
    # Ensure xm and xr are imported if used directly.
    # These are available after XLA runtime is initialized in the spawned process.
    try:
        # These imports are safe inside the spawned function
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr # Make sure this import is present

        # 'index' passed to _mp_fn by torch_xla.launch is the global ordinal.
        # xr.global_ordinal() would also return the same value as 'index'.
        logger.info(
            f"XLA Process (global_ordinal: {index}) started. " # Using 'index' as global_ordinal
            f"World Size: {xr.world_size()}, "
            f"Local Ordinal: {xr.local_ordinal()}, "
            f"Host Index: {xr.host_index()}, "
            f"Device: {str(xm.xla_device())}" # Convert device to string for logging
        )
    except Exception as e:
        # Fallback logging if full details can't be fetched, using only the passed index
        logger.error(f"XLA Process with passed index {index} encountered an error during initial XLA detail logging: {e}")
        logger.info(f"XLA Process (passed index: {index}) started with limited initial XLA detail logging.")



    # Unpack the raw_config_dict into the three expected by main_training_function
    model_args_config_dict = raw_config_dict_from_spawn.get("model", {})
    data_args_config_dict = raw_config_dict_from_spawn.get("data", {})
    training_args_flat_dict = {
        k: v for k, v in raw_config_dict_from_spawn.items() if k not in ["model", "data"]
    }
    try:
        main_training_function(model_args_config_dict, data_args_config_dict, training_args_flat_dict)
    except Exception as e:
        logger.error(f"Error in main_training_function for XLA process {index}: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        raise e

# # --- XLA Spawn Function ---
# def _mp_fn(index, model_args_dict, data_args_dict, training_args_dict):
#     """
#     `index` is the local rank given by xmp.spawn. Not directly used if Trainer handles rank.
#     """
#     # Set default dtype for XLA environment, bfloat16 is often good for TPUs
#     # torch.set_default_dtype(torch.bfloat16)
#     torch.set_default_dtype(torch.float32) # Or as per your model's requirement

#     # Environment variables for distributed training are typically set by the XLA launcher/environment
#     # main_training_function will re-parse TrainingArguments which will pick these up.
#     main_training_function(model_args_dict, data_args_dict, training_args_dict)

# # --- Script Entry Point ---
# if __name__ == "__main__":
#     # Parse arguments once at the beginning
#     # These will be passed as dicts to spawned processes to avoid issues with pickling complex objects.
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments))
#     args_dict = get_args_dict()
#     model_args, data_args, training_args = parser.parse_dict(args_dict)

#     # args_dict = get_args_dict() # Your custom function to get args as dict, ensure it returns a flat dict
    
#     # For demonstration, using a simplified arg parsing if get_args_dict is complex.
#     # Replace with your actual get_args_dict() call.
#     # Example: Simulating get_args_dict if it parses sys.argv and returns a flat dict
#     # This part needs to be robust in your actual script.
#     import sys
#     # A simple way to get args if they are like --key value --model.key value
#     # This is a placeholder for your get_args_dict.
#     # get_args_dict needs to return a flat dict compatible with HfArgumentParser.parse_dict
#     # Example: if sys.argv is ['script.py', '--output_dir', './results', '--model.num_layers', '12']
#     # HfArgumentParser.parse_dict expects something like:
#     # {'output_dir': './results', 'model': {'num_layers': '12'}}

#     # We will pass dicts of the parsed args to _mp_fn
#     # The HfArgumentParser will be used again inside main_training_function
#     # to correctly instantiate the dataclasses in each process.
    
#     # Initial parsing to get dicts.
#     # We need to convert the parsed args dataclasses back to dicts for xmp.spawn
#     # if get_args_dict is not already returning the structure parse_dict needs.
    
#     # Let's assume get_args_dict() returns a flat dictionary like:
#     # {'output_dir': '/tmp/results', 'learning_rate': 5e-5, 'model_name_or_path': 'bert-base-uncased', ...}
#     # And ModelArguments, DataArguments might be nested, e.g. --model.foo bar
#     # HfArgumentParser.parse_dict handles this if the dict is structured.
    
#     # Simplified: Assume get_args_dict returns a dict that parse_dict can handle.

#     # We need to separate these into the three groups for clarity if passing separately
#     # For HfArgumentParser.parse_dict, it can take a single dict with top-level keys
#     # matching the dataclass field names (e.g., 'model', 'data') or flat keys.
#     # To be safe for xmp.spawn, let's pre-parse to ensure we have the dicts.
    
#     # This initial parse is just to get the tpu_num_cores for xmp.spawn
#     # and to structure args for passing.

#     model_args_for_spawn = model_args.model # This is already a dict
#     data_args_for_spawn = data_args.data   # This is already a dict
#     # For training_args, convert the dataclass to a dict
#     training_args_for_spawn = vars(training_args)

#     # nprocs = training_args.tpu_num_cores
#     # if nprocs is None:
#     #     try:
#     #         # Try to get number of XLA devices available on the current host
#     #         nprocs = xr.world_size() # Gets devices per host if XRT_LOCAL_WORKER is not 'TPU_PLUGIN'
#     #                                      # Or could be total devices if master_addr/port are set for multi-host
#     #         logger.info(f"tpu_num_cores not specified, detected {nprocs} XLA devices on this host.")
#     #     except Exception as e:
#     #         logger.warning(f"Could not auto-detect XLA devices ({e}). Set --tpu_num_cores. Defaulting to 1 process.")
#     #         nprocs = 1
    
#     # if nprocs == 0 : # Should not happen if TPUs are available
#     #     logger.error("No XLA devices found or nprocs is 0. Exiting.")
#     #     sys.exit(1)

#     # logger.info(f"Spawning {nprocs} XLA processes.")
#     torch_xla.launch(
#         _mp_fn,
#         args=(model_args_for_spawn, data_args_for_spawn, training_args_for_spawn),)

# --- Script Entry Point ---
if __name__ == "__main__":
    # 1. Get the raw arguments first. This function should be very lightweight
    #    and AVOID any XLA initializations. It should just parse CLI/YAML.
    raw_args_from_config = get_args_dict() # Your custom function to get args as dict

    # 2. Extract nprocs for torch_xla.launch from the raw config if specified.
    #    This 'tpu_num_cores' from your YAML usually means the total cores for the job,
    #    but for torch_xla.launch's nprocs, it usually means cores *per host*
    #    if you're doing multi-host training where each host runs this script.
    #    If tpu_num_cores means total job cores (e.g., 32 for a v3-32),
    #    and you're on a machine with 8 cores, then nprocs for launch on this host should be 8.
    #    torch_xla.launch with nprocs=None will use all available local XLA devices.
    
    # Let torch_xla.launch determine nprocs if tpu_num_cores from config isn't specifically
    # for local process count. If 'tpu_num_cores' from your config IS the desired
    # local process count, then use it. Otherwise, None is safer.
    
    nprocs_for_launch = raw_args_from_config.get("tpu_num_cores") # Your YAML key is tpu_num_cores
                                                                # but in TrainingArguments it's tpu_num_cores.
                                                                # Assuming raw_config_from_config uses 'tpu_num_cores'
                                                                # matching your YAML.
    
    # Ensure training_args_dict passed to _mp_fn is just the part of raw_args_from_config
    # that pertains to TrainingArguments, to avoid parsing issues later.
    # The _mp_fn will then parse this raw_args_from_config.

    logger.info(f"Preparing to launch XLA processes. nprocs for torch_xla.launch: {nprocs_for_launch if nprocs_for_launch is not None else 'all available local'}")

    # torch_xla.launch will handle setting up the XLA environment for spawned processes.
    # Pass the raw_args_from_config directly.
    torch_xla.launch(
        _mp_fn,
        args=(raw_args_from_config,), # Pass as a tuple
        # start_method='fork' # launch usually defaults correctly
    )