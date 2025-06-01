from dataclasses import dataclass, field
from functools import partial
import os
from typing import Dict, Optional
import math
import time

import datasets
from easydict import EasyDict as edict
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer
from transformers.trainer import is_datasets_available

from accelerate import Accelerator
from accelerate.utils import set_seed

from data_module.dataset import get_dataset
from data_module.processor import VoRAProcessor
from data_module.sampler import GlobalGroupRandomSampler, GroupRandomSampler
from models.configuration_vora import VoRAConfig
from models.modeling_vora import VoRAForCausalLM
from utils import logging
from utils.training_utils import AdditionalState, MultiTaskModuleMixin
from utils.parser_utils import get_args_dict



# GCS support
import tempfile
import shutil
from pathlib import Path
try:
    from google.cloud import storage
except ImportError:
    storage = None
import subprocess
import json


logger = logging.get_logger("trainer")



def is_gcs_path(path: str) -> bool:
    """Check if a path is a GCS path"""
    return isinstance(path, str) and path.startswith("gs://")


def parse_gcs_path(gcs_path: str):
    """Parse GCS path into bucket and object path"""
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Not a GCS path: {gcs_path}")
    
    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    object_name = path_parts[1] if len(path_parts) > 1 else ""
    return bucket_name, object_name


def download_from_gcs(gcs_path: str, local_path: str, accelerator=None):
    """Download file or directory from GCS"""
    if accelerator and not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return local_path
    
    if accelerator and accelerator.is_main_process:
        logger.info(f"Downloading {gcs_path} to {local_path}")
    
    try:
        # Use gsutil for better performance
        cmd = ["gsutil", "-m", "cp", "-r", gcs_path, local_path]
        subprocess.run(cmd, check=True, capture_output=True)
        
        if accelerator and accelerator.is_main_process:
            logger.info(f"Successfully downloaded {gcs_path}")
    except subprocess.CalledProcessError as e:
        if storage is None:
            raise ImportError("google-cloud-storage is required for GCS support")
        
        if accelerator and accelerator.is_main_process:
            logger.warning(f"gsutil failed, using Python client: {e}")
        
        bucket_name, object_name = parse_gcs_path(gcs_path)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        
        if object_name.endswith('/') or not object_name:
            # Download directory
            blobs = bucket.list_blobs(prefix=object_name)
            for blob in blobs:
                if not blob.name.endswith('/'):
                    rel_path = blob.name[len(object_name):] if object_name else blob.name
                    local_file_path = os.path.join(local_path, rel_path)
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    blob.download_to_filename(local_file_path)
        else:
            # Download single file
            blob = bucket.blob(object_name)
            blob.download_to_filename(local_path)
    
    if accelerator:
        accelerator.wait_for_everyone()
    
    return local_path


def upload_to_gcs(local_path: str, gcs_path: str, accelerator=None):
    """Upload file or directory to GCS"""
    if accelerator and not accelerator.is_main_process:
        return
    
    if accelerator and accelerator.is_main_process:
        logger.info(f"Uploading {local_path} to {gcs_path}")
    
    try:
        # Use gsutil for better performance
        cmd = ["gsutil", "-m", "cp", "-r", local_path, gcs_path]
        subprocess.run(cmd, check=True, capture_output=True)
        
        if accelerator and accelerator.is_main_process:
            logger.info(f"Successfully uploaded to {gcs_path}")
    except subprocess.CalledProcessError:
        if storage is None:
            raise ImportError("google-cloud-storage is required for GCS support")
        
        bucket_name, object_name = parse_gcs_path(gcs_path)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        if os.path.isdir(local_path):
            # Upload directory
            for root, dirs, files in os.walk(local_path):
                for file in files:
                    local_file = os.path.join(root, file)
                    rel_path = os.path.relpath(local_file, local_path)
                    gcs_file_path = os.path.join(object_name, rel_path).replace("\\", "/")
                    blob = bucket.blob(gcs_file_path)
                    blob.upload_from_filename(local_file)
        else:
            # Upload single file
            blob = bucket.blob(object_name)
            blob.upload_from_filename(local_path)


def setup_gcs_paths(args_dict, accelerator=None):
    """Setup local paths for GCS resources"""
    temp_dir = tempfile.mkdtemp(prefix="vora_gcs_")
    gcs_mappings = {}
    
    if accelerator and accelerator.is_main_process:
        logger.info(f"Setting up GCS paths in temporary directory: {temp_dir}")
    
    # Handle data paths
    if 'data' in args_dict and 'train' in args_dict['data']:
        data_paths = args_dict['data']['train']['data_fetch']['data_paths']
        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        
        new_data_paths = []
        for i, path_config in enumerate(data_paths):
            new_path_config = dict(path_config)
            
            # Handle annotation path
            anno_path = path_config.get("anno_path", "")
            if is_gcs_path(anno_path):
                local_anno_path = os.path.join(temp_dir, f"annotations_{i}.jsonl")
                download_from_gcs(anno_path, local_anno_path, accelerator)
                new_path_config["anno_path"] = local_anno_path
                gcs_mappings[local_anno_path] = anno_path
            
            # Handle image folder
            image_folder = path_config.get("image_folder", "")
            if is_gcs_path(image_folder):
                local_image_folder = os.path.join(temp_dir, f"images_{i}")
                os.makedirs(local_image_folder, exist_ok=True)
                download_from_gcs(image_folder + "/", local_image_folder, accelerator)
                new_path_config["image_folder"] = local_image_folder
                gcs_mappings[local_image_folder] = image_folder
            
            new_data_paths.append(new_path_config)
        
        args_dict['data']['train']['data_fetch']['data_paths'] = new_data_paths
    
    # Handle aux_vision path
    if 'model' in args_dict and args_dict['model'].get("aux_vision"):
        aux_vision_path = args_dict['model']["aux_vision"]
        if is_gcs_path(aux_vision_path):
            local_aux_vision = os.path.join(temp_dir, "aux_vision.pt")
            download_from_gcs(aux_vision_path, local_aux_vision, accelerator)
            args_dict['model']["aux_vision"] = local_aux_vision
            gcs_mappings[local_aux_vision] = aux_vision_path
    
    # Handle HFImageTransform path
    if ('data' in args_dict and 'train' in args_dict['data'] and 
        'data_preprocess' in args_dict['data']['train']):
        frames_ops = args_dict['data']['train']['data_preprocess'].get("frames_ops", {})
        if "HFImageTransform" in frames_ops:
            hf_path = frames_ops["HFImageTransform"].get("path", "")
            if is_gcs_path(hf_path):
                local_hf_path = os.path.join(temp_dir, "hf_transform")
                os.makedirs(local_hf_path, exist_ok=True)
                download_from_gcs(hf_path + "/", local_hf_path, accelerator)
                frames_ops["HFImageTransform"]["path"] = local_hf_path
                gcs_mappings[local_hf_path] = hf_path
    
    # Handle pretrained model path
    if 'model' in args_dict and args_dict['model'].get("pretrained"):
        pretrained_path = args_dict['model']["pretrained"]
        if is_gcs_path(pretrained_path):
            local_pretrained = os.path.join(temp_dir, "pretrained_model")
            os.makedirs(local_pretrained, exist_ok=True)
            download_from_gcs(pretrained_path + "/", local_pretrained, accelerator)
            args_dict['model']["pretrained"] = local_pretrained
            gcs_mappings[local_pretrained] = pretrained_path
    
    return temp_dir, gcs_mappings


def _patching_module_base(module: nn.Module, additional_state: AdditionalState):
    if isinstance(module, nn.Module)             and hasattr(module, 'supports_report_metrics')             and module.supports_report_metrics             and MultiTaskModuleMixin not in module.__class__.__bases__:
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
    shuffle: bool = field(default=True)
    wandb_project: str = field(default="VoRA")
    tpu_cores: int = field(default=32)
    gradient_clipping: float = field(default=1.0)
    mixed_precision: str = field(default="bf16")


class VoRAAccelerateTrainer:
    def __init__(self, model, args, train_dataset, data_collator, accelerator):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.accelerator = accelerator
        self.additional_state = AdditionalState(args)
        
        if model is not None:
            report_patching = partial(_patching_module_base, additional_state=self.additional_state)
            model.apply(report_patching)
        
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0
        self.epoch = 0
        
    def create_optimizer_and_scheduler(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        num_update_steps_per_epoch = len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps
        max_steps = self.args.max_steps if self.args.max_steps > 0 else int(self.args.num_train_epochs * num_update_steps_per_epoch)
        
        self.lr_scheduler = transformers.get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=max_steps,
        )
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("No training dataset provided")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        sampler = None
        if self.args.group_by_data_source:
            lengths = self.train_dataset.datasets_length
            if self.accelerator.is_main_process:
                logger.info("Using GroupRandomSampler for mixed data!")
            sampler = GroupRandomSampler(self.train_dataset, lengths)
        elif self.args.group_by_data_modality:
            modality_group_indices = self.train_dataset.modality_group_indices
            if self.accelerator.is_main_process:
                logger.info("Using GlobalGroupRandomSampler for mixed data!")
            global_batchsize = self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps
            sampler = GlobalGroupRandomSampler(global_batchsize, modality_group_indices)
        
        dataloader_params = {
            "batch_size": self.args.per_device_train_batch_size,
            "collate_fn": data_collator,
            "num_workers": min(self.args.dataloader_num_workers, 4),
            "pin_memory": False,
            "drop_last": True,
        }
        
        if sampler is not None:
            dataloader_params["sampler"] = sampler
        else:
            dataloader_params["shuffle"] = self.args.shuffle
        
        return DataLoader(train_dataset, **dataloader_params)

    
    def save_model(self, output_dir, gcs_output_dir=None):
        local_output_dir = output_dir
        
        # If output_dir is a GCS path, save locally first
        if is_gcs_path(output_dir):
            local_output_dir = tempfile.mkdtemp(prefix="model_save_")
            gcs_output_dir = output_dir
        
        if self.accelerator.is_main_process:
            logger.info(f"Saving model to {local_output_dir}")
        
        self.accelerator.wait_for_everyone()
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if self.accelerator.is_main_process:
            os.makedirs(local_output_dir, exist_ok=True)
            torch.save(unwrapped_model.state_dict(), os.path.join(local_output_dir, "pytorch_model.bin"))
            
            if hasattr(unwrapped_model, 'config'):
                unwrapped_model.config.save_pretrained(local_output_dir)
        
        self.accelerator.wait_for_everyone()
        
        # Upload to GCS if needed
        if gcs_output_dir:
            if self.accelerator.is_main_process:
                logger.info(f"Uploading model to GCS: {gcs_output_dir}")
                upload_to_gcs(local_output_dir, gcs_output_dir, self.accelerator)
                shutil.rmtree(local_output_dir)  # Clean up local temp directory
            self.accelerator.wait_for_everyone()
    
    def log(self, logs, step=None):
        if not self.accelerator.is_main_process:
            return
        
        if step is None:
            step = self.global_step
            
        logs["step"] = step
        logs["epoch"] = self.epoch
        
        additional_logs = self.additional_state.pop_metrics() if hasattr(self, 'additional_state') else {}
        logs.update(additional_logs)
        
        self.accelerator.log(logs, step=step)

        
    def train(self):
        if self.accelerator.is_main_process:
            logger.info("Starting training...")
            logger.info(f"Number of processes: {self.accelerator.num_processes}")
            logger.info(f"Device: {self.accelerator.device}")
            logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        
        train_dataloader = self.get_train_dataloader()
        self.create_optimizer_and_scheduler()
        
        self.model, self.optimizer, train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, train_dataloader, self.lr_scheduler
        )
        
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps
        max_steps = self.args.max_steps if self.args.max_steps > 0 else int(self.args.num_train_epochs * num_update_steps_per_epoch)
        
        if self.accelerator.is_main_process:
            logger.info(f"Total training steps: {max_steps}")
            logger.info(f"Effective batch size: {self.args.per_device_train_batch_size * self.accelerator.num_processes * self.args.gradient_accumulation_steps}")
        
        self.model.train()
        
        for epoch in range(int(self.args.num_train_epochs)):
            self.epoch = epoch
            epoch_loss = 0.0
            
            if self.accelerator.is_main_process:
                logger.info(f"Starting epoch {epoch + 1}/{int(self.args.num_train_epochs)}")
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    
                    self.accelerator.backward(loss)
                    
                    if self.args.gradient_clipping > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
                    
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    self.global_step += 1
                    
                    loss_values = self.accelerator.gather_for_metrics({"loss": loss})
                    avg_loss = torch.mean(loss_values["loss"])
                    
                    epoch_loss += avg_loss.item()
                    
                    if self.global_step % self.args.logging_steps == 0:
                        logs = {
                            "train_loss": avg_loss.item(),
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "epoch": epoch + (step + 1) / len(train_dataloader),
                        }
                        self.log(logs)
                        
                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Step {self.global_step}: loss={avg_loss.item():.4f}, "
                                f"lr={self.lr_scheduler.get_last_lr()[0]:.6f}"
                            )
                    
                    if self.global_step % self.args.save_steps == 0:
                        if is_gcs_path(self.args.output_dir):
                            checkpoint_name = f"checkpoint-{self.global_step}"
                            gcs_checkpoint_dir = os.path.join(self.args.output_dir, checkpoint_name).replace("\\", "/")
                            self.save_model(gcs_checkpoint_dir)
                        else:
                            output_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                            self.save_model(output_dir)
                        
                        if self.accelerator.is_main_process:
                            logger.info(f"Saved checkpoint at step {self.global_step}")
                    
                    if self.global_step >= max_steps:
                        break
            
            avg_epoch_loss = epoch_loss / num_update_steps_per_epoch
            if self.accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            if self.global_step >= max_steps:
                break
        
        if self.accelerator.is_main_process:
            logger.info("Training completed. Saving final model...")
        
        final_output_dir = self.args.output_dir
        self.save_model(final_output_dir)
        
        if self.accelerator.is_main_process:
            logger.info(f"Training finished! Model saved to {final_output_dir}")



def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    args_dict = get_args_dict()
    
    # Setup GCS paths before parsing arguments
    temp_dir = None
    gcs_mappings = {}
    
    # Initialize accelerator early for GCS operations
    accelerator = Accelerator()
    
    # Check if we have any GCS paths and download them
    if any(is_gcs_path(str(v)) for v in str(args_dict).split() if isinstance(v, str)):
        temp_dir, gcs_mappings = setup_gcs_paths(args_dict, accelerator)
    
    model_args, data_args, training_args = parser.parse_dict(args_dict)
    
    # Store GCS info for cleanup
    training_args.temp_dir = temp_dir
    training_args.gcs_mappings = gcs_mappings
    
    # Reconfigure accelerator with training arguments
    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="wandb" if training_args.report_to == "wandb" else None,
        project_dir=training_args.output_dir if not is_gcs_path(training_args.output_dir) else None,
    )
    
    if accelerator.is_main_process:
        logger.info(f"Accelerator config: {accelerator.state}")
        logger.info(f"Number of processes: {accelerator.num_processes}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        
        if training_args.tpu_cores != accelerator.num_processes:
            logger.warning(
                f"Expected {training_args.tpu_cores} TPU cores, "
                f"but got {accelerator.num_processes} processes"
            )
        
        if training_args.report_to == "wandb":
            os.environ["WANDB_PROJECT"] = training_args.wandb_project
    
    set_seed(training_args.seed)
    
    data_args.data = edict(data_args.data)
    df_config = data_args.data.train.data_fetch
    dp_config = data_args.data.train.data_preprocess
    
    processor = VoRAProcessor(**dp_config)
    
    if not isinstance(df_config.data_paths, list):
        df_config.data_paths = [df_config.data_paths]
    
    train_dataset = get_dataset(
        data_paths=df_config.data_paths,
        processor=processor,
    )
    data_collator = processor.batch_transform
    
    config = VoRAConfig(**model_args.model)
    
    with accelerator.main_process_first():
        model = VoRAForCausalLM(config)
        
        if model_args.model.get("pretrained", ""):
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained model from {model_args.model['pretrained']}")
                from transformers.modeling_utils import load_sharded_checkpoint
                load_sharded_checkpoint(model, model_args.model["pretrained"], strict=False)
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    trainer = VoRAAccelerateTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        accelerator=accelerator
    )
    
    if accelerator.is_main_process and training_args.report_to == "wandb":
        accelerator.init_trackers(
            project_name=training_args.wandb_project,
            config={
                "model_args": model_args.__dict__,
                "data_args": data_args.__dict__,
                "training_args": training_args.__dict__,
            }
        )
    
    trainer.train()
    
    if accelerator.is_main_process and training_args.report_to == "wandb":
        accelerator.end_training()
    
    # Handle final model save with GCS
    final_save_path = training_args.output_dir
    if is_gcs_path(final_save_path):
        # The trainer.save_model will handle GCS upload
        pass
    
    # Cleanup temporary GCS files
    if temp_dir and accelerator.is_main_process:
        logger.info(f"Cleaning up temporary directory: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
