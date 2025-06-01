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


logger = logging.get_logger("trainer")


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

    
    def save_model(self, output_dir):
        if self.accelerator.is_main_process:
            logger.info(f"Saving model to {output_dir}")
        
        self.accelerator.wait_for_everyone()
        
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        if self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(unwrapped_model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
            
            if hasattr(unwrapped_model, 'config'):
                unwrapped_model.config.save_pretrained(output_dir)
        
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
    model_args, data_args, training_args = parser.parse_dict(args_dict)
    
    accelerator = Accelerator(
        mixed_precision=training_args.mixed_precision,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        log_with="wandb" if training_args.report_to == "wandb" else None,
        project_dir=training_args.output_dir,
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
    
    if accelerator.is_main_process:
        logger.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
