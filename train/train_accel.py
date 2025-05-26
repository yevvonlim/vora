from dataclasses import dataclass, field
from functools import partial
import os
from typing import Dict, Optional

import datasets
from easydict import EasyDict as edict
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer
from transformers.trainer import is_datasets_available

from data_module.dataset import get_dataset
from data_module.processor import VoRAProcessor
from data_module.sampler import GlobalGroupRandomSampler, GroupRandomSampler
from models.configuration_vora import VoRAConfig
from models.modeling_vora import VoRAForCausalLM
from utils import logging
from utils.training_utils import AdditionalState, MultiTaskModuleMixin
from utils.parser_utils import get_args_dict


logger = logging.get_logger("trainer")
processor = None


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
    shuffle: bool = field(default=True)
    wandb_project: str = field(default="VoRA")


class VoRATrainer(Trainer):
    def __init__(self, model: nn.Module, args: TrainingArguments, **kwargs):
        self.additional_state = AdditionalState(args)
        
        if model is not None:
            report_patching = partial(_patching_module_base, additional_state=self.additional_state)
            model.apply(report_patching)
        
        super().__init__(
            model=model,
            args=args,
            **kwargs,
        )

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # Copied from transformers 4.47.0
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        additional_logs = self.additional_state.pop_metrics(gather_func=self._nested_gather) if hasattr(self, 'additional_state') else dict()

        epoch = logs.pop('epoch', None)
        logs.update(additional_logs)
        logs['epoch'] = epoch

        # Copied from transformers 4.47.0  
        output = logs | {"step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.group_by_data_source:
            lengths = self.train_dataset.datasets_length
            if xm.is_master_ordinal():
                logger.info("using GroupRandomSampler for mixed data!")
            return GroupRandomSampler(self.train_dataset, lengths)
        elif self.args.group_by_data_modality:
            modality_group_indices = self.train_dataset.modality_group_indices
            if xm.is_master_ordinal():
                logger.info("using GlobalGroupRandomSampler for mixed data!")
            global_batchsize = self.args.train_batch_size * xm.xrt_world_size() * self.args.gradient_accumulation_steps
            return GlobalGroupRandomSampler(global_batchsize, modality_group_indices)
        else:
            return super()._get_train_sampler()

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Modified for TPU compatibility.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": False,  # TPU doesn't support pin_memory
            "persistent_workers": self.args.dataloader_persistent_workers,
            "shuffle": self.args.shuffle
        }
        
        # Add sampler if needed
        train_sampler = self._get_train_sampler()
        if train_sampler is not None:
            dataloader_params["sampler"] = train_sampler
            dataloader_params.pop("shuffle", None)
        
        return DataLoader(train_dataset, **dataloader_params)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """
    TPU-compatible model saving
    """
    xm.save(trainer.model.state_dict(), output_dir + "/pytorch_model.bin")
    if xm.is_master_ordinal():
        trainer.model.config.save_pretrained(output_dir)


def train_function():
    """
    Main training function that will be spawned across TPU cores
    """
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    args_dict = get_args_dict()
    model_args, data_args, training_args = parser.parse_dict(args_dict)

    # TPU-specific setup
    device = xm.xla_device()
    training_args.local_rank = xm.get_ordinal()
    training_args.world_size = xm.xrt_world_size()
    
    # setting default training_args
    if xm.is_master_ordinal():
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    training_args.remove_unused_columns = False

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
    model = VoRAForCausalLM(config).to(device)

    if model_args.model.get("pretrained", ""):
        from transformers.modeling_utils import load_sharded_checkpoint
        if xm.is_master_ordinal():
            logger.info(f"Loading pretrained model from {model_args.model['pretrained']}")
        load_sharded_checkpoint(model, model_args.model["pretrained"], strict=False)
        xm.rendezvous("loading_checkpoint")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if xm.is_master_ordinal():
        logger.info(f"Number of trainable parameters: {num_trainable_params:,}")

    # Create trainer
    trainer = VoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Train the model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    # Save model and state
    xm.rendezvous("training_complete")
    if xm.is_master_ordinal():
        trainer.save_state()
    
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )


def main():
    """
    Entry point - this will be called by the TPU spawning mechanism
    """
    # For TPU, we need to use xmp.spawn
    xmp.spawn(train_function, nprocs=None, start_method='fork')


if __name__ == "__main__":
    main()