import torch
from torch.utils.data import Sampler
from typing import List, Sized, Iterator, Optional

# Helper to get distributed information, prioritizing torch_xla.runtime (xr)
def _get_distributed_info():
    num_replicas = 1
    rank = 0
    is_distributed = False

    try:
        import torch_xla.runtime as xr
        # These calls will work if inside an XLA spawned process.
        # They might raise RuntimeError if XLA is not initialized.
        _xr_world_size = xr.world_size()
        if _xr_world_size > 0: # world_size is usually >= 1
            num_replicas = _xr_world_size
            rank = xr.global_ordinal()
            is_distributed = (num_replicas > 1)
        # If somehow _xr_world_size is 0 or less, it's not a valid distributed setup
        # and will default to non-distributed based on initial values.
            
    except (ImportError, RuntimeError, AttributeError): # Catches if xr not available or XLA not initialized
        # Fallback to torch.distributed if XLA fails or is not available
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            num_replicas = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            is_distributed = (num_replicas > 1)
        # If neither XLA nor torch.distributed is set up, it remains non-distributed (defaults)

    return num_replicas, rank, is_distributed

# --- DistributedGroupRandomSampler ---
class DistributedGroupRandomSampler(Sampler[int]):
    def __init__(self,
                 data_source: Sized,
                 lengths: List[int], # Lengths of the contiguous groups
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False):
        
        # Use helper if num_replicas or rank not provided
        _default_num_replicas, _default_rank, _ = _get_distributed_info()
        if num_replicas is None:
            num_replicas = _default_num_replicas
        if rank is None:
            rank = _default_rank

        self.data_source = data_source
        self.lengths = lengths
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.start_idx = [0] * (len(lengths) + 1)
        current_sum = 0
        for i, length in enumerate(lengths):
            self.start_idx[i] = current_sum
            current_sum += length
        self.start_idx[len(lengths)] = current_sum
        
        self.dataset_size = len(self.data_source) # type: ignore
        if self.dataset_size != self.start_idx[-1]:
            raise ValueError(f"Sum of lengths {self.start_idx[-1]} does not match data_source size {self.dataset_size}")

        if self.drop_last and self.dataset_size % self.num_replicas != 0:
            self.num_samples_per_replica = self.dataset_size // self.num_replicas
            self.total_size = self.num_samples_per_replica * self.num_replicas
        else:
            self.num_samples_per_replica = (self.dataset_size + self.num_replicas - 1) // self.num_replicas
            self.total_size = self.num_samples_per_replica * self.num_replicas
    
    # __iter__, __len__, set_epoch methods remain the same as previously defined
    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        all_indices: List[int] = []
        if self.shuffle:
            for i, length in enumerate(self.lengths):
                subgroup_local_indices = torch.randperm(length, generator=g).tolist()
                subgroup_global_indices = [self.start_idx[i] + local_idx for local_idx in subgroup_local_indices]
                all_indices.extend(subgroup_global_indices)
        else:
            for i, length in enumerate(self.lengths):
                all_indices.extend(range(self.start_idx[i], self.start_idx[i] + length))
        
        if not self.drop_last:
            padding_size = self.total_size - len(all_indices)
            if padding_size > 0:
                all_indices += (all_indices * (padding_size // len(all_indices) + 2))[:padding_size]
        else:
            all_indices = all_indices[:self.total_size]
        
        if len(all_indices) != self.total_size:
             raise RuntimeError(f"Logic error: all_indices length {len(all_indices)} != total_size {self.total_size}")

        indices_for_rank = all_indices[self.rank : self.total_size : self.num_replicas]
        
        if len(indices_for_rank) != self.num_samples_per_replica:
            raise RuntimeError(
                f"Logic error: indices_for_rank length {len(indices_for_rank)} "
                f"!= num_samples_per_replica {self.num_samples_per_replica} for rank {self.rank}"
            )
        return iter(indices_for_rank)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


# --- DistributedGlobalGroupRandomSampler ---
class DistributedGlobalGroupRandomSampler(Sampler[int]):
    def __init__(self,
                 per_replica_batch_size: int,
                 modality_group_indices: List[List[int]],
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False):

        _default_num_replicas, _default_rank, _ = _get_distributed_info()
        if num_replicas is None:
            num_replicas = _default_num_replicas
        if rank is None:
            rank = _default_rank

        self.per_replica_batch_size = per_replica_batch_size
        self.modality_group_indices_orig = modality_group_indices
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last 
        self.epoch = 0

        self.all_batches: List[List[int]] = self._prepare_all_batches()

        if self.drop_last and len(self.all_batches) % self.num_replicas != 0:
            self.num_batches_per_replica = len(self.all_batches) // self.num_replicas
            self.total_num_batches = self.num_batches_per_replica * self.num_replicas
        else:
            self.num_batches_per_replica = (len(self.all_batches) + self.num_replicas - 1) // self.num_replicas
            self.total_num_batches = self.num_batches_per_replica * self.num_replicas
        
        self.num_samples_per_replica = self.num_batches_per_replica * self.per_replica_batch_size

    # _prepare_all_batches, __iter__, __len__, set_epoch methods remain the same as previously defined
    def _prepare_all_batches(self) -> List[List[int]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        prepared_batches: List[List[int]] = []
        group_lengths = [len(group) for group in self.modality_group_indices_orig]

        for group_idx, length in enumerate(group_lengths):
            if length == 0:
                continue
            current_group_indices = list(self.modality_group_indices_orig[group_idx])
            if self.shuffle:
                perm = torch.randperm(len(current_group_indices), generator=g).tolist()
                shuffled_group_indices = [current_group_indices[i] for i in perm]
            else:
                shuffled_group_indices = current_group_indices
            num_full_batches = length // self.per_replica_batch_size
            remaining_indices_count = length % self.per_replica_batch_size
            ptr = 0
            for _ in range(num_full_batches):
                batch = shuffled_group_indices[ptr : ptr + self.per_replica_batch_size]
                prepared_batches.append(batch)
                ptr += self.per_replica_batch_size
            if remaining_indices_count > 0:
                partial_batch = list(shuffled_group_indices[ptr:])
                if len(partial_batch) < self.per_replica_batch_size:
                    num_to_add = self.per_replica_batch_size - len(partial_batch)
                    additional_indices_indices = torch.randint(0, len(shuffled_group_indices), (num_to_add,), generator=g).tolist()
                    additional_indices = [shuffled_group_indices[i] for i in additional_indices_indices]
                    partial_batch.extend(additional_indices)
                prepared_batches.append(partial_batch)
        if self.shuffle:
            batch_perm = torch.randperm(len(prepared_batches), generator=g).tolist()
            prepared_batches = [prepared_batches[i] for i in batch_perm]
        return prepared_batches

    def __iter__(self) -> Iterator[int]:
        current_epoch_all_batches = list(self.all_batches)
        if not self.drop_last:
            padding_size = self.total_num_batches - len(current_epoch_all_batches)
            if padding_size > 0:
                current_epoch_all_batches += (current_epoch_all_batches * (padding_size // len(current_epoch_all_batches) + 2))[:padding_size]
        else:
            current_epoch_all_batches = current_epoch_all_batches[:self.total_num_batches]
        if len(current_epoch_all_batches) != self.total_num_batches:
            raise RuntimeError(
                f"Logic error: current_epoch_all_batches length {len(current_epoch_all_batches)} "
                f"!= total_num_batches {self.total_num_batches}"
            )
        batches_for_rank = current_epoch_all_batches[self.rank : self.total_num_batches : self.num_replicas]
        if len(batches_for_rank) != self.num_batches_per_replica:
             raise RuntimeError(
                f"Logic error: batches_for_rank length {len(batches_for_rank)} "
                f"!= num_batches_per_replica {self.num_batches_per_replica} for rank {self.rank}"
            )
        final_indices_for_rank = [idx for batch in batches_for_rank for idx in batch]
        return iter(final_indices_for_rank)

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.all_batches = self._prepare_all_batches()