import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModel

from .configuration_vora import VoRAConfig
from .eva_model import EVAVisionTransformer

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class CosineLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(CosineLoss, self).__init__()
        self.reduction = reduction

    @staticmethod
    def interpolate_tokens_2d(self, teacher_tokens, target_size):
        """
        Interpolate teacher tokens to the target size using bilinear interpolation.
        """
        # teacher_tokens shape is (batch_size, height, width, feature_dim)
        teacher_tokens = teacher_tokens.permute(0, 3, 1, 2)  # Convert to (batch_size, feature_dim, height, width)
        interpolated = torch.nn.functional.interpolate(teacher_tokens, size=target_size, mode='bilinear', align_corners=True).flatten(2)  # Flatten height and width dimensions
        return interpolated.permute(0, 2, 1)  # Convert back to (batch_size, new_height * new_width, feature_dim)

    def forward(self, input: torch.Tensor, target: torch.Tensor, input_shape=None, target_shape=None) -> torch.Tensor:
        if input_shape is not None and target_shape is not None:
            input = input.reshape((input.shape[0], ) + input_shape + (-1, ))
            input = self.interpolate_tokens_2d(input, target_shape)

        cos_sim = nn.functional.cosine_similarity(input, target, dim=1)
        loss = 1 - cos_sim

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AuxVision(nn.Module):
    def __init__(self,
        config: VoRAConfig = None,
    ):
        super().__init__()
        self.skip_aux_cls = config.skip_aux_cls  # whether to skip the cls token in ViT
        # ---------------- Setup Aux Model ----------------
        # support jina clip encoder
        if 'jina' in config.aux_vision.lower() and 'clip' in config.aux_vision.lower():
            cfg = {
                "img_size": 512,
                "num_classes": 1024,
                "embed_dim": 1024,
                "patch_size": 14,
                "depth": 24,
                "qkv_bias": True,
                "naiveswiglu": True,
                "num_heads": 16,
                "patch_dropout":0.1,
                "subln": True,
                "mlp_ratio": 2.66666,
                "use_mean_pooling": False,
            } 
            self.aux_model = EVAVisionTransformer(**cfg)
            self.aux_model.load_state_dict(torch.load(config.aux_vision, map_location='cpu'), strict=False)
            vision_hidden_size = 1024
            num_hidden_layers = 24


        elif 'clip' in config.aux_vision.lower():
            self.aux_model = CLIPVisionModel.from_pretrained(config.aux_vision)
            vision_hidden_size = self.aux_model.vision_model.config.hidden_size
            num_hidden_layers = self.aux_model.vision_model.config.num_hidden_layers
        
        else:
            self.aux_model = AutoModel.from_pretrained(config.aux_vision, trust_remote_code=True)
            vision_hidden_size = self.aux_model.config.hidden_size
            num_hidden_layers = self.aux_model.config.num_hidden_layers
        for name, param in self.aux_model.named_parameters():
            param.requires_grad = False
        # -------------------------------------------------

        # ---------------- Setup Aux Heads ----------------
        self.aux_layers = list(range(num_hidden_layers))
        for layer_id in self.aux_layers:
            self.add_module(f"aux_layer_{layer_id}", self.build_aux_layer(config.hidden_size, vision_hidden_size))
        # -------------------------------------------------

        self.loss_function = CosineLoss()
        self.loss_keys = [f"loss_aux_layer_{layer_id}" for layer_id in self.aux_layers]

    def build_aux_layer(self, llm_hidden_size, vit_hidden_size):
        return nn.Sequential(
                    RMSNorm(llm_hidden_size),
                    nn.Linear(
                        llm_hidden_size,
                        vit_hidden_size,
                        bias=False,
                    )
                )

    def forward(self, frames, llm_hidden_states, vision_mask):
        vision_hidden_states = self.aux_model(frames, output_hidden_states=True).hidden_states
        losses = {}
        for layer_idx in self.aux_layers:
            aux_hidden_states = getattr(self, f"aux_layer_{layer_idx}")(llm_hidden_states[layer_idx][vision_mask == 1])
            start_id = 1 if self.skip_aux_cls else 0
            aux_loss = self.loss_function(vision_hidden_states[layer_idx][:, start_id:].reshape(aux_hidden_states.shape), aux_hidden_states)
            losses[f"loss_aux_layer_{layer_idx}"] = aux_loss
        return losses
