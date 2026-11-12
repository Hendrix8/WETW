# coding = utf-8

from torch import nn, Tensor
import torch.nn.functional as F
import torch 
from wetw._reference.model.commons import Squeeze, Reshape
import numpy as np 
from omegaconf import DictConfig
from wetw._reference.util.helper_functions import getActivation, getWeightNorm, getDilation, getLayerNorm, get_embedding_output_norm


class GlobalTemporalPool1d(nn.Module):
    """Collapse (B, C, T) -> (B, C, 1). Toggle via encoder_model.global_pooling: max | avg | attention."""

    def __init__(self, channels: int, mode: str = "max"):
        super().__init__()
        self.mode = str(mode).lower()
        if self.mode not in ("max", "avg", "attention"):
            raise ValueError("encoder_model.global_pooling must be one of: max, avg, attention")
        if self.mode == "attention":
            self._score = nn.Conv1d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "max":
            return F.adaptive_max_pool1d(x, 1)
        if self.mode == "avg":
            return F.adaptive_avg_pool1d(x, 1)
        logits = self._score(x)
        w = torch.softmax(logits, dim=2)
        return (x * w).sum(dim=2, keepdim=True)


class _OriginalResBlock(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, out_channels, dilation):
        super(_OriginalResBlock, self).__init__()
        self.cfg = cfg
        dim_series = cfg.dataset.dim_series
        kernel_size = cfg.encoder_model.conv_kernel_size
        padding = int(kernel_size / 2) * dilation
        activation_name = cfg.encoder_model.activation_conv
        bias = cfg.encoder_model.layernorm_type == 'none'

        self.__residual_link = nn.Sequential(getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg),
                                             getLayerNorm(dim_series, cfg), 
                                             getActivation(activation_name, cfg),

                                             getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg),
                                             getLayerNorm(dim_series, cfg))
        
        if in_channels != out_channels:
            self.__identity_link = getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias), cfg)
        else:
            self.__identity_link = nn.Identity()

        self.__after_addition = getActivation(activation_name, cfg)
        
        
    def forward(self, input: Tensor) -> Tensor:
        residual = self.__residual_link(input)
        identity = self.__identity_link(input)

        return self.__after_addition(identity + residual)



class _PreActivatedResBlock(nn.Module):
    def __init__(self, cfg: DictConfig, in_channels, out_channels, dilation, first = False, last = False):
        super(_PreActivatedResBlock, self).__init__()

        self.cfg = cfg
        self.dim_series = cfg.dataset.dim_series
        self.last = last
        kernel_size = cfg.encoder_model.conv_kernel_size
        padding = int(kernel_size / 2) * dilation
        activation_name = cfg.encoder_model.activation_conv

        bias = cfg.encoder_model.layernorm_type == 'none' or not cfg.encoder_model.layernorm_elementwise_affine
                
        if first:
           
            self.__first_block = getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg)
            in_channels = out_channels
        else:
            self.__first_block = nn.Identity()

        self.__residual_link = nn.Sequential(getLayerNorm(self.dim_series, cfg), 
                                             getActivation(activation_name, cfg),
                                             getWeightNorm(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg),
                                            
                                             getLayerNorm(self.dim_series, cfg),
                                             getActivation(activation_name, cfg),
                                             getWeightNorm(nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=bias), cfg))


        if in_channels != out_channels:
            self.__identity_link = getWeightNorm(nn.Conv1d(in_channels, out_channels, 1, bias=bias), cfg)
            
        else:
            self.__identity_link = nn.Identity()

        if last:
            self.__after_addition = nn.Sequential(getLayerNorm(self.dim_series, cfg), getActivation(activation_name, cfg))
        else:
            self.__after_addition = nn.Identity()


    def forward(self, input: Tensor) -> Tensor:
            
        input = self.__first_block(input)
        
        residual = input
        identity = input

        residual = self.__residual_link(residual)
                    
        identity = self.__identity_link(identity)
                
        out = self.__after_addition(identity + residual)
        
        return out


class _ResNet(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(_ResNet, self).__init__()

        self.cfg = cfg
        num_resblock = cfg.encoder_model.num_en_resblock

        if cfg.encoder_model.dilation_type == 'exponential':
            assert num_resblock > 1 and 2 ** (num_resblock + 1) <= cfg.dataset.dim_series + 1

        inner_channels = cfg.encoder_model.num_en_channels
        out_channels = cfg.encoder_model.dim_en_latent
      
        
        if cfg.encoder_model.resblock_pre_activation:
            layers = [_PreActivatedResBlock(cfg, 1, inner_channels, getDilation(1, cfg), first=True)]         
            layers += [_PreActivatedResBlock(cfg, inner_channels, inner_channels, getDilation(depth, cfg)) for depth in range(2, num_resblock)]
            layers += [_PreActivatedResBlock(cfg, inner_channels, out_channels, getDilation(num_resblock, cfg), last=True)]
       
        else:
            layers = [_OriginalResBlock(cfg, 1, inner_channels, getDilation(1, cfg))]
            layers += [_OriginalResBlock(cfg, inner_channels, inner_channels, getDilation(depth, cfg)) for depth in range(2, num_resblock)]
            layers += [_OriginalResBlock(cfg, inner_channels, out_channels, getDilation(num_resblock, cfg))]

        self.__model = nn.Sequential(*layers)

        
    def forward(self, input: Tensor) -> Tensor:
        
        input = self.__model(input)

        return input


class TransposeLayerNorm(nn.Module):
    """LayerNorm applied over the channel dimension (dim 1) for (B, C, L) inputs."""
    def __init__(self, channels: int, cfg: DictConfig):
        super().__init__()
        self.norm = nn.LayerNorm(channels, elementwise_affine=cfg.encoder_model.layernorm_elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        x = self.norm(x)
        # (B, L, C) -> (B, C, L)
        return x.transpose(1, 2)


class MultiScaleTemporalHead(nn.Module):
    def __init__(self, channels: int, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        kernel_sizes = getattr(cfg.encoder_model, "multi_temporal_head_kernels", [3, 5, 7])
        if cfg.encoder_model.flexible_kernels:
            kernel_sizes = self.generate_kernels(kernel_sizes, cfg.dataset.dim_series)
        bias = cfg.encoder_model.layernorm_type == 'none' or not cfg.encoder_model.layernorm_elementwise_affine
        if isinstance(kernel_sizes[0], float):
            kernel_sizes = [
                min(int(2 * k * cfg.dataset.dim_series) | 1,
                    cfg.dataset.dim_series - 1 if cfg.dataset.dim_series % 2 == 0 else cfg.dataset.dim_series - 2)
                    for k in kernel_sizes]
        print(f'TH kernel_sizes: {kernel_sizes}')
        self.heads = nn.ModuleList()
        for k in kernel_sizes:
            # For odd k, padding=(k-1)//2 keeps length same.
            # For even k, padding=k//2 leads to L+1. We use padding=k//2 and then slice or use asymmetric padding.
            # Best is to use padding='same' if supported, or handles it manually.
            padding = k // 2
            
            self.heads.append(nn.Sequential(
                getWeightNorm(nn.Conv1d(channels, channels, k, padding=padding, bias=bias), cfg),
                TransposeLayerNorm(channels, cfg) if cfg.encoder_model.layernorm_type != 'none' else nn.Identity(),
                getActivation(cfg.encoder_model.activation_conv, cfg)
            ))
        
        # Project back to original channel count
        self.fusion = getWeightNorm(nn.Conv1d(channels * len(kernel_sizes), channels, kernel_size=1, bias=bias), cfg)

    def generate_kernels(self, base, n):
        result = []
        
        k = 0
        while True:
            added = False
            for b in base:
                val = b + k
                if val <= n:
                    if val % 2 == 1:
                        result.append(val)
                    added = True
            if not added:
                break
            k += 100
        
        return result


    def forward(self, x: Tensor) -> Tensor:
        # Run heads in parallel and concatenate
        # If any head produced L+1 due to even k, we slice back to match input L
        L_in = x.shape[-1]
        outs = []
        for head in self.heads:
            h_out = head(x)
            if h_out.shape[-1] > L_in:
                h_out = h_out[..., :L_in]
            outs.append(h_out)
            
        out = torch.cat(outs, dim=1)
        return self.fusion(out)

class ResidualEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ResidualEncoder, self).__init__()
        self.cfg = cfg
        self.device = cfg.system.device
        dim_latent = cfg.encoder_model.dim_en_latent
        dim_embedding = cfg.encoder_model.dim_emb
        self.__resnet = _ResNet(cfg)
        
        # Store downsample layer for residual strided convs
        self.downsample_layer = nn.Conv1d(in_channels=dim_latent, out_channels=dim_latent, kernel_size=3, stride=2, padding=0).to(cfg.system.device)
        
        # Check if temporal head refines ResNet features before time fusion.
        self.time_fusion = bool(getattr(cfg.encoder_model, "time_fusion", False))
        self.temporal_head_enabled = (
            getattr(cfg.training, 'temporal_head_enabled', False)
            and self.time_fusion
        )

        if self.temporal_head_enabled:
            multi_enabled = getattr(cfg.encoder_model, "multi_temporal_head_enabled", False)
            if multi_enabled:
                self.temporal_head = MultiScaleTemporalHead(dim_latent, cfg)
            else:
                kernel_size = cfg.encoder_model.conv_kernel_size
                padding = kernel_size // 2
                bias = cfg.encoder_model.layernorm_type == 'none' or not cfg.encoder_model.layernorm_elementwise_affine
                self.temporal_head = nn.Sequential(
                    getWeightNorm(
                        nn.Conv1d(dim_latent, dim_latent, kernel_size, padding=padding, bias=bias),
                        cfg,
                    ),
                    getLayerNorm(cfg.dataset.dim_series, cfg),
                    getActivation(cfg.encoder_model.activation_conv, cfg),
                )
        else:
            self.temporal_head = nn.Identity()

        pool_mode = getattr(cfg.encoder_model, "global_pooling", "max")
        # When TFE is on, optionally drop the pre-concat embedding norm so `emb` enters the merge
        # linear un-rescaled (the post-merge norm `time_fusion_post_norm` still normalises the output).
        # When TFE is off, the pre-concat norm IS the only normalisation and must always run.
        time_fusion_pre_norm = bool(getattr(cfg.encoder_model, "time_fusion_pre_norm", True))
        if self.time_fusion and not time_fusion_pre_norm:
            embedding_tail_norm: nn.Module = nn.Identity()
        else:
            embedding_tail_norm = get_embedding_output_norm(cfg, dim_embedding)
        self.__model = nn.Sequential(
            GlobalTemporalPool1d(dim_latent, pool_mode),
            Squeeze(),

                                    nn.Linear(dim_latent, dim_latent),
                                    getActivation(cfg.encoder_model.activation_linear, cfg),

                                    nn.Linear(dim_latent, dim_embedding, bias=False),
                                    embedding_tail_norm)

        self.__model.to(self.device)

        # time_fusion: pool (B,C,T) -> channel summary, project, merge into final embedding (no side output)
        self.time_fusion_tc1 = None
        self.time_fusion_tc2 = None
        self.time_fusion_merge = None
        # Re-apply post-encoder normalization after time_fusion_merge so the merged embedding
        # keeps the same per-row scale as the no-TFE branch (otherwise distance-space calibration drifts).
        self.time_fusion_post_norm = nn.Identity()
        if self.time_fusion:
            _h = getattr(cfg.encoder_model, "time_fusion_hidden_dim", None)
            h = dim_latent if _h is None else int(_h)
            h = max(h, 1)
            # Optionally wrap the three TFE Linears with weight_norm so the optimizer can rescale
            # each output unit independently of its direction. Helps when the merge linear sees
            # asymmetric input scales (emb un-normed via time_fusion_pre_norm=False, u un-normed by construction).
            tfe_wn = bool(getattr(cfg.encoder_model, "time_fusion_weight_norm", False))
            try:
                from torch.nn.utils.parametrizations import weight_norm as _wn  # PyTorch >= 1.13
            except Exception:
                from torch.nn.utils import weight_norm as _wn  # legacy fallback
            def _maybe_wn(linear: nn.Linear) -> nn.Module:
                return _wn(linear) if tfe_wn else linear
            # For long series, pool temporally before the linear to avoid huge param count
            _ts = int(cfg.dataset.dim_series)
            _pool_t = getattr(cfg.encoder_model, "time_fusion_pool_target", None)
            if _pool_t is not None and _pool_t < _ts:
                self.time_fusion_pool = nn.AdaptiveAvgPool1d(_pool_t).to(self.device)
                self.time_fusion_tc1 = _maybe_wn(nn.Linear(_pool_t, h)).to(self.device)
            else:
                self.time_fusion_pool = nn.Identity().to(self.device)
                self.time_fusion_tc1 = _maybe_wn(nn.Linear(_ts, h)).to(self.device)
            self.time_fusion_tc2 = _maybe_wn(nn.Linear(h, dim_embedding)).to(self.device)
            self.time_fusion_merge = _maybe_wn(nn.Linear(2 * dim_embedding, dim_embedding, bias=True)).to(self.device)
            self.time_fusion_post_norm = get_embedding_output_norm(cfg, dim_embedding).to(self.device)

    def forward(self, input: Tensor): 
        # Use local variable instead of instance variable to avoid state mutation
        identity = None
        
        # Get ResNet output (temporal codes before pooling)
        temporal_codes = self.__resnet(input)
        temporal_codes_for_weight = (
            self.temporal_head(temporal_codes) if self.temporal_head_enabled else temporal_codes
        )
        
        # Process for embeddings (pool temporal dimension)
        input = temporal_codes
    
        for layer in self.__model:
            if self.cfg.encoder_model.residual_strided_convs:
                if type(layer) == torch.nn.modules.conv.Conv1d:
                    if identity is None:
                        identity = input 
                    else:
                        input = input + self.downsample_layer(identity)
                        identity = input 

                    input = layer(input)
                else:
                    input = layer(input)

            else:
                input = layer(input)

        emb = input
        if self.time_fusion and self.time_fusion_tc1 is not None:
            tc_mean = temporal_codes_for_weight.mean(dim=-2)
            if hasattr(self, "time_fusion_pool") and not isinstance(self.time_fusion_pool, nn.Identity):
                tc_mean = self.time_fusion_pool(tc_mean.unsqueeze(1)).squeeze(1)
            u = F.relu(self.time_fusion_tc1(tc_mean))
            u = self.time_fusion_tc2(u)
            # __model uses Squeeze() with dim=None, which drops *all* size-1 axes; for batch_size==1
            # the embedding path becomes 1D (dim_emb,) while u stays (1, dim_emb) from Linear.
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            if u.dim() == 1:
                u = u.unsqueeze(0)
            emb = self.time_fusion_merge(torch.cat([emb, u], dim=-1))
            emb = self.time_fusion_post_norm(emb)

        return emb


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:

        return input

