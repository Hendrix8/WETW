"""Four WESEE branch ablations sharing the exact same initialized ResNet.

Full matches the paper implementation. A branch-free ResNet needs a readout:
use parameter-free global pooling (the configured SFE pooling operator), without
the SFE projection MLP. All arms return the same normalized 128-vector.
"""
import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from wetw._reference.model.ResidualAE import ResidualEncoder as ReferenceEncoder, GlobalTemporalPool1d
from wetw._reference.util.helper_functions import get_embedding_output_norm

VARIANTS = ('resnet', 'sfe', 'tfe', 'full')


class AblationEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.variant = str(cfg.encoder_model.arch_ablation)
        assert self.variant in VARIANTS
        assert not cfg.encoder_model.residual_strided_convs
        assert cfg.encoder_model.dim_en_latent == cfg.encoder_model.dim_emb == 128
        # Always initialize the full model before taking its modules. This keeps
        # backbone/common-branch initializations identical for a given seed.
        full_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        assert full_cfg.encoder_model.time_fusion
        reference = ReferenceEncoder(full_cfg)
        self.backbone = reference._ResidualEncoder__resnet
        self.output_norm = get_embedding_output_norm(cfg, 128)
        if self.variant in ('sfe', 'full'):
            self.sfe = reference._ResidualEncoder__model
        if self.variant == 'resnet':
            self.pool = GlobalTemporalPool1d(128, cfg.encoder_model.global_pooling)
        if self.variant in ('tfe', 'full'):
            self.temporal_head = reference.temporal_head
            self.temporal_pool = reference.time_fusion_pool
            self.tc1, self.tc2 = reference.time_fusion_tc1, reference.time_fusion_tc2
        if self.variant == 'full':
            self.merge = reference.time_fusion_merge
            self.output_norm = reference.time_fusion_post_norm
        # Unused reference modules (including its dormant strided-conv layer)
        # are deliberately not registered, counted or optimized.

    def forward(self, x):
        codes = self.backbone(x)
        if self.variant == 'resnet':
            return self.output_norm(self.pool(codes).flatten(1))
        sfe = self.sfe(codes).reshape(len(x), 128) if hasattr(self, 'sfe') else None
        if self.variant == 'sfe':
            return self.output_norm(sfe)
        temporal = self.temporal_head(codes).mean(dim=1)
        temporal = self.temporal_pool(temporal[:, None]).squeeze(1)
        tfe = self.tc2(F.relu(self.tc1(temporal)))
        if self.variant == 'tfe':
            return self.output_norm(tfe)
        return self.output_norm(self.merge(torch.cat([sfe, tfe], dim=1)))


def install():
    import wetw._reference.model.builder as builder
    builder.ResidualEncoder = AblationEncoder


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def check_architectures(cfg):
    """Check faithful full output, shared initialization, gradients and shapes."""
    seed = 123
    counts, backbone_states = {}, []
    for variant in VARIANTS:
        torch.manual_seed(seed)
        cfg.encoder_model.arch_ablation = variant
        model = AblationEncoder(cfg).to(cfg.system.device)
        backbone_states.append({k: v.detach().cpu().clone() for k, v in model.backbone.state_dict().items()})
        for batch in (1, 3):
            x = torch.randn(batch, 1, int(cfg.dataset.dim_series), device=cfg.system.device)
            z = model(x)
            assert z.shape == (batch, 128) and torch.isfinite(z).all()
        # Non-invariant objective: a sum of normalized squares would be constant.
        (z * torch.randn_like(z)).mean().backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, (variant, 'inactive parameters', missing)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        counts[variant] = count_parameters(model)
        if variant == 'full':
            torch.manual_seed(seed)
            reference = ReferenceEncoder(cfg).to(cfg.system.device).eval()
            model.eval()
            with torch.no_grad():
                assert torch.allclose(model(x), reference(x), rtol=1e-5, atol=1e-6)
        del model
    for state in backbone_states[1:]:
        assert all(torch.equal(v, state[k]) for k, v in backbone_states[0].items())
    return counts
