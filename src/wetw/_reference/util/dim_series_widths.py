"""Derive encoder/weight widths from dataset.dim_series (largest 2^n with 2^n < dim_series)."""

from __future__ import annotations

import logging
from typing import Tuple

from omegaconf import DictConfig, OmegaConf

from wetw._reference.util.helper_functions import max_layers_before_collapse

logger = logging.getLogger(__name__)


def largest_power_of_two_strictly_below(dim_series: int) -> int:
    """Return the largest :math:`2^n` such that :math:`2^n < \\texttt{dim_series}`."""
    if dim_series <= 1:
        return 1
    p = 1
    while p * 2 < dim_series:
        p *= 2
    return p


def token_dim_num_tokens_for_channels(channels: int) -> Tuple[int, int]:
    """
    Choose ``token_dim`` and ``num_tokens`` with ``token_dim * num_tokens == channels``.
    Prefer ``token_dim == 4`` when it divides ``channels`` (matches common weight_model configs).
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")
    for td in (4, 8, 2, 16, 32, 64, 1):
        if channels % td == 0:
            return td, channels // td
    return 1, channels


def apply_dim_series_derived_widths(cfg: DictConfig) -> Tuple[int, int, int]:
    """
    Set encoder/weight channel sizes to ``largest_power_of_two_strictly_below(dim_series)`` and
    set ``weight_model.token_dim`` / ``num_tokens`` so their product equals ``weight_model_channels``.
    """
    dim_series = int(cfg.dataset.dim_series)
    
    # Cap the auto-width to a manageable size on our system (e.g., 512)
    # This prevents 14B+ parameter models on long sequences when using large kernels.
    max_auto_width = getattr(cfg.encoder_model, "auto_width_max", 512)
    cap = min(largest_power_of_two_strictly_below(dim_series), max_auto_width)

    OmegaConf.update(cfg, "encoder_model.dim_emb", cap)
    OmegaConf.update(cfg, "encoder_model.dim_en_latent", cap)
    OmegaConf.update(cfg, "encoder_model.num_en_channels", cap)

    if OmegaConf.select(cfg, "encoder_model.projection_head.hidden_dim") is not None:
        OmegaConf.update(cfg, "encoder_model.projection_head.hidden_dim", cap)

    OmegaConf.update(cfg, "weight_model.hidden_dim", cap)
    OmegaConf.update(cfg, "weight_model.weight_model_channels", cap)
    td, nt = token_dim_num_tokens_for_channels(cap)
    OmegaConf.update(cfg, "weight_model.token_dim", td)
    OmegaConf.update(cfg, "weight_model.num_tokens", nt)
    assert td * nt == cap
    return cap, td, nt


def prepare_model_config_from_dataset(cfg: DictConfig) -> None:
    """
    Merge optional ``dataset.encoder_model`` / ``dataset.weight_model`` overrides, set
    ``strided_conv_num`` from ``dim_series``, and optionally apply power-of-two widths
    (unless ``dataset.auto_width_from_dim_series`` is false).
    """
    if OmegaConf.select(cfg, "dataset.encoder_model") is not None:
        cfg.encoder_model = OmegaConf.merge(cfg.encoder_model, cfg.dataset.encoder_model)
    if OmegaConf.select(cfg, "dataset.weight_model") is not None:
        cfg.weight_model = OmegaConf.merge(cfg.weight_model, cfg.dataset.weight_model)

    cfg.encoder_model.strided_conv_num = max_layers_before_collapse(cfg.dataset.dim_series)

    if getattr(cfg.dataset, "auto_width_from_dim_series", True):
        cap, td, nt = apply_dim_series_derived_widths(cfg)
        logger.info(
            "auto_width_from_dim_series: dim_series=%s -> cap=%s (largest 2^n < dim_series); "
            "weight_model token_dim=%s num_tokens=%s (product=%s)",
            cfg.dataset.dim_series,
            cap,
            td,
            nt,
            td * nt,
        )
