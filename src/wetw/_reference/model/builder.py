# coding = utf-8

from torch import nn, Tensor
from omegaconf import DictConfig
from wetw._reference.model.ResidualAE import ResidualEncoder
from wetw._reference.model.dist_calib import LearnedPredDistanceCalib


class ModelBuilder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ModelBuilder, self).__init__()
        self.cfg = cfg
        device = cfg.system.device

        backbone = str(getattr(cfg.encoder_model, "backbone", "resnet")).lower()
        if backbone != "resnet":
            raise ValueError(f"encoder_model.backbone must be 'resnet' (the WESee backbone), got {backbone!r}")
        self.encoder = ResidualEncoder(cfg)

        self.encoder.to(device)

        cal_mode = getattr(cfg.training, "pred_distance_learned_calib", "off")
        cal_mode_s = str(cal_mode or "off").lower()
        if cal_mode_s in ("off", "false", "0", "none", ""):
            self.pred_dist_calib = None
        else:
            cal_bias0 = float(getattr(cfg.training, "pred_distance_learned_bias_init", 0.0))
            self.pred_dist_calib = LearnedPredDistanceCalib(
                mode=cal_mode_s, bias_init=cal_bias0
            ).to(device)

    def encode(self, input: Tensor):
        out = self.encoder(input)
        return out
                    
    def forward(self, input: Tensor):
        embedding = self.encode(input)
        return embedding

