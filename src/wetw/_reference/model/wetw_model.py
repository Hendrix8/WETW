import torch.nn as nn
import torch 
import torch.nn.functional as F


def _mha_nhead(embed_dim: int, preferred: int = 4) -> int:
    """Pick a head count that divides embed_dim (required by MultiheadAttention)."""
    for h in (preferred, 2, 1):
        if embed_dim % h == 0:
            return h
    return 1


class CrossSeriesTokenBlock(nn.Module):
    """Ablated cross-series interaction: queries on one side attend to the other token sequence."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        self.cross_ij = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_ji = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm_i = nn.LayerNorm(d_model)
        self.norm_j = nn.LayerNorm(d_model)

    def forward(self, tokens_i: torch.Tensor, tokens_j: torch.Tensor):
        ii, _ = self.cross_ij(tokens_i, tokens_j, tokens_j, need_weights=False)
        jj, _ = self.cross_ji(tokens_j, tokens_i, tokens_i, need_weights=False)
        return self.norm_i(tokens_i + ii), self.norm_j(tokens_j + jj)


class PairConvModel(nn.Module):
    def __init__(self, embed_dim=64, num_tokens=16, token_dim=8, num_layers=2, 
                 conv_channels=128, kernel_size=3, output_dim=128,
                 weight_mode='pairwise',
                 cross_attention_layers=0,
                 cross_attention_heads=4,
                 cross_attention_dropout=0.0,
                 token_self_attention_layers=0,
                 token_self_attention_dropout=0.1,
                 use_relational_tokens=False,
                 token_pooling='mean',
                 strictly_positive_weights=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.weight_mode = weight_mode  # 'pairwise' | 'additive'
        # Pairwise only: softplus(output_proj) so every dim > 0 (additive path already uses softplus)
        self.strictly_positive_weights = bool(strictly_positive_weights)
        
        if weight_mode == 'additive':
            # Additive: w(enc(x)) + w(enc(y)) with w >= 0. Maps single embedding -> per-dim weights.
            # Input is raw embedding (B, embed_dim), not tokenized - use embed_dim
            self.additive_proj = nn.Linear(embed_dim, output_dim)
            self.use_relational_tokens = False
            self.token_pooling = 'mean'
        else:
            if token_pooling not in ('mean', 'mean_max'):
                raise ValueError("token_pooling must be 'mean' or 'mean_max'")
            self.use_relational_tokens = use_relational_tokens
            self.token_pooling = token_pooling
            # Pairwise: w(enc(x), enc(y)) - relational weight from both embeddings
            self.token_proj_i = nn.Linear(token_dim, embed_dim)
            self.token_proj_j = nn.Linear(token_dim, embed_dim)
            
            # Convolutional stack (1D conv over token sequence)
            conv_layers = []
            in_channels = embed_dim
            for _ in range(num_layers):
                conv_layers.append(nn.Conv1d(in_channels, conv_channels, kernel_size, padding=kernel_size // 2))
                conv_layers.append(nn.ReLU())
                in_channels = conv_channels
            self.conv_net = nn.Sequential(*conv_layers)
            
            # Final reduction + output (mean_max doubles conv summary width)
            pool_mult = 2 if token_pooling == 'mean_max' else 1
            self.output_proj = nn.Linear(conv_channels * pool_mult, output_dim)

            # (1) Cross-series token mixing before concat (ablation: cross_attention_layers > 0)
            self.cross_blocks = None
            if cross_attention_layers and cross_attention_layers > 0:
                nh = _mha_nhead(embed_dim, cross_attention_heads)
                self.cross_blocks = nn.ModuleList(
                    [
                        CrossSeriesTokenBlock(embed_dim, nh, dropout=cross_attention_dropout)
                        for _ in range(cross_attention_layers)
                    ]
                )

            # (3) Optional self-attention on concatenated token sequence before conv
            self.token_self_attn = None
            if token_self_attention_layers and token_self_attention_layers > 0:
                nh = _mha_nhead(embed_dim, cross_attention_heads)
                ff = max(64, 4 * embed_dim)
                enc_layer = nn.TransformerEncoderLayer(
                    embed_dim,
                    nh,
                    dim_feedforward=ff,
                    dropout=token_self_attention_dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.token_self_attn = nn.TransformerEncoder(
                    enc_layer, num_layers=token_self_attention_layers
                )
    
    def forward(self, E_i, E_j , wandb=None, i=None, j=None, mask=None, warping_paths=None, curr_lr=None, init_lr=None):
        if self.weight_mode == 'additive':
            # Additive: w = |w(enc(x)) + w(enc(y))| with w(enc) >= 0 via softplus
            w_i = F.softplus(self.additive_proj(E_i))  # (B, output_dim)
            w_j = F.softplus(self.additive_proj(E_j))  # (B, output_dim)
            return w_i + w_j  # (B, output_dim), both >= 0 so sum >= 0
        
        # Pairwise: relational weight from both embeddings
        batch_size = E_i.size(0)
        
        # Step 1: Split into tokens
        tokens_i = E_i.view(batch_size, self.num_tokens, self.token_dim)  # (B, 16, 8)
        tokens_j = E_j.view(batch_size, self.num_tokens, self.token_dim)  # (B, 16, 8)
        
        # Step 2: Project to embed_dim
        tokens_i = self.token_proj_i(tokens_i)  # (B, 16, embed_dim)
        tokens_j = self.token_proj_j(tokens_j)  # (B, 16, embed_dim)

        if self.cross_blocks is not None:
            for blk in self.cross_blocks:
                tokens_i, tokens_j = blk(tokens_i, tokens_j)
        
        # Step 3: Combine (optional |Δ| and elementwise product, per TKDE-style relational features)
        if self.use_relational_tokens:
            diff = torch.abs(tokens_i - tokens_j)
            prod = tokens_i * tokens_j
            combined = torch.cat([tokens_i, tokens_j, diff, prod], dim=1)
        else:
            combined = torch.cat([tokens_i, tokens_j], dim=1)

        if self.token_self_attn is not None:
            combined = self.token_self_attn(combined)
        
        # Step 4: Convolution
        conv_input = combined.permute(0, 2, 1)  # (B, embed_dim, 32)
        conv_output = self.conv_net(conv_input)
        
        # Step 5: Pool over token positions
        if self.token_pooling == 'mean_max':
            pooled = torch.cat([conv_output.mean(dim=2), conv_output.amax(dim=2)], dim=1)
        else:
            pooled = conv_output.mean(dim=2)
        
        # Step 6: Final projection
        out = self.output_proj(pooled)  # (B, output_dim)
        if self.strictly_positive_weights:
            out = F.softplus(out)
        return out


def pair_conv_kwargs_from_cfg(cfg, output_dim: int) -> dict:
    """Build kwargs for PairConvModel from config (single place for train / eval loaders)."""
    wm = cfg.weight_model
    weight_mode = getattr(wm, "weight_mode", "pairwise")
    embed_dim = output_dim if weight_mode == "additive" else wm.hidden_dim
    return dict(
        embed_dim=embed_dim,
        num_tokens=wm.num_tokens,
        token_dim=wm.token_dim,
        num_layers=wm.weight_model_layers,
        conv_channels=wm.weight_model_channels,
        kernel_size=wm.kernel_size,
        output_dim=output_dim,
        weight_mode=weight_mode,
        cross_attention_layers=getattr(wm, "cross_attention_layers", 0),
        cross_attention_heads=getattr(wm, "cross_attention_heads", 4),
        cross_attention_dropout=getattr(wm, "cross_attention_dropout", 0.0),
        token_self_attention_layers=getattr(wm, "token_self_attention_layers", 0),
        token_self_attention_dropout=getattr(wm, "token_self_attention_dropout", 0.1),
        use_relational_tokens=getattr(wm, "use_relational_tokens", False),
        token_pooling=getattr(wm, "token_pooling", "mean"),
        strictly_positive_weights=bool(getattr(wm, "strictly_positive_weights", False)),
    )
