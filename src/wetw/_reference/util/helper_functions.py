import os
import warnings
import numpy as np
import torch
from typing import Optional, Tuple
from wetw._reference.model.activation import LeCunTanh


def sanitize_time_series_split(
    X: np.ndarray,
    dim_series: int,
    split_name: str = "data",
) -> np.ndarray:
    """
    Float32, NaN/inf -> 0, and align last dimension to ``dim_series`` (zero-pad or truncate)
    so train/valid/test match ``cfg.dataset.dim_series``.
    """
    if X is None:
        raise ValueError("sanitize_time_series_split: X is None")
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    if X.ndim == 1:
        n = int(X.size)
        if n % int(dim_series) != 0:
            raise ValueError(
                f"{split_name}: flat length {n} not divisible by dim_series={dim_series}"
            )
        X = X.reshape(-1, int(dim_series))
    elif X.ndim != 2:
        raise ValueError(f"{split_name}: expected 1D or 2D array, got shape {X.shape}")
    t = int(X.shape[1])
    d = int(dim_series)
    if t < d:
        warnings.warn(
            f"{split_name}: time length {t} < dim_series {d}; padding with zeros to match config.",
            UserWarning,
            stacklevel=2,
        )
        X = np.pad(X, ((0, 0), (0, d - t)), mode="constant", constant_values=0.0)
    elif t > d:
        warnings.warn(
            f"{split_name}: time length {t} > dim_series {d}; truncating to match config.",
            UserWarning,
            stacklevel=2,
        )
        X = X[:, :d].copy()
    return X.astype(np.float32, copy=False)


def get_data_load_dtype(cfg) -> np.dtype:
    """Get dtype for loading data from processed_train_fp / processed_test_fp.
    When preprocessed=True (default): float32 (preprocess.py always saves as float32).
    When preprocessed=False: use cfg.dataset.dtype (raw data, e.g. float64).
    Data is always converted to float32 after load. When skipping preprocessing,
    set processed_train_fp/processed_test_fp to raw data paths and preprocessed: false.
    """
    if getattr(cfg.dataset, 'preprocessed', True):
        return np.float32
    return parse_dtype(getattr(cfg.dataset, 'dtype', 'float64'))


def load_data_from_processed_files(cfg) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load X_train, X_test from processed_fp (.npz) or processed_train_fp/processed_test_fp (.npy).
    Returns (X_train, X_test) as float32 arrays, or None if files don't exist.
    """
    # Prefer single .npz
    if hasattr(cfg.dataset, 'processed_fp'):
        fp = cfg.dataset.processed_fp
        if os.path.exists(str(fp)) and str(fp).endswith('.npz'):
            d = np.load(fp)
            dim = int(cfg.dataset.dim_series)
            xt = sanitize_time_series_split(d["X_train"], dim, "X_train")
            xs = sanitize_time_series_split(d["X_test"], dim, "X_test")
            return xt, xs
    # Separate .npy files
    if not hasattr(cfg.dataset, 'processed_train_fp') or not hasattr(cfg.dataset, 'processed_test_fp'):
        return None
    train_fp = cfg.dataset.processed_train_fp
    test_fp = cfg.dataset.processed_test_fp
    if not os.path.exists(train_fp) or not os.path.exists(test_fp):
        return None
    if str(train_fp).endswith('.npy') and str(test_fp).endswith('.npy'):
        dim = int(cfg.dataset.dim_series)
        xt = sanitize_time_series_split(np.load(train_fp), dim, "train_sample")
        xs = sanitize_time_series_split(np.load(test_fp), dim, "test_sample")
        return xt, xs
    raise ValueError(f"Use .npz or .npy files. Got: {train_fp}")


def _infer_ucr_bundle_npz_from_train_path(train_data_fp: str) -> Optional[str]:
    """If ``train_data_fp`` is ``.../<Name>/samples_npy/train_sample.npy``, return ``.../<Name>/<Name>.npz``.

    Covers UCR tslearn bundles when ``dataset_fp`` is missing from config but the on-disk tree
    matches ``setup_ucr_dataset_bundle.py`` (one folder, one primary ``DatasetName.npz``).
    """
    if not train_data_fp:
        return None
    d = os.path.dirname(os.path.abspath(str(train_data_fp)))
    if os.path.basename(d) != "samples_npy":
        return None
    ds_dir = os.path.dirname(d)
    if not os.path.isdir(ds_dir):
        return None
    base = os.path.basename(ds_dir)
    preferred = os.path.join(ds_dir, f"{base}.npz")
    if os.path.isfile(preferred):
        return preferred
    npzs = sorted(
        f
        for f in os.listdir(ds_dir)
        if f.endswith(".npz") and os.path.isfile(os.path.join(ds_dir, f))
    )
    if len(npzs) == 1:
        return os.path.join(ds_dir, npzs[0])
    return None


def _ucr_bundle_train_y_aligned_to_train_sample(
    y_train_full: np.ndarray,
    train_sample_num_rows: int,
    valid_ratio: float = 0.2,
    seed: int = 42,
) -> Optional[np.ndarray]:
    """Apply the same train/valid index split as ``scripts_dtwrl/setup_ucr_dataset_bundle._export_samples_npy``.

    Train rows in ``train_sample.npy`` are ``X_train[tr_idx]`` with ``tr_idx = perm[:n_tr]``, not the first
    ``n_tr`` rows of the raw UCR array. Labels must be ``y_train_full[tr_idx]``, not ``y_train_full[:n_tr]``.
    """
    y_train_full = np.asarray(y_train_full)
    n = int(y_train_full.shape[0])
    if n == train_sample_num_rows:
        return y_train_full.copy()
    if n < train_sample_num_rows:
        return None
    n_valid = max(1, int(round(n * valid_ratio)))
    n_tr = n - n_valid
    if n_tr != train_sample_num_rows:
        return None
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    tr_idx = perm[:n_tr]
    return np.asarray(y_train_full)[tr_idx].copy()


def load_labels_from_processed_dir(
    train_data_fp: str,
    test_data_fp: Optional[str] = None,
    raw_data_dir: Optional[str] = None,
    processed_fp: Optional[str] = None,
    dataset_fp: Optional[str] = None,
    train_size: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load y_train, y_test. Tries (no internet required):
    1. Same dir as train_data_fp: y_train.npy, y_test.npy (prefer - already split when valid_split used)
    2. Derived names: {base}_y_train.npy, etc.
    3. raw_data_dir: y_train.npy, y_test.npy
    4. .npz with y_train/y_test: processed_fp, dataset_fp, or ``.../Name/samples_npy/`` → sibling ``Name.npz``.
       If the .npz has the full UCR training labels (longer than ``train_sample.npy``), the same
       random train/valid split as ``setup_ucr_dataset_bundle`` (valid_ratio=0.2, seed=42) is applied
       so labels match shuffled ``train_sample`` rows.

    If train_size is set and labels were loaded from .npy (already split), y_train may still be trimmed
    to train_size when longer (e.g. cap).

    Returns (y_train, y_test) as 0-indexed encoded labels, or None if not found.
    """
    from sklearn.preprocessing import LabelEncoder

    def _trim_train(yt: np.ndarray, yte: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if train_size is not None and len(yt) > train_size:
            yt = yt[:train_size]
        return yt, yte

    def _train_sample_row_count() -> Optional[int]:
        if train_size is not None:
            return int(train_size)
        if train_data_fp and str(train_data_fp).endswith(".npy") and os.path.isfile(str(train_data_fp)):
            try:
                return int(np.load(str(train_data_fp), mmap_mode="r").shape[0])
            except Exception:
                return None
        return None

    def _try_load(yt_fp: str, yte_fp: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not os.path.exists(yt_fp) or not os.path.exists(yte_fp):
            return None
        if not yt_fp.endswith('.npy'):
            return None
        y_train = np.load(yt_fp)
        y_test = np.load(yte_fp)
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)
        return y_train, y_test

    label_dir = os.path.dirname(train_data_fp)
    base = os.path.basename(train_data_fp)

    # 1. Standard names: y_train.npy, y_test.npy (prefer - preprocess saves split labels here)
    result = _try_load(
        os.path.join(label_dir, "y_train.npy"),
        os.path.join(label_dir, "y_test.npy"),
    )
    if result is not None:
        y_train, y_test = result
        y_train, y_test = _trim_train(y_train, y_test)
        return y_train, y_test

    # 2. Derived names: ArrowHead_y_train.npy from ArrowHead_train.npy
    y_train_name = base.replace("_train.npy", "_y_train.npy").replace("train.npy", "y_train.npy")
    y_test_name = base.replace("_train.npy", "_y_test.npy").replace("train.npy", "y_test.npy")
    if y_train_name != base:
        result = _try_load(os.path.join(label_dir, y_train_name), os.path.join(label_dir, y_test_name))
        if result is not None:
            y_train, y_test = result
            y_train, y_test = _trim_train(y_train, y_test)
            return y_train, y_test

    # 3. Raw data dir
    if raw_data_dir and os.path.isdir(raw_data_dir):
        result = _try_load(
            os.path.join(raw_data_dir, "y_train.npy"),
            os.path.join(raw_data_dir, "y_test.npy"),
        )
        if result is not None:
            y_train, y_test = result
            y_train, y_test = _trim_train(y_train, y_test)
            return y_train, y_test

    # 4. .npz with y_train / y_test: explicit paths, then sibling bundle (all UCR layouts)
    seen_npz = set()
    inferred = _infer_ucr_bundle_npz_from_train_path(str(train_data_fp))
    for nz in (processed_fp, dataset_fp, inferred):
        if not nz:
            continue
        nz_abs = os.path.normpath(os.path.abspath(str(nz)))
        if not nz_abs.endswith(".npz") or not os.path.isfile(nz_abs) or nz_abs in seen_npz:
            continue
        seen_npz.add(nz_abs)
        try:
            d = np.load(nz_abs)
        except Exception:
            continue
        if "y_train" in d and "y_test" in d:
            ytr_raw = np.asarray(d["y_train"])
            yte_raw = np.asarray(d["y_test"])
            ts_rows = _train_sample_row_count()
            if ts_rows is not None and ytr_raw.shape[0] > ts_rows:
                ytr_raw = _ucr_bundle_train_y_aligned_to_train_sample(ytr_raw, ts_rows)
                if ytr_raw is None:
                    warnings.warn(
                        "y_train from .npz could not be aligned to train_sample length using "
                        "bundle defaults (valid_ratio=0.2, seed=42). Expected y_train.npy beside "
                        "train_sample, or rebuild samples_npy with setup_ucr_dataset_bundle.py.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(ytr_raw)
            y_test = encoder.transform(yte_raw)
            y_train, y_test = _trim_train(y_train, y_test)
            return y_train, y_test

    return None


def load_labels_from_dataset_cfg(
    cfg,
    train_data_fp: Optional[str] = None,
    test_data_fp: Optional[str] = None,
    train_size: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load ``y_train``, ``y_test`` from paths on ``cfg.dataset`` (y_*.npy, ``processed_fp``, ``dataset_fp``, or inferred ``samples_npy`` sibling .npz)."""
    ds = cfg.dataset
    train_fp = train_data_fp or getattr(ds, "processed_train_fp", None)
    test_fp = test_data_fp or getattr(ds, "processed_test_fp", None)
    if not train_fp:
        return None
    raw_data_dir = None
    dfp = getattr(ds, "dataset_fp", None)
    if dfp:
        raw_data_dir = os.path.dirname(str(dfp))
    processed_fp = getattr(ds, "processed_fp", None)
    pf_ok = (
        processed_fp
        and os.path.exists(str(processed_fp))
        and str(processed_fp).endswith(".npz")
    )
    df_ok = dfp and os.path.exists(str(dfp)) and str(dfp).endswith(".npz")
    return load_labels_from_processed_dir(
        str(train_fp),
        test_data_fp=str(test_fp) if test_fp else None,
        raw_data_dir=raw_data_dir,
        processed_fp=str(processed_fp) if pf_ok else None,
        dataset_fp=str(dfp) if df_ok else None,
        train_size=train_size,
    )


def parse_dtype(dtype_str: Optional[str]) -> np.dtype:
    """Convert string dtype name to numpy dtype.
    Args:
        dtype_str: String like 'float32', 'float64', 'f32', etc. None/empty -> float32.
    Returns:
        numpy dtype object
    """
    if dtype_str is None or dtype_str == '':
        return np.float32
    dtype_str = str(dtype_str).lower().strip()
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'f32': np.float32,
        'f64': np.float64,
        'float': np.float32,
        'double': np.float64,
    }
    if dtype_str in dtype_map:
        return dtype_map[dtype_str]
    if hasattr(np, dtype_str):
        return getattr(np, dtype_str)
    raise ValueError(f"Unknown dtype: {dtype_str}. Supported: float32, float64, f32, f64, float, double")
from omegaconf import DictConfig
from typing import Union, List
from wetw._reference.model.normalization import AdaNorm

def pad_with_zeros(arr, target_length):
    if arr.shape[1] < target_length:
        padding = target_length - arr.shape[1]
        arr_padded = np.pad(arr, ((0, 0), (0, padding)), mode='constant', constant_values=0)
    else:
        arr_padded = arr[:, :target_length]  # Truncate if longer
    return arr_padded


def weighted_pairwise_distance(
    E_i,
    E_j,
    w,
    p=2,
    normalize_weight_sum_to_dim=False,
    scale_distance_inv_sqrt_dim=False,
):
    """
    Efficiently compute the weighted pairwise distance between two embeddings.

    Parameters:
    - E_i: First embedding (Tensor of shape [batch_size, embedding_dim])
    - E_j: Second embedding (Tensor of shape [batch_size, embedding_dim])
    - w: Weight vector (Tensor of shape [batch_size, embedding_dim])
    - p: Power parameter for the distance (p=2 for Euclidean distance)
    - normalize_weight_sum_to_dim: If True, rescale w per row so sum_k w_{ik} = D. With uniform w=1,
      matches standard Euclidean (sum-of-squares gauge; weights normalized to sum to d).
    - scale_distance_inv_sqrt_dim: If True, scale the returned distance by 1/sqrt(D). Optional; tune with targets/LR if training.

    Returns:
    - Weighted distance (Tensor of shape [batch_size])
    """
    # Compute weighted p-norm distance in a single operation
    # E_i = F.normalize(E_i, p=2, dim=1)
    # E_j = F.normalize(E_j, p=2, dim=1)
    epsilon = 1e-6
    w = torch.clamp(w, min=1e-6)
    if normalize_weight_sum_to_dim:
        d = float(E_i.size(-1))
        w = w * (d / (w.sum(dim=-1, keepdim=True) + epsilon))

    distance = torch.sum(w * torch.abs(E_i - E_j) ** p, dim=1).add(epsilon).pow(1.0 / p)
    if scale_distance_inv_sqrt_dim:
        d = E_i.size(-1)
        distance = distance / (d ** 0.5)

    return distance


def wetw_distance_kwargs_from_cfg(cfg):
    """Kwargs for weighted_pairwise_distance from cfg.weight_model.seanet_metric."""
    wm = getattr(cfg, "weight_model", None)
    if wm is None:
        return {}
    sm = getattr(wm, "seanet_metric", None)
    if sm is None or not bool(getattr(sm, "enabled", False)):
        return {}
    kwargs = {}
    if bool(getattr(sm, "normalize_weights_sum_to_embed_dim", True)):
        kwargs["normalize_weight_sum_to_dim"] = True
    if bool(getattr(sm, "scale_weighted_distance_by_inv_sqrt_dim", False)):
        kwargs["scale_distance_inv_sqrt_dim"] = True
    return kwargs


def training_scale_dists_enabled(cfg: DictConfig) -> bool:
    """
    When True, scale stored target distances by 1/sqrt(dim_series) and predicted
    pairwise distances by 1/sqrt(dim_emb) (see PairwiseDistDiff). ``scale_distances_by_dim``
    is a legacy alias for ``training.scale_dists``.
    """
    t = cfg.training
    return bool(getattr(t, "scale_dists", False) or getattr(t, "scale_distances_by_dim", False))


def getActivation(name: str, cfg: DictConfig) -> torch.nn.Module:
    if name == 'tanh':
        return torch.nn.Tanh()
    elif name == 'lecuntanh':
        return LeCunTanh()
    elif name == 'relu':
        return torch.nn.ReLU()
    elif name == 'leakyrelu':
        relu_slope = cfg.encoder_model.relu_slope
        return torch.nn.LeakyReLU(relu_slope)
    elif name == 'prelu':
        prelu_params = cfg.encoder_model.prelu_params
        return torch.nn.PReLU(prelu_params)
    return torch.nn.Identity()

def getWeightNorm(model: torch.nn.Module, cfg: DictConfig) -> torch.nn.Module:
    weightnorm_type = cfg.encoder_model.weightnorm_type

    if weightnorm_type == 'weightnorm':
        return torch.nn.utils.weight_norm(model, dim=cfg.encoder_model.weightnorm_dim)

    return model

def getLayerNorm(shape: Union[int, List[int], torch.Size], cfg: DictConfig) -> torch.nn.Module:
    
    layernorm_type = cfg.encoder_model.layernorm_type
    if layernorm_type == 'layernorm':
        return torch.nn.LayerNorm(shape, elementwise_affine=cfg.encoder_model.layernorm_elementwise_affine)
    
    elif layernorm_type == 'adanorm':
        return AdaNorm(shape, cfg.encoder_model.adanorm_k, cfg.encoder_model.adanorm_scale, cfg.encoder_model.eps, cfg.encoder_model.layernorm_elementwise_affine)

    return torch.nn.Identity()


class ZNormLastDim(torch.nn.Module):
    """
    Z-score along the last dimension independently for each leading batch index (no affine).
    Uses unbiased std by default (common "z-score" convention); differs slightly from
    ``nn.LayerNorm`` which uses the biased (population) variance estimator.
    """

    def __init__(self, eps: float, unbiased: bool = True):
        super().__init__()
        self.eps = eps
        self.unbiased = unbiased

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=self.unbiased)
        return (x - mean) / (std + self.eps)


def get_embedding_output_norm(cfg: DictConfig, dim: int) -> torch.nn.Module:
    """
    Normalization after the encoder's final linear / projection head.

    ``layernorm2_znorm=True`` selects explicit per-row z-scoring via :class:`ZNormLastDim`.
    Otherwise, ``encoder_normalize_embedding`` enables ``nn.LayerNorm(dim, elementwise_affine=False)``.
    """
    znorm = bool(getattr(cfg.encoder_model, "layernorm2_znorm", False))
    enc = bool(getattr(cfg.encoder_model, "encoder_normalize_embedding", False))
    eps = float(getattr(cfg.encoder_model, "eps", 1e-8))
    if znorm:
        return ZNormLastDim(eps=eps, unbiased=True)
    if enc:
        return torch.nn.LayerNorm(dim, elementwise_affine=False)
    return torch.nn.Identity()


def getDilation(depth: int, cfg: DictConfig) -> int:
    dilation_type = cfg.encoder_model.dilation_type

    if dilation_type == 'exponential':
        return int(2 ** (depth - 1))
    
    elif dilation_type == 'linear':
        return cfg.encoder_model.dilation_base + cfg.encoder_model.dilation_slope * (depth - 1)

    return cfg.encoder_model.dilation_constant


def max_layers_before_collapse(initial_dim, kernel_size=3, stride=2, padding=0):
    """
    Compute how many Strided conv layers can be applied 
    before spatial dimension collapses to <= 1.
    """
    dim = initial_dim
    num_layers = 0

    while True:
        next_dim = (dim - kernel_size + 2 * padding) // stride + 1
        if next_dim <= 1:
            break
        dim = next_dim
        num_layers += 1

    return num_layers


def get_divisors(N):
    divisors = []
    for i in range(1, int(N**0.5) + 1):
        if N % i == 0:
            divisors.append(i)
            if i != N // i:  # Avoid adding the square root twice for perfect squares
                divisors.append(N // i)
    return sorted(divisors)


def build_experiment_name_suffix(cfg) -> str:
    """Build a short descriptive suffix from config for experiment folder names.

    Naming convention (short names like ww0.1):
    - ww{frac}: warping window (e.g. ww0.1)
    - resnet: encoder_model.backbone
    - eucl / wetw: approximator (euclidean vs WETW)
    - pair / add: WETW weight_mode (pairwise vs additive, only when wetw)
    - r{n}: num resblocks (e.g. r6)
    - c{n}: num channels (e.g. c256)
    - strided / maxpool: downsampling (residual strided convs vs maxpool)
    - time_fusion: when encoder_model.time_fusion is true (fused temporal features in emb)
    - d{n}: dim_emb (e.g. d128)
    - rell1 | rell2 | relhub: training.distance_loss_type when not l1
    - {datetime}: YYYY-MM-DD-HH-MM-SS

    Example: sald_ww0.1_resnet_eucl_r6c256_maxpool-2026-03-12-12-55-16
    """
    parts = []

    # Warping window
    ww = getattr(getattr(cfg, 'training', None), 'warping_window', None)
    if ww is not None:
        parts.append(f"ww{ww}")

    # Encoder backbone (resnet)
    enc = getattr(cfg, 'encoder_model', None)
    if enc is not None:
        backbone = str(getattr(enc, 'backbone', 'resnet')).lower()
        parts.append(backbone)

    # Approximator
    wm = getattr(cfg, 'weight_model', None)
    approx = getattr(wm, 'approximator', 'wetw')
    parts.append('eucl' if approx == 'euclidean' else 'wetw')
    # WETW weight_mode: pairwise | additive (only when wetw)
    if approx == 'wetw' and wm:
        wmode = getattr(wm, 'weight_mode', 'pairwise')
        parts.append('pair' if wmode == 'pairwise' else 'add')

    # Encoder architecture (ResNet hyperparameters)
    if enc:
        r = getattr(enc, 'num_en_resblock', None)
        c = getattr(enc, 'num_en_channels', None)
        if r is not None and c is not None:
            parts.append(f"r{r}c{c}")
        elif r is not None:
            parts.append(f"r{r}")
        elif c is not None:
            parts.append(f"c{c}")
        strided = getattr(enc, 'residual_strided_convs', None)
        if strided is True:
            parts.append('strided')
        elif strided is False:
            parts.append('maxpool')
        d = getattr(enc, 'dim_emb', None)
        if d is not None:
            parts.append(f"d{d}")
        gp = getattr(enc, 'global_pooling', 'max') or 'max'
        if gp != 'max':
            parts.append(f"pool{gp[:3]}")
        if bool(getattr(enc, 'time_fusion', False)):
            parts.append('time_fusion')

    # Primary training loss on distances (when not default L1 on |pred-target|)
    tr = getattr(cfg, 'training', None)
    if tr is not None:
        dl = getattr(tr, 'distance_loss_type', 'l1') or 'l1'
        if dl != 'l1':
            short = {
                'relative_l1': 'rell1',
                'relative_l2': 'rell2',
                'relative_huber': 'relhub',
            }.get(dl, dl.replace('_', ''))
            parts.append(short)

    wm = getattr(cfg, 'weight_model', None)
    if wm is not None:
        ca = int(getattr(wm, 'cross_attention_layers', 0) or 0)
        if ca > 0:
            parts.append(f"xatt{ca}")
        tsa = int(getattr(wm, 'token_self_attention_layers', 0) or 0)
        if tsa > 0:
            parts.append(f"tsa{tsa}")

    return '_'.join(parts)
