"""
Profile I/O: save and load evaluation data for figure recreation.
One file per method per dataset under exps_dir/profiles/{dataset}/.
Format: .npz for arrays, .json for metadata and structure.

Structure: exps_dir/profiles/{method}.npz and {method}.json (inside each experiment folder)

Methods and data schema:
  rel_error: relative_errors, dtw_distances, embedding_distances, mean_error, std_error
  simsearch: recall_at_k_vs_n, n_values_valid, k_gt_vals, query, db, gt_indices, wetw_indices, gt_dists, wetw_dists
  ncc: recall_by_metric, precision_by_metric, f1_by_metric, classes, overall_metrics, y_test
  clust: nmi_train, nmi_test, X_train_emb, X_test_emb, cluster_labels_train_{dist}, cluster_labels_test_{dist}

Usage:
  from wetw._reference.util.profile_io import save_profile, load_profile, list_profiles

  save_profile(exps_dir, dataset_name, 'rel_error', data_dict, meta_dict)
  data, meta = load_profile(exps_dir, dataset_name, 'rel_error')
  methods = list_profiles(exps_dir, dataset_name)
"""
import os
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple


def _profile_dir(exps_dir: str, dataset_name: Optional[str]) -> str:
    """Return profiles/ directory path (inside experiment folder)."""
    return os.path.join(exps_dir, "profiles")


def get_profile_path(exps_dir: str, dataset_name: Optional[str], method: str) -> Tuple[str, str]:
    """Return (npz_path, json_path) for a method profile."""
    d = _profile_dir(exps_dir, dataset_name)
    return os.path.join(d, f"{method}.npz"), os.path.join(d, f"{method}.json")


def save_profile(
    exps_dir: str,
    dataset_name: Optional[str],
    method: str,
    data: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save profile data. Arrays go to .npz, metadata to .json.
    data: dict of arrays and scalars (arrays saved to npz, scalars to json)
    meta: optional extra metadata (saved to json)
    """
    d = _profile_dir(exps_dir, dataset_name)
    os.makedirs(d, exist_ok=True)
    npz_path, json_path = get_profile_path(exps_dir, dataset_name, method)

    # Separate arrays from scalars
    npz_data = {}
    json_data = {"method": method, "dataset_name": dataset_name, **(meta or {})}

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            npz_data[k] = v
        elif isinstance(v, (int, float, str, bool, type(None))):
            json_data[k] = v
        elif isinstance(v, (list, tuple)) and (not v or isinstance(v[0], (int, float, str, bool))):
            json_data[k] = list(v)
        elif isinstance(v, dict):
            # Flatten dict: arrays/lists of numbers -> npz, rest -> json
            for kk, vv in v.items():
                flat_key = f"{k}_{kk}"
                if isinstance(vv, np.ndarray):
                    npz_data[flat_key] = vv
                elif isinstance(vv, (list, tuple)) and vv and isinstance(vv[0], (int, float)):
                    npz_data[flat_key] = np.array(vv)
                elif isinstance(vv, dict):
                    json_data[flat_key] = vv
                elif isinstance(vv, (int, float, str, bool, type(None))):
                    json_data[flat_key] = vv
                else:
                    try:
                        npz_data[flat_key] = np.asarray(vv)
                    except (ValueError, TypeError):
                        json_data[flat_key] = str(vv)
            json_data[f"_{k}_keys"] = list(v.keys())
        else:
            json_data[k] = str(v)

    if npz_data:
        np.savez_compressed(npz_path, **npz_data)
    else:
        if os.path.exists(npz_path):
            os.remove(npz_path)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return d


def load_profile(
    exps_dir: str,
    dataset_name: Optional[str],
    method: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load profile. Returns (data, meta).
    Reconstructs dicts from flattened keys where _keys is present.
    """
    npz_path, json_path = get_profile_path(exps_dir, dataset_name, method)
    data = {}
    meta = {}

    if os.path.exists(json_path):
        with open(json_path) as f:
            meta = json.load(f)
        for k, v in meta.items():
            if k not in ("method", "dataset_name"):
                data[k] = v

    if os.path.exists(npz_path):
        npz = np.load(npz_path, allow_pickle=True)
        for k in npz.files:
            data[k] = npz[k]

    # Reconstruct dicts from flattened form
    keys_to_remove = []
    for k in list(data.keys()):
        if k.startswith("_") and k.endswith("_keys"):
            base = k[1:-5]
            sub_keys = data.pop(k)
            keys_to_remove.append(k)
            recon = {}
            for sk in sub_keys:
                flat_key = f"{base}_{sk}"
                if flat_key in data:
                    recon[sk] = data.pop(flat_key)
                keys_to_remove.append(flat_key)
            data[base] = recon

    return data, meta


def list_profiles(exps_dir: str, dataset_name: Optional[str] = None) -> list:
    """List available profile methods in the experiment's profiles folder."""
    d = _profile_dir(exps_dir, dataset_name)
    if not os.path.exists(d):
        return []
    return [f.replace(".json", "") for f in os.listdir(d) if f.endswith(".json")]
