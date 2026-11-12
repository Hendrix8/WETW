import copy
import logging  
import os
from timeit import default_timer as timer
from ctypes import CDLL, c_char_p, c_long
from _ctypes import dlclose
from omegaconf import DictConfig
import numpy as np
from wetw._reference.util.helper_functions import get_data_load_dtype, sanitize_time_series_split
from torch.utils.data import DataLoader
from mpmath import mp
import numpy as np
from multiprocessing import Pool
from dtaidistance import dtw
import ctypes
from tqdm import tqdm
import time
from wetw._reference.util.timing_utils import compute_time_start, compute_time_elapsed 


class Sampler: 
    def __init__(self, cfg: DictConfig, size_db, train_size, valid_size, logger=None, data_array=None):
        
            self.cfg = cfg 
            # Use provided logger or get existing logger (don't reconfigure logging)
            if logger is None:
                self.logger = logging.getLogger(self.__class__.__name__)
            else:
                self.logger = logger

    
            self.size_db = size_db
            # Resolve data path: data_array (pre-loaded), processed_fp (.npz), or processed_train_fp (.npy/.bin)
            if data_array is not None:
                self.dataset_filepath = None
                self._data_array = data_array.astype(np.float32)
            else:
                processed_fp = getattr(cfg.dataset, 'processed_fp', None)
                if processed_fp and str(processed_fp).endswith('.npz'):
                    self.dataset_filepath = processed_fp
                    self._data_array = np.load(processed_fp)['X_train'].astype(np.float32)
                else:
                    self.dataset_filepath = cfg.dataset.processed_train_fp
                    self._data_array = None
            self.db_sorted_indices_filepath = self.cfg.paths.db_sorted_indices_path

            self.dim_series = cfg.dataset.dim_series

            self.size_train = train_size
            self.size_valid = valid_size
            self.size_extra = 17

            self.batch_loading_size = cfg.training.batch_loading_size

            self.target_dist_function = cfg.training.target_dist_function

            self.warping_window = int(self.cfg.training.warping_window * self.dim_series)

    def sample(self, all_data=False) -> np.ndarray:
        
        # If pre-split train and valid files exist, use them as-is (no mixing/splitting)
        processed_train_fp = getattr(self.cfg.dataset, 'processed_train_fp', None)
        processed_valid_fp = getattr(self.cfg.dataset, 'processed_valid_fp', None)
        if (processed_train_fp and processed_valid_fp
                and os.path.exists(processed_train_fp) and os.path.exists(processed_valid_fp)):
            # Load train samples
            if str(processed_train_fp).endswith('.npy'):
                try:
                    train_samples = np.load(processed_train_fp, allow_pickle=False).astype(np.float32)
                except Exception:
                    train_samples = np.fromfile(processed_train_fp, dtype=np.float32)
                    train_samples = train_samples.reshape(-1, self.dim_series)
            else:
                load_dtype = get_data_load_dtype(self.cfg)
                train_samples = np.fromfile(processed_train_fp, dtype=load_dtype).reshape(-1, self.dim_series).astype(np.float32)
            # Load valid samples
            if str(processed_valid_fp).endswith('.npy'):
                try:
                    valid_samples = np.load(processed_valid_fp, allow_pickle=False).astype(np.float32)
                except Exception:
                    valid_samples = np.fromfile(processed_valid_fp, dtype=np.float32)
                    valid_samples = valid_samples.reshape(-1, self.dim_series)
            else:
                load_dtype = get_data_load_dtype(self.cfg)
                valid_samples = np.fromfile(processed_valid_fp, dtype=load_dtype).reshape(-1, self.dim_series).astype(np.float32)
            train_samples = sanitize_time_series_split(train_samples, self.dim_series, "train_sample")
            valid_samples = sanitize_time_series_split(valid_samples, self.dim_series, "valid_sample")
            self.logger.info(f'Loaded pre-split samples from {processed_train_fp} and {processed_valid_fp}')
            print(f'Loaded pre-split samples: train={train_samples.shape[0]}, valid={valid_samples.shape[0]}')
        else:
            # seasam sampling
            if not all_data:
                if not getattr(self.cfg.dataset, 'preprocessed', True):
                    raise ValueError(
                        "SEAsam sampling (all_data=False) requires preprocessed data (float32). "
                        "With raw float64 data, use all_data=True or run preprocessing first."
                    )
                self.logger.info('initialize sorted indices')

                seasam_libpath = self.cfg.paths.seasam_libpath
                if not os.path.isabs(seasam_libpath):
                    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                    seasam_libpath = os.path.join(project_root, seasam_libpath)

                # SEAsam C library needs a binary file path; for .npz/.npy we write to temp file
                data_path = self.dataset_filepath
                temp_path = None
                data_for_c = None
                if hasattr(self, '_data_array') and self._data_array is not None:
                    data_for_c = self._data_array
                elif self.dataset_filepath and str(self.dataset_filepath).endswith('.npy'):
                    data_for_c = np.load(self.dataset_filepath).astype(np.float32)
                if data_for_c is not None:
                    import tempfile
                    f = tempfile.NamedTemporaryFile(suffix='.bin', delete=False)
                    data_for_c.tofile(f.name)
                    f.close()
                    data_path = f.name
                    temp_path = f.name

                try:
                    c_functions = CDLL(seasam_libpath)
                    return_code = c_functions.sample_coconut(c_char_p(data_path.encode('ASCII')),
                                                        c_long(self.size_db),
                                                        c_char_p(self.db_sorted_indices_filepath.encode('ASCII')),
                                                        c_char_p(b'skip'),
                                                        self.size_train,
                                                        c_char_p(b'skip'),
                                                        self.size_valid,
                                                        self.dim_series,
                                                        self.cfg.training.seasam_cardinality,
                                                        self.cfg.training.dim_seasam)
                    dlclose(c_functions._handle)
                    assert return_code == 0
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)

                self.logger.info('SEAsam Done.')
                print("SEAsam Done.")

                indices = np.fromfile(self.db_sorted_indices_filepath, dtype=int, count=self.size_train + self.size_valid + self.size_extra)
                indices = np.sort(indices)
                samples = []
                samples_idx = []
                num_loaded = 0
                num2load = self.size_train + self.size_valid

                if hasattr(self, '_data_array') and self._data_array is not None:
                    data_arr = self._data_array
                elif self.dataset_filepath and str(self.dataset_filepath).endswith('.npy'):
                    data_arr = np.load(self.dataset_filepath).astype(np.float32)
                else:
                    data_dtype = get_data_load_dtype(self.cfg)
                    data_arr = np.fromfile(self.dataset_filepath, dtype=data_dtype, count=self.size_db * self.dim_series).reshape(-1, self.dim_series).astype(np.float32)

                for batch_loading_offset in range(0, self.size_db, self.batch_loading_size):
                    batch_end = min(batch_loading_offset + self.batch_loading_size, self.size_db)
                    series_batch = data_arr[batch_loading_offset:batch_end]

                    # as long as the number to load has not been reached (train_size + query_size) and the current index is not bigger than the batch
                    # add the new sample
                    while num_loaded < num2load and indices[num_loaded] < batch_end:
                        new_sample = series_batch[indices[num_loaded] - batch_loading_offset]

                        if not self.__is_nan(new_sample):
                            samples.append(copy.deepcopy(new_sample))
                            samples_idx.append(indices[num_loaded])

                            num_loaded += 1
                        else:
                            self.logger.info('nan found at {:d}: {:s}'.format(
                                indices[num_loaded], str(new_sample)))
                            num_loaded += 1

                rand_perm = np.random.permutation(len(samples))

                samples = np.asarray(samples)[rand_perm]
                samples_idx = np.asarray(samples_idx)[rand_perm]

                samples.tofile(self.cfg.paths.seasam_sample_filepath)
                samples_idx.tofile(self.cfg.paths.seasam_sample_idx_filepath)

                train_samples = samples[: self.size_train]
                # Limit validation samples to exactly self.size_valid (important when max_valid_samples is set)
                valid_samples = samples[self.size_train:self.size_train + self.size_valid]

                assert train_samples.shape[0] == self.size_train, f"train_samples.shape[0] != self.size_train: {train_samples.shape[0]} != {self.size_train}"
                assert valid_samples.shape[0] == self.size_valid, f"valid_samples.shape[0] != self.size_valid: {valid_samples.shape[0]} != {self.size_valid}"

            else:
                if hasattr(self, '_data_array') and self._data_array is not None:
                    samples = self._data_array.copy()
                elif self.dataset_filepath and str(self.dataset_filepath).endswith('.npy'):
                    samples = np.load(self.dataset_filepath).astype(np.float32)
                else:
                    data_dtype = get_data_load_dtype(self.cfg)
                    samples = np.fromfile(self.dataset_filepath, dtype=data_dtype, count=self.size_db * self.dim_series).reshape(-1, self.dim_series).astype(np.float32)

                rand_perm = np.random.permutation(len(samples))

                samples = np.asarray(samples)[rand_perm]

                train_samples = samples[: self.size_train]
                # Limit validation samples to exactly self.size_valid (important when max_valid_samples is set)
                valid_samples = samples[self.size_train:self.size_train + self.size_valid]

                assert train_samples.shape[0] == self.size_train, f"train_samples.shape[0] != self.size_train: {train_samples.shape[0]} != {self.size_train}"
                assert valid_samples.shape[0] == self.size_valid, f"valid_samples.shape[0] != self.size_valid: {valid_samples.shape[0]} != {self.size_valid}"

        dists = []
        # Check for distance file paths: dists_train_fp/dists_valid_fp, or dists_fp (suffix pattern)
        # If not given, infer from sample paths (same dir) or use fallback
        dists_train_fp = getattr(self.cfg.dataset, 'dists_train_fp', None)
        dists_valid_fp = getattr(self.cfg.dataset, 'dists_valid_fp', None)
        dists_fp = getattr(self.cfg.dataset, 'dists_fp', None)

        for idx, (sample_data, sample_type, sample_size) in enumerate(zip(
            [train_samples, valid_samples],
            ['-train-', '-valid-'],
            [self.size_train, self.size_valid])):
            
            # Determine distance file path (defined here so it's available for both DTW and Euclidean)
            if idx == 0 and dists_train_fp:
                dist_filepath = dists_train_fp
            elif idx == 1 and dists_valid_fp:
                dist_filepath = dists_valid_fp
            elif dists_fp:
                # Use dists_fp from dataset config (save as train/valid suffixes)
                dist_type = 'train' if idx == 0 else 'valid'
                dist_filepath = dists_fp.replace('.bin', f'_{dist_type}.bin')
            elif processed_train_fp and processed_valid_fp:
                # Pre-split: infer dist path from sample path (same dir, train_dists.npy / valid_dists.npy)
                sample_fp = processed_train_fp if idx == 0 else processed_valid_fp
                dist_name = 'train_dists.npy' if idx == 0 else 'valid_dists.npy'
                dist_filepath = os.path.join(os.path.dirname(sample_fp), dist_name)
            else:
                # Fall back to default path
                dist_filepath = f"{self.cfg.paths.train_val_dist_prefix}_{sample_type[1:-1]}.bin"
            
            # Check if distances file exists
            force_recalc = getattr(self.cfg.training, 'force_recalculate_dtw_train', False)
            if not force_recalc and os.path.exists(dist_filepath):
                print(f'Loading distances from {dist_filepath}...')
                self.logger.info(f'Loading distances from {dist_filepath}')
                if dist_filepath.endswith('.npy'):
                    try:
                        d = np.load(dist_filepath, allow_pickle=False).astype(np.float32)
                    except Exception:
                        # File may be raw binary (from tofile) with .npy extension
                        d = np.fromfile(dist_filepath, dtype=np.float32)
                else:
                    d = np.fromfile(dist_filepath, dtype=np.float32)
                if not np.isfinite(d).all():
                    raise ValueError(
                        f"Distance file contains NaN or Inf: {dist_filepath}. "
                        "Regenerate DTW caches for this split or verify the file is a float32 upper-triangle vector."
                    )
                dists.append(d)
                print(f'Loaded {len(d)} distances from file')
                self.logger.info(f'Loaded {len(d)} distances from file')
                
                # Print dataset information and distance statistics
                dataset_size = sample_data.shape[0]
                series_length = sample_data.shape[1]
                min_dist = d.min()
                max_dist = d.max()
                print(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                self.logger.info(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                continue
            
            # If file doesn't exist, compute distances
            if self.target_dist_function == 'dtw':

                # CPU-bound DTW: use process_time to exclude I/O wait, other processes
                start_time = compute_time_start(device=None)
                w = self.warping_window # getting the warping window of the sample

                # Show progress information
                dataset_size = sample_data.shape[0]
                series_length = sample_data.shape[1]
                n_pairs = int(dataset_size * (dataset_size - 1) / 2)
                print(f'Computing DTW distances for {sample_type} set: {dataset_size} samples, {series_length} length, {n_pairs} pairs...')
                self.logger.info(f'Computing DTW distances for {sample_type} set: {dataset_size} samples, {series_length} length, {n_pairs} pairs')

                # Use dtaidistance with C acceleration and parallel processing for maximum speed
                # Note: distance_matrix_fast doesn't support progress callbacks, but we show a spinner
                with tqdm(total=1, desc=f'Computing DTW distances {sample_type}', bar_format='{desc}: {elapsed}') as pbar:
                    d = dtw.distance_matrix_fast(   sample_data.astype(np.double),
                                                    window=w,
                                                    compact=True,
                                                    use_c=True,      # Enable C acceleration
                                                    parallel=True,
                                                    use_pruning=False)   # Enable parallel processing
                    pbar.update(1)

                d = np.array(d) ** 2
                d = d.astype(np.float32)
                elapsed_sec = compute_time_elapsed(start_time, device=None)
                elapsed_min = elapsed_sec / 60
                self.logger.info('distances calculated')
                self.logger.info(f'Elapsed time: {elapsed_min:.2f} min')
                print('distances calculated')
                print(f"Elapsed time: {elapsed_min:.2f} min ({elapsed_sec:.2f} sec)")

                dists.append(d)

                assert len(dists[-1]) == int(sample_data.shape[0]
                                            * (sample_data.shape[0] - 1) / 2)
                
                # Save distances to file
                os.makedirs(os.path.dirname(dist_filepath), exist_ok=True)
                d.tofile(dist_filepath)
                print(f'Saved distances to {dist_filepath}')
                self.logger.info(f'Saved distances to {dist_filepath}')

                # Print dataset information and distance statistics
                dataset_size = sample_data.shape[0]
                series_length = sample_data.shape[1]
                min_dist = d.min()
                max_dist = d.max()
                print(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                self.logger.info(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                  


            elif self.target_dist_function == 'l2' or self.target_dist_function == 'euclidean':
                # Check if distances file exists
                if os.path.exists(dist_filepath):
                    print(f'Loading distances from {dist_filepath}...')
                    self.logger.info(f'Loading distances from {dist_filepath}')
                    if dist_filepath.endswith('.npy'):
                        try:
                            local_dists = np.load(dist_filepath, allow_pickle=False).astype(np.float32)
                        except Exception:
                            local_dists = np.fromfile(dist_filepath, dtype=np.float32)
                    else:
                        local_dists = np.fromfile(dist_filepath, dtype=np.float32)
                    if not np.isfinite(local_dists).all():
                        raise ValueError(
                            f"Distance file contains NaN or Inf: {dist_filepath}. "
                            "Regenerate distances for this split or verify the file format."
                        )
                    dists.append(local_dists)
                    print(f'Loaded {len(local_dists)} distances from file')
                    self.logger.info(f'Loaded {len(local_dists)} distances from file')
                    
                    # Print dataset information and distance statistics
                    dataset_size = sample_data.shape[0]
                    series_length = sample_data.shape[1]
                    min_dist = local_dists.min()
                    max_dist = local_dists.max()
                    print(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                    self.logger.info(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                else:
                    # Compute distances
                    print(f'Computing Euclidean distances for {sample_type} set...')
                    self.logger.info(f'Computing Euclidean distances for {sample_type} set')
                    local_dists = []

                    for i in range(sample_data.shape[0]):
                        for j in range(i + 1, sample_data.shape[0]):
                            local_dists.append(np.linalg.norm(sample_data[i] - sample_data[j]))

                    local_dists = np.asarray(local_dists, dtype=np.float32)
                    dists.append(local_dists)
                    assert len(dists[-1]) == int(sample_data.shape[0]
                                                * (sample_data.shape[0] - 1) / 2)

                    # Save distances to file
                    os.makedirs(os.path.dirname(dist_filepath), exist_ok=True)
                    local_dists.tofile(dist_filepath)
                    print(f'Saved distances to {dist_filepath}')
                    self.logger.info(f'Saved distances to {dist_filepath}')
                    
                    # Print dataset information and distance statistics
                    dataset_size = sample_data.shape[0]
                    series_length = sample_data.shape[1]
                    min_dist = local_dists.min()
                    max_dist = local_dists.max()
                    print(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                    self.logger.info(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                
                # Print dataset information and distance statistics
                dataset_size = sample_data.shape[0]
                series_length = sample_data.shape[1]
                min_dist = local_dists.min()
                max_dist = local_dists.max()
                print(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')
                self.logger.info(f'Dataset size: {dataset_size}, Series length: {series_length}, min distance = {min_dist}, max distance = {max_dist}')

        
        return train_samples, dists[0], valid_samples, dists[1]


    def __is_nan(self, series: np.ndarray):
        return np.isnan(np.sum(series))
    
