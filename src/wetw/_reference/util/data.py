# coding = utf-8

import os
import struct
from os.path import isfile
from ctypes import CDLL, c_char_p, c_long
from _ctypes import dlclose

import torch
import numpy as np
from torch.utils.data import Dataset




class DatasetwithIdx(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, indices):
        return self.data[indices], indices


class TSDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, indices):
        return self.data[indices]

class FileContainer(object):
    def __init__(self, filename, binary=True):
        self.filename = filename
        self.binary = binary
        if self.binary:
            self.f = open(filename, "wb")
        else:
            self.f = open(filename, "w")

    def write(self, ts):
        if self.binary:
            s = struct.pack('f' * len(ts), *ts)
            self.f.write(s)
        else:
            self.f.write(" ".join(map(str, ts)) + "\n")

    def close(self):
        self.f.close()


def embedData(model, data_filepath, embedding_filepath, data_size,
              batch_size=2000, original_dim=256, embedded_dim=16, device='cuda',
              data_array=None):
    """Compute embeddings. Pass data_array (np.ndarray) for .npz/.npy, or data_filepath for .bin."""
    if data_size < batch_size:
        num_segments = 1
        batch_size = data_size
    else:
        num_segments = int(np.ceil(data_size / batch_size))

    nan_replacement_original = np.array([0.] * original_dim).reshape([1, original_dim])
    nan_replacement_embedding = [0.] * embedded_dim

    embeddings_list = []
    use_array = data_array is not None

    try:
        with torch.no_grad():
            total_nans = 0

            for segment in range(num_segments):
                if use_array:
                    start = segment * batch_size
                    end = min(start + batch_size, data_size)
                    batch = data_array[start:end].astype(np.float32)
                elif num_segments > 1 and segment == num_segments - 1 and data_size % batch_size != 0:
                    last_batch_size = data_size % batch_size
                    batch = np.fromfile(data_filepath, dtype=np.float32,
                                        count=original_dim * last_batch_size,
                                        offset=4 * original_dim * batch_size * segment)
                else:
                    batch = np.fromfile(data_filepath, dtype=np.float32,
                                        count=original_dim * batch_size,
                                        offset=4 * original_dim * batch_size * segment)

                batch = batch.reshape(-1, 1, original_dim)

                nan_indices = set()
                for i, sequence in zip(range(batch.shape[0]), batch):
                    if np.isnan(np.sum(sequence)):
                        nan_indices.add(i)
                        batch[i] = nan_replacement_original

                encoded = model.encode(torch.from_numpy(batch).to(device))
                if isinstance(encoded, tuple):
                    embedding = encoded[0].detach().cpu().numpy()
                else:
                    embedding = encoded.detach().cpu().numpy()

                for i in nan_indices:
                    embedding[i] = nan_replacement_embedding

                embeddings_list.append(embedding)
                total_nans += len(nan_indices)

            print('nans = {:d}'.format(total_nans))
            out = np.vstack(embeddings_list).astype(np.float32)
            if embedding_filepath.endswith('.npy'):
                np.save(embedding_filepath, out)
            else:
                out.tofile(embedding_filepath)
    except Exception:
        raise
