#!/usr/bin/env python3
"""NeuralWarp_WD in Figures 1 and 5 (search over the full database): ranks of all ten exact neighbours for each of the 100 queries, per-query resume, and one-thread
CPU seconds per pair. Reuses neuralwarp_v2.py (model, PAA, predict) with the validation-selected checkpoints.
usage: nw_fig5.py search|cpu --lab L [--shard S --shards K] [--queries Q]   (root = this folder, specs.json)"""
import argparse, json, os, time
from pathlib import Path
import sys
import numpy as np
ROOT = Path(__file__).resolve().parent
NV2 = Path(os.environ.get('NW_BUNDLE', Path(__file__).resolve().parent))  # folder holding neuralwarp_v2.py
sys.path.insert(0, str(NV2))
import neuralwarp_v2 as nv2  # noqa: E402

def ckpt(spec):
    d = Path(spec['best'])
    return str(max(d.glob('model-*.index'), key=lambda p: int(p.stem.split('-')[1])).with_suffix(''))

def run(a):
    spec = json.loads((ROOT / 'specs.json').read_text())[a.lab]
    q = nv2.paa(np.load(spec['queries']), spec['paa'])
    db = np.load(spec['database'], mmap_mode='r')
    ex = np.fromfile(spec['exact'], dtype=np.int32).reshape(100, -1)[:, :10]
    (ROOT / 'out').mkdir(exist_ok=True)
    if a.mode == 'cpu':
        import tensorflow as tf0  # the bundle refuses to run without a GPU; this timing is deliberately on one CPU thread
        tf0.compat.v1.test.is_gpu_available = lambda *x, **k: True
        tf0.test.is_gpu_available = lambda *x, **k: True
        _, tf, model, config = nv2.make_model(NV2, q.shape[1])
        config.intra_op_parallelism_threads = 1; config.inter_op_parallelism_threads = 1
        with tf.compat.v1.Session(config=config) as s:
            tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables()).restore(s, ckpt(spec))
            right = nv2.paa(db[:1000], spec['paa']); left = np.repeat(q[:1], 1000, axis=0)
            nv2.predict(s, model, left[:100], right[:100])
            t = time.perf_counter(); nv2.predict(s, model, left, right); sec = (time.perf_counter() - t) / 1000
        (ROOT / 'out' / f'{a.lab}.cpu.json').write_text(json.dumps(dict(lab=a.lab, seconds_per_pair=sec, pairs=1000, N=len(db), paa=spec['paa'], threads=1)))
        print('CPU_OK', a.lab, sec, flush=True)
        return
    ids = list(range(a.shard, 100, a.shards))[:a.queries or None]
    p = ROOT / 'out' / f'{a.lab}.{a.shard}.json'
    done = json.loads(p.read_text()) if p.exists() else []
    _, tf, model, config = nv2.make_model(NV2, q.shape[1])
    with tf.compat.v1.Session(config=config) as s:
        tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables()).restore(s, ckpt(spec))
        for qi in ids[len(done):]:
            t = time.monotonic(); ch = []
            for st in range(0, len(db), 5000):
                right = nv2.paa(db[st:st + 5000], spec['paa'])
                ch.append(nv2.predict(s, model, np.repeat(q[qi:qi + 1], len(right), axis=0), right))
            d = np.concatenate(ch)
            done.append(dict(query=qi, ranks=[nv2.rank_of_target(d, int(x)) for x in ex[qi]], seconds=time.monotonic() - t))
            tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(done)); os.replace(tmp, p)
            print(qi, done[-1]['ranks'][:3], round(done[-1]['seconds'], 1), flush=True)
    print('SEARCH_OK', a.lab, a.shard, len(done), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--lab', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--shards', type=int, default=1); ap.add_argument('--queries', type=int, default=0)
    run(ap.parse_args())
