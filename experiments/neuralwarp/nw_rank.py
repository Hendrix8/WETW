#!/usr/bin/env python3
"""NeuralWarp_WD + ranking term (component table, last column), as close as NeuralWarp allows to the refinement the
other encoders receive (stage iii: frozen encoder, pair scorer refined with WarpDistill + log hinge).

Start from the validation-selected NeuralWarp_WD checkpoint of bundle neuralwarp_v2_20261025T115612_c300d5. The CNN
encoder is frozen (batch normalization in inference mode, no dropout); the warping network and the WarpDistill scale are
refined with  mean Huber_0.1 of the relative error on 50 random training pairs  +  mean max(0, 0.1 + log d_ap - log d_an)
on 25 triplets (anchor, one of its 10 DTW nearest neighbours, a random series), i.e. beta * L_WD + L_rank with
beta = 0.1 as in equation (refinement_objective). NeuralWarp's graph takes exactly 100 pairs per batch, hence 50 + 2*25.
Neighbours come from the first min(n, 20000) training series (their full DTW block is read as contiguous condensed row
segments; the 100k-series collections hold 5e9 pair distances). Checkpoints are selected on the same objective over a
fixed validation set from the first 2000 validation series (NeuralWarp's own selection rule), every 100 steps, for
--hours of training. Search and aggregation are neuralwarp_v2.py's, unchanged, on this bundle's runs.
usage: nw_rank.py train|search|aggregate --bundle <this bundle> --task K [--smoke] [--hours H] [--shard S] [--if-ready]
"""
import argparse
import fcntl
import json
import os
import math
from pathlib import Path
import pickle
import sys
import time

import numpy as np

# Bundle created by neuralwarp_v2.py (its code, specs and trained runs); set NW_BUNDLE.
NV2 = Path(os.environ.get('NW_BUNDLE', Path(__file__).resolve().parent))
sys.path.insert(0, str(NV2))
import neuralwarp_v2 as nv2  # noqa: E402

PAIRS, REG, TRI, POOL, VPOOL, K, MARGIN = nv2.PAIRS, 50, 25, 20000, 2000, 10, 0.1


def block(dists, n_full, m):
    """Symmetric m x m DTW block of the first m series of a condensed matrix over n_full series."""
    out = np.zeros((m, m), np.float32)
    for i in range(m - 1):
        a = nv2.condensed_index(n_full, np.array([i]), np.array([i + 1]))[0]
        out[i, i + 1:] = dists[a:a + m - i - 1]
    out += out.T
    np.fill_diagonal(out, np.inf)
    return out


def triplets(rng, nn, count, m):
    a = rng.randint(len(nn), size=count)
    p = nn[a, rng.randint(K, size=count)]
    n = rng.randint(m, size=count)
    n = np.where(n == a, (n + 1) % m, n)
    return a, p, n


def make_rank_model(bundle, length):
    tensorflow, tf, cnn_models, HyperParams = nv2.model_api(bundle)
    tf.set_random_seed(0)
    config = HyperParams().get_uniwarp_config([])
    config['uniwarp:length'], config['dataset:num_channels'] = length, 1

    class WarpDistillRank(cnn_models.CNNWarpedSim):
        def create_optimization_routine(self):
            with tf.variable_scope('OptimizationRoutines'):  # same names as NeuralWarp_WD: log_scale restores
                self.true_dists = tf.placeholder(tf.float32, shape=(PAIRS,))
                self.log_scale = tf.get_variable('log_scale', initializer=0.)
                pred = tf.exp(self.log_scale) * self.pair_dists
                reg = tf.losses.huber_loss(tf.zeros((REG,)), (pred[:REG] - self.true_dists[:REG]) / self.true_dists[:REG],
                                           delta=.1)
                logd = tf.log(tf.maximum(self.pair_dists[REG:], 1e-6))
                rank = tf.reduce_mean(tf.nn.relu(MARGIN + logd[0::2] - logd[1::2]))
                self.reg_loss, self.rank_loss, self.loss = reg, rank, reg + rank
            variables = [v for v in tf.trainable_variables() if 'CNNWarpedSimDistPair' in v.name or 'log_scale' in v.name]
            assert variables and all('conv1d' not in v.name and 'batch_normalization' not in v.name for v in variables)
            self.refined_variables = variables
            with tf.variable_scope('RankRefine'):
                gradients, _ = tf.clip_by_global_norm(tf.gradients(self.loss, variables), config['uniwarp:max_grad_norm'])
                self.update_rule = tf.train.AdamOptimizer(config['uniwarp:eta']).apply_gradients(zip(gradients, variables))

    model = WarpDistillRank(config)
    model.create_model()
    session_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    session_config.gpu_options.allow_growth = True
    return tensorflow, tf, model, session_config


def batch(x, xfull_len, dists, n_full, nn, rng_or_fixed, m):
    """One 100-pair batch: 50 regression pairs (targets from the condensed matrix), then 25 (a,p),(a,n) pairs."""
    rng = rng_or_fixed
    i, j = nv2.random_pairs(rng, len(x), REG)
    t = nv2.normalized_targets(dists, n_full, i, j, xfull_len)
    a, p, n = triplets(rng, nn, TRI, m)
    left = np.concatenate([i, np.repeat(a, 2)])
    right = np.concatenate([j, np.stack([p, n], 1).ravel()])
    targets = np.concatenate([t, np.ones(2 * TRI, np.float32)])  # ranking pairs carry no regression target
    return nv2.batch_input(x, left, right), targets


def train(args, bundle, spec, run):
    done = run / 'training_complete.json'
    if done.exists():
        print('Training already complete:', done, flush=True)
        return
    sd = Path(spec['sample_dir'])
    xfull = np.load(sd / 'train_sample.npy', mmap_mode='r')
    vfull = np.load(sd / 'valid_sample.npy', mmap_mode='r')
    L = xfull.shape[1]
    x = nv2.paa(xfull, spec['paa'])
    nv = min(VPOOL, len(vfull))
    xv = nv2.paa(vfull[:nv], spec['paa'])
    dtrain = np.load(sd / 'train_dists.npy', mmap_mode='r')
    dvalid = np.load(sd / 'valid_dists.npy', mmap_mode='r')
    m = min(POOL if not args.smoke else 500, len(x))
    t0 = time.monotonic()
    nn = np.argpartition(block(dtrain, len(x), m), K, axis=1)[:, :K]  # the K nearest, any order
    vnn = np.argpartition(block(dvalid, len(vfull), nv), K, axis=1)[:, :K]
    print(f'neighbour pools: train {m} valid {nv} in {time.monotonic() - t0:.0f}s', flush=True)
    vrng = np.random.RandomState(12345)
    nvb = (args.valid_pairs // PAIRS) if not args.smoke else 1
    vbatches = []
    for _ in range(nvb):  # fixed validation set, regression pairs and triplets within the first nv series
        i, j = nv2.random_pairs(vrng, nv, REG)
        t = nv2.normalized_targets(dvalid, len(vfull), i, j, L)
        a, p, n = triplets(vrng, vnn, TRI, nv)
        left, right = np.concatenate([i, np.repeat(a, 2)]), np.concatenate([j, np.stack([p, n], 1).ravel()])
        vbatches.append((nv2.batch_input(xv, left, right), np.concatenate([t, np.ones(2 * TRI, np.float32)])))
    tensorflow, tf, model, session_config = make_rank_model(bundle, x.shape[1])
    source = json.loads((Path(spec.get('source_bundle', NV2)) / 'runs' / spec['source_run'] / 'best.json').read_text())['checkpoint']
    in_ckpt = {name for name, _ in tf.train.list_variables(source)}
    restore = [v for v in tf.global_variables() if v.op.name in in_ckpt and 'Adam' not in v.op.name
               and 'power' not in v.op.name and not v.op.name.startswith('RankRefine')]
    missing = [v.op.name for v in tf.trainable_variables() if v not in restore]
    assert not missing, ('model variables not in the WD checkpoint', missing)
    for d in ('best', 'resume'):
        (run / d).mkdir(exist_ok=True)
    saver, best_saver = tf.train.Saver(max_to_keep=2), tf.train.Saver(max_to_keep=1)
    rng = np.random.RandomState(0)
    state_path = run / 'resume' / 'state.pkl'
    state = dict(step=0, elapsed=0., best_loss=None, best_step=None)
    if state_path.exists():
        state = pickle.loads(state_path.read_bytes())
        rng.set_state(state['rng_state'])
    params = dict(spec=spec, hours=args.hours, source_checkpoint=source, regression_pairs=REG, triplets=TRI,
                  margin=MARGIN, neighbour_pool=m, valid_series=nv, valid_batches=nvb, frozen='CNN encoder (BN inference)',
                  refined=[v.op.name for v in model.refined_variables], tensorflow=tensorflow.__version__)
    pp = run / 'parameters.json'
    if pp.exists():
        assert json.loads(pp.read_text()) == params, 'refusing to resume with changed parameters'
    else:
        nv2.exclusive_json(pp, params)
    start, elapsed0 = time.monotonic(), state['elapsed']
    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.Saver(var_list=restore).restore(sess, source)
        if 'checkpoint' in state:
            saver.restore(sess, state['checkpoint'])
        frozen = [v for v in tf.trainable_variables() if v not in model.refined_variables]
        encoder0 = sess.run(frozen)

        def validate():
            vals = [sess.run([model.loss, model.reg_loss, model.rank_loss],
                             {model.X_batch: xb, model.true_dists: tb, model.is_training: False}) for xb, tb in vbatches]
            loss, reg, rank = (float(np.mean(c)) for c in zip(*vals))
            if not math.isfinite(loss):
                raise RuntimeError('Nonfinite validation loss')
            if state['best_loss'] is None or loss < state['best_loss']:
                ck = best_saver.save(sess, str(run / 'best' / 'model'), global_step=state['step'])
                state.update(best_loss=loss, best_step=state['step'])
                nv2.atomic_json(run / 'best.json', dict(checkpoint=ck, step=state['step'], validation_loss=loss,
                                                        validation_reg=reg, validation_rank=rank))
            state['checkpoint'] = saver.save(sess, str(run / 'resume' / 'model'), global_step=state['step'])
            state['elapsed'] = elapsed0 + time.monotonic() - start
            state['rng_state'] = rng.get_state()
            tmp = state_path.with_suffix('.tmp'); tmp.write_bytes(pickle.dumps(state)); tmp.replace(state_path)
            rec = dict(step=state['step'], elapsed=state['elapsed'], validation_loss=loss, reg=reg, rank=rank)
            with (run / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print(json.dumps(rec), flush=True)

        if state['step'] == 0:
            validate()  # step 0 = the NeuralWarp_WD checkpoint itself, so refinement can only be selected if it helps
        while elapsed0 + time.monotonic() - start < args.hours * 3600 and not (args.max_steps and state['step'] >= args.max_steps):
            xb, tb = batch(x, L, dtrain, len(x), nn, rng, m)
            loss, _ = sess.run([model.loss, model.update_rule], {model.X_batch: xb, model.true_dists: tb, model.is_training: False})
            if not np.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            state['step'] += 1
            if state['step'] % args.valid_every == 0:
                validate()
        validate()
        enc1 = sess.run(frozen)
        assert all(np.array_equal(a, b) for a, b in zip(encoder0, enc1)), 'encoder changed'
    best = json.loads((run / 'best.json').read_text())
    nv2.exclusive_json(done, dict(best=best, steps=state['step'], elapsed=state['elapsed'], complete=True,
                                  finished_at_utc=nv2.utc(), smoke=args.smoke))
    print('TRAINING_COMPLETE', spec['name'], best, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('train', 'search', 'aggregate'))
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--hours', type=float, default=2.)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--valid-every', type=int, default=100)
    ap.add_argument('--valid-pairs', type=int, default=2000)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--if-ready', action='store_true')
    args = ap.parse_args()
    bundle = args.bundle.resolve()
    spec = json.loads((bundle / 'manifest.json').read_text())['runs'][args.task]
    run = bundle / ('smoke' if args.smoke else 'runs') / spec['name']
    run.mkdir(parents=True, exist_ok=True)
    lock_name = ('search_%d' % args.shard) if args.stage == 'search' else args.stage
    with (run / (lock_name + '.lock')).open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            if args.stage == 'aggregate' and args.if_ready:
                return
            raise
        {'train': train, 'search': nv2.search, 'aggregate': nv2.aggregate}[args.stage](args, bundle, spec, run)


if __name__ == '__main__':
    main()
