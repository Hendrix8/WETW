#!/usr/bin/env python3
"""Isolated NeuralWarp reruns: validation selection and resumable 100-query search.

Uses the original CNNWarpedSim graph through TensorFlow's v1 compatibility API.
All writable paths are inside a newly created experiment bundle; old runs are
never opened for writing. Native runs use labels; WD runs use full-length DTW.
"""
import argparse
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

RHOS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
PAIRS = 100


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def exclusive_json(path, data):
    # Publish only after a complete write, without replacing an existing result.
    temporary = path.with_name(path.name + '.publish.%d' % os.getpid())
    with temporary.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def condensed_index(n, i, j):
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    if np.any(lo == hi) or np.any(lo < 0) or np.any(hi >= n):
        raise ValueError('Invalid condensed-distance pair')
    return n * lo - lo * (lo + 1) // 2 + hi - lo - 1


def paa(x, factor):
    x = np.asarray(x, dtype=np.float32)
    if factor == 1:
        return x
    if x.shape[1] % factor:
        raise ValueError('PAA must divide the original series length exactly')
    return x.reshape(len(x), x.shape[1] // factor, factor).mean(2)


def random_pairs(rng, n, count):
    if n < 2:
        raise ValueError('At least two series required')
    i, j = rng.randint(n, size=count), rng.randint(n, size=count)
    return i, np.where(i == j, (j + 1) % n, j)


def labelled_pairs(rng, labels, count):
    """Authors' rejection sampling: alternating positive/negative pairs."""
    if len(np.unique(labels)) < 2:
        raise ValueError('Native objective needs at least two classes')
    left, right, target = [], [], []
    for k in range(count):
        positive = k % 2 == 0
        while True:
            i, j = rng.randint(len(labels), size=2)
            if bool(labels[i] == labels[j]) == positive:
                break
        left.append(i); right.append(j); target.append(float(positive))
    return np.asarray(left), np.asarray(right), np.asarray(target, np.float32)


def normalized_targets(dists, n_full, i, j, original_length):
    d = np.asarray(dists[condensed_index(n_full, i, j)], np.float64)
    if not np.all(np.isfinite(d)) or np.any(d < 0):
        raise ValueError('Invalid DTW targets')
    return np.maximum(d / np.sqrt(original_length), 1e-6).astype(np.float32)


def model_api(bundle):
    import tensorflow as tensorflow
    tf = tensorflow.compat.v1
    tf.disable_v2_behavior()
    # The upstream modules use `import tensorflow as tf` and require v1 names.
    sys.modules['tensorflow'] = tf
    sys.path.insert(0, str(bundle / 'upstream'))
    import cnn_models
    from hyper_parameters import HyperParams
    if not tf.test.is_gpu_available(cuda_only=True):
        raise RuntimeError('A CUDA GPU is required; refusing CPU fallback')
    return tensorflow, tf, cnn_models, HyperParams


def make_model(bundle, length, objective=None, seed=0):
    tensorflow, tf, cnn_models, HyperParams = model_api(bundle)
    tf.set_random_seed(seed)
    config = HyperParams().get_uniwarp_config([])
    config['uniwarp:length'], config['dataset:num_channels'] = length, 1
    if objective == 'wd':
        class WarpDistill(cnn_models.CNNWarpedSim):
            def create_optimization_routine(self):
                with tf.variable_scope('OptimizationRoutines'):
                    self.true_dists = tf.placeholder(tf.float32, shape=(PAIRS,))
                    self.log_scale = tf.get_variable('log_scale', initializer=0.)
                    pred = tf.exp(self.log_scale) * self.pair_dists
                    self.loss = tf.losses.huber_loss(
                        tf.zeros_like(self.true_dists),
                        (pred - self.true_dists) / self.true_dists, delta=.1)
                    variables = tf.trainable_variables()
                    gradients, _ = tf.clip_by_global_norm(
                        tf.gradients(self.loss, variables), config['uniwarp:max_grad_norm'])
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        self.update_rule = tf.train.AdamOptimizer(config['uniwarp:eta']).apply_gradients(
                            zip(gradients, variables))
        model = WarpDistill(config)
    else:
        model = cnn_models.CNNWarpedSim(config)
    if objective:
        model.create_model()
    else:
        model.create_encoder(); model.create_similarity()
    session_config = tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)
    session_config.gpu_options.allow_growth = True
    return tensorflow, tf, model, session_config


def batch_input(x, i, j):
    xb = np.empty((2 * len(i), x.shape[1], 1), np.float32)
    xb[0::2, :, 0], xb[1::2, :, 0] = x[i], x[j]
    return xb


def train(args, bundle, spec, run):
    done = run / 'training_complete.json'
    if done.exists():
        marker = json.loads(done.read_text())
        if marker.get('complete') is not True or marker.get('smoke') != args.smoke:
            raise ValueError('Invalid existing training completion marker')
        print('Training already complete:', done, flush=True)
        return
    sd = Path(spec['sample_dir'])
    xfull = np.load(sd / 'train_sample.npy', mmap_mode='r')
    vfull = np.load(sd / 'valid_sample.npy', mmap_mode='r')
    full_length = xfull.shape[1]
    x = paa(xfull, spec['paa'])
    nv = min(2000, len(vfull))
    xv = paa(vfull[:nv], spec['paa'])
    dtrain = np.load(sd / 'train_dists.npy', mmap_mode='r')
    dvalid = np.load(sd / 'valid_dists.npy', mmap_mode='r')
    if len(dtrain) != len(x) * (len(x) - 1) // 2 or len(dvalid) != len(vfull) * (len(vfull) - 1) // 2:
        raise ValueError('Distance arrays do not match their samples')
    labels = valid_labels = None
    if spec['objective'] == 'native':
        labels = np.load(bundle / spec['train_labels'])
        valid_labels = np.load(bundle / spec['valid_labels'])[:nv]
        if len(labels) != len(x) or len(valid_labels) != nv:
            raise ValueError('Labels are not aligned to training/validation samples')
    validation_rng = np.random.RandomState(12345)
    count = args.valid_pairs
    if count % PAIRS:
        raise ValueError('Validation pairs must be a multiple of 100')
    if spec['objective'] == 'native':
        vi, vj, vt = labelled_pairs(validation_rng, valid_labels, count)
    else:
        vi, vj = random_pairs(validation_rng, nv, count)
        vt = normalized_targets(dvalid, len(vfull), vi, vj, full_length)
    # Materializing this small fixed validation set avoids repeated Lustre reads.
    vx = batch_input(xv, vi, vj)
    tensorflow, tf, model, session_config = make_model(bundle, x.shape[1], spec['objective'])
    for directory in ('best', 'resume'):
        (run / directory).mkdir(exist_ok=True)
    saver = tf.train.Saver(max_to_keep=2)
    best_saver = tf.train.Saver(max_to_keep=1)
    target_placeholder = model.true_dists if spec['objective'] == 'wd' else model.true_similarities
    rng = np.random.RandomState(0)
    state_path = run / 'resume' / 'state.pkl'
    state = dict(step=0, elapsed=0., best_loss=None, best_step=None)
    if state_path.exists():
        with state_path.open('rb') as stream:
            state = pickle.load(stream)
        rng.set_state(state['rng_state'])
    start = time.monotonic()
    initial_elapsed = state['elapsed']
    parameters = dict(spec=spec, hours=args.hours, max_steps=args.max_steps,
                      valid_pairs=count, valid_series=nv, valid_every=args.valid_every,
                      selection='minimum validation value of own training objective',
                      tensorflow=tensorflow.__version__, batch_pairs=PAIRS,
                      full_length=full_length, model_length=x.shape[1], seed=0)
    parameters_path = run / 'parameters.json'
    if parameters_path.exists():
        if json.loads(parameters_path.read_text()) != parameters:
            raise ValueError('Refusing to resume with changed training parameters')
    else:
        exclusive_json(parameters_path, parameters)
    with tf.Session(config=session_config) as sess:
        sess.run(tf.global_variables_initializer())
        if 'checkpoint' in state:
            saver.restore(sess, state['checkpoint'])
        def validate_and_save():
            losses = []
            for st in range(0, count, PAIRS):
                losses.append(float(sess.run(model.loss, {
                    model.X_batch: vx[2 * st:2 * (st + PAIRS)],
                    target_placeholder: vt[st:st + PAIRS], model.is_training: False})))
            validation_loss = float(np.mean(losses))
            if not math.isfinite(validation_loss):
                raise RuntimeError('Nonfinite validation loss')
            if state['best_loss'] is None or validation_loss < state['best_loss']:
                checkpoint = best_saver.save(sess, str(run / 'best' / 'model'), global_step=state['step'])
                state.update(best_loss=validation_loss, best_step=state['step'])
                atomic_json(run / 'best.json', dict(checkpoint=checkpoint, step=state['step'],
                            validation_loss=validation_loss, checked_at_utc=utc()))
            state['checkpoint'] = saver.save(sess, str(run / 'resume' / 'model'), global_step=state['step'])
            state['elapsed'] = initial_elapsed + time.monotonic() - start
            state['rng_state'] = rng.get_state()
            temporary = state_path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                pickle.dump(state, stream)
            temporary.replace(state_path)
            record = dict(step=state['step'], elapsed=state['elapsed'], validation_loss=validation_loss,
                          best_loss=state['best_loss'], best_step=state['best_step'])
            with (run / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record), flush=True)

        while initial_elapsed + time.monotonic() - start < args.hours * 3600:
            if args.max_steps and state['step'] >= args.max_steps:
                break
            if spec['objective'] == 'native':
                i, j, targets = labelled_pairs(rng, labels, PAIRS)
            else:
                i, j = random_pairs(rng, len(x), PAIRS)
                targets = normalized_targets(dtrain, len(x), i, j, full_length)
            loss, _ = sess.run([model.loss, model.update_rule], {
                model.X_batch: batch_input(x, i, j), target_placeholder: targets, model.is_training: True})
            if not np.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            state['step'] += 1
            if state['step'] == 1 or state['step'] % args.valid_every == 0:
                validate_and_save()
        if not state['step']:
            raise RuntimeError('No training steps completed')
        validate_and_save()
    best = json.loads((run / 'best.json').read_text())
    exclusive_json(done, dict(best=best, steps=state['step'], elapsed=state['elapsed'],
                             complete=True, finished_at_utc=utc(), smoke=args.smoke))
    print('TRAINING_COMPLETE', spec['name'], best, flush=True)


def predict(sess, model, left, right):
    output = []
    for start in range(0, len(left), PAIRS):
        a, b = left[start:start + PAIRS], right[start:start + PAIRS]
        n = len(a)
        if n < PAIRS:
            a = np.concatenate((a, np.repeat(a[-1:], PAIRS - n, axis=0)))
            b = np.concatenate((b, np.repeat(b[-1:], PAIRS - n, axis=0)))
        xb = np.empty((2 * PAIRS, a.shape[1], 1), np.float32)
        xb[0::2, :, 0], xb[1::2, :, 0] = a, b
        values = sess.run(model.pair_dists, {model.X_batch: xb, model.is_training: False})[:n]
        if not np.all(np.isfinite(values)):
            raise RuntimeError('Nonfinite predicted distances')
        output.append(values)
    return np.concatenate(output)


def diagnostics(sess, model, sd, factor, count=2000):
    from scipy.stats import spearmanr
    results = {}
    predictions, targets = {}, {}
    for split, seed in [('valid', 101), ('test', 102)]:
        full = np.load(sd / (split + '_sample.npy'), mmap_mode='r')
        d = np.load(sd / (split + '_dists.npy'), mmap_mode='r')
        i, j = random_pairs(np.random.RandomState(seed), len(full), count)
        left, right = paa(full[i], factor), paa(full[j], factor)
        predictions[split] = predict(sess, model, left, right).astype(np.float64)
        raw = np.asarray(d[condensed_index(len(full), i, j)], np.float64) / np.sqrt(full.shape[1])
        if not np.all(np.isfinite(raw)) or np.any(raw < 0):
            raise RuntimeError('Invalid diagnostic targets')
        targets[split] = raw
    mask = targets['valid'] > 0
    if not np.any(mask):
        raise RuntimeError('No positive validation targets for calibration')
    scale = float(np.median(targets['valid'][mask] / np.maximum(predictions['valid'][mask], 1e-12)))
    mask = targets['test'] > 0
    if not np.any(mask):
        raise RuntimeError('No positive test targets for diagnostics')
    correlation = float(spearmanr(predictions['test'][mask], targets['test'][mask]).correlation)
    results.update(calibration_split='validation', calibration_scale=scale, diagnostic_pairs=count,
                   spearman=correlation if math.isfinite(correlation) else None,
                   mean_abs_re=float(np.mean(np.abs(scale * predictions['test'][mask] - targets['test'][mask]) / targets['test'][mask])))
    return results


def rank_of_target(distances, target):
    """Deterministic ascending distance, then ascending database ID for ties."""
    value = distances[target]
    return int(np.count_nonzero(distances < value) + np.count_nonzero(distances[:target] == value))


def search(args, bundle, spec, run):
    training = json.loads((run / 'training_complete.json').read_text())
    if training['complete'] is not True or training['smoke'] != args.smoke:
        raise ValueError('Invalid training completion marker')
    checkpoint = training['best']['checkpoint']
    final_path = run / ('search_%d.json' % args.shard)
    if final_path.exists():
        old = json.loads(final_path.read_text())
        if old.get('search_complete') is not True or old.get('checkpoint') != checkpoint:
            raise ValueError('Existing search result is incomplete or uses another checkpoint')
        print('Shard already complete:', final_path, flush=True)
        return
    qfull = np.load(spec['queries'])
    if len(qfull) != 100:
        raise ValueError('Expected the same 100 query series as the other baselines')
    queries = paa(qfull, spec['paa'])
    db = np.load(spec['database'], mmap_mode='r')
    exact = np.fromfile(spec['exact'], dtype=np.int32).reshape(100, -1)
    if exact.shape[1] < 1 or np.any(exact < 0) or np.any(exact >= len(db)):
        raise ValueError('Invalid exact nearest-neighbour IDs')
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard ID')
    query_ids = list(range(args.shard, 100, args.shards))
    if args.smoke:
        query_ids = query_ids[:1]
        ndb = min(1000, len(db))
    else:
        ndb = len(db)
    progress_path = run / ('search_%d.progress.json' % args.shard)
    progress = dict(checkpoint=checkpoint, query_ids=query_ids, completed=[], inference_seconds=0.,
                    dataset=spec['dataset'], objective=spec['objective'], smoke=args.smoke,
                    total_database=len(db), evaluated_database=ndb, shards=args.shards,
                    shard=args.shard, tie_policy='distance_then_database_id', search_complete=False)
    if progress_path.exists():
        saved = json.loads(progress_path.read_text())
        for key in ('checkpoint', 'query_ids', 'total_database', 'evaluated_database', 'smoke', 'shards'):
            if saved[key] != progress[key]:
                raise ValueError('Search resume parameter mismatch: ' + key)
        progress = saved
    _, tf, model, config = make_model(bundle, queries.shape[1])
    saver = tf.train.Saver(var_list=tf.global_variables())
    with tf.Session(config=config) as sess:
        saver.restore(sess, checkpoint)
        finished = [entry['query'] for entry in progress['completed']]
        if finished != query_ids[:len(finished)]:
            raise ValueError('Invalid resumed query prefix')
        for query_id in query_ids[len(finished):]:
            started = time.monotonic()
            chunks = []
            for st in range(0, ndb, 5000):
                right = paa(db[st:min(st + 5000, ndb)], spec['paa'])
                left = np.repeat(queries[query_id:query_id + 1], len(right), axis=0)
                chunks.append(predict(sess, model, left, right))
            distances = np.concatenate(chunks)
            seconds = time.monotonic() - started
            # Smoke tests exercise inference/search mechanics on a small database;
            # their target may fall outside it and is never a reported result.
            target = int(exact[query_id, 0])
            rank = rank_of_target(distances, target) if target < ndb else None
            entry = dict(query=query_id, rank=rank, seconds=seconds,
                         hits={str(r): bool(rank is not None and rank < max(1, int(r * ndb))) for r in RHOS})
            progress['completed'].append(entry)
            progress['inference_seconds'] += seconds
            atomic_json(progress_path, progress)
            print(json.dumps(entry), flush=True)
        if args.shard == 0:
            progress['diagnostics'] = diagnostics(sess, model, Path(spec['sample_dir']), spec['paa'],
                                                  count=100 if args.smoke else 2000)
    progress.update(search_complete=True, finished_at_utc=utc(), search_queries_done=len(query_ids))
    exclusive_json(final_path, progress)
    print('SEARCH_SHARD_COMPLETE', spec['name'], args.shard, flush=True)


def aggregate(args, bundle, spec, run):
    output = run / 'eval.json'
    if output.exists():
        result = json.loads(output.read_text())
        if result.get('search_complete') is not True:
            raise ValueError('Existing aggregate is incomplete')
        print('Aggregate already complete:', output, flush=True)
        return
    paths = [run / ('search_%d.json' % k) for k in range(args.shards)]
    if args.if_ready and not all(path.exists() for path in paths):
        print('Waiting for the remaining query shards', flush=True)
        return
    shards = [json.loads(path.read_text()) for path in paths]
    checkpoint = json.loads((run / 'training_complete.json').read_text())['best']['checkpoint']
    for shard in shards:
        if shard['search_complete'] is not True or shard['checkpoint'] != checkpoint or shard['smoke'] != args.smoke:
            raise ValueError('Inconsistent/incomplete shards')
    entries = sorted([row for shard in shards for row in shard['completed']], key=lambda row: row['query'])
    expected = [0, 1, 2, 3] if args.smoke and args.shards == 4 else list(range(100))
    if [row['query'] for row in entries] != expected:
        raise ValueError('Missing, duplicate or unexpected query IDs')
    result = dict(dataset=spec['dataset'], objective=spec['objective'], checkpoint=checkpoint,
                  paa=spec['paa'], search_complete=True, search_queries_done=len(entries),
                  query_ids=[row['query'] for row in entries], smoke=args.smoke,
                  search_gpu_seconds=sum(shard['inference_seconds'] for shard in shards),
                  diagnostics=shards[0]['diagnostics'], completed=entries, finished_at_utc=utc())
    result.update({'R@1_' + str(r): float(np.mean([row['hits'][str(r)] for row in entries])) for r in RHOS})
    exclusive_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'completed'}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('train', 'search', 'aggregate'))
    ap.add_argument('--bundle', type=Path, required=True)
    ap.add_argument('--task', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--hours', type=float, default=6.)
    ap.add_argument('--max-steps', type=int, default=0)
    ap.add_argument('--valid-every', type=int, default=100)
    ap.add_argument('--valid-pairs', type=int, default=2000)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=4)
    ap.add_argument('--if-ready', action='store_true')
    args = ap.parse_args()
    if args.hours <= 0 or args.valid_every <= 0 or args.valid_pairs <= 0 or args.shards != 4:
        raise ValueError('Invalid experiment parameters (four search shards required)')
    bundle = args.bundle.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    spec = manifest['runs'][args.task]
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
        {'train': train, 'search': search, 'aggregate': aggregate}[args.stage](args, bundle, spec, run)


if __name__ == '__main__':
    main()
