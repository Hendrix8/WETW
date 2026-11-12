"""Validation-only AudioMNIST pilot and checkpointed Figure 6 reruns.

Warm-starts the existing WESee architecture/checkpoints; does not change splits,
embedding dimension, DTW targets or the paper's test queries. No test/query data
is loaded by either hyperparameter pilot.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(os.environ.get('FIG6_ROOT', Path(__file__).resolve().parent))
DEVICE = torch.device(os.environ.get('FIG6_DEVICE', 'cuda'))
DEADLINE = time.monotonic() + float(os.environ.get('PIECE_SECONDS', '6000'))
STOP = False
signal.signal(signal.SIGUSR1, lambda *_: globals().__setitem__('STOP', True))


class PieceEnd(Exception):
    pass


def check_time():
    if STOP or time.monotonic() >= DEADLINE:
        raise PieceEnd()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    q.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    q.replace(p)


def save(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.tmp')
    torch.save(obj, q)
    q.replace(p)


def seed(n=42):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(n)


def rng_state():
    return dict(py=random.getstate(), np=np.random.get_state(), cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(r):
    random.setstate(r['py'])
    np.random.set_state(r['np'])
    torch.set_rng_state(r['cpu'].cpu())
    if r['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in r['cuda']])


def load(p):
    return torch.load(p, map_location=DEVICE, weights_only=False)


class Head(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3*d, 128), nn.ReLU(), nn.Linear(128, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.log_s = nn.Parameter(torch.zeros(()))

    def forward(self, a, b, euclidean=False):
        if euclidean:
            return self.log_s.exp() * (a-b).norm(dim=-1)
        features = torch.cat([(a-b).abs(), a*b, a+b], -1)
        w = F.softplus(self.net(features) + .5413)
        return self.log_s.exp() * (((a-b).square()*w).sum(-1)+1e-12).sqrt()


def ci(i, j, n):
    a, b = np.minimum(i, j), np.maximum(i, j)
    return n*a-a*(a+1)//2+b-a-1


def matrix(path, n):
    condensed = np.load(path, mmap_mode='r')
    a, b = np.triu_indices(n, 1)
    d = np.zeros((n, n), dtype=np.float32)
    d[a, b] = condensed[ci(a, b, n)]
    d += d.T
    return d


def neighbors(d, limit=2000):
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1, kind='stable')[:, :min(limit, len(d)-1)]


def score(emb, distances, head=None):
    """Only the held-out validation pool; self matches excluded, ties pessimistic."""
    e = torch.as_tensor(emb, device=DEVICE)
    n = len(e)
    gt = neighbors(distances, 10)
    hits1, hits10 = [], []
    with torch.no_grad():
        for q in range(n):
            pred = (e-e[q]).norm(dim=1) if head is None else head(e[q:q+1].expand(n, -1), e)
            if not torch.isfinite(pred).all():
                return dict(r1=0., r10=0.)
            pred[q] = torch.inf
            rank = int((pred <= pred[int(gt[q, 0])]).sum())-1
            hits1.append(rank < max(1, int(.02*n)))
            ix = torch.argsort(pred, stable=True)[:max(1, int(.02*n))].cpu().numpy()
            hits10.append(len(set(ix) & set(gt[q]))/10)
    return dict(r1=float(np.mean(hits1)), r10=float(np.mean(hits10)))


def key(v):
    return (v['r10'], v['r1'])


def encoder(task):
    from omegaconf import OmegaConf
    from wetw._reference.model.builder import ModelBuilder
    cfg = OmegaConf.load(task['config'])
    cfg.system.device = str(DEVICE)
    cfg.encoder_model.eps = 1e-8  # same exporter as the existing Figure 6
    net = ModelBuilder(cfg).to(DEVICE)
    net.load_state_dict(load(task['encoder']))
    return net


def encode(net, raw, length, dest=None):
    """Each completed embedding chunk is resumable, including database encoding."""
    net.eval()
    batch = max(1, min(128, 128000//length))
    if dest is None:
        result = np.empty((len(raw), 128), np.float32)
        start = 0
    else:
        dest = Path(dest)
        mark = dest.with_suffix('.progress.json')
        start = read(mark)['rows'] if mark.exists() else 0
        result = np.lib.format.open_memmap(dest, mode='r+' if dest.exists() else 'w+',
                                         dtype=np.float32, shape=(len(raw), 128))
    with torch.no_grad():
        for a in range(start, len(raw), batch):
            x = torch.as_tensor(np.array(raw[a:a+batch], dtype=np.float32), device=DEVICE)[:, None]
            z = net.encode(x)
            z = z[0] if isinstance(z, tuple) else z
            assert torch.isfinite(z).all(), 'nonfinite embeddings'
            result[a:a+len(z)] = z.cpu().numpy()
            if dest is not None:
                result.flush()
                write(mark, dict(rows=a+len(z)))
                check_time()
    return result


def head_from(task):
    h = Head().to(DEVICE)
    st = load(task['wetw'])
    h.net.load_state_dict({k[4:]: v for k, v in st.items() if k.startswith('net.')})
    return h


def data(task):
    d = Path(task['data_dir'])
    x = np.load(d/'train_sample.npy', mmap_mode='r')
    v = np.load(d/'valid_sample.npy', mmap_mode='r')
    return x, v, matrix(d/'train_dists.npy', len(x)), matrix(d/'valid_dists.npy', len(v))


def calibrated(head, emb, d):
    e = torch.as_tensor(emb[:128], device=DEVICE)
    with torch.no_grad():
        a, b = torch.triu_indices(len(e), len(e), 1, device=DEVICE)
        target = torch.as_tensor(d[:128, :128], device=DEVICE)[a, b]
        head.log_s.zero_()
        ratio = target/(head(e[a], e[b])+1e-9)
        ratio = ratio[torch.isfinite(ratio) & (ratio > 0)]
        head.log_s.copy_(ratio.median().log())


def encoder_fit(task, hp, out, steps):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'encoder_done.json').exists():
        return
    x, v, dt, dv = data(task)
    near = torch.as_tensor(neighbors(dt), device=DEVICE)
    target = torch.as_tensor(dt, device=DEVICE)
    net, head = encoder(task), head_from(task)
    if hp['optimizer'] == 'sgd':
        opt = torch.optim.SGD(net.parameters(), lr=hp['lr'], momentum=.9, weight_decay=hp['decay'])
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=hp['lr'], weight_decay=hp['decay'])
    oh = torch.optim.Adam(head.parameters(), lr=5e-5)
    ck = out/'encoder_resume.pt'
    st = 0
    best = None
    if ck.exists():
        r = load(ck)
        net.load_state_dict(r['encoder']); head.load_state_dict(r['head'])
        opt.load_state_dict(r['optimizer']); oh.load_state_dict(r['head_optimizer'])
        st, best = r['step'], r['best']
        restore_rng(r['rng'])
    else:
        calibrated(head, encode(net, x[:128], task['length']), dt)
        ve = encode(net, v, task['length'])
        # Keep the published refined heads as candidates, not just the co-trained head.
        candidates = [head]
        for path in Path(task['stock_heads']).glob('head_mlp_w_*.pt'):
            old = load(path)
            assert not old.get('norm', False), 'stock head has a different feature normalization'
            h = Head().to(DEVICE)
            h.load_state_dict(old['state'])
            candidates.append(h)
        head.load_state_dict(max(candidates, key=lambda h: key(score(ve, dv, h))).state_dict())
        baseline = max((score(ve, dv), score(ve, dv, head)), key=key)
        best = baseline
        save(out/'encoder_best.pt', dict(encoder=net.state_dict(), head=head.state_dict(), validation=best, step=0))

    def checkpoint():
        save(ck, dict(step=st, encoder=net.state_dict(), head=head.state_dict(),
                     optimizer=opt.state_dict(), head_optimizer=oh.state_dict(), best=best, rng=rng_state()))

    while st < steps and hp['lr'] > 0:
        net.train(); head.train()
        opt.zero_grad(); oh.zero_grad()
        bs = hp['batch']
        a = torch.randint(len(x), (bs,), device=DEVICE)
        b = torch.randint(len(x), (bs,), device=DEVICE)
        b = torch.where(a == b, (b+1) % len(x), b)
        micro = max(1, min(bs, 64000//(2*task['length'])))
        total = 0.
        for i in range(0, bs, micro):
            ai, bi = a[i:i+micro], b[i:i+micro]
            xx = np.array(x[torch.cat([ai, bi]).cpu().numpy()], dtype=np.float32)
            z = net.encode(torch.as_tensor(xx, device=DEVICE)[:, None])
            z = z[0] if isinstance(z, tuple) else z
            za, zb = z.chunk(2)
            t = target[ai, bi].clamp_min(1e-6)
            loss = F.huber_loss((head(za, zb)-t)/t, torch.zeros_like(t), delta=.1)
            if hp.get('ed_weight', 0):
                ed = head.log_s.exp()*(za-zb).norm(dim=1)
                loss = loss + hp['ed_weight']*F.huber_loss((ed-t)/t, torch.zeros_like(t), delta=.1)
            assert torch.isfinite(loss), 'nonfinite encoder loss'
            (loss*len(ai)/bs).backward()
            total += float(loss.detach())*len(ai)/bs
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        opt.step(); oh.step(); st += 1
        if st % 50 == 0 or st == steps:
            ve = encode(net, v, task['length'])
            val = max((score(ve, dv), score(ve, dv, head)), key=key)
            if key(val) > key(best):
                best = val
                save(out/'encoder_best.pt', dict(encoder=net.state_dict(), head=head.state_dict(), validation=best, step=st))
            checkpoint()
            print('ENCODER', task['name'], st, total, val, 'best', best, flush=True)
        if STOP or time.monotonic() >= DEADLINE:
            checkpoint(); raise PieceEnd()
    checkpoint()
    write(out/'encoder_done.json', dict(hp=hp, steps=st, validation=best, warm_start=True))


def embeddings(task, out, splits=('train', 'valid')):
    net = encoder(task)
    net.load_state_dict(load(out/'encoder_best.pt')['encoder'])
    paths = {s: Path(task['data_dir'])/(s+'_sample.npy') for s in ('train', 'valid', 'test')}
    paths.update(db=Path(task['database']), q=Path(task['queries']))
    for s in splits:
        encode(net, np.load(paths[s], mmap_mode='r'), task['length'], out/f'emb_{s}.npy')
    del net
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def refine(task, source, hp, out, steps, smoke_stop=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        return
    e = torch.as_tensor(np.load(source/'emb_train.npy'), device=DEVICE)
    v = np.load(source/'emb_valid.npy')
    d = Path(task['data_dir'])
    dt = torch.as_tensor(matrix(d/'train_dists.npy', len(e)), device=DEVICE)
    dv = matrix(d/'valid_dists.npy', len(v))
    near = torch.as_tensor(neighbors(dt.cpu().numpy()), device=DEVICE)
    h = Head().to(DEVICE)
    h.load_state_dict(load(source/'encoder_best.pt')['head'])
    opt = torch.optim.Adam(h.parameters(), lr=hp['lr'])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    ck = out/'resume.pt'
    st, best = 0, score(v, dv, h)
    if ck.exists():
        r = load(ck); h.load_state_dict(r['head']); opt.load_state_dict(r['optimizer'])
        sch.load_state_dict(r['scheduler']); st, best = r['step'], r['best']; restore_rng(r['rng'])
    else:
        save(out/'best.pt', dict(head=h.state_dict(), validation=best, step=0))

    def checkpoint():
        save(ck, dict(step=st, head=h.state_dict(), optimizer=opt.state_dict(),
                     scheduler=sch.state_dict(), best=best, rng=rng_state(), hp=hp, steps=steps))

    while st < steps:
        h.train()
        bs = hp.get('batch', 4096)
        a = torch.randint(len(e), (bs,), device=DEVICE)
        b = torch.randint(len(e)-1, (bs,), device=DEVICE); b += b >= a
        t = dt[a, b].clamp_min(1e-6)
        loss = F.huber_loss((h(e[a], e[b])-t)/t, torch.zeros_like(t), delta=hp.get('delta', .1))
        if hp['rank_weight']:
            ia = torch.randint(len(e), (1024,), device=DEVICE)
            pos = near[ia, torch.randint(10, (1024,), device=DEVICE)]
            neg = torch.randint(len(e), (1024,), device=DEVICE)
            hard = near[ia, torch.randint(10, min(hp.get('hard_rank', 2000), near.shape[1]), (1024,), device=DEVICE)]
            neg = torch.where(torch.rand(1024, device=DEVICE) < hp['hard_fraction'], hard, neg)
            dp, dn = h(e[ia], e[pos]), h(e[ia], e[neg])
            loss = loss + hp['rank_weight']*F.relu(hp['margin']+dp.clamp_min(1e-6).log()-dn.clamp_min(1e-6).log()).mean()
        assert torch.isfinite(loss), 'nonfinite refinement loss'
        opt.zero_grad(); loss.backward(); opt.step(); sch.step(); st += 1
        every = min(2000, max(10, steps//5))
        if st % every == 0 or st == steps:
            h.eval(); val = score(v, dv, h)
            if key(val) > key(best):
                best = val
                save(out/'best.pt', dict(head=h.state_dict(), validation=best, step=st))
            checkpoint()
            print('REFINE', task['name'], st, val, 'best', best, flush=True)
        if (smoke_stop is not None and st == smoke_stop) or STOP or time.monotonic() >= DEADLINE:
            checkpoint(); raise PieceEnd()
    checkpoint()
    write(out/'result.json', dict(hp=hp, steps=steps, validation=best))


def select_prefilter(emb, distances, head, euclidean=False):
    e = torch.as_tensor(emb, device=DEVICE)
    n, rows = len(e), {}
    gt = neighbors(distances, 10)
    factors = [1] if euclidean else [1, 2, 4, 8, 16, 50]
    with torch.no_grad():
        for factor in factors:
            r1, r10 = [], []
            for q in range(n):
                ed = (e-e[q]).norm(dim=1); ed[q] = torch.inf
                prefix = torch.argsort(ed, stable=True)[:min(n-1, factor*max(1, int(.02*n)))]
                values = head(e[q:q+1].expand(len(prefix), -1), e[prefix], euclidean=euclidean)
                top = prefix[torch.argsort(values, stable=True)[:max(1, int(.02*n))]].cpu().numpy()
                r1.append(int(gt[q, 0]) in top)
                r10.append(len(set(top) & set(gt[q]))/10)
            rows[str(factor)] = dict(r1=float(np.mean(r1)), r10=float(np.mean(r10)))
    chosen = max(factors, key=lambda f: key(rows[str(f)]))  # ascending order breaks ties toward cheaper scoring
    return dict(kappa=chosen, validation=rows[str(chosen)], candidates=rows)


def evaluate(task, out, head_file, euclidean=False, kappa=50):
    """Test queries are first accessed after validation-only model selection."""
    h = Head().to(DEVICE)
    h.load_state_dict(load(head_file)['head']); h.eval()
    e, q = [torch.as_tensor(np.load(out/f'emb_{s}.npy'), device=DEVICE) for s in ('db', 'q')]
    sizes = [3250, 6500, 13000, 26000] if task['length'] == 47998 else [26000]
    for n in sizes:
        dest = out/f'order_N{n}.npy'; mark = out/f'order_N{n}.progress.json'
        start = read(mark)['queries'] if mark.exists() else 0
        order = np.lib.format.open_memmap(dest, mode='r+' if dest.exists() else 'w+', dtype=np.int32, shape=(len(q), n))
        with torch.no_grad():
            for i in range(start, len(q)):
                base = torch.argsort((e[:n]-q[i]).norm(dim=1), stable=True)
                prefix = base[:min(n, kappa*max(10, int(.02*n)))]
                scores = h(q[i:i+1].expand(len(prefix), -1), e[prefix], euclidean=euclidean)
                assert torch.isfinite(scores).all()
                ranked = prefix[torch.argsort(scores, stable=True)]
                order[i] = torch.cat([ranked, base[len(prefix):]]).cpu().numpy()
                order.flush(); write(mark, dict(queries=i+1)); check_time()
        ex = Path(task['exact']) if n == 26000 else Path(task['exact']).with_name(f'gt_N{n}.nn.bin')
        truth = np.fromfile(ex, dtype=np.int32).reshape(-1, 10)
        r = {}
        for rho in [.02]:
            m = max(10, int(rho*n))
            r[str(rho)] = float(np.mean([len(set(order[i, :m]) & set(truth[i]))/10 for i in range(len(q))]))
        write(out/f'search_N{n}.json', dict(N=n, recall10=r, kappa=kappa,
                                           ranking_sha256=hashlib.sha256(order.tobytes()).hexdigest()))


def main(mode, index):
    seed()
    manifest = read(ROOT/'manifest.json')
    task = manifest['tasks'][index if mode == 'production' else manifest['pilot_task']]
    if mode == 'encoder_pilot':
        out = ROOT/'encoder_pilot'/str(index)
        encoder_fit(task, manifest['encoder_grid'][index], out, manifest['encoder_pilot_steps'])
        embeddings(task, out)
        hp = manifest['head_grid'][0]
        refine(task, out, hp, out/'head', manifest['encoder_pilot_head_steps'])
        vals = [read(out/'encoder_done.json')['validation'], read(out/'head/result.json')['validation']]
        write(out/'result.json', dict(validation=max(vals, key=key), hp=manifest['encoder_grid'][index]))
    elif mode == 'head_pilot':
        chosen = read(ROOT/'encoder_selection.json')
        source = ROOT/'encoder_pilot'/str(chosen['index'])
        refine(task, source, manifest['head_grid'][index], ROOT/'head_pilot'/str(index), manifest['head_pilot_steps'])
    elif mode == 'production':
        out = ROOT/'production'/task['name']
        if (out/'result.json').exists(): return
        ep, hp = read(ROOT/'encoder_selection.json')['hp'], read(ROOT/'head_selection.json')['hp']
        encoder_fit(task, ep, out, manifest['production_encoder_steps'])
        embeddings(task, out)
        for steps in [30000, 100000]:
            refine(task, out, hp, out/f'head_{steps}', steps)
        # Selection is committed before opening database/query embeddings.
        if not (out/'selection.json').exists():
            v = np.load(out/'emb_valid.npy'); dv = matrix(Path(task['data_dir'])/'valid_dists.npy', len(v))
            candidates = [('ed', score(v, dv))] + [(str(st), read(out/f'head_{st}/result.json')['validation']) for st in [30000,100000]]
            selected, val = max(candidates, key=lambda t: key(t[1]))
            write(out/'selection.json', dict(selected=selected, validation=val, candidates=dict(candidates), protocol='validation only'))
        selected = read(out/'selection.json')['selected']
        fp = out/'encoder_best.pt' if selected == 'ed' else out/f'head_{selected}/best.pt'
        if not (out/'prefilter_selection.json').exists():
            v = np.load(out/'emb_valid.npy'); dv = matrix(Path(task['data_dir'])/'valid_dists.npy', len(v))
            h = Head().to(DEVICE); h.load_state_dict(load(fp)['head']); h.eval()
            write(out/'prefilter_selection.json', select_prefilter(v, dv, h, selected == 'ed'))
        kappa = read(out/'prefilter_selection.json')['kappa']
        embeddings(task, out, ('db', 'q'))
        evaluate(task, out, fp, selected == 'ed', kappa)
        write(out/'result.json', dict(complete=True, task=task, selection=read(out/'selection.json'),
                                     note='Timing requires fresh measurement; old figure timings are not reused.'))
    elif mode == 'smoke':
        # A real encoder forward/backward, then bit-exact uninterrupted vs resumed head training.
        out = ROOT/'smoke_v2'
        encoder_fit(task, manifest['encoder_grid'][1], out, 2)
        embeddings(task, out)
        seed(42)
        interrupted = out/'encoder_piece'
        encoder_fit(task, manifest['encoder_grid'][1], interrupted, 1)
        (interrupted/'encoder_done.json').unlink()
        seed(777)
        encoder_fit(task, manifest['encoder_grid'][1], interrupted, 2)
        ea, eb = load(out/'encoder_resume.pt'), load(interrupted/'encoder_resume.pt')
        assert ea['step'] == eb['step'] == 2
        for k, v in ea['encoder'].items():
            torch.testing.assert_close(v, eb['encoder'][k], rtol=1e-5, atol=1e-6)
        for k, v in ea['head'].items():
            torch.testing.assert_close(v, eb['head'][k], rtol=1e-5, atol=1e-6)
        seed(13); refine(task, out, manifest['head_grid'][0], out/'straight', 20)
        seed(13)
        try: refine(task, out, manifest['head_grid'][0], out/'resumed', 20, smoke_stop=7)
        except PieceEnd: pass
        seed(999); refine(task, out, manifest['head_grid'][0], out/'resumed', 20)
        a, b = load(out/'straight/resume.pt'), load(out/'resumed/resume.pt')
        assert a['step'] == b['step'] == 20
        for k, v in a['head'].items(): torch.testing.assert_close(v, b['head'][k], rtol=0, atol=0)
        assert a['scheduler'] == b['scheduler']
        dv = matrix(Path(task['data_dir'])/'valid_dists.npy', len(np.load(out/'emb_valid.npy')))
        ve = np.load(out/'emb_valid.npy')
        prefilter = select_prefilter(ve, dv, head_from(task), euclidean=True)
        assert prefilter['kappa'] == 1
        write(out/'result.json', dict(passed=True, resume_exact=True, encoder_resume=True,
                                      encoder_forward_backward=True, prefilter=True,
                                      code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        print('SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode'); p.add_argument('index', type=int, nargs='?', default=0)
    args = p.parse_args()
    try: main(args.mode, args.index)
    except PieceEnd:
        print('PIECE_END', flush=True)
        sys.exit(3)
