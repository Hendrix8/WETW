import sys
p = sys.argv[1]; s = open(p).read()
rep = [
("""        if os.path.isfile(train_sample_p) and os.path.isfile(train_dists_p):
            X_train = np.load(train_sample_p).astype(np.float32)
            train_dists_compact = np.load(train_dists_p).astype(np.float32)""",
"""        native = getattr(cfg.training, "objective", "dtw") == "native"  # no DTW targets needed
        if os.path.isfile(train_sample_p) and (native or os.path.isfile(train_dists_p)):
            X_train = np.load(train_sample_p).astype(np.float32)
            train_dists_compact = None if native else np.load(train_dists_p).astype(np.float32)"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w").write(s); print("ok", p)
