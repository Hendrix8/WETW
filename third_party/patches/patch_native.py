#!/usr/bin/env python3
"""Native-objective switches for the encoder study (default = DTW objective, unchanged).

RF-DTAN  (+training.objective=native): upstream RF-DTAN loss, ICAE (inverse-consistency averaging
         error) with no smoothness prior, all series in one class (label-free joint alignment),
         computed by upstream models.train_utils.one_batch. DTW targets are not built.
usage: patch_native.py <train_rfdtan.py>
"""
import sys

rf = sys.argv[1]


def patch(p, rep):
    s = open(p).read()
    for a, b in rep:
        assert s.count(a) == 1, (p, a[:70])
        s = s.replace(a, b)
    open(p, "w").write(s)


patch(rf, [
("""from models.RFDTAN import RFDTAN
""",
"""from models.RFDTAN import RFDTAN
from models.train_utils import one_batch
from loss.alignment_loss import ICAE
"""),
("""            precomputed_dists = compact_to_square(train_dists_compact, len(X_train))""",
"""            precomputed_dists = (compact_to_square(train_dists_compact, len(X_train))
                                 if getattr(cfg.training, "objective", "dtw") == "dtw" else None)"""),
("            outputs = model(batch_x)\n            embeddings = outputs[\"embedding\"]\n" + " " * 12 + "\n"
 "            loss = loss_fn(embeddings, batch_x, batch_indices=batch_idx, warping_window=cfg.dataset.warping_window)",
"""            if objective == "native":  # upstream RF-DTAN objective: ICAE, one class (label-free)
                loss = one_batch(batch_x, torch.zeros(len(batch_x), dtype=torch.long), model, device,
                                 native_losses)["loss"]
            else:
                outputs = model(batch_x)
                embeddings = outputs["embedding"]
                loss = loss_fn(embeddings, batch_x, batch_indices=batch_idx, warping_window=cfg.dataset.warping_window)"""),
("""    model.train()
    best_loss = float('inf')""",
"""    objective = getattr(cfg.training, "objective", "dtw")  # "dtw" (DTW-preserving) or "native"
    native_losses = {"ICAE_loss": ICAE()}
    print(f"objective: {objective}")
    model.train()
    best_loss = float('inf')"""),
])

print("patched")
