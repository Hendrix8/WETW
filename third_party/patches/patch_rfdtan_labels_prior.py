#!/usr/bin/env python3
"""RF-DTAN trainer options (defaults unchanged):
  +training.labels_fp=<y_train.npy>  native ICAE per class, with the real class labels (as published);
                                     without it all series form one class (unlabelled collections)
  +training.smooth_prior=true        add RF-DTAN's own smoothness prior on the warps (upstream PriorLoss,
                                     its default lambda_smooth=0.5, lambda_var=0.01, weight 1) to the loss
usage: patch_rfdtan_labels_prior.py <train_rfdtan.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
rep = [
("""from loss.alignment_loss import ICAE
""", """from loss.alignment_loss import ICAE
from loss.smoothness_prior import PriorLoss
"""),
("""    # Reshape for RF-DTAN (N, C, L)""",
 """    labels_fp = getattr(cfg.training, "labels_fp", None)
    if labels_fp:  # class labels for the native per-class objective
        y_lab = np.load(labels_fp)
        if y_lab.dtype == np.float32 and np.abs(y_lab).max() < 1e-30:
            y_lab = y_lab.view(np.int32)  # int32 class ids stored in a float32 container
        y_train = y_lab.astype(np.int64)
        assert len(y_train) == len(X_train), (len(y_train), len(X_train))
        print(f"class labels: {np.unique(y_train).size} classes", flush=True)

    # Reshape for RF-DTAN (N, C, L)"""),
("""        for batch_x, _, batch_idx in pbar:""", """        for batch_x, batch_y, batch_idx in pbar:"""),
("""                loss = one_batch(batch_x, torch.zeros(len(batch_x), dtype=torch.long), model, device,""",
 """                yb = batch_y.long() if labels_fp else torch.zeros(len(batch_x), dtype=torch.long)
                loss = one_batch(batch_x, yb, model, device,"""),
("""                loss = loss_fn(embeddings, batch_x, batch_indices=batch_idx, warping_window=cfg.dataset.warping_window)""",
 """                loss = loss_fn(embeddings, batch_x, batch_indices=batch_idx, warping_window=cfg.dataset.warping_window)
                if prior is not None:  # RF-DTAN's own smoothness prior on the warps
                    loss = loss + prior(thetas=outputs["theta"])"""),
("""    native_losses = {"ICAE_loss": ICAE()}""",
 """    native_losses = {"ICAE_loss": ICAE()}
    prior = PriorLoss(model.T, lambda_smooth=0.5, lambda_var=0.01) if getattr(cfg.training, "smooth_prior", False) else None
    print(f"smoothness prior: {prior is not None}", flush=True)"""),
]
for a, b in rep:
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w").write(s)
print("ok")
