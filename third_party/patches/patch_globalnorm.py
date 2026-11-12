#!/usr/bin/env python3
"""LAB_FEATNORM=global: rescale the generator's pair features by one global std of the training
embeddings (no centring). Embeddings with std ~1 are unchanged; large-scale ones no
longer saturate the MLP. LAB_FEATNORM=1 keeps the per-dimension standardisation."""
p = "lab_heads.py"
s = open(p).read()
rep = [
("""FEATNORM = os.environ.get("LAB_FEATNORM", "0") == "1"
""",
"""FEATNORM = os.environ.get("LAB_FEATNORM", "0")  # "0" off, "1" per-dimension, "global" one scalar
"""),
("""        head = Head(kind, d, norm=FEATNORM).to(DEV)
        if head.norm:
            head.mu.copy_(E["train"].mean(0)); head.sd.copy_(E["train"].std(0) + 1e-6)
""",
"""        head = Head(kind, d, norm=FEATNORM != "0").to(DEV)
        if head.norm and FEATNORM == "global":
            head.sd.fill_(float(E["train"].std()) + 1e-6)  # mu stays 0: pure rescaling
        elif head.norm:
            head.mu.copy_(E["train"].mean(0)); head.sd.copy_(E["train"].std(0) + 1e-6)
"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w").write(s)
print("ok")
