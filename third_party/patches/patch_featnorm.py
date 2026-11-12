#!/usr/bin/env python3
"""Add optional per-dimension feature standardisation to the lab heads (LAB_FEATNORM=1)."""
p = "lab_heads.py"
s = open(p).read()
rep = [
("""    def __init__(self, kind, d, hidden=128, rank=64):
        super().__init__()
        self.kind = kind""",
"""    def __init__(self, kind, d, hidden=128, rank=64, norm=False):
        super().__init__()
        self.kind = kind
        # norm: standardise embeddings (train mean/std per dimension) before the generator pair
        # features; the distance itself stays on raw differences. Needed for encoders whose
        # embeddings have a large scale (std ~50), where e_i*e_j saturates the MLP.
        self.norm = norm and kind == "mlp_w"
        if self.norm:
            self.register_buffer("mu", torch.zeros(d)); self.register_buffer("sd", torch.ones(d))"""),
("""        if self.kind == "mlp_w":
            w = F.softplus(self.net(feats(a, b)) + 0.5413)""",
"""        fa, fb = ((a - self.mu) / self.sd, (b - self.mu) / self.sd) if self.norm else (a, b)
        if self.kind == "mlp_w":
            w = F.softplus(self.net(feats(fa, fb)) + 0.5413)"""),
("""            return s * F.softplus(self.net(feats(a, b)).squeeze(1) + torch.linalg.norm(a - b, dim=1))""",
"""            return s * F.softplus(self.net(feats(fa, fb)).squeeze(1) + torch.linalg.norm(a - b, dim=1))"""),
("""        head = Head(kind, d).to(DEV)
""",
"""        head = Head(kind, d, norm=FEATNORM).to(DEV)
        if head.norm:
            head.mu.copy_(E["train"].mean(0)); head.sd.copy_(E["train"].std(0) + 1e-6)
"""),
("""        torch.save({"kind": kind, "d": d, "state": head.state_dict()},""",
"""        torch.save({"kind": kind, "d": d, "norm": head.norm, "state": head.state_dict()},"""),
("""        hc = Head(kind, d); hc.load_state_dict""", """        hc = Head(kind, d, norm=head.norm); hc.load_state_dict"""),
("""DEV = torch.device("cuda")
""", """DEV = torch.device("cuda")
FEATNORM = os.environ.get("LAB_FEATNORM", "0") == "1"
"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
open(p, "w").write(s)
old, new = 'Head(ck["kind"], ck["d"])', 'Head(ck["kind"], ck["d"], norm=ck.get("norm", False))'
for p in ("lab_db.py", "ann_lab.py", "retime_rank.py"):
    s = open(p).read()
    print(p, s.count(old))
    open(p, "w").write(s.replace(old, new))
