#!/usr/bin/env python3
"""NeuTS_WD switch (+training.objective=wd): same NeuTS network, sampling and budget, but trained with
WarpDistill -- the relative Huber error (delta 0.1) between the Euclidean distance of the two NeuTS
embeddings and the DTW target (recovered from the exp(-alpha * d) batch targets; self-pairs excluded).
Distances are computed directly from the embeddings (no exp), so large distances do not underflow.
usage: patch_neuts_wd.py <train_neuts.py>
"""
import sys

p = sys.argv[1]
s = open(p).read()
rep = [
("""        mse_loss_m = WeightedRankingLoss(sampling_num=self.sampling_num)
        mse_loss_m.cuda()
""",
"""        mse_loss_m = WeightedRankingLoss(sampling_num=self.sampling_num)
        mse_loss_m.cuda()
        objective = getattr(self.cfg.training, "objective", "native")  # "native" or "wd" (WarpDistill)
        logger.info(f"objective: {objective}")
        if objective == "wd":
            import types

            def _distances(net, inputs_arrays, inputs_len_arrays):
                # NeuSeq_Network.forward up to the embeddings; returns Euclidean distances, not exp(-d)
                emb = []
                for k in range(3):
                    x = torch.Tensor(inputs_arrays[k])
                    h = net._get_initial_hidden(x.size(0))
                    emb.append(net.rnn([x.cuda(), inputs_len_arrays[k]], h))
                return (F.pairwise_distance(emb[0], emb[1], p=2), F.pairwise_distance(emb[0], emb[2], p=2))

            spatial_net.forward = types.MethodType(_distances, spatial_net)

            def _warpdistill(dp, tp, dn, tn):
                d_hat = torch.cat([dp.view(-1), dn.view(-1)])
                d = -torch.log(torch.cat([tp.view(-1), tn.view(-1)]).clamp_min(1e-30)) / config.mail_pre_degree
                m = d > 1e-8  # drop the anchor-with-itself entries (target exp(0) = 1)
                return F.huber_loss((d_hat[m] - d[m]) / d[m], torch.zeros_like(d[m]), delta=0.1)

            mse_loss_m = _warpdistill
"""),
]
for a, b in rep:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
if "import torch.nn.functional as F" not in s:
    s = s.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n", 1)
open(p, "w").write(s)
print("ok")
