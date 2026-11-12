p = "src/util/train.py"
s = open(p).read()
old_head = """        for pe in range(1, n_pretrain + 1):
            self.encoder_model.train()
            if use_identity:"""
new_head = """        for pe in range(1, n_pretrain + 1):
            self.encoder_model.train()
            self.encoder_optimizer.zero_grad()
            if decoder_opt is not None:
                decoder_opt.zero_grad()
            if use_identity:"""
old_id = """                pred_chunks = []
                for start in range(0, n_pairs_cap, chunk):
                    end = min(start + chunk, n_pairs_cap)
                    i_np = gi[start:end].cpu().numpy()
                    j_np = gj[start:end].cpu().numpy()
                    ts_i = torch.from_numpy(self.train_data_np[i_np]).float().to(self.device).unsqueeze(1)
                    ts_j = torch.from_numpy(self.train_data_np[j_np]).float().to(self.device).unsqueeze(1)
                    enc_i = self.encoder_model.encode(ts_i)
                    enc_j = self.encoder_model.encode(ts_j)
                    emb_i = enc_i[0] if isinstance(enc_i, tuple) else enc_i
                    emb_j = enc_j[0] if isinstance(enc_j, tuple) else enc_j
                    pred_chunks.append(l2(emb_i, emb_j))
                pred_dists = torch.cat(pred_chunks, dim=0)
                if scale:
                    pred_dists = pred_dists / math.sqrt(dim_emb)
                true_dists = self.lossF_distances_train._targets_from_global_pairs(gi, gj)
                loss = F.mse_loss(pred_dists, true_dists)
"""
new_id = """                # MSE over all pairs, back-propagated chunk by chunk (gradient
                # accumulation): identical gradients to one big graph, but memory
                # no longer grows with n_pairs_cap * L (needed for L >= ~3000).
                true_dists = self.lossF_distances_train._targets_from_global_pairs(gi, gj)
                loss_sum = 0.0
                for start in range(0, n_pairs_cap, chunk):
                    end = min(start + chunk, n_pairs_cap)
                    i_np = gi[start:end].cpu().numpy()
                    j_np = gj[start:end].cpu().numpy()
                    ts_i = torch.from_numpy(self.train_data_np[i_np]).float().to(self.device).unsqueeze(1)
                    ts_j = torch.from_numpy(self.train_data_np[j_np]).float().to(self.device).unsqueeze(1)
                    enc_i = self.encoder_model.encode(ts_i)
                    enc_j = self.encoder_model.encode(ts_j)
                    emb_i = enc_i[0] if isinstance(enc_i, tuple) else enc_i
                    emb_j = enc_j[0] if isinstance(enc_j, tuple) else enc_j
                    pred = l2(emb_i, emb_j)
                    if scale:
                        pred = pred / math.sqrt(dim_emb)
                    part = F.mse_loss(pred, true_dists[start:end], reduction='sum') / n_pairs_cap
                    part.backward()
                    loss_sum += part.item()
                loss = torch.tensor(loss_sum)
"""
old_dec = """                loss = F.mse_loss(dec, batch.squeeze(1))

            self.encoder_optimizer.zero_grad()
            if decoder_opt is not None:
                decoder_opt.zero_grad()
            loss.backward()
"""
new_dec = """                loss = F.mse_loss(dec, batch.squeeze(1))
                loss.backward()

"""
for o, n in ((old_head, new_head), (old_id, new_id), (old_dec, new_dec)):
    assert s.count(o) == 1, o[:80]
    s = s.replace(o, n)
open(p, "w").write(s)
print("patched")
