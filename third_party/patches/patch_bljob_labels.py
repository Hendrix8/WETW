import sys
p = sys.argv[1]; s = open(p).read()
rep = [
("""      +training.objective=native) >> $log 2>&1 ;;""",
 """      +training.objective=native $LBLARG) >> $log 2>&1 ;;"""),
("""      model.dim_emb=128 model.n_recurrences=4 system.wandb_mode=disabled training.time_limit_hours=6) >> $log 2>&1 ;;""",
 """      model.dim_emb=128 model.n_recurrences=4 system.wandb_mode=disabled training.time_limit_hours=6 \\
      +training.smooth_prior=true) >> $log 2>&1 ;;"""),
("""  rfdtan_native)  # upstream RF-DTAN objective (ICAE, one class) and upstream default of 4 recurrences""",
 """  rfdtan_native)  # upstream RF-DTAN objective (ICAE): per class with the real labels where the collection has
                  # labels, one class otherwise; upstream default of 4 recurrences
    LBL=/data/cpanourg/1-dtwrl/data/$ds/samples_random0.1_npy/y_train.npy
    LBLARG=; [ -f $LBL ] && LBLARG=+training.labels_fp=$LBL"""),
("""  rfdtan4)  # DTW objective with 4 recurrences, so that the warping layers are trained by it""",
 """  rfdtan4)  # WarpDistill with 4 recurrences (warp trained), plus the RF-DTAN smoothness prior on the warps"""),
]
for a, b in rep:
    assert s.count(a) == 1, (s.count(a), a[:60])
    s = s.replace(a, b)
open(p + ".new", "w").write(s); print("ok")
