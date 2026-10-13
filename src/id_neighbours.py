# ============================================================
# 11.4a src/id_neighbours.py: optional row-order model (LEAK)
# ============================================================
"""
id_neighbours.py - OPTIONAL row-order model (notebook section 11, USE_ID_NEIGHBOURS)

The posts are scraped comment sections stored in id order: neighbouring ids are
comments on the same video, so they share topic and often label. This "model"
predicts a post from the labels of the TRAINING posts with nearby ids.

It does not read the text at all. It exploits how the data file was ordered
(a leak), the competition rules do not mention it, and it would not work on
new posts. It is therefore off by default and its outputs are named
id_neighbours, so any ensemble or submission using it is clearly labelled.

Honest out-of-fold scores: for fold k, the features of both training and
validation rows are built only from training-fold labels (a row never sees its
own label). Test features use all training labels.
Writes preds/oof_id_neighbours_f{k}.csv + preds/test_id_neighbours_f{k}.csv.
"""
VERSION = "2026-09-28.2"

import os, sys, time, argparse
if "--smoke" in sys.argv:
    os.environ["HERWILL_SMOKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.linear_model import LogisticRegression
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
NAME, WINDOWS = "id_neighbours", (3, 10, 30, 100)
print(f"id_neighbours.py VERSION {VERSION} | common.py VERSION {C.VERSION} | smoke={C.SMOKE} | windows +/-{WINDOWS} ids")

def features(query_ids, ref_ids, ref_y):
    """Share of each label among reference rows within +/-w ids (self excluded), per window."""
    order = np.argsort(ref_ids); r, ry = ref_ids[order], ref_y[order]
    cum = np.vstack([np.zeros(3), np.cumsum(np.eye(3)[ry], axis=0)])
    is_ref = np.isin(query_ids, r)
    out = []
    for w in WINDOWS:
        cnt = cum[np.searchsorted(r, query_ids + w, "right")] - cum[np.searchsorted(r, query_ids - w, "left")]
        cnt[is_ref] -= np.eye(3)[ry[np.searchsorted(r, query_ids[is_ref])]]
        out.append((cnt + 1) / (cnt.sum(1, keepdims=True) + 3))     # smoothed label shares
    return np.hstack(out)

train_df, test_df = C.load_data()
ids, y, F = train_df["id"].values, train_df["y"].values, train_df["fold"].values
for k in range(C.N_FOLDS):
    if C.fold_done(NAME, k) and not args.overwrite:
        print(f"fold {k}: exists, skipping"); continue
    t0 = time.time()
    tr, va = F != k, F == k
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=C.SEED)
    clf.fit(features(ids[tr], ids[tr], y[tr]), y[tr])
    va_prob = clf.predict_proba(features(ids[va], ids[tr], y[tr]))
    te_prob = clf.predict_proba(features(test_df["id"].values, ids[tr], y[tr]))
    res = C.report(y[va], va_prob.argmax(1), train_df["lang"].values[va], name=f"{NAME} fold {k}")
    C.save_preds("oof", NAME, va_prob, ids[va], fold=k, overwrite=args.overwrite)
    C.save_preds("test", NAME, te_prob, test_df["id"].values, fold=k, overwrite=args.overwrite)
    C.log_run(step="11_id_neighbours", model=NAME, script_version=VERSION, fold=k, gpu="CPU",
              runtime_min=round((time.time() - t0) / 60, 2), smoke=C.SMOKE, leak=True, **res)
oof, _ = C.load_model_preds(NAME, train_df, test_df)
C.report(y, oof.argmax(1), train_df["lang"].values, name=f"{NAME} OOF (LEAK, no text)")