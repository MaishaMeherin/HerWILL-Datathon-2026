# ============================================================
# 7.1 src/train_tfidf.py: TF-IDF + logistic regression, 5 folds
# ============================================================
"""
train_tfidf.py - TF-IDF + logistic regression baseline (notebook section 7, CPU)

Features: character 2-5-grams (char_wb) + word 1-2-grams (token pattern includes
the Bangla block, because Python's \\w misses Bangla vowel signs), sublinear tf.
The vocabulary is fitted on train + test text (no labels are used).
Model: LogisticRegression(C=4.0, class_weight="balanced") - kept from our first baseline, not tuned.
Text goes through common.clean(), the same cleaning every other model uses.

Writes preds/oof_tfidf_lr_f{k}.csv + preds/test_tfidf_lr_f{k}.csv per fold,
one row per fold in results/runs.csv, and submissions/sub_tfidf_lr.csv.
Folds that already have both files are skipped (use --overwrite to redo).
"""
VERSION = "2026-09-28.2"

import os, sys, time, argparse
if "--smoke" in sys.argv:
    os.environ["HERWILL_SMOKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", action="store_true", help="redo folds whose predictions exist")
ap.add_argument("--smoke", action="store_true", help="small sample, outputs in _smoke/")
ap.add_argument("--n_jobs", type=int, default=min(C.N_FOLDS, os.cpu_count() or 1))
args = ap.parse_args()
NAME = "tfidf_lr"
print(f"train_tfidf.py VERSION {VERSION} | common.py VERSION {C.VERSION} | smoke={C.SMOKE} | n_jobs={args.n_jobs}")

train_df, test_df = C.load_data()
todo = [k for k in range(C.N_FOLDS) if args.overwrite or not C.fold_done(NAME, k)]
if not todo:
    print("all 5 folds exist, skipping (use --overwrite to redo)")
else:
    t0 = time.time()
    y = train_df["y"].values
    tr_txt = train_df["text"].map(C.clean)
    te_txt = test_df["text"].map(C.clean)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                               max_features=400_000, sublinear_tf=True)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                               token_pattern=r"[\wঀ-৿]+", max_features=200_000, sublinear_tf=True)
    all_txt = pd.concat([tr_txt, te_txt])            # unsupervised fit only (no labels)
    char_vec.fit(all_txt); word_vec.fit(all_txt)
    X = sp.hstack([char_vec.transform(tr_txt), word_vec.transform(tr_txt)]).tocsr()
    X_te = sp.hstack([char_vec.transform(te_txt), word_vec.transform(te_txt)]).tocsr()
    print(f"features: train {X.shape}, test {X_te.shape} ({time.time() - t0:.0f}s) | folds to train: {todo}")

    def run_fold(k):
        t = time.time()
        tr, va = np.where(train_df["fold"] != k)[0], np.where(train_df["fold"] == k)[0]
        clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced", random_state=C.SEED)
        clf.fit(X[tr], y[tr])
        return k, va, clf.predict_proba(X[va]), clf.predict_proba(X_te), (time.time() - t) / 60

    for k, va, va_prob, te_prob, minutes in Parallel(n_jobs=args.n_jobs)(delayed(run_fold)(k) for k in todo):
        res = C.report(y[va], va_prob.argmax(1), train_df["lang"].values[va], name=f"{NAME} fold {k}")
        C.save_preds("oof", NAME, va_prob, train_df["id"].values[va], fold=k, overwrite=args.overwrite)
        C.save_preds("test", NAME, te_prob, test_df["id"].values, fold=k, overwrite=args.overwrite)
        C.log_run(step="07_baseline", model=NAME, script_version=VERSION, fold=k, C_reg=4.0,
                  class_weight="balanced", features=X.shape[1], gpu="CPU", runtime_min=round(minutes, 1),
                  smoke=C.SMOKE, **res)

# ---------------------------------------------------------------- all folds together
oof, test_prob = C.load_model_preds(NAME, train_df, test_df)
C.report(train_df["y"].values, oof.argmax(1), train_df["lang"].values, name=f"{NAME} OOF (5 folds)", show_cm=True)
sub = pd.DataFrame({"id": test_df["id"].values, "y_pred": test_prob.argmax(1).astype(int)})
path = C.write_csv(sub, f"{C.SUB_DIR}/sub_{NAME}.csv")
print(f"wrote {os.path.relpath(path, C.ROOT)} {sub.shape} | predicted class mix "
      f"{np.round(np.bincount(sub['y_pred'], minlength=3) / len(sub), 3).tolist()}")