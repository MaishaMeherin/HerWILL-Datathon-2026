# ============================================================
# 10.6a src/make_soft_labels.py: nested-CV TF-IDF second opinion
# ============================================================
"""
make_soft_labels.py - TF-IDF "second opinion" for soft labels (notebook section 10, optional)

Some identical posts got different labels from annotators, so some training
labels are wrong or ambiguous. Soft labels let a
model learn a mix of the given label and a second opinion:

    target = (1 - alpha) * one_hot(y) + alpha * teacher_probs

Teacher = TF-IDF + logistic regression. For each outer fold k it is trained
with 4-fold CV *inside* the other 4 folds, so each training row's second
opinion comes from a model that never saw that row and never saw fold k
(no leakage into fold k's validation score).

Writes results/soft_tfidf_f{k}.csv: id, y, p0, p1, p2, p_label
(one file per outer fold, covering that fold's ~37.6K training rows).

--evaluate also trains TF-IDF students on the soft targets and scores them
on the ORIGINAL labels of each held-out fold (honest check that it helps).
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
ap.add_argument("--evaluate", action="store_true", help="also test soft labels on TF-IDF students")
ap.add_argument("--overwrite", action="store_true", help="rebuild files that already exist")
ap.add_argument("--n_jobs", type=int, default=min(C.N_FOLDS, os.cpu_count() or 1))
ap.add_argument("--smoke", action="store_true", help="small sample, outputs in _smoke/")
args = ap.parse_args()
print(f"make_soft_labels.py VERSION {VERSION} | common.py VERSION {C.VERSION} | smoke={C.SMOKE} | n_jobs {args.n_jobs}")
t_start = time.time()

# ---------------------------------------------------------------- data + features
train_df, test_df = C.load_data()
y, F = train_df["y"].values, train_df["fold"].values
tr_txt = train_df["text"].map(C.clean)
te_txt = test_df["text"].map(C.clean)

# same features as the TF-IDF baseline; vocabulary fitted on train + test text (no labels)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           max_features=400_000, sublinear_tf=True)
word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                           token_pattern=r"[\wঀ-৿]+", max_features=200_000, sublinear_tf=True)
all_txt = pd.concat([tr_txt, te_txt])
char_vec.fit(all_txt); word_vec.fit(all_txt)
X = sp.hstack([char_vec.transform(tr_txt), word_vec.transform(tr_txt)]).tocsr()
print(f"features {X.shape} ({time.time() - t_start:.0f}s)")

# 'balanced' class weights, fixed from the full label counts
CW = len(y) / (3 * np.bincount(y, minlength=3))

def fit_lr(rows, labels, weights):
    return LogisticRegression(C=4.0, max_iter=3000).fit(X[rows], labels, sample_weight=weights * CW[labels])

# ---------------------------------------------------------------- 1. teacher (nested CV)
def teacher_for_fold(k):
    """Second-opinion probs for the training rows of outer fold k (never sees fold k)."""
    probs = np.zeros((len(y), 3))
    for j in sorted(set(range(C.N_FOLDS)) - {k}):
        fit_rows = np.where((F != k) & (F != j))[0]
        pred_rows = np.where(F == j)[0]
        probs[pred_rows] = fit_lr(fit_rows, y[fit_rows], np.ones(len(fit_rows))).predict_proba(X[pred_rows])
    rows = np.where(F != k)[0]
    out = pd.DataFrame(probs[rows], columns=C.COLS)
    out.insert(0, "y", y[rows]); out.insert(0, "id", train_df["id"].values[rows])
    out["p_label"] = probs[rows, y[rows]]
    return k, out

soft_path = lambda k: f"{C.RESULTS_DIR}/soft_tfidf_f{k}.csv"
os.makedirs(C.RESULTS_DIR, exist_ok=True)
todo = [k for k in range(C.N_FOLDS) if args.overwrite or not os.path.exists(soft_path(k))]
print(f"\n1. teacher: building folds {todo} (existing files kept: "
      f"{[k for k in range(C.N_FOLDS) if k not in todo]})")
t = time.time()
for k, out in Parallel(n_jobs=args.n_jobs)(delayed(teacher_for_fold)(k) for k in todo):
    C.write_csv(out, soft_path(k))
print(f"   done in {time.time() - t:.0f}s")

soft = {k: pd.read_csv(soft_path(k)) for k in range(C.N_FOLDS)}
for k, s in soft.items():
    expect = set(train_df.loc[F != k, "id"])
    assert set(s["id"]) == expect and len(s) == len(expect), f"fold {k}: rows do not match training rows"
    assert np.allclose(s[C.COLS].sum(axis=1), 1, atol=1e-4), f"fold {k}: probabilities do not sum to 1"
print("   checks passed: every file covers exactly its fold's training rows")

# ---------------------------------------------------------------- 2. noise summary
# each row is a training row for 4 outer folds; average its 4 second opinions
p_label = pd.concat(soft.values()).groupby("id")["p_label"].mean()
d = train_df[["id", "y", "lang"]].assign(p_label=train_df["id"].map(p_label))
d["suspect"] = d["p_label"] < 0.1
print(f"\n2. label noise summary: {d['suspect'].mean():.1%} of rows are strong suspects "
      f"(teacher gives the given label < 10%)")
tab = (pd.crosstab(d["lang"], d["y"], values=d["suspect"], aggfunc="mean") * 100).round(1)
tab.columns = [f"% suspects, y={c} ({C.LABELS[c]})" for c in tab.columns]
print(tab.reindex([lg for lg in C.LANGS if lg in tab.index]).to_string())
C.log_run(step="10_soft_labels", smoke=C.SMOKE, model="tfidf_teacher", script_version=VERSION,
          suspect_rate=round(d["suspect"].mean(), 4), runtime_min=round((time.time() - t_start) / 60, 1))

# ---------------------------------------------------------------- 3. optional: does it help?
if args.evaluate:
    VARIANTS = {"plain": None, "soft_a0.3": 0.3, "soft_a0.5": 0.5, "downweight_suspects": "w"}

    def student(k, name):
        rows = np.where(F != k)[0]; val = np.where(F == k)[0]
        s = soft[k].set_index("id").loc[train_df["id"].values[rows]]
        a = VARIANTS[name]
        if a is None:
            m = fit_lr(rows, y[rows], np.ones(len(rows)))
        elif a == "w":   # weight 0.3 for rows the teacher strongly doubts
            m = fit_lr(rows, y[rows], np.where(s["p_label"].values < 0.2, 0.3, 1.0))
        else:            # soft targets via 3 weighted copies of each row
            target = (1 - a) * np.eye(3)[y[rows]] + a * s[C.COLS].values
            rep = np.repeat(rows, 3); lab = np.tile([0, 1, 2], len(rows))
            m = LogisticRegression(C=4.0, max_iter=3000).fit(
                X[rep], lab, sample_weight=target.ravel() * CW[lab])
        return k, name, m.predict_proba(X[val])

    print(f"\n3. evaluating {list(VARIANTS)} on original labels (takes a while)...")
    t = time.time()
    oof = {n: np.zeros((len(y), 3)) for n in VARIANTS}
    for k, n, p in Parallel(n_jobs=args.n_jobs)(delayed(student)(k, n) for n in VARIANTS for k in range(C.N_FOLDS)):
        oof[n][F == k] = p
    print(f"   done in {time.time() - t:.0f}s")

    grid = np.arange(-1.5, 1.51, 0.1)
    def honest_shifted(logp):
        """class shifts tuned on 4 folds, applied to the 5th"""
        pred = np.zeros(len(y), int)
        for k in range(C.N_FOLDS):
            a, b = F != k, F == k
            best = max(((C.macro_f1(y[a], (logp[a] + [0, b1, b2]).argmax(1)), (0, b1, b2))
                        for b1 in grid for b2 in grid))
            pred[b] = (logp[b] + best[1]).argmax(1)
        return pred

    base_folds = None
    print(f"   {'variant':22s} {'argmax':>7s} {'shifted':>8s}   per-fold gain vs plain (shifted)")
    for n in VARIANTS:
        logp = np.log(oof[n] + 1e-9)
        p_arg, p_sh = logp.argmax(1), honest_shifted(logp)
        folds = np.array([C.macro_f1(y[F == k], p_sh[F == k]) for k in range(C.N_FOLDS)])
        base_folds = folds if base_folds is None else base_folds
        gain = folds - base_folds
        print(f"   {n:22s} {C.macro_f1(y, p_arg):7.4f} {C.macro_f1(y, p_sh):8.4f}   "
              f"{np.round(gain, 4).tolist()}  better in {(gain > 0).sum()}/5")
        C.log_run(step="10_soft_labels_eval", smoke=C.SMOKE, model=f"tfidf_student_{n}", script_version=VERSION,
                  macro_f1_argmax=round(C.macro_f1(y, p_arg), 4), macro_f1=round(C.macro_f1(y, p_sh), 4),
                  **{f"fold{k}": round(v, 4) for k, v in enumerate(folds)})

print(f"\nfinished in {(time.time() - t_start) / 60:.1f} min")