# ============================================================
# 11.1 src/ensemble.py: honest weight + class-shift tuning, submission
# ============================================================
"""
ensemble.py - combine models honestly and write a submission (notebook sections 11-12)

Blend: weighted sum of the models' log-probabilities (a weighted geometric mean).
Tuning, on out-of-fold (OOF) predictions:
  1. weights: equal weights + random candidates (fixed seed); keep the best Macro F1
  2. class shifts: add b1 to class 1 and b2 to class 2 log-scores (grid -1.5..1.5,
     step 0.05); Macro F1 rewards getting the small classes right
  3. optional --lang_shifts: a separate class-shift pair for Bangla posts
     (Bangla posts are rarely neutral)

Honest score: tune on 4 folds, score the 5th, rotate over all 5 folds. The score
from tuning and scoring on the same rows is printed next to it, labelled
"optimistic". The final weights/shifts (tuned on all OOF rows) are applied to
the test predictions.

Refuses to run if any chosen model is missing a fold. Output names contain the
models used: results/ensemble_{tag}.json, submissions/sub_ens_{tag}.csv.
--use_overlap additionally writes sub_ens_{tag}_OVERLAP.csv, where test posts
that also appear in train take the (unambiguous) train label.
"""
VERSION = "2026-09-28.2"

import os, sys, time, argparse
if "--smoke" in sys.argv:
    os.environ["HERWILL_SMOKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True, help="comma-separated model names, e.g. tfidf_lr,xlm_roberta_large")
ap.add_argument("--lang_shifts", action="store_true", help="separate class shifts for Bangla posts")
ap.add_argument("--use_overlap", action="store_true", help="also write a submission using train labels of overlapping test posts")
ap.add_argument("--n_weights", type=int, default=1000, help="random weight candidates")
ap.add_argument("--overwrite", action="store_true")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
MODELS = args.models.split(",")
TAG = "+".join(MODELS) + ("_langshift" if args.lang_shifts else "") + ("_IDLEAK" if "id_neighbours" in MODELS else "")
JSON, SUB = f"{C.RESULTS_DIR}/ensemble_{TAG}.json", f"{C.SUB_DIR}/sub_ens_{TAG}.csv"
print(f"ensemble.py VERSION {VERSION} | common.py VERSION {C.VERSION} | smoke={C.SMOKE} | models {MODELS}")
if "id_neighbours" in MODELS:
    print("WARNING: id_neighbours uses row order (a leak), not text. Label any submission from it clearly.")
def fingerprint():
    """Size + modification time of every prediction file used: changes when any fold is retrained."""
    return {os.path.basename(C.pred_path(kind, m, k)): [os.path.getsize(C.pred_path(kind, m, k)), int(os.path.getmtime(C.pred_path(kind, m, k)))]
            for m in MODELS for k in range(C.N_FOLDS) for kind in ("oof", "test") if os.path.exists(C.pred_path(kind, m, k))}

expected = [JSON, SUB] + ([SUB.replace(".csv", "_OVERLAP.csv")] if args.use_overlap else [])
if all(os.path.exists(p) for p in expected) and not args.overwrite:
    import json
    if json.load(open(JSON)).get("inputs") == fingerprint():
        print(f"exists and its prediction files are unchanged, skipping: {os.path.relpath(JSON, C.ROOT)} (use --overwrite to redo)")
        sys.exit(0)
    print("prediction files changed since the last run (or no fingerprint saved): recomputing")
t0 = time.time()

# ---------------------------------------------------------------- load (refuses on missing folds)
train_df, test_df = C.load_data()
y, F = train_df["y"].values, train_df["fold"].values
is_bn = (train_df["lang"] == "bangla").values
is_bn_test = (test_df["lang"] == "bangla").values
loaded = [C.load_model_preds(m, train_df, test_df) for m in MODELS]
L_oof = np.stack([np.log(o + 1e-9) for o, _ in loaded])      # (models, rows, 3)
L_test = np.stack([np.log(t + 1e-9) for _, t in loaded])
for m, (o, _) in zip(MODELS, loaded):
    C.report(y, o.argmax(1), name=f"single model: {m}")

GRID = np.round(np.arange(-1.5, 1.5001, 0.05), 2)
rng = np.random.default_rng(C.SEED)
W_CAND = [np.ones(len(MODELS)) / len(MODELS)] + (list(rng.dirichlet(np.ones(len(MODELS)), size=args.n_weights))
                                                  if len(MODELS) > 1 else [])

def predict(p, L, bangla):
    s = np.tensordot(p["w"], L, axes=1) + np.array([0, *p["b"]])
    if p.get("b_bn") is not None:
        s[bangla] += np.array([0, *p["b_bn"]]) - np.array([0, *p["b"]])
    return s.argmax(1)

def best_shift(score, yy, extra_mask=None, base=None):
    """Grid search (b1, b2); with extra_mask only those rows get the new shift."""
    best, best_b = -1, (0.0, 0.0)
    for b1 in GRID:
        for b2 in GRID:
            s = score.copy()
            if extra_mask is None:
                s += np.array([0, b1, b2])
            else:
                s += np.where(extra_mask[:, None], np.array([0, b1, b2]), np.array([0, *base]))
            f = C.macro_f1(yy, s.argmax(1))
            if f > best:
                best, best_b = f, (float(b1), float(b2))
    return best_b

def tune(rows):
    L, yy = L_oof[:, rows], y[rows]
    w = max(W_CAND, key=lambda w: C.macro_f1(yy, np.tensordot(w, L, axes=1).argmax(1)))
    score = np.tensordot(w, L, axes=1)
    p = {"w": [round(float(v), 4) for v in w], "b": best_shift(score, yy)}
    p["b_bn"] = best_shift(score, yy, extra_mask=is_bn[rows], base=p["b"]) if args.lang_shifts else None
    return p

# ---------------------------------------------------------------- honest: tune on 4 folds, score the 5th
honest_pred, fold_params, fold_f1 = np.zeros(len(y), int), [], []
for k in range(C.N_FOLDS):
    p = tune(np.where(F != k)[0])
    va = F == k
    honest_pred[va] = predict(p, L_oof[:, va], is_bn[va])
    fold_params.append(p); fold_f1.append(round(float(C.macro_f1(y[va], honest_pred[va])), 4))
honest = C.report(y, honest_pred, train_df["lang"].values, name="HONEST (tuned on 4 folds)", show_cm=True)
print(f"{'':29s}per fold {fold_f1}")

# ---------------------------------------------------------------- final parameters: tuned on all OOF rows
final = tune(np.arange(len(y)))
optimistic = C.report(y, predict(final, L_oof, is_bn), name="optimistic (same rows)", quiet=True)
print(f"{'optimistic (same rows)':28s} macro-F1 {optimistic['macro_f1']:.4f}  <- tuned and scored on the same rows; NOT a real estimate")
print(f"final weights {dict(zip(MODELS, final['w']))} | class shifts (class 1, class 2) {final['b']}"
      + (f" | Bangla shifts {final['b_bn']}" if args.lang_shifts else ""))

test_pred = predict(final, L_test, is_bn_test)
sub = pd.DataFrame({"id": test_df["id"].values, "y_pred": test_pred.astype(int)})
C.write_csv(sub, SUB)
outputs = [SUB]
if args.use_overlap:
    ov = pd.read_csv(C.TEST_OVERLAP).dropna(subset=["train_label"])
    lab = dict(zip(ov["id"], ov["train_label"].astype(int)))
    sub_ov = sub.assign(y_pred=[lab.get(i, p) for i, p in zip(sub["id"], sub["y_pred"])])
    changed = int((sub_ov["y_pred"] != sub["y_pred"]).sum())
    outputs.append(C.write_csv(sub_ov, SUB.replace(".csv", "_OVERLAP.csv")))
    print(f"overlap: {int(sub['id'].isin(lab).sum())} test posts take their train label ({changed} predictions changed)")

C.write_json({"models": MODELS, "tag": TAG, "lang_shifts": args.lang_shifts, "leak": "id_neighbours" in MODELS,
              "final_weights": dict(zip(MODELS, final["w"])), "final_shifts": final["b"], "final_bangla_shifts": final["b_bn"],
              "honest": honest, "honest_per_fold": fold_f1, "optimistic_macro_f1": optimistic["macro_f1"],
              "fold_params": fold_params, "submission": os.path.relpath(SUB, C.ROOT),
              "test_class_mix": np.round(np.bincount(test_pred, minlength=3) / len(test_pred), 4).tolist(),
              "inputs": fingerprint(), "script_version": VERSION, "smoke": C.SMOKE}, JSON)
C.log_run(step="11_ensemble", model=f"ens:{TAG}", script_version=VERSION, gpu="CPU", smoke=C.SMOKE,
          runtime_min=round((time.time() - t0) / 60, 1), optimistic_macro_f1=optimistic["macro_f1"],
          weights=str(final["w"]), shifts=str(final["b"]), **honest)
print(f"wrote {[os.path.relpath(p, C.ROOT) for p in outputs + [JSON]]} ({(time.time() - t0) / 60:.1f} min)")