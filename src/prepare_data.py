# ============================================================
# 4.1 src/prepare_data.py: drop outlier, merge duplicates, overlap list, frozen folds
# ============================================================
"""
prepare_data.py - cleaning decisions + frozen folds (notebook sections 4 and 6)

  1. drop contaminated rows: train posts longer than 10,000 characters
     (id 57606, a 42K-character pasted chat dump; the longest test post is ~2.1K)
  2. near-duplicate key (common.norm_key) and language tag (common.lang_tag)
  3. merge near-duplicates: clear majority -> one row with that label;
     tie -> one row per tied label (a soft 50/50 target)
  4. test posts whose key also appears in train -> data/test_overlap.csv
     (only used if USE_OVERLAP is switched on)
  5. folds: StratifiedGroupKFold, 5 folds, seed 42, stratified on class x language,
     grouped by key, so copies of one post never sit on both sides of a split

Writes data/train_folds.csv, data/test_lang.csv, data/test_overlap.csv.
The folds are frozen: this script refuses to overwrite them, and compares them
with _archive_v1/train_folds.csv (our first version's folds) when that file is present.
"""
VERSION = "2026-09-28.2"

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--overwrite", action="store_true", help="rebuild files that already exist")
args = ap.parse_args()
print(f"prepare_data.py VERSION {VERSION} | common.py VERSION {C.VERSION} | scikit-learn {sklearn.__version__}")

MAX_TRAIN_CHARS = 10_000
SKLEARN_PIN = "1.7.2"   # StratifiedGroupKFold shuffles differently from scikit-learn 1.8 on
outputs = [C.TRAIN_FOLDS, C.TEST_LANG, C.TEST_OVERLAP]
if all(os.path.exists(p) for p in outputs) and not args.overwrite:
    print("all outputs exist, skipping (frozen). Files:", [os.path.relpath(p, C.ROOT) for p in outputs])
    sys.exit(0)
if sklearn.__version__ != SKLEARN_PIN:
    sys.exit(f"STOP: scikit-learn {sklearn.__version__} is loaded, but the frozen folds need "
             f"{SKLEARN_PIN}. Rerun section 0 (restart the runtime if it asks).")

train_raw = pd.read_csv(C.TRAIN_RAW)
test_raw = pd.read_csv(C.TEST_RAW)
train_raw["text"] = train_raw["text"].astype(str)
test_raw["text"] = test_raw["text"].astype(str)
print(f"loaded train {train_raw.shape}, test {test_raw.shape}")

# ---------------------------------------------------------------- 1. contaminated rows
too_long = train_raw["text"].str.len() > MAX_TRAIN_CHARS
print(f"1. dropped {too_long.sum()} train row(s) longer than {MAX_TRAIN_CHARS:,} chars: "
      f"ids {train_raw.loc[too_long, 'id'].tolist()} (longest test post: {test_raw['text'].str.len().max():,} chars)")
train_df = train_raw[~too_long].reset_index(drop=True)
test_df = test_raw.copy()

# ---------------------------------------------------------------- 2. key + language
train_df["key"] = train_df["text"].map(C.norm_key)
test_df["key"] = test_df["text"].map(C.norm_key)
train_df["lang"] = train_df["text"].map(C.lang_tag)
test_df["lang"] = test_df["text"].map(C.lang_tag)

# ---------------------------------------------------------------- 3. merge near-duplicates
def labels_to_keep(ys):
    vc = ys.value_counts()
    return sorted(vc[vc == vc.max()].index.tolist())   # 1 label = majority, >1 = tie

keep_labels = train_df.groupby("key")["y"].agg(labels_to_keep)
n_labels = train_df.groupby("key")["y"].nunique()
pairs = keep_labels.explode().rename("y").reset_index()
pairs["y"] = pairs["y"].astype(int)
first_rows = train_df.drop_duplicates(["key", "y"])        # first row of each (text, label)
dedup = first_rows.merge(pairs, on=["key", "y"], how="inner").reset_index(drop=True)
print(f"3. duplicate groups: {(train_df.groupby('key').size() > 1).sum()} | conflicting: {(n_labels > 1).sum()} "
      f"(majority {((n_labels > 1) & (keep_labels.str.len() == 1)).sum()}, tied {(keep_labels.str.len() > 1).sum()}) "
      f"| rows {len(train_df):,} -> {len(dedup):,}")

# ---------------------------------------------------------------- 4. train/test overlap
unambiguous = keep_labels[keep_labels.str.len() == 1].str[0]
overlap = test_df.loc[test_df["key"].isin(keep_labels.index), ["id", "key"]].copy()
overlap["train_label"] = overlap["key"].map(unambiguous).astype("Int64")   # <NA> where train is tied
print(f"4. test posts whose key appears in train: {len(overlap)} ({len(overlap) / len(test_df):.1%}); "
      f"tied in train (no usable label): {overlap['train_label'].isna().sum()}")

# ---------------------------------------------------------------- 5. group-aware folds
strat = dedup["y"].astype(str) + "_" + dedup["lang"].replace({"other": "english"})
sgkf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
dedup["fold"] = -1
for f, (_, va) in enumerate(sgkf.split(dedup, strat, groups=dedup["key"])):
    dedup.loc[va, "fold"] = f
assert (dedup.groupby("key")["fold"].nunique() == 1).all(), "a text was split across folds"
print(f"5. fold sizes: {dedup['fold'].value_counts().sort_index().tolist()}")

archived = f"{C.ARCHIVE_DIR}/train_folds.csv"
if os.path.exists(archived):
    old = pd.read_csv(archived, usecols=["id", "y", "fold"])
    same = (len(old) == len(dedup) and (old["id"].values == dedup["id"].values).all()
            and (old["y"].values == dedup["y"].values).all() and (old["fold"].values == dedup["fold"].values).all())
    print(f"   folds identical to _archive_v1/train_folds.csv: {same}")
    if not same:
        sys.exit("STOP: new folds differ from our first version's folds. Nothing written. Check the scikit-learn version.")
else:
    print("   _archive_v1/train_folds.csv not found: comparison skipped")

C.write_csv(dedup[["id", "text", "y", "lang", "fold", "key"]], C.TRAIN_FOLDS)
C.write_csv(test_df[["id", "text", "lang"]], C.TEST_LANG)
C.write_csv(overlap, C.TEST_OVERLAP)
print(f"wrote {[os.path.relpath(p, C.ROOT) for p in outputs]}")