# ============================================================
# 0.2 src/common.py: paths, seed, clean(), metrics, prediction files, run log
# ============================================================
"""
common.py - shared code for every script and notebook cell: paths, seed,
text cleaning, metrics, prediction files and the run log.
Written by the notebook (section 0.2); we edit it there, not by hand.
"""
import os, re, glob, json, unicodedata
from datetime import datetime
import numpy as np
import pandas as pd

VERSION = "2026-09-28.2"

# ---------------------------------------------------------------- paths
# ROOT = the project folder (the one holding data/train.csv). common.py lives in
# ROOT/src, so ROOT is its parent folder; HERWILL_ROOT overrides it (tests).
ROOT = os.environ.get("HERWILL_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_RAW    = f"{ROOT}/data/train.csv"                  # competition files: read-only
TEST_RAW     = f"{ROOT}/data/test.csv"
SAMPLE_SUB   = f"{ROOT}/data/sample_submission.csv"
ARCHIVE_DIR  = f"{ROOT}/_archive_v1"                # our first-version files (used only to check the folds)
DATA_DIR     = f"{ROOT}/data"
TRAIN_FOLDS  = f"{DATA_DIR}/train_folds.csv"        # FROZEN: every prediction depends on its folds
TEST_LANG    = f"{DATA_DIR}/test_lang.csv"
TEST_OVERLAP = f"{DATA_DIR}/test_overlap.csv"

# SMOKE mode (quick end-to-end test on a small sample): model outputs go to
# ROOT/_smoke/..., so a test run can never overwrite real results.
SMOKE = os.environ.get("HERWILL_SMOKE") == "1"
OUT_DIR     = f"{ROOT}/_smoke" if SMOKE else ROOT
PRED_DIR    = f"{OUT_DIR}/preds"
RESULTS_DIR = f"{OUT_DIR}/results"
HISTORY_DIR = f"{RESULTS_DIR}/history"
FIG_DIR     = f"{RESULTS_DIR}/figures"
SUB_DIR     = f"{OUT_DIR}/submissions"
RUNLOG      = f"{RESULTS_DIR}/runs.csv"

SEED, N_FOLDS = 42, 5
LABELS = {0: "explicit", 1: "subtle", 2: "neutral"}
COLS = ["p0", "p1", "p2"]
LANGS = ["english", "bangla", "banglish", "mixed_script"]
SMOKE_TRAIN_PER_FOLD, SMOKE_TEST_ROWS = 100, 200

# ---------------------------------------------------------------- text processing
# clean(): we use this one function for every model, so no two models see
# differently cleaned text.
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(\D)\1{3,}")   # 4+ repeats -> 3; digits untouched

def clean(t):
    t = unicodedata.normalize("NFKC", str(t))
    t = URL_RE.sub("URL", t)
    t = MENTION_RE.sub("@USER", t)
    t = REPEAT_RE.sub(r"\1\1\1", t)
    return re.sub(r"\s+", " ", t).strip()

# norm_key(): near-duplicate key (lower case, no URLs/mentions/punctuation,
# emojis kept). lang_tag(): script-based language tag. We kept both unchanged
# from our first version, which is why the folds reproduce exactly.
def norm_key(t):
    k = unicodedata.normalize("NFKC", t).lower()
    k = re.sub(r"https?://\S+|www\.\S+|@\w+", " ", k)
    k = re.sub(r"[^\w\sঀ-৿\U0001F000-\U0001FAFF☀-➿]+", " ", k)
    k = re.sub(r"\s+", " ", k).strip()
    return k if k else "raw::" + t

BANGLISH = set("""
ami amra tumi tomra apni apnara tui tora se tara amar tomar apnar tor tar oder
ki keno kemne kemon kobe kothay kake kar koto kire
na nai nah naki ni hoy hoi hobe hoilo hoise hoyeche hoye hocche hoche ache ase asi
chilo silo thake theke thik kore korse korche korlo korbo korben koren korish kor
bole bolo bollo bolche bolse dekhe dekho dekhlam dekhen lage lagche lagse lagbe
valo bhalo vallo kharap khub onek beshi besi kom sob shob sobai shobai kichu kisu
ekta eta oita ota ei oi etai shei sei tai tahole jodi kintu abar arekta
bhai vai bhaiya apu dada didi mama kaka shala sala shaala magi pagol pagla
jonno jnno diye dia niye nia pore age pare parbe parish mone jani janina bujhi
desh manush manus kaj kotha din rat shesh sesh hashi haha lol vua bhua
""".split())

def lang_tag(t):
    bn = len(re.findall(r"[ঀ-৿]", t))
    lat = len(re.findall(r"[A-Za-z]", t))
    if bn + lat == 0:
        return "other"
    r = bn / (bn + lat)
    if r > 0.85:
        return "bangla"
    if r >= 0.15:
        return "mixed_script"
    toks = re.findall(r"[a-z]+", t.lower())
    hits = sum(w in BANGLISH for w in toks)
    if hits >= 2 or (toks and hits / len(toks) >= 0.25):
        return "banglish"
    return "english"

# ---------------------------------------------------------------- data
def load_data():
    """Prepared train (with frozen folds) and test, restricted to the SMOKE
    sample when SMOKE is on. Every script uses this, so all models always
    cover exactly the same rows."""
    train_df = pd.read_csv(TRAIN_FOLDS)
    test_df = pd.read_csv(TEST_LANG)
    train_df["text"] = train_df["text"].fillna("").astype(str)
    test_df["text"] = test_df["text"].fillna("").astype(str)
    if SMOKE:
        train_df = train_df.groupby("fold").head(SMOKE_TRAIN_PER_FOLD).reset_index(drop=True)
        test_df = test_df.head(SMOKE_TEST_ROWS).reset_index(drop=True)
    return train_df, test_df

# ---------------------------------------------------------------- metrics
def confusion(y, pred, k=3):
    return np.bincount(np.asarray(y) * k + np.asarray(pred), minlength=k * k).reshape(k, k)

def f1_per_class(y, pred, k=3):
    cm = confusion(y, pred, k)
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)

def macro_f1(y, pred, k=3):
    return f1_per_class(y, pred, k).mean()

def report(y, pred, lang=None, name="", show_cm=False, quiet=False):
    """Compact summary; returns the numbers as a flat dict (for the run log)."""
    y, pred = np.asarray(y), np.asarray(pred)
    pc = f1_per_class(y, pred)
    out = {"macro_f1": round(macro_f1(y, pred), 4),
           **{f"f1_{c}": round(v, 4) for c, v in enumerate(pc)}, "n": len(y)}
    if lang is not None:
        lang = np.asarray(lang)
        for lg in LANGS:
            m = lang == lg
            if m.any():
                out[f"f1_{lg}"] = round(macro_f1(y[m], pred[m]), 4)
    if not quiet:
        print(f"{name:28s} macro-F1 {out['macro_f1']:.4f} | per class "
              + " / ".join(f"{v:.3f}" for v in pc) + f" | n={len(y)}")
        if lang is not None:
            print(" " * 29 + "by language: " + " | ".join(
                f"{lg} {out[f'f1_{lg}']:.3f}" for lg in LANGS if f"f1_{lg}" in out))
        if show_cm:
            print(" " * 29 + "confusion (rows=true):", confusion(y, pred).tolist())
    return out

# ---------------------------------------------------------------- safe writing
def write_csv(df, path):
    """Write to a temporary file, then rename: a disconnect mid-write can
    never leave a half-written file behind."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return path

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)
    return path

# ---------------------------------------------------------------- prediction files
# preds/oof_{model}_f{k}.csv   out-of-fold probabilities for fold k   (p0, p1, p2, id)
# preds/test_{model}_f{k}.csv  test probabilities from the fold-k model (p0, p1, p2, id)
def pred_path(kind, model, fold):
    assert kind in ("oof", "test")
    return f"{PRED_DIR}/{kind}_{model}_f{fold}.csv"

def save_preds(kind, model, prob, ids, fold, overwrite=False):
    path = pred_path(kind, model, fold)
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True only if you mean it")
    return write_csv(pd.DataFrame(np.asarray(prob), columns=COLS).assign(id=np.asarray(ids)), path)

def fold_done(model, fold):
    return os.path.exists(pred_path("oof", model, fold)) and os.path.exists(pred_path("test", model, fold))

def list_models():
    """Model names that have at least one OOF file in preds/."""
    return sorted({re.sub(r"_f\d$", "", os.path.basename(f)[4:-4])
                   for f in glob.glob(f"{PRED_DIR}/oof_*_f[0-9].csv")})

def missing_folds(model, train_df, test_df):
    """Folds whose files are missing or do not cover exactly the right rows."""
    bad = []
    for k in range(N_FOLDS):
        if not fold_done(model, k):
            bad.append(k); continue
        oof_ids = pd.read_csv(pred_path("oof", model, k), usecols=["id"])["id"]
        test_ids = pd.read_csv(pred_path("test", model, k), usecols=["id"])["id"]
        if (set(oof_ids) != set(train_df.loc[train_df["fold"] == k, "id"]) or len(oof_ids) != (train_df["fold"] == k).sum()
                or len(test_ids) != len(test_df) or (test_ids.values != test_df["id"].values).any()):
            bad.append(k)
    return bad

def load_model_preds(model, train_df, test_df):
    """OOF probabilities in train_df row order and test probabilities (mean of
    the 5 fold models) in test_df order. Refuses if any fold is missing."""
    bad = missing_folds(model, train_df, test_df)
    if bad:
        raise SystemExit(f"REFUSED: model '{model}' is missing or has incomplete folds {bad}")
    oof = pd.concat([pd.read_csv(pred_path("oof", model, k)) for k in range(N_FOLDS)]).set_index("id")
    oof = oof.loc[train_df["id"].values, COLS].values
    test = np.mean([pd.read_csv(pred_path("test", model, k))[COLS].values for k in range(N_FOLDS)], axis=0)
    return oof, test

# ---------------------------------------------------------------- run log + training history
def log_run(**row):
    """Append one row to results/runs.csv (new columns are added automatically)."""
    row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "common_version": VERSION, **row}
    new = pd.DataFrame([row])
    if os.path.exists(RUNLOG):
        new = pd.concat([pd.read_csv(RUNLOG), new], ignore_index=True)
    write_csv(new, RUNLOG)
    return row

def history_path(model, fold):
    return f"{HISTORY_DIR}/{model}_f{fold}.csv"

def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    except ImportError:
        return "CPU"