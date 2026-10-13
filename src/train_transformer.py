# ============================================================
# 10.1 src/train_transformer.py: k-fold fine-tuning for every neural model
# ============================================================
"""
train_transformer.py - k-fold fine-tuning of a pretrained model (notebook section 10)

One script for every neural model, so all of them share folds, cleaning,
output format and the run log:
  * encoders (XLM-R, BanglishBERT, ...): full fine-tuning
  * decoder LLMs (Gemma 3 1B, ...): --lora 16  -> only small LoRA adapters + the
    3-class head are trained; the LLM's own weights stay frozen

Each fold writes preds/oof_{name}_f{k}.csv and preds/test_{name}_f{k}.csv,
its training history (loss per logging step, F1 per epoch) to
results/history/{name}_f{k}.csv, and appends a row to results/runs.csv, so folds
can be split across Colab sessions (re-running the same command skips folds
that are already done).

Honest scores: a fixed number of epochs, no best-epoch picking on the
validation fold (the F1 after each epoch is only recorded).

Label options:
  default          hard labels + mild label smoothing (as in our first runs)
  --soft_alpha A   target = (1-A) * one_hot(y) + A * TF-IDF second opinion
                   (results/soft_tfidf_f{k}.csv); label smoothing is then off

Examples:
  python src/train_transformer.py --model FacebookAI/xlm-roberta-large --lr 1e-5 --folds 0,1
  python src/train_transformer.py --model csebuetnlp/banglishbert --normalize --lr 2e-5 --folds 0,1
  python src/train_transformer.py --model unsloth/gemma-3-1b-pt --lora 16 --lr 1e-4 \
        --epochs 1 --max_len 64 --precision fp32 --folds 0
  add --smoke for a quick end-to-end test (10 steps, small sample, outputs in _smoke/)
"""
VERSION = "2026-09-28.2"

import os, sys, gc, time, math, inspect, argparse
if "--smoke" in sys.argv:
    os.environ["HERWILL_SMOKE"] = "1"      # must be set before common is imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if "ipykernel" in sys.modules:
    raise SystemExit("Run as a script: !python src/train_transformer.py --model ... --folds ...")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import datasets
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer,
                          DataCollatorWithPadding, TrainerCallback, PrinterCallback, set_seed)
import common as C

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--name", default=None, help="output name; default derived from --model (+ _soft)")
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--lr", type=float, default=1e-5)             # encoders: large 1e-5, base 2e-5; LoRA: 1e-4
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--batch_size", type=int, default=16)
ap.add_argument("--grad_accum", type=int, default=2)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--label_smoothing", type=float, default=0.05)
ap.add_argument("--soft_alpha", type=float, default=0.0)
ap.add_argument("--lora", type=int, default=0, help="LoRA rank; 0 = full fine-tuning")
ap.add_argument("--precision", choices=["fp16", "fp32", "bf16"], default="fp16",
                help="we use fp32 for Gemma on a T4 (fp16 can overflow; the T4 has no bf16)")
ap.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves memory)")
ap.add_argument("--normalize", action="store_true", help="csebuetnlp normalizer (BanglaBERT/BanglishBERT)")
ap.add_argument("--seed", type=int, default=C.SEED)
ap.add_argument("--overwrite", action="store_true", help="re-train folds whose predictions exist")
ap.add_argument("--smoke", action="store_true", help="quick end-to-end test, outputs in _smoke/")
args = ap.parse_args()
datasets.disable_progress_bars()

NAME = args.name or (args.model.split("/")[-1].replace("-", "_").replace(".", "_")
                     + ("_norm" if args.normalize else "") + ("_soft" if args.soft_alpha > 0 else ""))
if args.seed != C.SEED and args.name is None:
    NAME += f"_s{args.seed}"
CKPT = "/content/ckpt" if os.path.isdir("/content") else os.path.join(C.RESULTS_DIR, "ckpt")
RUN_FOLDS = [int(f) for f in args.folds.split(",")]
gpu = C.gpu_name()
print(f"train_transformer.py VERSION {VERSION} | common.py VERSION {C.VERSION} | smoke={C.SMOKE}")
print(f"name={NAME} model={args.model} folds={RUN_FOLDS} device={gpu} precision={args.precision}")
print({k: v for k, v in vars(args).items() if k not in ("model", "folds")})
if args.precision == "fp16" and "gemma" in args.model.lower():
    print("WARNING: Gemma in fp16 can overflow to NaN; use --precision fp32 on a T4")
todo = [k for k in RUN_FOLDS if args.overwrite or not C.fold_done(NAME, k)]
if not todo:
    print(f"folds {RUN_FOLDS}: predictions exist, nothing to do (use --overwrite to retrain)")
    sys.exit(0)

# ---------------------------------------------------------------- data
train_df, test_df = C.load_data()
if args.normalize:
    try:
        from normalizer import normalize as bn_normalize  # pip install git+https://github.com/csebuetnlp/normalizer
    except ImportError:
        sys.exit("STOP: --normalize needs the csebuetnlp normalizer; rerun notebook section 0.1 (it installs it)")
    prep = lambda t: bn_normalize(C.clean(t))
else:
    prep = C.clean
train_df["text"] = train_df["text"].map(prep)
test_df["text"] = test_df["text"].map(prep)
y = train_df["y"].values

tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lens = np.array([len(tok(t)["input_ids"]) for t in train_df["text"].sample(min(2000, len(train_df)), random_state=0)])
print(f"token length p50/p95/p99: {np.percentile(lens, [50, 95, 99]).round(0).tolist()} "
      f"| truncated at {args.max_len}: {np.mean(lens > args.max_len):.1%}")
if tok.unk_token_id is not None:        # how much text the vocabulary cannot represent
    unk = {}
    for lg in ["english", "bangla", "banglish"]:
        smp = train_df.loc[train_df["lang"] == lg, "text"]
        ids = [i for t in smp.sample(min(500, len(smp)), random_state=0) for i in tok(t, add_special_tokens=False)["input_ids"]]
        if ids:
            unk[lg] = f"{np.mean(np.array(ids) == tok.unk_token_id):.2%}"
    print("unknown-token rate:", unk)

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=args.max_len)

test_ds = Dataset.from_pandas(test_df[["text"]], preserve_index=False).map(
    tokenize, batched=True, remove_columns=["text"])

# class weights: square root of inverse class frequency (the rare neutral class counts more)
freq = np.bincount(y, minlength=3) / len(y)
class_w = torch.tensor(np.sqrt(1 / freq) / np.sqrt(1 / freq).mean(), dtype=torch.float)
print("class weights:", [round(float(w), 3) for w in class_w])

# ---------------------------------------------------------------- loss
class LabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        soft = inputs.pop("soft", None)
        logits = model(**inputs).logits.float()
        w = class_w.to(logits.device)
        if soft is None:     # hard labels (validation, or --soft_alpha 0)
            loss = F.cross_entropy(logits, labels, weight=w, label_smoothing=args.label_smoothing)
        else:                # soft targets, class-weighted like F.cross_entropy(weight=...)
            soft = soft.float()
            per_row = -(soft * w * F.log_softmax(logits, -1)).sum(-1)
            loss = per_row.sum() / (soft * w).sum()
        return (loss, {"logits": logits}) if return_outputs else loss

class CompactLog(TrainerCallback):
    """One line per epoch (validation F1) + a heartbeat every 500 steps, instead of
    one log dict per 100 steps. The full history is saved to results/history/."""
    def on_log(self, targs, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "eval_macro_f1" in logs:
            print(f"   epoch {logs.get('epoch', 0):.1f}: validation macro-F1 {logs['eval_macro_f1']:.4f} "
                  f"| {(time.time() - t0) / 60:.1f} min", flush=True)
        elif "loss" in logs and state.global_step % 500 == 0:
            print(f"   step {state.global_step}/{state.max_steps}: train loss per batch "
                  f"{logs['loss'] / args.grad_accum:.3f}", flush=True)

def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    return {"macro_f1": C.macro_f1(p.label_ids, logits.argmax(-1))}

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def build_model():
    dtype = {"fp16": torch.float16 if args.lora else torch.float32,   # LoRA: frozen base in fp16
             "fp32": torch.float32, "bf16": torch.bfloat16}[args.precision]
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=3, dtype=dtype)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    if args.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(task_type="SEQ_CLS", r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        for p in model.parameters():          # trainable weights in fp32 (needed for fp16 training)
            if p.requires_grad:
                p.data = p.data.float()
        if args.grad_ckpt:
            model.enable_input_require_grads()
        model.print_trainable_parameters()
    return model

# ---------------------------------------------------------------- CV loop
for fold in RUN_FOLDS:
    if fold not in todo:
        print(f"\nfold {fold}: predictions exist, skipping (use --overwrite to retrain)")
        continue
    t0 = time.time()
    set_seed(args.seed + fold)
    tr_idx = np.where(train_df["fold"] != fold)[0]
    va_idx = np.where(train_df["fold"] == fold)[0]
    print(f"\n========== {NAME} fold {fold}: train {len(tr_idx)}, valid {len(va_idx)} ==========")

    tr = pd.DataFrame({"text": train_df["text"].values[tr_idx], "labels": y[tr_idx]})
    if args.soft_alpha > 0:
        teacher = pd.read_csv(f"{C.RESULTS_DIR}/soft_tfidf_f{fold}.csv").set_index("id")
        tp = teacher.reindex(train_df["id"].values[tr_idx])[C.COLS].values
        assert not np.isnan(tp).any(), "soft label file does not cover this fold's training rows"
        tr["soft"] = list(((1 - args.soft_alpha) * np.eye(3)[y[tr_idx]] + args.soft_alpha * tp).astype(np.float32))
    va = pd.DataFrame({"text": train_df["text"].values[va_idx], "labels": y[va_idx]})
    tr_ds = Dataset.from_pandas(tr, preserve_index=False).map(tokenize, batched=True, remove_columns=["text"])
    va_ds = Dataset.from_pandas(va, preserve_index=False).map(tokenize, batched=True, remove_columns=["text"])

    model = build_model()
    steps_per_epoch = math.ceil(len(tr) / (args.batch_size * args.grad_accum))
    ta = dict(
        output_dir=f"{CKPT}/{NAME}_f{fold}",
        learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 4,
        gradient_accumulation_steps=args.grad_accum, weight_decay=0.01,
        warmup_steps=int(0.1 * steps_per_epoch * args.epochs), lr_scheduler_type="linear",
        eval_strategy="epoch", save_strategy="no",          # fixed epochs: no best-epoch picking
        fp16=args.precision == "fp16" and torch.cuda.is_available(),
        bf16=args.precision == "bf16",
        use_cpu=not torch.cuda.is_available(),                # no CUDA -> plain CPU (never Apple MPS)
        gradient_checkpointing=args.grad_ckpt,
        remove_unused_columns=False, logging_steps=2 if C.SMOKE else 100, report_to="none",
        disable_tqdm=True, seed=args.seed + fold, max_steps=10 if C.SMOKE else -1,
    )
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    if "eval_strategy" not in supported:
        ta["evaluation_strategy"] = ta.pop("eval_strategy")
    skipped = [k for k in ta if k not in supported]
    if skipped:
        print("note: this transformers version lacks", skipped)
    trainer = LabelTrainer(
        model=model, args=TrainingArguments(**{k: v for k, v in ta.items() if k in supported}),
        train_dataset=tr_ds, eval_dataset=va_ds,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=compute_metrics,
    )
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(CompactLog())
    print(f"   training {args.epochs} epochs x {steps_per_epoch} steps (logged loss = sum over "
          f"{args.grad_accum} accumulation steps; divided by {args.grad_accum} in these lines)", flush=True)
    trainer.train()

    # training history: one row per logging step (train loss) or epoch (validation F1)
    hist = pd.DataFrame([h for h in trainer.state.log_history if "loss" in h or "eval_macro_f1" in h])
    C.write_csv(hist.assign(model=NAME, fold=fold, grad_accum=args.grad_accum), C.history_path(NAME, fold))
    epoch_f1 = [round(h["eval_macro_f1"], 4) for h in trainer.state.log_history if "eval_macro_f1" in h]
    va_logits = trainer.predict(va_ds).predictions
    te_logits = trainer.predict(test_ds).predictions
    va_logits = va_logits[0] if isinstance(va_logits, tuple) else va_logits
    te_logits = te_logits[0] if isinstance(te_logits, tuple) else te_logits
    va_prob, te_prob = softmax(va_logits.astype(np.float64)), softmax(te_logits.astype(np.float64))
    assert np.isfinite(va_prob).all(), "NaN in predictions (fp16 overflow?) - rerun with --precision fp32"

    res = C.report(y[va_idx], va_prob.argmax(1), train_df["lang"].values[va_idx], name=f"{NAME} fold {fold}")
    print(f"   macro-F1 after each epoch (recorded only, never used to pick an epoch): {epoch_f1}")
    C.save_preds("oof", NAME, va_prob, train_df["id"].values[va_idx], fold=fold, overwrite=args.overwrite)
    C.save_preds("test", NAME, te_prob, test_df["id"].values, fold=fold, overwrite=args.overwrite)
    C.log_run(step="10_transformer", model=NAME, script_version=VERSION, base_model=args.model, fold=fold,
              lr=args.lr, epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
              effective_batch=args.batch_size * args.grad_accum, max_len=args.max_len, warmup_steps=ta["warmup_steps"],
              weight_decay=ta["weight_decay"], n_params=n_params, n_trainable=n_train,
              label_smoothing=args.label_smoothing if args.soft_alpha == 0 else 0, soft_alpha=args.soft_alpha,
              lora=args.lora, precision=args.precision, normalize=args.normalize, seed=args.seed,
              epoch_f1=str(epoch_f1), gpu=gpu, runtime_min=round((time.time() - t0) / 60, 1),
              smoke=C.SMOKE, **res)
    print(f"   saved, {(time.time() - t0) / 60:.1f} min")

    del model, trainer; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.system(f"rm -rf {CKPT}/{NAME}_f{fold}")
print("done.")