"""Compress DONDO by depth reduction: warm-start a shallower student and fine-tune.

Loss is CTC on the labels, optionally plus a knowledge-distillation term matching
the student's logits to the frozen parent's (`--kd-alpha 0` disables it). Trains on
WAXAL Twi + Ewe, evaluates on the held-out validation split, and saves the student.

    modal run distill.py --layers 12                 # CTC + KD
    modal run distill.py --layers 12 --kd-alpha 0    # CTC only (ablation)
    modal run distill.py --layers 12 --smoke         # tiny sanity run
"""
from __future__ import annotations
import modal

from dondo_nanos.config import LANGS, PARENT, SR, blended
from dondo_nanos.infra import IMAGE, SECRETS, VOL

app = modal.App("dondo-nano-distill")

LAYERS, EPOCHS, LR, BS = 12, 6, 1e-4, 8
KD_ALPHA, KD_T = 1.0, 2.0
N_TRAIN, N_VAL, SEED = 6000, 200, 42


@app.function(image=IMAGE, volumes={"/data": VOL}, secrets=SECRETS,
              gpu="H100", timeout=6 * 60 * 60)
def distill(layers: int = LAYERS, kd_alpha: float = KD_ALPHA,
            epochs: int = EPOCHS, smoke: bool = False):
    import json, os, random, time
    import numpy as np, torch, jiwer
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoProcessor

    from dondo_nanos.audio import norm
    from dondo_nanos.data import load_split
    from dondo_nanos.models import build_student

    torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    dev = "cuda"
    n_train = 300 if smoke else N_TRAIN
    n_val = 40 if smoke else N_VAL
    epochs = 1 if smoke else epochs

    proc = AutoProcessor.from_pretrained(PARENT)
    tok = proc.tokenizer

    train_rows, val_rows = [], {}
    per_lang = max(1, n_train // len(LANGS))
    for lg in LANGS:
        train_rows += load_split(lg, "train", per_lang)
        val_rows[lg] = load_split(lg, "validation", n_val)
    random.shuffle(train_rows)
    print(f"train={len(train_rows)} val={sum(len(v) for v in val_rows.values())} "
          f"layers={layers} kd_alpha={kd_alpha} epochs={epochs}")

    student, teacher, sel, copied = build_student(layers, dtype=torch.bfloat16)
    print(f"warm-start: copied {copied} tensors from teacher layers {sel}")
    n_params = sum(p.numel() for p in student.parameters())
    print(f"student params: {n_params / 1e6:.1f}M")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher = teacher.to(dev).eval()
    student = student.to(dev).train()

    class DS(Dataset):
        def __init__(s, rows): s.rows = rows
        def __len__(s): return len(s.rows)
        def __getitem__(s, i): return s.rows[i]

    def collate(batch):
        feats = proc([b[0] for b in batch], sampling_rate=SR,
                     return_tensors="pt", padding=True)
        labs = tok([b[1] for b in batch], padding=True, return_tensors="pt")
        labels = labs["input_ids"].masked_fill(labs["attention_mask"].ne(1), -100)
        return feats["input_features"], feats.get("attention_mask"), labels

    dl = DataLoader(DS(train_rows), batch_size=(4 if smoke else BS), shuffle=True,
                    collate_fn=collate, num_workers=8, drop_last=True)
    opt = torch.optim.AdamW(student.parameters(), lr=LR)
    steps = len(dl) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.1)

    def feat_lengths(attn):
        return None if attn is None else teacher._get_feat_extract_output_lengths(attn.sum(-1)).long()

    def evaluate():
        student.eval()
        out = {}
        for lg, rows in val_rows.items():
            refs = [t for _, t in rows]; hyp = []
            for i in range(0, len(rows), 8):
                fb = proc([a for a, _ in rows[i:i + 8]], sampling_rate=SR,
                          return_tensors="pt", padding=True)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    lg_ = student(fb["input_features"].to(dev)).logits
                hyp += [norm(t) for t in proc.batch_decode(lg_.argmax(-1).cpu())]
            hyp = [h or "na" for h in hyp]
            w, c = jiwer.wer(refs, hyp), jiwer.cer(refs, hyp)
            out[lg] = {"wer": round(w, 4), "cer": round(c, 4), "blended": round(blended(w, c), 4)}
        student.train()
        return out

    for ep in range(epochs):
        t0 = time.perf_counter()
        for feats, attn, labels in dl:
            feats, labels = feats.to(dev), labels.to(dev)
            attn = attn.to(dev) if attn is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = student(feats, attention_mask=attn, labels=labels)
                s_logits, ctc = out.logits.float(), out.loss.float()
            if kd_alpha > 0:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_logits = teacher(feats, attention_mask=attn).logits.float()
                kd = F.kl_div(F.log_softmax(s_logits / KD_T, -1),
                              F.softmax(t_logits / KD_T, -1),
                              reduction="none").sum(-1) * (KD_T * KD_T)
                fl = feat_lengths(attn)
                if fl is not None:
                    m = torch.arange(s_logits.shape[1], device=dev)[None] < fl[:, None]
                    kd = (kd * m).sum() / m.sum().clamp(min=1)
                else:
                    kd = kd.mean()
            else:
                kd = torch.zeros((), device=dev)
            loss = ctc + kd_alpha * kd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step()
        ev = evaluate()
        mean = float(np.mean([v["blended"] for v in ev.values()]))
        print(f"[epoch {ep}] {time.perf_counter() - t0:.0f}s val={ev} mean_blended={mean:.4f}")

    tag = f"nano_L{layers}" + ("" if kd_alpha > 0 else "_ctc")
    out_dir = f"/data/models/{tag}/best"
    os.makedirs(out_dir, exist_ok=True)
    student.save_pretrained(out_dir); proc.save_pretrained(out_dir)
    final = evaluate()
    report = {"tag": tag, "layers": layers, "kd_alpha": kd_alpha,
              "params_M": round(n_params / 1e6, 1),
              "val": final,
              "mean_blended": round(float(np.mean([v["blended"] for v in final.values()])), 4)}
    os.makedirs("/data/results", exist_ok=True)
    json.dump(report, open(f"/data/results/{tag}.json", "w"), indent=2)
    VOL.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(layers: int = LAYERS, kd_alpha: float = KD_ALPHA,
         epochs: int = EPOCHS, smoke: bool = False):
    print(distill.remote(layers=layers, kd_alpha=kd_alpha, epochs=epochs, smoke=smoke))
