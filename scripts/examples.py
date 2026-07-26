"""Qualitative transcripts: reference vs a trained nano vs the zero-shot parent.

    modal run examples.py --ckpt /data/models/nano_L12/best
"""
from __future__ import annotations
import modal

from dondo_nanos.config import LANGS, PARENT, SR
from dondo_nanos.infra import IMAGE, SECRETS, VOL

app = modal.App("dondo-nano-examples")
N = 6


@app.function(image=IMAGE, volumes={"/data": VOL}, secrets=SECRETS,
              gpu="L40S", timeout=60 * 60)
def run(ckpt: str = "/data/models/nano_L12/best"):
    import json
    import torch
    from transformers import AutoProcessor, AutoModelForCTC

    from dondo_nanos.audio import norm
    from dondo_nanos.data import load_val

    dev = "cuda"
    proc = AutoProcessor.from_pretrained(PARENT)
    parent = AutoModelForCTC.from_pretrained(PARENT, torch_dtype=torch.bfloat16).to(dev).eval()
    nano = AutoModelForCTC.from_pretrained(ckpt, torch_dtype=torch.bfloat16).to(dev).eval()

    def tx(model, y):
        inp = proc(y, sampling_rate=SR, return_tensors="pt")
        with torch.no_grad():
            lg = model(inp.input_features.to(dev, dtype=torch.bfloat16)).logits
        return norm(proc.batch_decode(lg.argmax(-1).cpu())[0]) or "(blank)"

    out = {}
    for folder, name in LANGS.items():
        rows = sorted(load_val(folder, 250, min_s=1.0, max_s=20.0), key=lambda r: len(r[0]))[:N]
        out[name] = [{"seconds": round(len(y) / SR, 1), "reference": t,
                      "nano": tx(nano, y), "parent_zeroshot": tx(parent, y)} for y, t in rows]
        for e in out[name]:
            print(f"\n[{name} {e['seconds']}s]\n REF : {e['reference']}\n NANO: {e['nano']}\n PAR : {e['parent_zeroshot']}")

    import os
    os.makedirs("/data/results", exist_ok=True)
    json.dump(out, open("/data/results/examples.json", "w"), indent=2, ensure_ascii=False)
    VOL.commit()
    return out


@app.local_entrypoint()
def main(ckpt: str = "/data/models/nano_L12/best"):
    print(run.remote(ckpt=ckpt))
