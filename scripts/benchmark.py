"""Efficiency table: params, on-disk size, CPU/GPU real-time-factor, and accuracy.

Measures the parent and any trained students on WAXAL Twi/Ewe validation. RTF is
compute-seconds per audio-second (lower is better; <1 is realtime).

    modal run benchmark.py
"""
from __future__ import annotations
import modal

from dondo_nanos.config import LANGS, PARENT, SR, blended
from dondo_nanos.infra import IMAGE, SECRETS, VOL

app = modal.App("dondo-nano-benchmark")

# label -> checkpoint path (None = the parent from the Hub)
MODELS = {
    "parent": None,
    "nano_L12": "/data/models/nano_L12/best",
    "nano_L6": "/data/models/nano_L6/best",
    "nano_L3": "/data/models/nano_L3/best",
}
N_ACC, N_CPU, N_WARM, THREADS = 200, 40, 3, 4


@app.function(image=IMAGE, volumes={"/data": VOL}, secrets=SECRETS,
              gpu="L40S", cpu=8.0, memory=32768, timeout=3 * 60 * 60)
def bench():
    import glob, json, os, time
    import numpy as np, torch, jiwer
    from transformers import AutoProcessor, AutoModelForCTC

    from dondo_nanos.audio import norm
    from dondo_nanos.data import load_val

    torch.set_num_threads(THREADS)
    proc = AutoProcessor.from_pretrained(PARENT)
    data = {f: load_val(f, N_ACC) for f in LANGS}

    def transcribe(model, auds, dev, dtype):
        out = []
        for i in range(0, len(auds), 8):
            inp = proc(auds[i:i + 8], sampling_rate=SR, return_tensors="pt", padding=True)
            feats = inp.get("input_features", inp.get("input_values")).to(dev, dtype=dtype)
            with torch.no_grad():
                lg = model(feats).logits
            out += [norm(t) for t in proc.batch_decode(lg.argmax(-1).cpu())]
        return out

    def rtf(model, auds, dev, dtype):
        for _ in range(N_WARM):
            transcribe(model, auds[:8], dev, dtype)
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        transcribe(model, auds, dev, dtype)
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / (sum(len(a) for a in auds) / SR)

    def disk_mb(path):
        from huggingface_hub import snapshot_download
        path = path or snapshot_download(PARENT)
        return sum(os.path.getsize(f) for f in glob.glob(f"{path}/*.safetensors")) / 1e6

    report = {}
    for label, path in MODELS.items():
        src = path or PARENT
        try:
            mg = AutoModelForCTC.from_pretrained(src, torch_dtype=torch.bfloat16).to("cuda").eval()
        except Exception as e:
            print(f"{label}: skip ({type(e).__name__})"); continue
        entry = {"params_M": round(sum(p.numel() for p in mg.parameters()) / 1e6, 1),
                 "disk_MB": round(disk_mb(path), 1), "langs": {}}
        for f in LANGS:
            refs = [t for _, t in data[f]]
            hyp = [h or "na" for h in transcribe(mg, [a for a, _ in data[f]], "cuda", torch.bfloat16)]
            w, c = jiwer.wer(refs, hyp), jiwer.cer(refs, hyp)
            entry["langs"][f] = {"wer": round(w, 4), "cer": round(c, 4), "blended": round(blended(w, c), 4)}
        gpu = [a for a, _ in data["aka"]]
        entry["gpu_rtf"] = round(rtf(mg, gpu, "cuda", torch.bfloat16), 5)
        del mg; torch.cuda.empty_cache()
        mc = AutoModelForCTC.from_pretrained(src, torch_dtype=torch.float32).cpu().eval()
        entry["cpu_rtf"] = round(rtf(mc, [a for a, _ in data["aka"]][:N_CPU], "cpu", torch.float32), 4)
        del mc
        entry["mean_blended"] = round(np.mean([v["blended"] for v in entry["langs"].values()]), 4)
        report[label] = entry
        print(f"{label}: {entry['params_M']}M {entry['disk_MB']}MB "
              f"cpu_rtf={entry['cpu_rtf']} blended={entry['mean_blended']}")

    os.makedirs("/data/results", exist_ok=True)
    json.dump(report, open("/data/results/benchmark.json", "w"), indent=2)
    VOL.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main():
    print(bench.remote())
