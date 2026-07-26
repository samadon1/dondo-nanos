"""Export a trained student to ONNX and apply int8 dynamic quantization.

Torch's own dynamic quant only touches nn.Linear, which is a no-op on a Conformer.
ONNX Runtime also quantizes the attention/FFN matmuls, so it actually shrinks the
model. Reports on-disk size, CPU real-time-factor, and accuracy for fp32 vs int8.

    modal run quantize_onnx.py --ckpt /data/models/nano_L12/best
"""
from __future__ import annotations
import modal

from dondo_nanos.config import LANGS, PARENT, SR, blended
from dondo_nanos.infra import SECRETS, VOL, with_onnx

app = modal.App("dondo-nano-quantize-onnx")

N_ACC, N_CPU, N_WARM, THREADS = 200, 40, 3, 4


@app.function(image=with_onnx(), volumes={"/data": VOL}, secrets=SECRETS,
              gpu="L40S", cpu=8.0, memory=32768, timeout=3 * 60 * 60)
def quantize(ckpt: str = "/data/models/nano_L12/best"):
    import glob, json, os, time
    import numpy as np, torch, jiwer, torch.nn as nn
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from transformers import AutoProcessor, AutoModelForCTC

    from dondo_nanos.audio import norm
    from dondo_nanos.data import load_val

    os.environ["OMP_NUM_THREADS"] = str(THREADS)
    torch.set_num_threads(THREADS)
    proc = AutoProcessor.from_pretrained(ckpt)
    data = {f: load_val(f, N_ACC) for f in LANGS}

    out_dir = ckpt.rstrip("/") + "_onnx"; os.makedirs(out_dir, exist_ok=True)
    fp32 = f"{out_dir}/model.onnx"; int8 = f"{out_dir}/model_int8.onnx"

    class Wrap(nn.Module):
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, input_features): return s.m(input_features=input_features).logits

    tmodel = AutoModelForCTC.from_pretrained(ckpt, torch_dtype=torch.float32).cpu().eval()
    fp32_disk = sum(os.path.getsize(f) for f in glob.glob(f"{ckpt}/*.safetensors")) / 1e6
    feat_dim = proc(data["aka"][0][0], sampling_rate=SR, return_tensors="pt").input_features.shape[-1]
    torch.onnx.export(Wrap(tmodel).eval(), (torch.randn(1, 200, feat_dim),), fp32,
                      input_names=["input_features"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"input_features": {0: "b", 1: "t"}, "logits": {0: "b", 1: "t"}})
    del tmodel
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)

    def acc_rtf(path):
        so = ort.SessionOptions(); so.intra_op_num_threads = THREADS
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        res = {}
        for f in LANGS:
            refs = [t for _, t in data[f]]; hyp = []
            for y, _ in data[f]:
                feats = proc(y, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)
                lg = sess.run(None, {"input_features": feats})[0]
                hyp.append(norm(proc.batch_decode(torch.tensor(lg).argmax(-1))[0]) or "na")
            w, c = jiwer.wer(refs, hyp), jiwer.cer(refs, hyp)
            res[f] = blended(w, c)
        cpu = [a for a, _ in data["aka"]][:N_CPU]
        for y in cpu[:N_WARM]:
            sess.run(None, {"input_features": proc(y, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)})
        t0 = time.perf_counter()
        for y in cpu:
            sess.run(None, {"input_features": proc(y, sampling_rate=SR, return_tensors="np").input_features.astype(np.float32)})
        return {k: round(v, 4) for k, v in res.items()}, round((time.perf_counter() - t0) / (sum(len(a) for a in cpu) / SR), 4)

    af, rf = acc_rtf(fp32)
    aq, rq = acc_rtf(int8)
    report = {
        "torch_fp32_disk_MB": round(fp32_disk, 1),
        "onnx_int8_disk_MB": round(os.path.getsize(int8) / 1e6, 1),
        "shrink_vs_fp32": round(fp32_disk / (os.path.getsize(int8) / 1e6), 2),
        "onnx_fp32": {"acc": af, "cpu_rtf": rf, "mean_blended": round(np.mean(list(af.values())), 4)},
        "onnx_int8": {"acc": aq, "cpu_rtf": rq, "mean_blended": round(np.mean(list(aq.values())), 4)},
    }
    os.makedirs("/data/results", exist_ok=True)
    json.dump(report, open("/data/results/quantize_onnx.json", "w"), indent=2)
    VOL.commit()
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main(ckpt: str = "/data/models/nano_L12/best"):
    print(quantize.remote(ckpt=ckpt))
