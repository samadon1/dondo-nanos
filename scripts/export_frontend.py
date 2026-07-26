"""Bake the Kaldi-fbank frontend into a tiny standalone ONNX (raw audio -> features).

The on-device app then feeds raw 16 kHz samples to `frontend.onnx`, whose output goes
straight into the int8 transformer model. No feature-extraction code in the app.

The frontend reproduces the SeamlessM4T fbank (int16 scaling, DC removal, preemphasis
0.97, povey window, 512-FFT via a fixed DFT matmul, 257->80 mel, log, per-bin CMVN,
stride-2 stacking). The FFT is a constant matmul so the whole thing exports cleanly.

Validates against the real processor on GPU (where the model produces real transcripts),
then exports frontend.onnx to the volume.

    modal run export_frontend.py
"""
from __future__ import annotations
import modal

from dondo_nanos.config import LANGS, PARENT, SR
from dondo_nanos.infra import SECRETS, VOL, with_onnx

app = modal.App("dondo-nano-export-frontend")
MODEL = "samwell/dondo-nano-twi-ewe"


@app.function(image=with_onnx(), volumes={"/data": VOL}, secrets=SECRETS,
              gpu="L40S", timeout=60 * 60)
def export():
    import io, os
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, soundfile as sf
    import onnxruntime as ort
    from transformers import AutoProcessor, AutoModelForCTC
    from dondo_nanos.audio import norm
    from dondo_nanos.data import load_val

    proc = AutoProcessor.from_pretrained(MODEL)
    fe = proc.feature_extractor

    class FbankFront(nn.Module):
        def __init__(self, fe, fft=512, flen=400, hop=160, preemph=0.97,
                     floor=1.192092955078125e-07):
            super().__init__()
            n = torch.arange(fft).float(); k = torch.arange(fft // 2 + 1).float()
            ang = 2 * np.pi * torch.outer(n, k) / fft
            self.register_buffer("cos", torch.cos(ang)); self.register_buffer("sin", torch.sin(ang))
            self.register_buffer("window", torch.tensor(np.asarray(fe.window), dtype=torch.float32))
            self.register_buffer("mel", torch.tensor(np.asarray(fe.mel_filters), dtype=torch.float32))
            self.register_buffer("kern", torch.eye(flen).view(flen, 1, flen))
            self.fft, self.flen, self.hop = fft, flen, hop
            self.preemph, self.floor, self.stride = preemph, floor, fe.stride

        def forward(self, wav):
            x = wav * (2 ** 15)
            f = F.conv1d(x.view(1, 1, -1), self.kern, stride=self.hop)[0].transpose(0, 1)
            f = f - f.mean(-1, keepdim=True)
            pre = f.clone()
            pre[:, 1:] = f[:, 1:] - self.preemph * f[:, :-1]; pre[:, 0] = f[:, 0] * (1 - self.preemph)
            w = torch.cat([pre * self.window, torch.zeros(f.shape[0], self.fft - self.flen)], 1)
            power = (w @ self.cos) ** 2 + (w @ self.sin) ** 2
            mel = torch.log(torch.clamp(power @ self.mel, min=self.floor))
            mel = (mel - mel.mean(0, keepdim=True)) / torch.sqrt(mel.var(0, unbiased=True, keepdim=True) + 1e-7)
            T2 = mel.shape[0] - (mel.shape[0] % self.stride)
            return mel[:T2].reshape(T2 // self.stride, 80 * self.stride).unsqueeze(0)

    front = FbankFront(fe).eval()
    model = AutoModelForCTC.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("cuda").eval()

    def dec_gpu(feats):  # feats: (1,T,160) float
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(input_features=feats.to("cuda", dtype=torch.bfloat16)).logits
        return norm(proc.batch_decode(lg.argmax(-1).cpu())[0])

    print("== validating frontend vs processor (real transcripts) ==")
    matches = 0; total = 0
    for folder in LANGS:
        for y, ref in sorted(load_val(folder, 60, min_s=1.0, max_s=20.0),
                             key=lambda r: len(r[0]))[:4]:
            proc_feat = proc(y, sampling_rate=SR, return_tensors="pt").input_features.float()
            fb_feat = front(torch.tensor(y))
            a, b = dec_gpu(proc_feat), dec_gpu(fb_feat)
            total += 1; matches += (a == b)
            print(f"[{folder}] proc : {a[:70]}")
            print(f"      fbank: {b[:70]}  {'OK' if a==b else 'DIFF'}")
    print(f"\nfrontend == processor on {matches}/{total} clips")

    os.makedirs("/data/models", exist_ok=True)
    out = "/data/models/frontend.onnx"
    dummy = torch.randn(16000 * 4)
    torch.onnx.export(front, (dummy,), out, input_names=["waveform"],
                      output_names=["input_features"], opset_version=17,
                      dynamo=False,  # legacy tracer; newer torch defaults to dynamo which fails here
                      dynamic_axes={"waveform": {0: "n"}, "input_features": {1: "t"}})
    # sanity: ORT frontend features close to torch
    y = load_val("aka", 30, min_s=1.0, max_s=20.0)
    y = sorted(y, key=lambda r: len(r[0]))[0][0]
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    fo = sess.run(None, {"waveform": y.astype(np.float32)})[0]
    ft = front(torch.tensor(y)).numpy()
    print(f"frontend.onnx size: {os.path.getsize(out)/1e6:.2f} MB | "
          f"ORT-vs-torch feat maxdiff: {np.abs(fo - ft).max():.2e}")
    VOL.commit()
    return {"frontend_processor_match": f"{matches}/{total}", "onnx_MB": round(os.path.getsize(out)/1e6, 2)}


@app.local_entrypoint()
def main():
    print(export.remote())
