"""Zero-shot reference points: the parent DONDO and Meta MMS-1b on WAXAL Twi/Ewe.

Neither model is trained on WAXAL here, so this is an out-of-the-box comparison that
gives context for the (fine-tuned) nano numbers. Both are plain CTC greedy decode.

    modal run eval_zeroshot.py
"""
from __future__ import annotations
import modal

from dondo_nanos.config import LANGS, PARENT, SR, blended
from dondo_nanos.infra import IMAGE, SECRETS, VOL

app = modal.App("dondo-nano-eval-zeroshot")

# MMS-1b adapter codes for the WAXAL folders
MMS_CODE = {"aka": "aka", "ewe": "ewe"}
N_EVAL = 200


@app.function(image=IMAGE, volumes={"/data": VOL}, secrets=SECRETS,
              gpu="L40S", timeout=3 * 60 * 60)
def evaluate():
    import json
    import torch, jiwer
    from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2ForCTC

    from dondo_nanos.audio import norm
    from dondo_nanos.data import load_val

    dev = "cuda"
    dproc = AutoProcessor.from_pretrained(PARENT)
    dmodel = AutoModelForCTC.from_pretrained(PARENT, torch_dtype=torch.bfloat16).to(dev).eval()

    def ctc_decode(model, proc, auds, input_key):
        out = []
        for i in range(0, len(auds), 8):
            inp = proc(auds[i:i + 8], sampling_rate=SR, return_tensors="pt", padding=True)
            feats = inp[input_key].to(dev, dtype=torch.bfloat16)
            kw = {"attention_mask": inp.attention_mask.to(dev)} if "attention_mask" in inp else {}
            with torch.no_grad():
                lg = model(feats, **kw).logits
            out += [norm(t) for t in proc.batch_decode(lg.argmax(-1).cpu())]
        return [h or "na" for h in out]

    results = {}
    for folder, name in LANGS.items():
        rows = load_val(folder, N_EVAL)
        auds, refs = [a for a, _ in rows], [t for _, t in rows]
        dhyp = ctc_decode(dmodel, dproc, auds, "input_features")
        dw, dc = jiwer.wer(refs, dhyp), jiwer.cer(refs, dhyp)
        entry = {"n": len(auds),
                 "dondo": {"wer": round(dw, 4), "cer": round(dc, 4), "blended": round(blended(dw, dc), 4)}}
        try:
            mproc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
            mmodel = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", torch_dtype=torch.bfloat16)
            mproc.tokenizer.set_target_lang(MMS_CODE[folder]); mmodel.load_adapter(MMS_CODE[folder])
            mmodel.to(dev, dtype=torch.bfloat16).eval()
            mhyp = ctc_decode(mmodel, mproc, auds, "input_values")
            mw, mc = jiwer.wer(refs, mhyp), jiwer.cer(refs, mhyp)
            entry["mms"] = {"wer": round(mw, 4), "cer": round(mc, 4), "blended": round(blended(mw, mc), 4)}
            del mmodel; torch.cuda.empty_cache()
        except Exception as e:
            entry["mms"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        results[name] = entry
        print(name, json.dumps(entry))

    import os
    os.makedirs("/data/results", exist_ok=True)
    json.dump(results, open("/data/results/eval_zeroshot.json", "w"), indent=2)
    VOL.commit()
    print(json.dumps(results, indent=2))
    return results


@app.local_entrypoint()
def main():
    print(evaluate.remote())
