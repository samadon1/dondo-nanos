# DONDO-nanos

Post-training compression of [Khaya AI's DONDO](https://huggingface.co/KhayaAI/w2v-bert-ada_ewe_fat_fra_gaa_nzi_twi_en)
speech model for on-device ASR in Twi and Ewe.

DONDO is the best open speech model I've found for these languages, but the multilingual
version is a 605.8M-parameter Wav2Vec2-BERT Conformer, about 1.2 GB in half precision. That is
too big for a mid-range phone, which is where it would get used. This repo compresses it and
measures every step.

- **Blog:** https://samadon1.github.io/dondo-nanos/
- **Models:** https://huggingface.co/samwell/dondo-nano-twi-ewe

## Results

Blended error = 0.5·WER + 0.5·CER on WAXAL validation (Twi/Ewe mean), greedy CTC decode,
normalized text. All rows are trained on the same WAXAL data (CTC, no distillation).

| Model | Layers | Params | Disk (fp32) | CPU RTF | Blended |
|---|---|---|---|---|---|
| Fine-tuned parent | 24 | 605.8M | 2423 MB | 0.252 | 0.216 |
| **Nano-L12** | 12 | 315.6M | 1262 MB | 0.132 | **0.261** |
| Nano-L6 | 6 | 170.5M | 682 MB | 0.069 | 0.387 |
| Nano-L3 | 3 | 98.0M | 392 MB | 0.036 | ~0.57 |

Halving the model costs 0.045 blended error. int8 (ONNX Runtime) shrinks Nano-L12 to 319 MB
(~4×) for almost no accuracy change. RTF is compute-seconds per audio-second; every model is
already faster than real time on CPU, so the point of compression here is size, not speed.

## The recipe

1. Build a shallower Wav2Vec2-BERT student and **warm-start** it from evenly-spaced layers of
   the parent (24 → 6 takes layers 0, 5, 9, 14, 18, 23), copying the feature projection, adapter,
   and CTC head verbatim.
2. Fine-tune on WAXAL Twi + Ewe with a CTC loss.
3. Quantize to int8 with ONNX Runtime for deployment.

Knowledge distillation from the parent was tested and dropped: the parent is out-of-domain on
WAXAL, so its soft targets drag the student below plain fine-tuning. This is the same
layer-initialization idea as DistilBERT and DistilHuBERT; the contribution here is the applied
result for Ghanaian languages and the documented failure modes (naive int8 is a no-op on a
Conformer; an out-of-domain teacher hurts).

## Layout

```
scripts/          Python jobs (run on Modal), sharing the dondo_nanos package
  dondo_nanos/    config, audio, data loaders, warm-start builder, Modal infra
  distill.py      warm-start + fine-tune (+ optional KD); --kd-alpha 0 for the ablation
  benchmark.py    params / size / RTF / accuracy table
  quantize_onnx.py  ONNX export + int8 (features -> logits)
  export_frontend.py  bake the Kaldi-fbank into a tiny frontend.onnx (raw audio -> features)
  eval_zeroshot.py  parent vs MMS-1b, zero-shot references
  examples.py     qualitative transcripts
  push_to_hf.py   upload students + ONNX + model card
ios-app/          example iPhone app (ONNX Runtime, on-device)
```

## Reproduce

The jobs run on [Modal](https://modal.com) (cloud GPUs). Set up once:

```bash
pip install -r requirements.txt
modal token new
modal secret create hf-token HF_TOKEN=hf_xxx      # your HuggingFace token
cd scripts
```

Then:

```bash
modal run distill.py --layers 12 --smoke          # sanity check
modal run distill.py --layers 12                   # train Nano-L12 (CTC + KD)
modal run distill.py --layers 12 --kd-alpha 0      # ablation: CTC only
modal run benchmark.py                             # efficiency table
modal run quantize_onnx.py --ckpt /data/models/nano_L12/best
modal run eval_zeroshot.py                         # parent vs MMS references
modal run export_frontend.py                       # raw-audio -> features frontend.onnx
modal run push_to_hf.py --repo <you>/dondo-nano-twi-ewe
```

## On-device

The app feeds raw 16 kHz samples to `onnx/frontend.onnx` (the Kaldi-fbank baked into a ~1.8 MB
graph, its FFT written as a constant matmul so it exports cleanly), whose features go straight
into `onnx/model_int8.onnx`. No DSP in the app. The frontend was checked to match the reference
processor to ~3e-4, and the int8 transformer measures 0.269 blended via ONNX Runtime. See
`ios-app/` for the SwiftUI example.

## Attribution

- Base model: Khaya AI DONDO (Apache-2.0)
- Data: Google WAXAL / WaxalNLP (CC-BY-SA-4.0 / CC-BY-4.0)

Code is MIT (see `LICENSE`); released model weights are CC-BY-SA-4.0.
