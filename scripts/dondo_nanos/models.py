"""Building the compressed student by warm-starting from the parent.

The student is the same Wav2Vec2-BERT architecture with fewer encoder layers. It
is initialized from evenly-spaced layers of the parent, plus the parent's feature
projection, adapter, and CTC head copied verbatim. This is the DistilBERT /
DistilHuBERT style of layer initialization.
"""
from __future__ import annotations

from .config import PARENT

_LAYER_KEY = ".encoder.layers."


def pick_layers(n_teacher: int, n_student: int) -> list[int]:
    """Evenly-spaced teacher layer indices to seed the student.

    e.g. 24 -> 6 gives [0, 5, 9, 14, 18, 23].
    """
    if n_student <= 1:
        return [0]
    return [round(i * (n_teacher - 1) / (n_student - 1)) for i in range(n_student)]


def build_student(layers: int, parent_id: str = PARENT, dtype=None):
    """Return a warm-started `Wav2Vec2BertForCTC` student with `layers` blocks.

    Also returns the teacher (frozen) and the list of teacher layers used, so the
    caller can run knowledge distillation if desired.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCTC

    teacher = AutoModelForCTC.from_pretrained(
        parent_id, torch_dtype=dtype or torch.float32)
    n_teacher = teacher.config.num_hidden_layers

    sconf = AutoConfig.from_pretrained(parent_id)
    sconf.num_hidden_layers = layers
    student = AutoModelForCTC.from_config(sconf)

    sel = pick_layers(n_teacher, layers)
    tsd, ssd, new = teacher.state_dict(), student.state_dict(), {}
    copied = 0
    for k in ssd:
        if _LAYER_KEY in k:
            head, tail = k.split(_LAYER_KEY, 1)
            sidx = int(tail.split(".", 1)[0])
            suffix = tail.split(".", 1)[1]
            tkey = f"{head}{_LAYER_KEY}{sel[sidx]}.{suffix}"
            if tkey in tsd and tsd[tkey].shape == ssd[k].shape:
                new[k] = tsd[tkey].float()
                copied += 1
                continue
        if k in tsd and tsd[k].shape == ssd[k].shape:
            new[k] = tsd[k].float()
            copied += 1
        else:
            new[k] = ssd[k]
    student.load_state_dict(new, strict=False)
    student.config.ctc_zero_infinity = True
    return student, teacher, sel, copied
