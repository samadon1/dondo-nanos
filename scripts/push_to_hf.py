"""Push trained students + int8 ONNX to a HuggingFace repo, with a model card.

Layout in the repo: L12 at the root (default `from_pretrained` load), the smaller
variants under nano-l6/ and nano-l3/, and the int8 ONNX under onnx/.

    modal run push_to_hf.py --repo <user>/dondo-nano-twi-ewe
"""
from __future__ import annotations
import modal

from dondo_nanos.infra import IMAGE, SECRETS, VOL

app = modal.App("dondo-nano-push")

# (local path on the volume, path within the repo). Folders use upload_folder;
# single files (*.onnx) use upload_file.
UPLOADS = [
    ("/data/models/nano_L12/best", "."),
    ("/data/models/nano_L6/best", "nano-l6"),
    ("/data/models/nano_L3/best", "nano-l3"),
    ("/data/models/nano_L12_onnx", "onnx"),
    ("/data/models/frontend.onnx", "onnx/frontend.onnx"),   # raw-audio -> features
]


@app.function(image=IMAGE, volumes={"/data": VOL}, secrets=SECRETS, timeout=60 * 60)
def push(repo: str, private: bool = False, card_path: str = "/data/MODEL_CARD.md"):
    import os
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    assert token, "set a Modal secret exporting HF_TOKEN (see infra.py)"
    api = HfApi(token=token)
    create_repo(repo, private=private, exist_ok=True, repo_type="model", token=token)

    if os.path.exists(card_path):
        api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                        repo_id=repo, token=token)
    for src, dest in UPLOADS:
        if os.path.isdir(src):
            api.upload_folder(folder_path=src, path_in_repo=dest, repo_id=repo, token=token)
            print("uploaded", src, "->", dest or "(root)")
        elif os.path.isfile(src):
            api.upload_file(path_or_fileobj=src, path_in_repo=dest, repo_id=repo, token=token)
            print("uploaded", src, "->", dest)
    url = f"https://huggingface.co/{repo}"
    print("pushed ->", url)
    return url


@app.local_entrypoint()
def main(repo: str, private: bool = False):
    print(push.remote(repo=repo, private=private))
