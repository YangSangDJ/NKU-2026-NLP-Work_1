"""下载 BERT 模型必需文件到 code/artifacts/bert-base-uncased/（绕开 DNS 问题）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download

BASE = "https://hf-mirror.com/google-bert/bert-base-uncased/resolve/main/"
OUT = Path(__file__).resolve().parent / "artifacts" / "bert-base-uncased"
FILES = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
]
for f in FILES:
    dest = OUT / f
    print(f"--- {f} ---", flush=True)
    ok = download.download(BASE + f, dest, max_attempts=25)
    if not ok:
        print(f"FAILED: {f}", flush=True)
        sys.exit(1)
print("ALL DONE", flush=True)
