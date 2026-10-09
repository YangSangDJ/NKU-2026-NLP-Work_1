"""分析 BERT 训练数据截断情况与分类错误分布（供报告分析用）。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from transformers import AutoTokenizer

from common import get_split, load_nyt

MODEL_NAME = "google-bert/bert-base-uncased"
MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "bert-base-uncased"

# 与 task3_bert.py 一致：优先用本地权重，缺失时回退到 Hugging Face
tok = AutoTokenizer.from_pretrained(
    str(MODEL_DIR) if (MODEL_DIR / "config.json").exists() else MODEL_NAME
)
train, valid, test = get_split()

# 训练集 token 长度分布
lens = [len(tok.encode(t, add_special_tokens=True)) for t in train["text"]]
lens = np.asarray(lens)
print(f"train texts: {len(lens)}")
print(f"token length: mean={lens.mean():.0f} median={np.median(lens):.0f} "
      f"p25={np.percentile(lens,25):.0f} p75={np.percentile(lens,75):.0f}")
print(f"fraction > 64 tokens (会被截断): {(lens>64).mean()*100:.1f}%")
print(f"fraction > 128: {(lens>128).mean()*100:.1f}%")
print(f"fraction > 200: {(lens>200).mean()*100:.1f}%")

# 各类别平均长度
import pandas as pd
df = train.copy()
df["len"] = lens
print(df.groupby("label")["len"].agg(["count", "mean", "median"]))
