"""生成实验结果对比图（Accuracy 与 Macro-F1），输出到 Report/images/results.png。

从 code/results/*.json 读取所有模型的测试集指标。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "code" / "results"
OUT = ROOT / "Report" / "images" / "results.png"

ORDER = [
    ("task1_binary", "Binary BoW"),
    ("task1_freq", "Word Freq."),
    ("task2_glove", "GloVe 100d"),
    ("task2_word2vec_ag", "W2V (AG News)"),
    ("task2_word2vec_nyt", "W2V (NYT)"),
    ("task3_bert", "BERT"),
]

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    acc, f1 = [], []
    labels = []
    for key, name in ORDER:
        p = RESULTS / f"{key}.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        acc.append(data["accuracy"] * 100)
        f1.append(data["macro_f1"] * 100)
        labels.append(name)

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, acc, width, label="Accuracy", color="#4C72B0")
    b2 = ax.bar(x + width / 2, f1, width, label="Macro-F1", color="#DD8452")
    ax.set_ylabel("Score (%)")
    ax.set_title("NYT Test Set 上的分类性能对比")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(90, 101)
    ax.legend()
    for bar in list(b1) + list(b2):
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.axhline(100, color="gray", lw=0.6, ls="--")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
