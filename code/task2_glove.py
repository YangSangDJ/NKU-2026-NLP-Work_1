"""Task 2.1：预训练 GloVe 词向量（glove.6B.100d）+ Logistic Regression。

步骤：
  1. 加载公开预训练 GloVe Embedding（glove.6B.100d.txt，100 维）；
  2. 对 NYT 每篇文档分词，取所有有效单词词向量的平均作为文档向量
     e(d) = (1/n) * sum(e(w_i))；无有效词时文档向量取全零向量；
  3. 使用 Logistic Regression 训练分类器，在验证集选择 C，
     最终在测试集报告 Accuracy 与 Macro-F1。

依赖：code/HW-1/glove.6B.100d.txt（由 HW-1/glove.6B.zip 解压得到，
      或直接放置同文件名的文本文件）。

运行：python code/task2_glove.py
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, get_split, evaluate, save_results, tokenize

GLOVE_PATH = DATA_DIR / "glove.6B.100d.txt"
DIM = 100


def load_glove(path) -> dict[str, np.ndarray]:
    """读取 glove.6B.100d.txt：每行 'word v1 v2 ... v100'。"""
    vectors: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) != DIM + 1:
                continue
            try:
                vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
    return vectors


def doc_to_vector(tokens: list[str], glove: dict[str, np.ndarray], dim: int) -> np.ndarray:
    vecs = [glove[t] for t in tokens if t in glove]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def main():
    print(f"[GloVe] loading {GLOVE_PATH} ...")
    glove = load_glove(GLOVE_PATH)
    print(f"[GloVe] loaded {len(glove)} word vectors, dim={DIM}")

    train, valid, test = get_split()
    x_train = np.vstack([doc_to_vector(tokenize(t), glove, DIM) for t in train["text"]])
    x_valid = np.vstack([doc_to_vector(tokenize(t), glove, DIM) for t in valid["text"]])
    x_test = np.vstack([doc_to_vector(tokenize(t), glove, DIM) for t in test["text"]])

    # 对特征做标准化，帮助 LR 收敛
    scaler = StandardScaler().fit(x_train)
    x_train, x_valid, x_test = scaler.transform(x_train), scaler.transform(x_valid), scaler.transform(x_test)

    best = None
    for c in (0.1, 1.0, 10.0, 100.0):
        model = OneVsRestClassifier(
            LogisticRegression(C=c, max_iter=2000, random_state=42)
        )
        model.fit(x_train, train["label"])
        score = evaluate(valid["label"], model.predict(x_valid))["macro_f1"]
        if best is None or score > best[0]:
            best = (score, c)
    _, best_c = best

    model = OneVsRestClassifier(
        LogisticRegression(C=best_c, max_iter=2000, random_state=42)
    )
    model.fit(x_train, train["label"])
    metrics = evaluate(test["label"], model.predict(x_test))
    save_results("task2_glove", metrics, {"method": "GloVe 100d + LR", "best_C": best_c})
    print(f"[GloVe] C={best_c:g}  Accuracy={metrics['accuracy']:.4f}  "
          f"Macro-F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
