"""Task 2.2 / 2.3：用 gensim 训练 100 维 Word2Vec + Logistic Regression。

Task 2.2 —— Word2Vec Trained on AG News：
  使用 AG News 的 90,000 条新闻文本训练 100 维 Word2Vec（Skip-gram，
  window=5, min_count=2, epochs=10），再把 NYT 文档表示为词向量平均，
  训练 Logistic Regression 分类。

Task 2.3 —— Word2Vec Trained on NYT：
  使用 NYT 训练集文本训练同样的 100 维 Word2Vec，其余流程相同。
  （只使用训练集文本训练词向量，避免测试集信息泄漏。）

用法：
  python code/task2_word2vec.py ag    # Task 2.2
  python code/task2_word2vec.py nyt   # Task 2.3

依赖：gensim（Python 3.12 环境；本仓库附运行说明）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, get_split, evaluate, save_results, tokenize, load_ag

W2V_DIR = Path(__file__).resolve().parent / "artifacts"
DIM = 100


def iter_sentences(texts):
    """把每篇文档分词后作为一条训练句子（词元已过滤标点）。"""
    for t in texts:
        toks = tokenize(t)
        if toks:
            yield toks


def train_word2vec(sentences, workers: int = 4):
    print(f"[Word2Vec] training {DIM}-d Skip-gram, window=5, min_count=2, epochs=10 ...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=DIM,
        window=5,
        min_count=2,
        sg=1,               # Skip-gram
        negative=5,
        sample=1e-4,
        epochs=10,
        workers=workers,
        seed=42,
    )
    return model


def doc_to_vector(tokens, kv, dim: int) -> np.ndarray:
    vecs = [kv[t] for t in tokens if t in kv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def run(corpus: str):
    assert corpus in ("ag", "nyt"), "corpus must be 'ag' or 'nyt'"
    W2V_DIR.mkdir(exist_ok=True)
    train, valid, test = get_split()

    if corpus == "ag":
        ag = load_ag()
        sentences = list(iter_sentences(ag["text"]))
        model_path = W2V_DIR / "word2vec_ag_100d.model"
    else:
        sentences = list(iter_sentences(train["text"]))
        model_path = W2V_DIR / "word2vec_nyt_100d.model"

    if model_path.exists():
        model = Word2Vec.load(str(model_path))
        print(f"[Word2Vec] loaded cached model: {model_path.name}")
    else:
        model = train_word2vec(sentences)
        model.save(str(model_path))
        print(f"[Word2Vec] saved model: {model_path.name}")

    kv = model.wv
    print(f"[Word2Vec] vocab size = {len(kv)}")
    x_train = np.vstack([doc_to_vector(tokenize(t), kv, DIM) for t in train["text"]])
    x_valid = np.vstack([doc_to_vector(tokenize(t), kv, DIM) for t in valid["text"]])
    x_test = np.vstack([doc_to_vector(tokenize(t), kv, DIM) for t in test["text"]])

    scaler = StandardScaler().fit(x_train)
    x_train, x_valid, x_test = scaler.transform(x_train), scaler.transform(x_valid), scaler.transform(x_test)

    best = None
    for c in (0.1, 1.0, 10.0, 100.0):
        model_lr = OneVsRestClassifier(
            LogisticRegression(C=c, max_iter=2000, random_state=42)
        )
        model_lr.fit(x_train, train["label"])
        score = evaluate(valid["label"], model_lr.predict(x_valid))["macro_f1"]
        if best is None or score > best[0]:
            best = (score, c)
    _, best_c = best

    model_lr = OneVsRestClassifier(
        LogisticRegression(C=best_c, max_iter=2000, random_state=42)
    )
    model_lr.fit(x_train, train["label"])
    metrics = evaluate(test["label"], model_lr.predict(x_test))
    name = f"Word2Vec ({'AG News' if corpus == 'ag' else 'NYT'}) 100d + LR"
    save_results(f"task2_word2vec_{corpus}", metrics, {"method": name, "best_C": best_c})
    print(f"[Word2Vec-{corpus}] C={best_c:g}  Accuracy={metrics['accuracy']:.4f}  "
          f"Macro-F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "ag")
