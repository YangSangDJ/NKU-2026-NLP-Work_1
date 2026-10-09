"""Task 1：Bag-of-Words 文本表示 + Logistic Regression 文本分类。

实现作业要求的两种词袋表示：
  1. Binary Bag of Words —— 只记录单词是否出现（出现为 1，否则为 0）；
  2. Word Frequency —— 记录单词在文档中的出现次数。

词表由训练集文本构建（覆盖训练语料全部词元）；文档被表示为 |V| 维向量。
分类器统一使用 Logistic Regression（One-vs-Rest + L2 正则，liblinear 求解），
正则化强度 C 在验证集上按 Macro-F1 选择，最终在测试集上报告 Accuracy 与 Macro-F1。

运行：python code/task1_bow.py
"""
from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from common import get_split, evaluate, save_results, tokenize

CANDIDATES = [0.1, 1.0, 10.0]


def build_lr(C: float) -> OneVsRestClassifier:
    return OneVsRestClassifier(
        LogisticRegression(C=C, solver="liblinear", max_iter=500, random_state=42)
    )


def run_representation(name: str, kind: str):
    """kind: 'binary'（二值词袋）| 'freq'（词频）"""
    train, valid, test = get_split()

    vectorizer = CountVectorizer(
        tokenizer=tokenize, binary=(kind == "binary"), lowercase=True,
    )
    x_train = vectorizer.fit_transform(train["text"])
    x_valid = vectorizer.transform(valid["text"])
    x_test = vectorizer.transform(test["text"])
    print(f"[{name}] vocabulary size |V| = {len(vectorizer.vocabulary_)}")

    # 验证集选择正则化强度 C
    best = None
    for c in CANDIDATES:
        model = build_lr(c)
        model.fit(x_train, train["label"])
        score = evaluate(valid["label"], model.predict(x_valid))["macro_f1"]
        if best is None or score > best[0]:
            best = (score, c)
    _, best_c = best

    # 用最佳 C 重新训练并在测试集评估
    model = build_lr(best_c)
    model.fit(x_train, train["label"])
    metrics = evaluate(test["label"], model.predict(x_test))
    save_results(
        f"task1_{kind}",
        metrics,
        {"method": name, "best_C": best_c, "vocab_size": len(vectorizer.vocabulary_)},
    )
    print(f"[{name}] C={best_c:g}  Accuracy={metrics['accuracy']:.4f}  "
          f"Macro-F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    run_representation("Binary Bag of Words", "binary")
    run_representation("Word Frequency", "freq")
