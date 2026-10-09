"""共享工具：数据加载、数据划分、分词、评价指标。

所有实验（Task 1/2/3）统一使用本模块：
- 同一份 NYT 80/10/10 随机划分（种子 42，分层抽样保证类别比例一致）；
- 同一套英文分词规则（nltk.word_tokenize，小写，仅保留长度>=2 的字母词元；
  若 nltk 数据不可用则回退到正则分词，保证流程可复现）；
- 统一的 Accuracy 与 Macro-F1 计算。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]          # Lab1 项目根目录
DATA_DIR = ROOT / "HW-1"                           # 原始数据目录
SPLIT_DIR = ROOT / "code" / "split"                 # 划分结果缓存目录
RESULTS_DIR = ROOT / "code" / "results"             # 实验指标结果目录

SEED = 42
RANDOM_STATE = SEED

TRAIN_RATIO, VALID_RATIO, TEST_RATIO = 0.8, 0.1, 0.1


def load_nyt() -> pd.DataFrame:
    """读取 NYT 数据集（text / label 两列），丢弃缺失行。"""
    df = pd.read_csv(DATA_DIR / "nyt.csv").dropna(subset=["text", "label"])
    df = df.reset_index(drop=True)
    return df


def load_ag() -> pd.DataFrame:
    """读取 AG News 数据集（text 列，无标签），用于训练 Word2Vec。"""
    df = pd.read_csv(DATA_DIR / "ag.csv").dropna(subset=["text"])
    df = df.reset_index(drop=True)
    return df


def get_split(force: bool = False):
    """返回 (train, valid, test) 三个 DataFrame。

    数据先整体随机打乱，再按 80%/10%/10% 分层划分为训练/验证/测试集。
    划分结果缓存为 CSV，保证所有实验使用完全相同的样本集合。
    """
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    files = [SPLIT_DIR / "train.csv", SPLIT_DIR / "valid.csv", SPLIT_DIR / "test.csv"]
    if not force and all(p.exists() for p in files):
        return (pd.read_csv(f) for f in files)

    data = load_nyt()
    train, holdout = train_test_split(
        data, test_size=TEST_RATIO + VALID_RATIO,
        random_state=RANDOM_STATE, stratify=data["label"],
    )
    valid, test = train_test_split(
        holdout, test_size=0.5,
        random_state=RANDOM_STATE, stratify=holdout["label"],
    )
    for df, p in zip((train, valid, test), files):
        df.to_csv(p, index=False)
    return train, valid, test


def tokenize(text: str) -> list[str]:
    """英文分词：小写 -> nltk.word_tokenize -> 仅保留长度>=2 的字母词元。

    nltk 数据不可用（例如 punkt 下载失败）时回退到正则分词，
    保证在任何环境下都能运行且结果基本一致。
    """
    text = str(text).lower()
    try:
        from nltk import word_tokenize
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return [t for t in tokens if t.isalpha() and len(t) >= 2]


def evaluate(y_true, y_pred) -> dict:
    """计算 Accuracy 与 Macro-F1。"""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def save_results(name: str, metrics: dict, extra: dict | None = None) -> None:
    """把实验指标写入 code/results/<name>.json，供报告引用。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    (RESULTS_DIR / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[results] {name}: " + json.dumps(payload, ensure_ascii=False))
