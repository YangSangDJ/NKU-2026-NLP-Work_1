"""Task 3：BERT（google-bert/bert-base-uncased）Fine-tuning 文本分类。

设置（按作业要求）：
  - Pre-trained model：bert-base-uncased
  - Maximum sequence length：64
  - Number of epochs：3

流程：
  1. 加载预训练 BERT + 分类头（3 类）；
  2. 用 BERT tokenizer 将 NYT 文本编码为 max_length=64 的输入；
  3. 微调 3 个 epoch（AdamW，lr=2e-5，batch_size=32）；
  4. 在测试集报告 Accuracy 与 Macro-F1。

国内网络下可通过环境变量使用镜像：
  $env:HF_ENDPOINT = 'https://hf-mirror.com'
运行：python code/task3_bert.py
"""
from __future__ import annotations

import os
import random
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from common import get_split, evaluate, save_results
from pathlib import Path

MODEL_NAME = "google-bert/bert-base-uncased"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "bert-base-uncased"
MAX_LENGTH = 64
EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BERT] device = {device}")
    if device.type == "cuda":
        print(f"[BERT] GPU: {torch.cuda.get_device_name(0)}")

    train, valid, test = get_split()
    labels = sorted(set(train["label"]))
    label2id = {lb: i for i, lb in enumerate(labels)}
    print(f"[BERT] classes: {label2id}")

    if (LOCAL_MODEL_DIR / "config.json").exists():
        print(f"[BERT] loading model from local dir: {LOCAL_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(LOCAL_MODEL_DIR), num_labels=len(labels)
        ).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(labels)
        ).to(device)

    def encode(texts):
        return tokenizer(
            list(texts), max_length=MAX_LENGTH, truncation=True, padding="max_length",
            return_tensors="pt",
        )

    enc = encode(train["text"])
    train_ds = TensorDataset(enc["input_ids"], enc["attention_mask"],
                             torch.tensor(train["label"].map(label2id).values))
    enc_v = encode(valid["text"])
    valid_ds = TensorDataset(enc_v["input_ids"], enc_v["attention_mask"],
                             torch.tensor(valid["label"].map(label2id).values))
    enc_t = encode(test["text"])
    test_ds = TensorDataset(enc_t["input_ids"], enc_t["attention_mask"],
                            torch.tensor(test["label"].map(label2id).values))

    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    total_steps = len(loader) * EPOCHS
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    def predict(dl):
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for ids, mask, y in dl:
                logits = model(ids.to(device), attention_mask=mask.to(device)).logits
                preds.extend(logits.argmax(dim=1).cpu().tolist())
                trues.extend(y.tolist())
        return trues, preds

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for i, (ids, mask, y) in enumerate(loader):
            optimizer.zero_grad()
            out = model(ids.to(device), attention_mask=mask.to(device), labels=y.to(device))
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        v_trues, v_preds = predict(valid_loader)
        v = evaluate(v_trues, v_preds)
        print(f"[BERT] epoch {epoch}/{EPOCHS}  loss={total_loss/len(loader):.4f}  "
              f"val_acc={v['accuracy']:.4f}  val_macroF1={v['macro_f1']:.4f}  "
              f"({time.time()-t0:.0f}s)")

    trues, preds = predict(test_loader)
    metrics = evaluate(trues, preds)
    save_results(
        "task3_bert",
        metrics,
        {
            "method": "BERT-base-uncased fine-tuning",
            "model": MODEL_NAME, "max_length": MAX_LENGTH,
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
        },
    )
    print(f"[BERT] TEST  Accuracy={metrics['accuracy']:.4f}  "
          f"Macro-F1={metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
