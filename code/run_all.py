"""一键运行全部实验（Task 1 / 2 / 3），结果写入 code/results/*.json。

用法（在 code 目录下）：
    python run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = sys.executable


def run(script: str, *args: str):
    print(f"\n{'='*70}\n>>> python {script} {' '.join(args)}\n{'='*70}")
    subprocess.run([PYTHON, str(HERE / script), *args], cwd=HERE, check=True)


if __name__ == "__main__":
    run("task1_bow.py")
    run("task2_glove.py")
    run("task2_word2vec.py", "ag")
    run("task2_word2vec.py", "nyt")
    run("task3_bert.py")
    print("\n全部实验完成，指标见 code/results/*.json")
