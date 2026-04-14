import os
import subprocess
from pathlib import Path

assert Path("train_gpt_fixed.py").exists(), "Run: python3 patch_sdpa.py first"

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ.update({
    "RUN_ID":                "exp1_baseline",
    "DATA_PATH":             "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH":        "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE":            "1024",
    "ITERATIONS":            "300",
    "VAL_LOSS_EVERY":        "150",
    "MAX_WALLCLOCK_SECONDS": "0",
    "TRAIN_BATCH_TOKENS":    "131072",
    "VAL_BATCH_SIZE":        "131072",
    "NUM_LAYERS":            "9",
    "MODEL_DIM":             "512",
    "NUM_HEADS":             "8",
    "NUM_KV_HEADS":          "4",
    "MLP_MULT":              "2",
    "TRAIN_SEQ_LEN":         "1024",
})

print("=" * 60)
print("Exp 1 — INT8 Baseline")
print("  Layers    : 9")
print("  MLP_MULT  : 2")
print("  Quant     : INT8 PTQ")
print("  Target BPB: ~1.224")
print("  Target MB : ~10 MB")
print("=" * 60)

subprocess.run(["python3", "train_gpt_fixed.py"], check=True)