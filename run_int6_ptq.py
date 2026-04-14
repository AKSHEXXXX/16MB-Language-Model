import os
import subprocess
from pathlib import Path

assert Path("train_gpt_fixed.py").exists(), "Run: python3 patch_sdpa.py first"

# Create INT6 version from fixed baseline
if not Path("train_gpt_int6.py").exists():
    subprocess.run(["python3", "patch_int6.py"], check=True)

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ.update({
    "RUN_ID":                "exp2_int6_ptq",
    "DATA_PATH":             "./data/datasets/fineweb10B_sp1024/",
    "TOKENIZER_PATH":        "./data/tokenizers/fineweb_1024_bpe.model",
    "VOCAB_SIZE":            "1024",
    "ITERATIONS":            "300",
    "VAL_LOSS_EVERY":        "150",
    "MAX_WALLCLOCK_SECONDS": "0",
    "TRAIN_BATCH_TOKENS":    "131072",
    "VAL_BATCH_SIZE":        "131072",
    "NUM_LAYERS":            "11",
    "MODEL_DIM":             "512",
    "NUM_HEADS":             "8",
    "NUM_KV_HEADS":          "4",
    "MLP_MULT":              "3",
    "TRAIN_SEQ_LEN":         "1024",
})

print("=" * 60)
print("Exp 2 — INT6 PTQ")
print("  Layers    : 11 (+2 from INT6 budget)")
print("  MLP_MULT  : 3 (wider from INT6 budget)")
print("  Quant     : INT6 PTQ range[-31, 31]")
print("  Target BPB: ~1.17")
print("  Target MB : ~11 MB")
print("=" * 60)

subprocess.run(["python3", "train_gpt_int6.py"], check=True)