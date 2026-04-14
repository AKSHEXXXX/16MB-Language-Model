import os
import subprocess
from pathlib import Path

assert Path("train_gpt_fixed.py").exists(), "Run: python3 patch_sdpa.py first"

# Build INT6 script if needed
if not Path("train_gpt_int6.py").exists():
    subprocess.run(["python3", "patch_int6.py"], check=True)

# Apply QAT patch on top of INT6 script
subprocess.run(["python3", "patch_qat.py"], check=True)

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.environ.update({
    "RUN_ID":                "exp3_int6_qat",
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
print("Exp 3 — INT6 QAT")
print("  Layers    : 11")
print("  MLP_MULT  : 3")
print("  Quant     : INT6 QAT (STE fake quant during training)")
print("  Target BPB: ~1.15")
print("  Target MB : ~11 MB")
print("=" * 60)

subprocess.run(["python3", "train_gpt_int6.py"], check=True)