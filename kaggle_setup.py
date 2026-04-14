# ============================================================
# KAGGLE NOTEBOOK — Copy each block into a separate cell
# ============================================================

# ── CELL 1: Clone repos ──────────────────────────────────────
"""
!pip install sentencepiece huggingface-hub datasets -q

import os
if not os.path.exists("/kaggle/working/parameter-golf"):
    !git clone https://github.com/openai/parameter-golf.git /kaggle/working/parameter-golf
if not os.path.exists("/kaggle/working/my-scripts"):
    !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /kaggle/working/my-scripts

os.chdir("/kaggle/working/parameter-golf")
print("✅ Ready")
"""

# ── CELL 2: Download dataset ─────────────────────────────────
"""
!python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
"""

# ── CELL 3: Copy patch scripts ───────────────────────────────
"""
import shutil, os
for f in ["patch_sdpa.py", "patch_int6.py", "patch_qat.py",
          "run_baseline.py", "run_int6_ptq.py", "run_int6_qat.py"]:
    shutil.copy(f"/kaggle/working/my-scripts/{f}", f".")
print("✅ Scripts copied")
"""

# ── CELL 4: Create train_gpt_fixed.py ───────────────────────
"""
!python3 patch_sdpa.py
"""

# ── CELL 5: Run Exp 1 — INT8 Baseline ───────────────────────
"""
!python3 run_baseline.py
"""

# ── CELL 6: Check baseline results ──────────────────────────
"""
!tail -20 logs/exp1_baseline.txt
"""

# ── CELL 7: Create train_gpt_int6.py ────────────────────────
"""
!python3 patch_int6.py
"""

# ── CELL 8: Run Exp 2 — INT6 PTQ ────────────────────────────
"""
!python3 run_int6_ptq.py
"""

# ── CELL 9: Check INT6 PTQ results ──────────────────────────
"""
!tail -20 logs/exp2_int6_ptq.txt
"""

# ── CELL 10: Run Exp 3 — INT6 QAT ───────────────────────────
"""
!python3 run_int6_qat.py
"""

# ── CELL 11: Check INT6 QAT results + size ───────────────────
"""
!tail -20 logs/exp3_int6_qat.txt
import os
size_mb = os.path.getsize("final_model.int8.ptz") / 1e6
print(f"Model size: {size_mb:.2f} MB (limit: 16 MB)")
assert size_mb < 16, f"Model too large: {size_mb:.2f} MB"
print("✅ Under 16 MB limit!")
"""