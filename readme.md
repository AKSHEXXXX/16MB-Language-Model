# 🔩 NanoForge — 16MB Language Model

> Training the most accurate language model that fits inside **16 MB** from scratch.  
> Built for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

***

## 🎯 What Is This?

NanoForge is a from-scratch language model trained to maximize accuracy under a strict **16 MB file size constraint** (zlib compressed). It uses a stack of compression and architecture techniques to pack as much intelligence as possible into a tiny model.

The evaluation metric is `val_bpb` (bits-per-byte) — a tokenizer-agnostic compression score. **Lower is better.**

***

## 📊 Results

| | Before | After |
|---|---|---|
| **val_bpb** | 1.224 | **1.192** |
| **Model size** | ~10 MB | ~11 MB |
| **Layers** | 9 | 11 |
| **MLP Width** | 2× | 3× |
| **Quantization** | INT8 PTQ | INT6 QAT |

| Experiment | Layers | MLP Width | Quantization | val_bpb | Size |
|------------|--------|-----------|--------------|---------|------|
| Exp 1 — INT8 Baseline | 9 | 2× | INT8 PTQ | 1.224 | ~10 MB |
| Exp 2 — INT6 PTQ | 11 | 3× | INT6 PTQ | ~1.200 | ~11 MB |
| Exp 3 — INT6 QAT ✅ | 11 | 3× | INT6 QAT | **1.192** | ~11 MB |

All runs are **under 16 MB**. Each experiment builds on the previous one.

***

## 🧠 Algorithms & Techniques

### 1. 🔤 Small Vocabulary (1024 tokens)

Standard GPT-2 uses 50,257 tokens. NanoForge uses a custom SentencePiece BPE tokenizer with only **1024 tokens**, eliminating ~74 MB of embedding parameters. The `val_bpb` metric is tokenizer-agnostic (measures raw bytes), so this is a free win — no quality penalty.

***

### 2. 🔗 Tied Embeddings

The input embedding matrix and the output projection (`lm_head`) **share the same weight matrix**. One matrix does both encoding and decoding. This eliminates one full `vocab_size × model_dim` matrix, saving ~2 MB with a tiny quality trade-off.

```python
# Shared embedding in forward pass
logits = F.linear(x, self.tok_emb.weight)  # reuses input embedding
```

***

### 3. 👁️ Grouped Query Attention (GQA)

Standard multi-head attention uses 8 Q, K, and V heads. NanoForge uses **8 query heads but only 4 KV heads** — each pair of query heads shares one K and V head. This halves the size of K/V projections (~1.5 MB saved) with negligible quality loss. The same technique is used in Llama 2, Llama 3, and Mistral.

```
num_heads    = 8   ← query heads
num_kv_heads = 4   ← key/value heads (shared)
```

***

### 4. 🏗️ U-Net Skip Connections

Instead of a flat stack of transformer blocks, NanoForge uses a **U-Net-style architecture**. The first half (encoder) saves residual activations. The second half (decoder) re-injects them in reverse order via learnable skip weights.

This dramatically improves gradient flow through deep networks, giving better quality at the same parameter count — essentially a free accuracy boost.

```python
# Encoder stores skip connections
for i in range(num_encoder_layers):
    x = blocks[i](x, x0)
    skips.append(x)

# Decoder reuses them in reverse
for i in range(num_decoder_layers):
    x = x + skip_weights[i] * skips.pop()
    x = blocks[num_encoder_layers + i](x, x0)
```

***

### 5. ⚡ Muon Optimizer

All 2D weight matrices (attention projections, MLP weights) are trained with the **Muon optimizer** instead of Adam. Muon applies Newton-Schulz orthogonalization to gradients before the update, giving faster convergence and better final quality within the same number of iterations.

Scalar parameters, embeddings, and control tensors use Adam as usual.

```python
# Orthogonalize gradient update
g = zeropower_via_newtonschulz5(g, steps=5)
g *= max(1, g.size(0) / g.size(1)) ** 0.5
model_weight -= lr * g
```

***

### 6. 🟦 relu² Activation

The MLP blocks use `relu(x)²` instead of GeLU or standard ReLU. This promotes sparsity in activations, which improves both generalization and compressibility of the model's internal representations.

```python
def forward(self, x):
    x = torch.relu(self.fc(x))
    return self.proj(x.square())  # relu²
```

***

### 7. 🔢 INT6 Post-Training Quantization (PTQ)

After training in bf16/fp32, all large 2D weight matrices are quantized to **6-bit integers** stored as int8 (range [-31, 31]):

- **Per-row scaling**: each output row gets its own scale factor for better accuracy
- **Outlier clipping**: weights are clipped at the 99.9984th percentile before quantizing
- **Small tensor passthrough**: tensors with < 65,536 elements kept as fp16
- **Control tensor passthrough**: scales, norms, skip weights kept as fp32

The key advantage over INT8: INT6-range values have lower entropy, which means **zlib compresses them ~25% better**, freeing budget for more layers.

```python
INT6_QUANT_MAX = 31  # 6-bit signed symmetric

# Per-row INT6 quantization
scale = (clip_abs / 31.0).clamp_min(1.0 / 31.0)
q = torch.clamp(
    torch.round(clipped / scale[:, None]),
    -31, 31
).to(torch.int8)
```

***

### 8. 🎯 INT6 Quantization-Aware Training (QAT)

PTQ introduces a small accuracy gap because weights were not trained expecting INT6 noise. QAT closes this gap by **simulating INT6 quantization during training** using the Straight-Through Estimator (STE):

- **Forward pass**: weights are fake-quantized to INT6 range
- **Backward pass**: gradients flow through as if no quantization happened
- **Result**: weights learn distributions that survive INT6 rounding, recovering accuracy lost in PTQ

This is what pushed us from **1.224 → 1.192 val_bpb**.

```python
def fake_quant_int6(w):
    scale = w.float().abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 31.0
    w_quantized = torch.clamp(torch.round(w.float() / scale), -31.0, 31.0) * scale
    # Straight-Through Estimator: fake forward, real backward
    return w + (w_quantized.to(w.dtype) - w).detach()

class CastedLinear(nn.Linear):
    def forward(self, x):
        w = fake_quant_int6(self.weight) if self.training else self.weight
        return F.linear(x, w.to(x.dtype))
```

***

### 9. 📦 zlib Compression (Level 9)

The final exported model is zlib-compressed at maximum level. Because INT6 values are restricted to [-31, 31] (63 distinct values vs 255 for INT8), they compress significantly better. The submission includes a decompressor that restores weights to fp32/bf16 for evaluation.

***

### 10. 💰 INT6 Budget → Extra Capacity

The space saved by INT6 over INT8 compression (~2.5 MB) is reinvested into making the model larger and more capable — same final file size, better quality:

| | INT8 Baseline | INT6 NanoForge |
|---|---|---|
| Layers | 9 | **11** (+2) |
| MLP width | 2× | **3×** (+50%) |
| Quantization | INT8 | INT6 QAT |
| val_bpb | 1.224 | **1.192** ✅ |
| Model size | ~10 MB | ~11 MB |

***

## 🔁 Full Compression Pipeline

```
┌─────────────────────────────────────────────────┐
│  Train in bf16/fp32                             │
│  • 11 transformer layers                        │
│  • 512 model dim, 8/4 GQA heads                 │
│  • relu² MLP (3× expansion)                     │
│  • U-Net skip connections                        │
│  • Muon optimizer for matrices                  │
│  • QAT: fake INT6 during training (STE)         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  INT6 Post-Training Quantization                │
│  • 2D weights → per-row INT6 [-31, 31]          │
│  • Small tensors  → fp16 passthrough            │
│  • Control tensors (scales, norms) → fp32       │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│  zlib compress (level 9)                        │
│  INT6 entropy → ~25% better than INT8           │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
         final_model.int8.ptz
         ✅ Under 16 MB  |  val_bpb: 1.192
```

***

## 🚀 Getting Started

### Prerequisites

```bash
# Clone the base challenge repo
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

# Install dependencies
pip install sentencepiece huggingface-hub datasets torch numpy

# Download dataset
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### Clone NanoForge scripts

```bash
git clone https://github.com/YOUR_USERNAME/nanoforge.git scripts
```

### Build patched training scripts

```bash
# Step 1: Fix SDPA backend for T4 compatibility (creates train_gpt_fixed.py)
python3 scripts/patch_sdpa.py

# Step 2: Apply INT6 quantization + extra layers (creates train_gpt_int6.py)
python3 scripts/patch_int6.py

# Step 3: Apply QAT fake quantization (modifies train_gpt_int6.py)
python3 scripts/patch_qat.py
```

### Run experiments

```bash
# Exp 1: INT8 Baseline — validate setup (val_bpb: ~1.224)
python3 scripts/run_baseline.py

# Exp 2: INT6 PTQ — more layers, better quality (val_bpb: ~1.200)
python3 scripts/run_int6_ptq.py

# Exp 3: INT6 QAT — best quality, final submission (val_bpb: 1.192)
python3 scripts/run_int6_qat.py
```

### Verify model size

```bash
python3 -c "
import os
size = os.path.getsize('final_model.int8.ptz') / 1e6
print(f'Model size: {size:.2f} MB')
assert size < 16, f'Too large: {size:.2f} MB'
print('✅ Under 16 MB limit')
"
```

***

## 🖥️ Hardware

Developed and tested on **Kaggle T4 GPU (14.5 GB VRAM)**.

T4-specific optimizations applied:

| Setting | Value | Why |
|---------|-------|-----|
| `TORCH_COMPILE_DISABLE` | `1` | Saves 2–3 GB VRAM |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Fixes memory fragmentation |
| `TRAIN_BATCH_TOKENS` | `131072` | Reduced from default 524,288 to fit 14 GB |
| SDPA backend | Math backend | Fixes `Invalid backend` crash on T4 |

***

## 📁 Repository Structure

```
nanoforge/
├── requirements.txt         ← Python dependencies
├── .gitignore
├── patch_sdpa.py            ← Step 1: creates train_gpt_fixed.py
├── patch_int6.py            ← Step 2: creates train_gpt_int6.py (INT6 + extra layers)
├── patch_qat.py             ← Step 3: adds QAT fake quant to train_gpt_int6.py
├── run_baseline.py          ← Exp 1: INT8 baseline  (val_bpb: 1.224)
├── run_int6_ptq.py          ← Exp 2: INT6 PTQ       (val_bpb: ~1.200)
├── run_int6_qat.py          ← Exp 3: INT6 QAT ✅    (val_bpb: 1.192)
└── kaggle_setup.py          ← Kaggle notebook cell reference
```

> `train_gpt_fixed.py` and `train_gpt_int6.py` are **generated files** — not committed.  
> Run `patch_sdpa.py` → `patch_int6.py` to regenerate them fresh from the original repo.

***

## 🙏 Credits

- [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) — challenge framework, base training script, evaluation harness
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/) by Keller Jordan
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) — U-Net skip connections, relu² MLP, training setup
- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb) by HuggingFace

***

<p align="center">
  Built to be small. Trained to be sharp. 🔩<br/>
  <strong>1.224 → 1.192 val_bpb &nbsp;|&nbsp; Under 16 MB</strong>
</p>
