# 🔩 NanoForge — A 16MB Language Model

<p align="center">
  <img src="https://img.shields.io/badge/OpenAI-Parameter%20Golf-412991?style=for-the-badge&logo=openai&logoColor=white" />
  <img src="https://img.shields.io/badge/Model%20Size-Under%2016MB-00b894?style=for-the-badge" />
  <img src="https://img.shields.io/badge/val__bpb-1.192-0984e3?style=for-the-badge" />
  <img src="https://img.shields.io/badge/GPU-Kaggle%20T4-f39c12?style=for-the-badge" />
</p>

<p align="center">
  <strong>A challenge-driven language model built for the OpenAI Parameter Golf competition.</strong><br/>
  Goal: train the strongest possible language model that stays under 16 MB — weights, code, and all.
</p>

<p align="center">
  <a href="https://github.com/openai/parameter-golf">🏆 View the Challenge</a> &nbsp;·&nbsp;
  <a href="#-results">📊 Results</a> &nbsp;·&nbsp;
  <a href="#-getting-started">🚀 Getting Started</a> &nbsp;·&nbsp;
  <a href="#-algorithms--techniques">🧠 Techniques</a>
</p>

---

## 🏆 About the Challenge

**OpenAI Parameter Golf** is an open ML research competition launched March 18, 2026.

The premise is deceptively simple: train the best language model you can, but the **entire artifact — model weights plus training code combined — must fit inside 16 megabytes**. For reference, a single iPhone photo is 3–5 MB. GPT-2 Small alone is 548 MB.

Training must complete in under 10 minutes on 8×H100 GPUs and models are ranked by **bits-per-byte (BPB)** on the FineWeb validation set — a tokenizer-agnostic compression metric where lower is better.

| | |
|---|---|
| **Organizer** | OpenAI |
| **Prize pool** | $1,000,000 in compute credits |
| **Deadline** | April 30, 2026 |
| **Metric** | val_bpb on FineWeb (lower = better) |
| **Baseline** | 1.2244 BPB — 9 layers, INT8, 512 dim |
| **Current SOTA** | ~1.119 BPB |

> OpenAI CRO Mark Chen described the core question as: *"Can you come up with creative solutions in a sandbox setting?"* — the same quality they test for in frontier research roles. Top participants may be invited to interview.

🔗 [openai/parameter-golf on GitHub](https://github.com/openai/parameter-golf)

---

## 🎯 What Is NanoForge?

NanoForge is a **from-scratch language model engineering effort** built entirely around the constraints of the Parameter Golf challenge. It is not a fine-tuned wrapper. It is not a toy notebook. It is a complete, end-to-end compression pipeline that starts from the OpenAI baseline and applies a stacked sequence of architectural and quantization improvements — reinvesting every freed byte back into model capacity.

The central insight driving the design: **INT6 quantization gives you ~25% smaller weights than INT8, and that freed space buys you two entire extra transformer layers at no additional cost.**

---

## 📊 Results

<table>
<tr>
<th>Metric</th><th>Baseline</th><th>NanoForge ✅</th>
</tr>
<tr>
<td><strong>val_bpb</strong></td><td>1.224</td><td><strong>1.192</strong></td>
</tr>
<tr>
<td><strong>Model size</strong></td><td>~10 MB</td><td>~11 MB</td>
</tr>
<tr>
<td><strong>Layers</strong></td><td>9</td><td>11 (+2)</td>
</tr>
<tr>
<td><strong>MLP width</strong></td><td>2×</td><td>3× (+50%)</td>
</tr>
<tr>
<td><strong>Quantization</strong></td><td>INT8 PTQ</td><td>INT6 QAT</td>
</tr>
</table>

### Experiment progression

| # | Config | Layers | MLP | Quantization | val_bpb | Size |
|---|--------|--------|-----|--------------|---------|------|
| Exp 1 | INT8 Baseline | 9 | 2× | INT8 PTQ | 1.224 | ~10 MB |
| Exp 2 | INT6 PTQ | 11 | 3× | INT6 PTQ | ~1.200 | ~11 MB |
| Exp 3 | INT6 QAT ✅ | 11 | 3× | INT6 QAT | **1.192** | ~11 MB |

Every experiment stays under 16 MB. Each one builds directly on the previous.

---

## 🧠 Algorithms & Techniques

Every technique below directly solves one of two problems: **make the model smaller** or **make it smarter within the same size**.

---

### 1. 🔤 Small Vocabulary (1,024 tokens)

Standard GPT-2 uses a 50,257-token vocabulary, requiring a 50K × 512 embedding matrix — over 100 MB before training even begins. NanoForge uses a **custom SentencePiece BPE tokenizer with 1,024 tokens**, eliminating ~74 MB of embedding parameters instantly.

The `val_bpb` metric is tokenizer-agnostic — it measures raw bytes, not tokens — so a smaller vocabulary is a genuinely free win with no quality penalty.

---

### 2. 🔗 Tied Embeddings

The input embedding matrix and the output projection (`lm_head`) **share the same weight matrix**. Encoding and decoding use a single matrix in opposite directions. This eliminates one full `vocab_size × model_dim` parameter block, saving ~2 MB with negligible quality trade-off.

A model with rare tokens benefits especially: the shared matrix receives gradients from both input and output paths simultaneously, training rare tokens more effectively.

```python
# Output projection reuses the input embedding weights
logits = F.linear(x, self.tok_emb.weight)
```

---

### 3. 👁️ Grouped Query Attention (GQA)

Standard multi-head attention replicates K and V projections for every query head. NanoForge uses **8 query heads but only 4 KV heads** — each pair of query heads shares one K and V projection. This halves the size of K/V weight matrices (~1.5 MB saved) with negligible quality loss at this scale.

The same architecture is used in Llama 2, Llama 3, and Mistral.

```
num_heads    = 8   ← query heads (full resolution)
num_kv_heads = 4   ← key/value heads (shared, 2:1)
```

---

### 4. 🏗️ U-Net Skip Connections

A standard transformer stacks N identical blocks in sequence. Deep networks suffer from gradient degradation — gradients weaken as they travel backward through many layers. NanoForge uses a **U-Net-style architecture** borrowed from image segmentation:

- The **first half** (encoder layers) saves residual activations at each step
- The **second half** (decoder layers) re-injects them in reverse order via learned skip weights

This creates direct gradient highways from output to input, enabling deeper networks to train stably and improving final BPB at no parameter cost beyond the small skip weight scalars.

```python
# Encoder: process and store
for i in range(num_encoder_layers):
    x = blocks[i](x, x0)
    skips.append(x)

# Decoder: process with skip injection
for i in range(num_decoder_layers):
    x = x + skip_weights[i] * skips.pop()
    x = blocks[num_encoder_layers + i](x, x0)
```

---

### 5. ⚡ Muon Optimizer

All 2D weight matrices — attention projections and MLP weights — are trained with **Muon** instead of Adam. The key insight: transformer weight matrix gradients tend to be dominated by a small number of directions (near low-rank). Adam ignores this structure and updates all directions equally.

Muon applies **Newton-Schulz orthogonalization** to the gradient update, spreading it evenly across all directions in weight space. This makes rare features and edge cases update more aggressively — exactly what language modeling needs for rare tokens and unusual patterns.

Empirically: Muon achieved a **35% training speed improvement** over Adam on the NanoGPT benchmark this challenge is based on.

```python
# Core of Muon: orthogonalize the gradient
g = zeropower_via_newtonschulz5(g, steps=5)
g *= max(1, g.size(0) / g.size(1)) ** 0.5
model_weight -= lr * g
```

Embeddings, biases, and scalar parameters continue to use Adam, since orthogonalization only applies to 2D matrices.

---

### 6. 🟦 relu² Activation

MLP blocks use `relu(x)²` instead of GeLU or standard ReLU. This produces **sparser activations** — a higher fraction of neurons output exactly zero. Sparsity helps in two ways: better generalization (fewer neurons "fire" for any given input, reducing overfitting) and better compressibility of internal representations.

```python
def forward(self, x):
    x = torch.relu(self.fc(x))
    return self.proj(x.square())  # element-wise square after relu
```

---

### 7. 🔢 INT6 Post-Training Quantization (PTQ)

After training in bf16, all large 2D weight matrices are quantized to **6-bit integers** stored as int8 (range `[-31, 31]`, 63 distinct values):

| Setting | Value | Why |
|---------|-------|-----|
| Range | `[-31, 31]` | 6-bit signed symmetric |
| Scale | Per-row | One scale factor per output neuron |
| Clipping | 99.9984th percentile | Remove outliers before scaling |
| Small tensors | fp16 passthrough | Tensors < 65K elements kept full precision |
| Control tensors | fp32 passthrough | Scales, norms, skip weights untouched |

**Why INT6 over INT8?** INT6 values live in a smaller range (63 values vs 255). Smaller range = lower entropy = better zlib compression ratio (~25% more compression). The freed bytes are directly reinvested into two extra transformer layers — same final artifact size, meaningfully better model.

```python
INT6_QUANT_MAX = 31  # 6-bit signed range

# Per-row quantization with outlier clipping
clip_abs = torch.quantile(w.abs().flatten(), 0.9999984).item()
scale = clip_abs / 31.0
q = torch.clamp(torch.round(w.clamp(-clip_abs, clip_abs) / scale), -31, 31).to(torch.int8)
```

---

### 8. 🎯 INT6 Quantization-Aware Training (QAT)

PTQ introduces a small accuracy gap: the model trained in full precision, then got compressed — it never adapted to INT6 noise. **QAT closes this gap** by simulating INT6 quantization in every forward pass during training using the Straight-Through Estimator (STE):

- **Forward pass**: weights are fake-quantized to INT6 range — the model sees what it will look like compressed
- **Backward pass**: gradients flow through the quantization operation as if it were identity — training continues normally
- **Result**: weights learn to cluster around values that survive INT6 rounding with minimal precision loss

**This single change drove the final improvement from 1.200 → 1.192 val_bpb.**

```python
def fake_quant_int6(w: torch.Tensor) -> torch.Tensor:
    scale = w.float().abs().max(dim=1, keepdim=True).values.clamp(min=1e-8) / 31.0
    w_q = torch.clamp(torch.round(w.float() / scale), -31.0, 31.0) * scale
    # STE: quantized in forward, full-precision in backward
    return w + (w_q.to(w.dtype) - w).detach()

class CastedLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quant_int6(self.weight) if self.training else self.weight
        return F.linear(x, w.to(x.dtype))
```

---

### 9. 📦 zlib Compression (Level 9)

The final model is zlib-compressed at maximum level before size measurement. Because INT6 values are restricted to 63 distinct values (vs 255 for INT8), they compress significantly better — the compressed artifact is ~25% smaller than an INT8 model with the same architecture.

The submission includes a self-contained decompressor that restores weights to fp32 for evaluation, with no external dependencies.

---

### 10. 💰 The INT6 Budget Equation

The full picture of how INT6 compression creates model capacity:

| | INT8 Baseline | INT6 NanoForge |
|---|---|---|
| Compression ratio | ~1.8× | ~2.4× |
| Compressed size | ~10 MB | ~11 MB |
| Layers | 9 | **11** |
| MLP width | 2× | **3×** |
| val_bpb | 1.224 | **1.192** |

Every byte freed by moving from INT8 to INT6 is directly converted into model capacity. **The constraint becomes the opportunity.**

---

## 🔁 Full Compression Pipeline

```
  Training (bf16/fp32)
  ┌──────────────────────────────────────────────┐
  │  11 transformer layers                       │
  │  512 model dim · 8/4 GQA heads              │
  │  relu² MLP (3× expansion)                   │
  │  U-Net skip connections                      │
  │  Muon optimizer (matrices)                   │
  │  QAT: fake INT6 every forward pass (STE)    │
  └─────────────────────┬────────────────────────┘
                        │
                        ▼
  INT6 Post-Training Quantization
  ┌──────────────────────────────────────────────┐
  │  2D weights  →  per-row INT6 [-31, 31]      │
  │  Small tensors  →  fp16 passthrough          │
  │  Control tensors  →  fp32 passthrough        │
  └─────────────────────┬────────────────────────┘
                        │
                        ▼
  zlib Compression (level 9)
  ┌──────────────────────────────────────────────┐
  │  INT6 entropy → ~25% better than INT8       │
  │  Final: final_model.int8.ptz                │
  └─────────────────────┬────────────────────────┘
                        │
                        ▼
         ✅ val_bpb: 1.192 · Size: ~11 MB · Under 16 MB
```

---

## 🚀 Getting Started

### 1. Set up the base challenge repo

```bash
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf

pip install sentencepiece huggingface-hub datasets torch numpy

python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

### 2. Clone NanoForge

```bash
git clone https://github.com/AKSHEXXXX/nanoforge.git scripts
```

### 3. Patch the training script

```bash
# Fix SDPA backend for T4 compatibility → generates train_gpt_fixed.py
python3 scripts/patch_sdpa.py

# Apply INT6 quantization + extra layers → generates train_gpt_int6.py
python3 scripts/patch_int6.py

# Apply QAT fake quantization → modifies train_gpt_int6.py in-place
python3 scripts/patch_qat.py
```

### 4. Run experiments

```bash
# Exp 1 — INT8 Baseline (val_bpb: 1.224)
python3 scripts/run_baseline.py

# Exp 2 — INT6 PTQ: more layers, tighter quantization (val_bpb: ~1.200)
python3 scripts/run_int6_ptq.py

# Exp 3 — INT6 QAT: quantization-aware training, final result (val_bpb: 1.192)
python3 scripts/run_int6_qat.py
```

### 5. Verify model size

```bash
python3 -c "
import os
size = os.path.getsize('final_model.int8.ptz') / 1e6
print(f'Model size: {size:.2f} MB')
assert size < 16, f'Over limit: {size:.2f} MB'
print('✅ Under 16 MB')
"
```

---

## 🖥️ Hardware & Kaggle Setup

Developed on **Kaggle T4 GPU (14.5 GB VRAM)** — completely free.

| Setting | Value | Reason |
|---------|-------|--------|
| `TORCH_COMPILE_DISABLE` | `1` | Saves 2–3 GB VRAM on T4 |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Prevents memory fragmentation |
| `TRAIN_BATCH_TOKENS` | `131072` | Fits within 14 GB VRAM |
| SDPA backend | Math (via `patch_sdpa.py`) | Fixes `Invalid backend` crash on T4 |

The `patch_sdpa.py` script surgically patches the three lines inside `train_gpt.py` that control attention backend selection, forcing math-only SDP which runs correctly on any CUDA GPU regardless of capability level.

---

## 📁 Repository Structure

```
nanoforge/
├── README.md                ← this file
├── requirements.txt         ← Python dependencies
├── .gitignore
│
├── patch_sdpa.py            ← Step 1: patches SDP backend for T4
├── patch_int6.py            ← Step 2: adds INT6 quantization + 11 layers
├── patch_qat.py             ← Step 3: adds QAT with STE
│
├── run_baseline.py          ← Exp 1: INT8 baseline   (val_bpb: 1.224)
├── run_int6_ptq.py          ← Exp 2: INT6 PTQ        (val_bpb: ~1.200)
├── run_int6_qat.py          ← Exp 3: INT6 QAT ✅     (val_bpb: 1.192)
│
└── kaggle_setup.py          ← Full Kaggle notebook reference
```

> `train_gpt_fixed.py` and `train_gpt_int6.py` are generated files and not committed.
> Run `patch_sdpa.py` → `patch_int6.py` to regenerate them from the base repo.

---

## 🙏 Credits

- **[OpenAI Parameter Golf](https://github.com/openai/parameter-golf)** — challenge framework, base training script, evaluation harness, FineWeb dataset pipeline
- **[Muon Optimizer](https://kellerjordan.github.io/posts/muon/)** by Keller Jordan — Newton-Schulz orthogonalization for transformer training
- **[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)** — U-Net skip connections, relu² MLP, training setup patterns
- **[FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** by HuggingFace — training and evaluation corpus

---

<p align="center">
  <strong>Built to be small. Trained to be sharp.</strong><br/><br/>
  1.224 → 1.192 val_bpb &nbsp;·&nbsp; Under 16 MB &nbsp;·&nbsp; OpenAI Parameter Golf Challenge<br/><br/>
  <a href="https://github.com/openai/parameter-golf">🔗 Challenge Repo</a> &nbsp;·&nbsp;
  <a href="https://openai.com/index/parameter-golf/">🔗 Challenge Page</a>
</p>
