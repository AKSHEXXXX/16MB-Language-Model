from pathlib import Path

src = Path("train_gpt_fixed.py")
dst = Path("train_gpt_int6.py")

assert src.exists(), "train_gpt_fixed.py not found. Run: python3 patch_sdpa.py first"

text = src.read_text()

# Fix 1: INT6 clip percentile (slightly more aggressive for smaller range)
old_clip = "INT8_CLIP_PERCENTILE = 99.99984"
new_clip = "INT8_CLIP_PERCENTILE = 99.9984"

assert old_clip in text, "INT8_CLIP_PERCENTILE not found"
text = text.replace(old_clip, new_clip)
print("✅ Fix 1: INT6 clip percentile")

# Fix 2: Replace quantize_float_tensor with INT6 version (127 -> 31)
old_quant = """def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale"""

new_quant = """# INT6: symmetric range [-31, 31] stored in int8
# Values are still stored as int8 but only use 63 distinct values
# zlib compresses INT6-range int8 ~25% better than full INT8 range
INT6_QUANT_MAX = 31

def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        # Per-row INT6: one scale per output row
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / float(INT6_QUANT_MAX)).clamp_min(1.0 / float(INT6_QUANT_MAX))
        q = torch.clamp(
            torch.round(clipped / scale[:, None]),
            -INT6_QUANT_MAX, INT6_QUANT_MAX
        ).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    # Vectors / scalars: per-tensor INT6
    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(
        clip_abs / float(INT6_QUANT_MAX) if clip_abs > 0 else 1.0,
        dtype=torch.float32
    )
    q = torch.clamp(
        torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale),
        -INT6_QUANT_MAX, INT6_QUANT_MAX
    ).to(torch.int8).contiguous()
    return q, scale"""

assert old_quant in text, "quantize_float_tensor not found"
text = text.replace(old_quant, new_quant)
print("✅ Fix 2: INT6 quantize_float_tensor (range [-31, 31])")

# Fix 3: More layers using freed budget from INT6
old_layers = '    num_layers = int(os.environ.get("NUM_LAYERS", 9))'
new_layers = '    num_layers = int(os.environ.get("NUM_LAYERS", 11))'

assert old_layers in text, "NUM_LAYERS not found"
text = text.replace(old_layers, new_layers)
print("✅ Fix 3: NUM_LAYERS default 9 -> 11")

# Fix 4: Wider MLP using freed budget from INT6
old_mlp = '    mlp_mult = int(os.environ.get("MLP_MULT", 2))'
new_mlp = '    mlp_mult = int(os.environ.get("MLP_MULT", 3))'

assert old_mlp in text, "MLP_MULT not found"
text = text.replace(old_mlp, new_mlp)
print("✅ Fix 4: MLP_MULT default 2 -> 3")

dst.write_text(text)
print(f"\n✅ {dst} created")
print("   INT6 quantization: range [-31, 31] stored as int8")
print("   Extra capacity: 11 layers (was 9), MLP_MULT=3 (was 2)")
print("   Expected model size: ~10-13 MB (under 16 MB limit)")