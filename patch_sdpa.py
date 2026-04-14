from pathlib import Path

src = Path("train_gpt.py")
dst = Path("train_gpt_fixed.py")

assert src.exists(), "train_gpt.py not found. Clone: https://github.com/openai/parameter-golf"

text = src.read_text()

# Fix 1: Wrap SDPA call in math backend context
old_sdpa = """        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            enable_gqa=(self.num_kv_heads != self.num_heads),
        )"""

new_sdpa = """        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )"""

assert old_sdpa in text, "SDPA block not found"
text = text.replace(old_sdpa, new_sdpa)
print("✅ Fix 1: SDPA math backend")

# Fix 2: Global backend settings — disable flash, enable math
old_backends = """    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)"""

new_backends = """    enable_cudnn_sdp(False)
    enable_flash_sdp(False)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(True)"""

assert old_backends in text, "Backend settings not found"
text = text.replace(old_backends, new_backends)
print("✅ Fix 2: Global backends updated")

# Fix 3: Update log line to match new backend settings
old_log = '    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")'
new_log = '    log0("sdp_backends:cudnn=False flash=False mem_efficient=False math=True")'

assert old_log in text, "Backend log line not found"
text = text.replace(old_log, new_log)
print("✅ Fix 3: Backend log line updated")

dst.write_text(text)
print(f"\n✅ {dst} created from {src}")