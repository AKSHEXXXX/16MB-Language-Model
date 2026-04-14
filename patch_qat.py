from pathlib import Path

path = Path("train_gpt_int6.py")
assert path.exists(), "train_gpt_int6.py not found. Run: python3 patch_int6.py first"

text = path.read_text()

# Replace CastedLinear with QAT version
old_linear = """class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)"""

new_linear = """def fake_quant_int6(w: Tensor) -> Tensor:
    # Simulate INT6 quantization noise during training (QAT).
    # Forward pass: quantized weights. Backward pass: straight-through (real gradients).
    # This forces weights to learn distributions that survive INT6 quantization.
    w32 = w.float()
    if w32.ndim == 2:
        clip_abs = w32.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = clip_abs / 31.0
    else:
        clip_abs = w32.abs().max().clamp(min=1e-8)
        scale = clip_abs / 31.0
    w_quantized = torch.clamp(torch.round(w32 / scale), -31.0, 31.0) * scale
    # Straight-Through Estimator: replace value in forward, keep gradient in backward
    return w + (w_quantized.to(w.dtype) - w).detach()


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    # In training mode: apply INT6 fake quantization (QAT via Straight-Through Estimator).
    # In eval mode: real INT6 PTQ is applied at export time as usual.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        w = fake_quant_int6(self.weight) if self.training else self.weight
        return F.linear(x, w.to(x.dtype), bias)"""

assert old_linear in text, "CastedLinear not found"
text = text.replace(old_linear, new_linear)

path.write_text(text)
print("✅ QAT: fake_quant_int6 added to CastedLinear")
print("   Training: weights simulate INT6 noise via STE")
print("   Export:   real INT6 PTQ applied as usual")