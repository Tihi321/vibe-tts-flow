"""Scratch (torchvenv only): export a BIDIRECTIONAL Qwen3 LLM (inputs_embeds -> hidden_states) from the upstream
k2-fsa/OmniVoice safetensors to ONNX, and compare numerics with the community (causal GQA) ONNX LLM.
usage: torch_ref.py <hf_upstream_dir> <out_dir>
"""
import sys, os, json
import numpy as np
import torch
from safetensors import safe_open
from transformers.models.qwen3 import Qwen3Config, Qwen3Model

hf_dir, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
cfg = json.load(open(os.path.join(hf_dir, "config.json")))
lc = {k: v for k, v in cfg["llm_config"].items() if k != "architectures"}
llm_cfg = Qwen3Config(**lc)
llm_cfg._attn_implementation = "sdpa"
llm = Qwen3Model(llm_cfg).eval()
sd = {}
with safe_open(os.path.join(hf_dir, "model.safetensors"), "pt") as f:
    for k in f.keys():
        if k.startswith("llm."):
            sd[k[4:]] = f.get_tensor(k)
res = llm.load_state_dict(sd, strict=False)
print("missing", res.missing_keys[:5], "unexpected", res.unexpected_keys[:5])


class Wrap(torch.nn.Module):
    """inputs_embeds -> hidden_states, bidirectional (all-True 4D mask, like upstream)."""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, inputs_embeds):
        B, S, _ = inputs_embeds.shape
        mask = torch.ones(B, 1, S, S, dtype=torch.bool)
        return self.m(inputs_embeds=inputs_embeds, attention_mask=mask, use_cache=False).last_hidden_state


w = Wrap(llm).eval()
x = torch.randn(1, 50, 1024)
with torch.no_grad():
    ref = w(x).numpy()
print("torch ok", ref.shape, float(np.abs(ref).mean()))
path = os.path.join(out_dir, "llm_bidir_fp32.onnx")
with torch.no_grad():
    torch.onnx.export(w, (x,), path, input_names=["inputs_embeds"], output_names=["hidden_states"],
                      dynamic_axes={"inputs_embeds": {0: "B", 1: "S"}, "hidden_states": {0: "B", 1: "S"}},
                      opset_version=20, dynamo=False)
print("exported", path)
import onnxruntime as ort
so = ort.SessionOptions()
so.log_severity_level = 3
s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
o = s.run(None, {"inputs_embeds": x.numpy()})[0]
print("onnx vs torch max abs diff", float(np.abs(o - ref).max()))
x2 = torch.randn(1, 83, 1024)
with torch.no_grad():
    r2 = w(x2).numpy()
o2 = s.run(None, {"inputs_embeds": x2.numpy()})[0]
print("dynamic shape diff", float(np.abs(o2 - r2).max()))
