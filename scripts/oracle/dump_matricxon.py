"""Dumps matricxon's own per-layer activations for the first N decoder layers.

Run in its own (ideally memory-capped) process via scripts/run_m3_oracle_check.sh
- see scripts/README_oracle.md. Builds the real production layer classes from
app/architectures/mistral3.py, but only the first `--n-layers` of them, so peak
RSS stays a small fraction of the full 26-layer model.
"""

import argparse

import torch
from safetensors.torch import save_file

from app.architectures.layers import RMSNorm, unpermute_rope_rows
from app.architectures.mistral3 import Mistral3DecoderLayer
from app.architectures.rope import YarnRotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from scripts.oracle.common import DEFAULT_GGUF_PATH, DEFAULT_N_LAYERS, DUMP_DIR, build_input_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf-path", default=str(DEFAULT_GGUF_PATH))
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--out", default=str(DUMP_DIR / "matricxon_activations.safetensors"))
    args = parser.parse_args()

    with GGUFModelLoader(args.gguf_path, dtype=torch.bfloat16) as loader:
        meta = loader.metadata
        arch = meta.arch_key
        rope = YarnRotaryEmbedding(
            head_dim=meta.get_u32(arch("attention.key_length")),
            rope_theta=meta.get_f32(arch("rope.freq_base")),
            factor=meta.get_f32(arch("rope.scaling.factor")),
            beta_fast=meta.get_f32(arch("rope.scaling.beta_fast")),
            beta_slow=meta.get_f32(arch("rope.scaling.beta_slow")),
            original_context_length=meta.get_u32(arch("rope.scaling.original_context_length")),
            mscale=meta.get_f32(arch("rope.scaling.mscale")),
            mscale_all_dim=meta.get_f32(arch("rope.scaling.mscale_all_dim")),
        )

        # Cast up front: matricxon's real forward pass runs in float32 (Mistral3DecoderLayer's
        # nn.Linear/RMSNorm weights default to float32 and from_gguf's `.copy_()` never casts
        # the module - only the loader's transient dequant tensor is bf16). Using this tensor
        # directly for both the embedding lookup and the tied lm_head matmul without upcasting
        # would make `x` start as bf16 and silently promote to float32 partway through the
        # first layer instead - technically fine numerically, but not what production runs.
        embed_weight = loader.load_tensor("token_embd.weight").to(torch.float32)
        layers = []
        with torch.no_grad():
            n_head = meta.get_u32(arch("attention.head_count"))
            n_head_kv = meta.get_u32(arch("attention.head_count_kv"))
            for i in range(args.n_layers):
                layer = Mistral3DecoderLayer(
                    n_embd=meta.get_u32(arch("embedding_length")),
                    n_head=n_head,
                    n_head_kv=n_head_kv,
                    head_dim=meta.get_u32(arch("attention.key_length")),
                    ffn_len=meta.get_u32(arch("feed_forward_length")),
                    rms_eps=meta.get_f32(arch("attention.layer_norm_rms_epsilon")),
                )
                prefix = f"blk.{i}."
                layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
                layer.post_attention_layernorm.weight.copy_(
                    loader.load_tensor(prefix + "ffn_norm.weight")
                )
                layer.self_attn.q_proj.weight.copy_(
                    unpermute_rope_rows(loader.load_tensor(prefix + "attn_q.weight"), n_head)
                )
                layer.self_attn.k_proj.weight.copy_(
                    unpermute_rope_rows(loader.load_tensor(prefix + "attn_k.weight"), n_head_kv)
                )
                layer.self_attn.v_proj.weight.copy_(loader.load_tensor(prefix + "attn_v.weight"))
                layer.self_attn.o_proj.weight.copy_(
                    loader.load_tensor(prefix + "attn_output.weight")
                )
                layer.mlp.gate_proj.weight.copy_(loader.load_tensor(prefix + "ffn_gate.weight"))
                layer.mlp.up_proj.weight.copy_(loader.load_tensor(prefix + "ffn_up.weight"))
                layer.mlp.down_proj.weight.copy_(loader.load_tensor(prefix + "ffn_down.weight"))
                layers.append(layer.eval())

            final_norm = RMSNorm(
                meta.get_u32(arch("embedding_length")),
                meta.get_f32(arch("attention.layer_norm_rms_epsilon")),
            )
            final_norm.weight.copy_(loader.load_tensor("output_norm.weight"))

    input_ids = build_input_ids()
    activations = {}
    with torch.no_grad():
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)
        cos, sin = rope(position_ids)
        x = torch.nn.functional.embedding(input_ids, embed_weight)
        for i, layer in enumerate(layers):
            # Inlines Mistral3DecoderLayer.forward to additionally capture the
            # attn-only and mlp-only sub-outputs (debugging aid for localizing
            # where a mismatch enters - see scripts/README_oracle.md).
            residual = x
            attn_out = layer.self_attn(layer.input_layernorm(x), cos, sin)
            activations[f"attn_{i}"] = attn_out.clone().to(torch.float32)
            x = residual + attn_out

            residual = x
            mlp_out = layer.mlp(layer.post_attention_layernorm(x))
            activations[f"mlp_{i}"] = mlp_out.clone().to(torch.float32)
            x = residual + mlp_out

            activations[f"layer_{i}"] = x.clone().to(torch.float32)

        normed = final_norm(x)
        logits = torch.nn.functional.linear(normed, embed_weight)
        activations["final_norm"] = normed.to(torch.float32)
        activations["logits"] = logits.to(torch.float32)

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    save_file(activations, args.out)
    print(f"Wrote {len(activations)} tensors to {args.out}")


if __name__ == "__main__":
    main()
