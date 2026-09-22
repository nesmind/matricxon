"""Dumps the real HF `Ministral3ForCausalLM` per-layer activations, for the
same first N decoder layers matricxon's own dump script computes.

Run in its own (ideally memory-capped) process - see
scripts/README_oracle.md. Never touches the vision tower or multimodal
projector (Ministral3ForCausalLM is the text-only half of the checkpoint,
registered separately from the multimodal Mistral3ForConditionalGeneration),
and only fetches the checkpoint bytes for the layers it actually builds.
"""

import argparse

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, Ministral3ForCausalLM

from scripts.oracle.common import (
    DEFAULT_N_LAYERS,
    DUMP_DIR,
    HF_REPO,
    RemoteSafetensorsReader,
    build_input_ids,
)


class LayerActivationRecorder:
    """Captures each hooked module's output under a caller-chosen name."""

    def __init__(self) -> None:
        self.activations: dict[str, torch.Tensor] = {}
        self._handles = []

    def attach(self, name: str, module: torch.nn.Module) -> None:
        def hook(_module: torch.nn.Module, _inputs: tuple, output: torch.Tensor | tuple) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            self.activations[name] = tensor.detach().clone().to(torch.float32)

        self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def load_layer_weights(model: Ministral3ForCausalLM, reader: RemoteSafetensorsReader) -> None:
    prefix = "language_model.model."
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            reader.get_dequantized(prefix + "embed_tokens.weight")
        )
        model.model.norm.weight.copy_(reader.get_dequantized(prefix + "norm.weight"))

        for i, layer in enumerate(model.model.layers):
            layer_prefix = f"{prefix}layers.{i}."
            layer.input_layernorm.weight.copy_(
                reader.get_dequantized(layer_prefix + "input_layernorm.weight")
            )
            layer.post_attention_layernorm.weight.copy_(
                reader.get_dequantized(layer_prefix + "post_attention_layernorm.weight")
            )
            layer.self_attn.q_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "self_attn.q_proj.weight")
            )
            layer.self_attn.k_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "self_attn.k_proj.weight")
            )
            layer.self_attn.v_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "self_attn.v_proj.weight")
            )
            layer.self_attn.o_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "self_attn.o_proj.weight")
            )
            layer.mlp.gate_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "mlp.gate_proj.weight")
            )
            layer.mlp.up_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "mlp.up_proj.weight")
            )
            layer.mlp.down_proj.weight.copy_(
                reader.get_dequantized(layer_prefix + "mlp.down_proj.weight")
            )

    model.tie_weights()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=HF_REPO)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--out", default=str(DUMP_DIR / "hf_oracle_activations.safetensors"))
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.repo).text_config
    config.num_hidden_layers = args.n_layers

    # float32 throughout, matching matricxon's own compute dtype (see dump_matricxon.py) -
    # this isolates GGUF-dequant-vs-fp8-dequant numerical drift instead of also mixing in
    # bf16-vs-fp32 compute-precision noise.
    model = Ministral3ForCausalLM(config).to(torch.float32).eval()

    reader = RemoteSafetensorsReader(args.repo)
    try:
        load_layer_weights(model, reader)
    finally:
        reader.close()

    recorder = LayerActivationRecorder()
    for i, layer in enumerate(model.model.layers):
        recorder.attach(f"attn_{i}", layer.self_attn)
        recorder.attach(f"mlp_{i}", layer.mlp)
        recorder.attach(f"layer_{i}", layer)
    recorder.attach("final_norm", model.model.norm)

    input_ids = build_input_ids(args.repo)
    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits
    recorder.remove()

    activations = dict(recorder.activations)
    activations["logits"] = logits.to(torch.float32)

    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    save_file(activations, args.out)
    print(f"Wrote {len(activations)} tensors to {args.out}")


if __name__ == "__main__":
    main()
