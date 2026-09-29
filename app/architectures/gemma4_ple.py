"""Gemma4's Per-Layer Embeddings (PLE) - a real, active mechanism confirmed against a real
`google/gemma-4-E2B-it` GGUF (2026-09-29, `unsloth/gemma-4-E2B-it-GGUF`): a huge auxiliary
embedding table (`per_layer_token_embd.weight`, one row per vocab entry, packed
`n_layer * per_layer_dim` wide - 50.5% of that checkpoint's own real total parameters) feeds a
small residual signal into *every* decoder layer, on top of the ordinary token embedding. Ported
from real HF `transformers` source (`transformers/models/gemma4/modular_gemma4.py`,
`Gemma4TextModel.get_per_layer_inputs`/`project_per_layer_inputs`) and cross-checked against
real llama.cpp GGUF-inference source (`src/models/gemma4.cpp`,
`build_inp_per_layer`/`project_per_layer_inputs`) - both agree on the exact math below.

Two components combine per layer: a "token identity" term (this token's own row of the packed
per-layer table) and a "context" term (the main hidden state projected down + renormalized) -
`(context + token_identity) / sqrt(2)`. `Gemma4Architecture._forward_impl` calls this once per
model forward (not once per layer) and slices the result per layer index.
"""

import torch
from torch import nn

from app.architectures.layers import RMSNorm


class Gemma4PerLayerEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_layer: int,
        per_layer_dim: int,
        n_embd: int,
        rms_eps: float,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.n_layer = n_layer
        self.per_layer_dim = per_layer_dim
        self.embed_tokens_per_layer = nn.Embedding(vocab_size, n_layer * per_layer_dim, dtype=dtype)
        self.per_layer_model_projection = nn.Linear(
            n_embd, n_layer * per_layer_dim, bias=False, dtype=dtype
        )
        self.per_layer_projection_norm = RMSNorm(per_layer_dim, rms_eps, dtype=dtype)
        self._embed_scale = per_layer_dim**0.5
        self._proj_scale = n_embd**-0.5
        self._combine_scale = 2.0**-0.5

    def forward(self, input_ids: torch.Tensor, scaled_inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Returns `(batch, seq, n_layer, per_layer_dim)` - `Gemma4Architecture` slices
        `[:, :, i, :]` for layer `i`. `scaled_inputs_embeds` is the main token embedding *after*
        its own `sqrt(n_embd)` scale (real HF passes the same already-scaled `inputs_embeds`)."""
        token_identity = self.embed_tokens_per_layer(input_ids) * self._embed_scale
        token_identity = token_identity.reshape(*input_ids.shape, self.n_layer, self.per_layer_dim)

        context = self.per_layer_model_projection(scaled_inputs_embeds) * self._proj_scale
        context = context.reshape(
            *scaled_inputs_embeds.shape[:-1], self.n_layer, self.per_layer_dim
        )
        context = self.per_layer_projection_norm(context)

        return (context + token_identity) * self._combine_scale
