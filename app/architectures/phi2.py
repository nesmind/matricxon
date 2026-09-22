import logging
import time
from collections.abc import Callable

import torch
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.phi2_layers import Phi2DecoderLayer
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class Phi2Architecture(ModelArchitecture):
    """Phi-2 (moondream2's text half) - real, confirmed differences from every other architecture
    here (via a real moondream2 GGUF pull's own header, not assumed): a **parallel residual**
    decoder block (one shared LayerNorm feeds both the attention and MLP branches - see
    `Phi2DecoderLayer`), **partial rotary embeddings** (RoPE applies to only the first
    `rope.dimension_count` dims of each head, not the whole head - see
    `apply_rotary_pos_emb_partial`), plain (non-grouped) multi-head attention with real biases on
    every projection, and a real `LayerNorm` (not RMSNorm) output norm with an untied, biased
    `lm_head`.

    `rope.freq_base`/`vocab_size` have no real metadata key on this GGUF (confirmed) - falls back
    to Phi-2's own real HF default (10000.0) and to `len(tokenizer.ggml.tokens)` respectively, the
    same fallback `BertArchitecture`/`NomicBertArchitecture`/`Gemma4Architecture` already use for
    the latter.
    """

    NAME = "phi2"

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> None:
        super().__init__()
        # Unlike mistral3/llama/gemma4, q_proj/k_proj/v_proj here are sliced at materialize time
        # from one real *fused* `attn_qkv` GGUF tensor (see _materialize_weights) - splitting a
        # packed/quantized tensor's raw bytes the same way would need real per-quant-type-aware
        # byte-range slicing, extra complexity not attempted this round (real, scoped-out follow-
        # up, not silently dropped). o_proj/up_proj/down_proj/lm_head - each its own real,
        # standalone GGUF tensor - go quantized-native when this is on.
        self._enable_quantized_native = enable_quantized_native
        self._dtype = dtype
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.head_dim = self.n_embd // self.n_head
        self.rope_dim = metadata.get_u32(arch("rope.dimension_count"))
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.layer_norm_eps = metadata.get_f32(arch("attention.layer_norm_epsilon"))
        self.vocab_size = len(metadata.require("tokenizer.ggml.tokens"))

        # Plain floats (not tensors), so unaffected by _construct_without_init's meta-device
        # trick - kept so _rebuild_derived_buffers can recreate self.rope for real afterward. See
        # Mistral3TextArchitecture's identical _rope_kwargs for the full reasoning. head_dim is
        # deliberately rope_dim here, not self.head_dim - see RotaryEmbedding's own docstring:
        # inv_freq only ever depends on the width actually being rotated.
        self._rope_kwargs = dict(
            head_dim=self.rope_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"), 10000.0)
        )
        self.rope = RotaryEmbedding(**self._rope_kwargs)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Phi2DecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.head_dim,
                    self.rope_dim,
                    self.ffn_len,
                    self.layer_norm_eps,
                    dtype=dtype,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.output_norm = nn.LayerNorm(self.n_embd, eps=self.layer_norm_eps, dtype=dtype)
        self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=True, dtype=dtype)

    def _rebuild_derived_buffers(self) -> None:
        """self.rope.inv_freq is a real computed value, not a GGUF tensor from_gguf ever
        `.copy_()`'s in - see ModelArchitecture._rebuild_derived_buffers's own docstring for why
        this has to be recreated for real after _construct_without_init."""
        self.rope = RotaryEmbedding(**self._rope_kwargs)

    @property
    def kv_cache_layer_shapes(self) -> list[tuple[int, int]]:
        """Same `(n_head, head_dim)` pair repeated per layer - plain MHA (no GQA: this
        architecture's real `attention.head_count_kv` always equals `attention.head_count`), see
        `KVCache`'s own docstring for why this is a per-layer list at all."""
        return [(self.n_head, self.head_dim)] * self.n_layer

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> "Phi2Architecture":
        model = cls._construct_without_init(
            loader.metadata, dtype=dtype, enable_quantized_native=enable_quantized_native
        )
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        stage_started = time.monotonic()
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        self.output_norm.bias.copy_(loader.load_tensor("output_norm.bias"))
        self.lm_head = self._load_projection(
            loader,
            "output.weight",
            self.lm_head,
            self._dtype,
            self._enable_quantized_native,
            bias_tensor_name="output.bias",
        )
        logger.debug(
            "materializing: token_embd + output_norm + lm_head in %.1fs",
            time.monotonic() - stage_started,
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
            layer.input_layernorm.bias.copy_(loader.load_tensor(prefix + "attn_norm.bias"))

            # attn_qkv is one real fused (3*n_embd, n_embd) tensor (confirmed against a real
            # moondream2 GGUF) - split into three equal n_embd-row thirds for this model's own
            # separate q/k/v projections (see Phi2Attention's own docstring for why those stay
            # separate rather than also fusing them here). Always today's exact path - see
            # __init__'s own comment on why splitting a packed tensor's raw bytes isn't attempted
            # this round.
            qkv_weight = loader.load_tensor(prefix + "attn_qkv.weight")
            qkv_bias = loader.load_tensor(prefix + "attn_qkv.bias")
            n_embd = self.n_embd
            layer.self_attn.q_proj.weight.copy_(qkv_weight[0:n_embd])
            layer.self_attn.k_proj.weight.copy_(qkv_weight[n_embd : 2 * n_embd])
            layer.self_attn.v_proj.weight.copy_(qkv_weight[2 * n_embd : 3 * n_embd])
            layer.self_attn.q_proj.bias.copy_(qkv_bias[0:n_embd])
            layer.self_attn.k_proj.bias.copy_(qkv_bias[n_embd : 2 * n_embd])
            layer.self_attn.v_proj.bias.copy_(qkv_bias[2 * n_embd : 3 * n_embd])

            enabled = self._enable_quantized_native
            layer.self_attn.o_proj = self._load_projection(
                loader,
                prefix + "attn_output.weight",
                layer.self_attn.o_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "attn_output.bias",
            )
            layer.mlp.up_proj = self._load_projection(
                loader,
                prefix + "ffn_up.weight",
                layer.mlp.up_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "ffn_up.bias",
            )
            layer.mlp.down_proj = self._load_projection(
                loader,
                prefix + "ffn_down.weight",
                layer.mlp.down_proj,
                self._dtype,
                enabled,
                bias_tensor_name=prefix + "ffn_down.bias",
            )
            logger.debug(
                "loading weights: layer %d/%d (not computing yet) in %.1fs",
                i + 1,
                self.n_layer,
                time.monotonic() - layer_started,
            )
            if stop_check is not None and stop_check():
                raise GenerationCancelledError(f"stopped loading weights at layer {i + 1}")

    def _forward_impl(
        self,
        input_ids: torch.Tensor,
        kv_cache: KVCache | None = None,
        position_ids: torch.Tensor | None = None,
        stop_check: Callable[[], bool] | None = None,
        image_embeddings: list[tuple[int, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        """`image_embeddings` - see `LlamaArchitecture._forward_impl`'s own docstring for the full
        contract; identical handling here (this is the second real vision-fusion-capable
        architecture - see `app.models.capabilities._VISION_FUSION_ARCHITECTURES`)."""
        _, seq_len = input_ids.shape
        logger.debug("forward start: seq_len=%d", seq_len)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        stage_started = time.monotonic()
        cos, sin = self.rope(position_ids)
        logger.debug("rope: %.1fms", (time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        x = self.token_embd(input_ids)
        if image_embeddings:
            for start, embeds in image_embeddings:
                x[:, start : start + embeds.shape[0], :] = embeds.to(x.dtype)
            logger.debug(
                "spliced %d real image-embedding span(s) into input", len(image_embeddings)
            )
        logger.debug("embedding lookup: %.1fms", (time.monotonic() - stage_started) * 1000)

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            x = layer(x, cos, sin, kv_cache, i)
            logger.debug(
                "layer %d/%d: %.1fms (seq_len=%d)",
                i + 1,
                self.n_layer,
                (time.monotonic() - layer_started) * 1000,
                seq_len,
            )
            if stop_check is not None and stop_check():
                logger.info(
                    "generation stop requested - cancelling after layer %d/%d", i + 1, self.n_layer
                )
                raise GenerationCancelledError(f"stopped after layer {i + 1}/{self.n_layer}")

        stage_started = time.monotonic()
        x = self.output_norm(x)
        logits = self.lm_head(x)
        logger.debug(
            "output_norm + lm_head: %.1fms (logits shape=%s)",
            (time.monotonic() - stage_started) * 1000,
            tuple(logits.shape),
        )
        return logits
