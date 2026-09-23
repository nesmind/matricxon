import logging
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.layers import RMSNorm
from app.architectures.mistral3_layers import Mistral3DecoderLayer
from app.architectures.quantized_embedding import QuantizedEmbedding
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class LlamaArchitecture(ModelArchitecture):
    """Plain (no YaRN) GQA decoder - the "simpler special case" M10's own

    ROADMAP entry anticipated once `Mistral3TextArchitecture` existed:
    RMSNorm + grouped-query attention + SwiGLU MLP + un-scaled rotary
    embeddings, confirmed identical enough in real tensor layout (against a
    real `TinyLlama-1.1B-Chat-v1.0` GGUF pull, not assumed) to reuse
    `Mistral3DecoderLayer`/`GroupedQueryAttention`/`SwiGLUMLP` directly
    rather than duplicating them - only the RoPE class (plain
    `RotaryEmbedding`, no `rope.scaling.*` metadata exists for this
    architecture at all) and the output projection differ.

    A real `llama`-arch GGUF may or may not carry its own separate
    `output.weight` tensor: a real `TinyLlama-1.1B-Chat-v1.0` pull has one,
    but real `Llama-3.2-1B/3B-Instruct` pulls don't (`tie_word_embeddings:
    true` in their HF configs) - llama.cpp's own conversion omits
    `output.weight` entirely for those, the same way it always has for
    `mistral3` (see that architecture's own comment). Confirmed live
    (2026-09-22): loading a real `Llama-3.2-3B-Instruct-GGUF` crashed with
    `UnknownModelError: GGUF file has no tensor named 'output.weight'` -
    `tied_embeddings` (detected once, in `from_gguf`, via
    `loader.has_tensor("output.weight")`) now takes the same tied path
    `mistral3` always does instead of assuming every llama-arch file has a
    separate projection.
    """

    NAME = "llama"

    def __init__(
        self,
        metadata: GGUFMetadata,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
        tied_embeddings: bool = False,
    ) -> None:
        super().__init__()
        # When this is on, every per-layer projection goes quantized-native - q_proj/k_proj too,
        # with unpermute_rope_rows applied to their packed rows (see
        # PackedWeightLoading._load_projection) - and so does token_embd (QuantizedEmbedding).
        # `lm_head` is `output.weight` for an untied checkpoint, or that same packed embedding
        # table for a tied one (QuantizedEmbedding.as_linear), exactly like mistral3.
        self._enable_quantized_native = enable_quantized_native
        self._dtype = dtype
        self._tied_embeddings = tied_embeddings
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.n_head_kv = metadata.get_u32(arch("attention.head_count_kv"))
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.rms_eps = metadata.get_f32(arch("attention.layer_norm_rms_epsilon"))
        # Real, confirmed gap (2026-09-21): a real LLaVA-v1.6-Vicuna GGUF pull has no
        # `llama.vocab_size` metadata key at all - falls back to the real tokenizer vocab's own
        # length, the same fallback `BertArchitecture`/`NomicBertArchitecture`/`Gemma4Architecture`/
        # `Phi2Architecture` already established for exactly this situation, rather than crashing
        # on `nn.Embedding(None, ...)`.
        vocab_size = metadata.get_u32(arch("vocab_size"))
        self.vocab_size = (
            vocab_size if vocab_size is not None else len(metadata.require("tokenizer.ggml.tokens"))
        )

        # Plain floats (not tensors), so unaffected by _construct_without_init's meta-device
        # trick - kept so _rebuild_derived_buffers can recreate self.rope for real afterward.
        # See Mistral3TextArchitecture's identical _rope_kwargs for the full reasoning.
        self._rope_kwargs = dict(
            head_dim=self.head_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"))
        )
        self.rope = RotaryEmbedding(**self._rope_kwargs)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Mistral3DecoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.n_head_kv,
                    self.head_dim,
                    self.ffn_len,
                    self.rms_eps,
                    dtype=dtype,
                )
                for _ in range(self.n_layer)
            ]
        )
        self.output_norm = RMSNorm(self.n_embd, self.rms_eps, dtype=dtype)
        if not tied_embeddings:
            self.lm_head = nn.Linear(self.n_embd, self.vocab_size, bias=False, dtype=dtype)

    def _rebuild_derived_buffers(self) -> None:
        """self.rope.inv_freq is a real computed value, not a GGUF tensor from_gguf ever
        `.copy_()`'s in - see ModelArchitecture._rebuild_derived_buffers's own docstring for why
        this has to be recreated for real after _construct_without_init."""
        self.rope = RotaryEmbedding(**self._rope_kwargs)

    @property
    def kv_cache_layer_shapes(self) -> list[tuple[int, int]]:
        """Same `(n_head_kv, head_dim)` pair repeated per layer - see

        `KVCache`'s own docstring for why this is a per-layer list at all.
        """
        return [(self.n_head_kv, self.head_dim)] * self.n_layer

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls,
        loader: GGUFModelLoader,
        dtype: torch.dtype = torch.float32,
        enable_quantized_native: bool = False,
    ) -> "LlamaArchitecture":
        model = cls._construct_without_init(
            loader.metadata,
            dtype=dtype,
            enable_quantized_native=enable_quantized_native,
            tied_embeddings=not loader.has_tensor("output.weight"),
        )
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        stage_started = time.monotonic()
        enabled = self._enable_quantized_native
        self.token_embd = self._load_token_embedding(loader, self.token_embd, self._dtype, enabled)
        self.output_norm.weight.copy_(loader.load_tensor("output_norm.weight"))
        if not self._tied_embeddings:
            self.lm_head = self._load_projection(
                loader, "output.weight", self.lm_head, self._dtype, enabled
            )
        elif isinstance(self.token_embd, QuantizedEmbedding):
            self.lm_head = self.token_embd.as_linear()
        logger.debug(
            "materializing: token_embd + output_norm + lm_head in %.1fs",
            time.monotonic() - stage_started,
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.input_layernorm.weight.copy_(loader.load_tensor(prefix + "attn_norm.weight"))
            layer.post_attention_layernorm.weight.copy_(
                loader.load_tensor(prefix + "ffn_norm.weight")
            )
            layer.self_attn.q_proj = self._load_projection(
                loader,
                prefix + "attn_q.weight",
                layer.self_attn.q_proj,
                self._dtype,
                enabled,
                rope_heads=self.n_head,
            )
            layer.self_attn.k_proj = self._load_projection(
                loader,
                prefix + "attn_k.weight",
                layer.self_attn.k_proj,
                self._dtype,
                enabled,
                rope_heads=self.n_head_kv,
            )
            layer.self_attn.v_proj = self._load_projection(
                loader, prefix + "attn_v.weight", layer.self_attn.v_proj, self._dtype, enabled
            )
            layer.self_attn.o_proj = self._load_projection(
                loader, prefix + "attn_output.weight", layer.self_attn.o_proj, self._dtype, enabled
            )
            layer.mlp.gate_proj = self._load_projection(
                loader, prefix + "ffn_gate.weight", layer.mlp.gate_proj, self._dtype, enabled
            )
            layer.mlp.up_proj = self._load_projection(
                loader, prefix + "ffn_up.weight", layer.mlp.up_proj, self._dtype, enabled
            )
            layer.mlp.down_proj = self._load_projection(
                loader, prefix + "ffn_down.weight", layer.mlp.down_proj, self._dtype, enabled
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
        """`image_embeddings` (see `ModelArchitecture.forward`'s own docstring for the full
        contract): applied immediately after the normal token-embedding lookup, before any
        decoder layer runs - each `(start, embeds)` pair overwrites
        `x[:, start:start+embeds.shape[0], :]` in place with real projected image-patch
        embeddings. The input ids at those positions are placeholder values only (see
        `app.runtime.vision_fusion.build_prompt_with_images`) - their own looked-up embedding is
        computed and then immediately discarded, never influencing anything downstream.
        """
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
        if self._last_logits_only:
            x = x[:, -1:, :]
        x = self.output_norm(x)
        lm_head = getattr(self, "lm_head", None)
        if lm_head is not None:
            logits = lm_head(x)
        else:
            logits = F.linear(x.to(self.token_embd.weight.dtype), self.token_embd.weight)
        logger.debug(
            "output_norm + lm_head: %.1fms (logits shape=%s)",
            (time.monotonic() - stage_started) * 1000,
            tuple(logits.shape),
        )
        return logits
