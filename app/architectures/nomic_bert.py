import logging
import time
from collections.abc import Callable

import torch
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.nomic_bert_layers import NomicBertEncoderLayer
from app.architectures.rope import RotaryEmbedding
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class NomicBertArchitecture(ModelArchitecture):
    """`nomic-bert`: a modernized BERT variant (fused QKV, no bias, rotary
    position embeddings instead of learned absolute ones, SwiGLU FFN) -
    still encoder-only/non-causal, still post-LN, so it shares the same
    overall shape as `BertArchitecture` (token+type embeddings -> LN ->
    encoder stack -> final hidden states) but differs enough in the actual
    layer internals to warrant its own class rather than parameterizing one
    shared implementation (see `nomic_bert_layers.py`).

    No `position_embd` tensor exists in this GGUF at all (confirmed) - RoPE
    replaces it entirely, computed once here and passed to every layer, the
    same pattern `Mistral3TextArchitecture` uses.
    """

    NAME = "nomic-bert"

    def __init__(self, metadata: GGUFMetadata, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.head_dim = self.n_embd // self.n_head
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.layer_norm_eps = metadata.get_f32(arch("attention.layer_norm_epsilon"))
        self.vocab_size = len(metadata.require("tokenizer.ggml.tokens"))

        # Plain floats/ints (not tensors), so unaffected by _construct_without_init's meta-device
        # trick - kept so _rebuild_derived_buffers can recreate self.rope for real afterward. See
        # Mistral3TextArchitecture's identical _rope_kwargs for the full reasoning.
        self._rope_kwargs = dict(
            head_dim=self.head_dim, rope_theta=metadata.get_f32(arch("rope.freq_base"))
        )
        self.rope = RotaryEmbedding(**self._rope_kwargs)

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.token_types = nn.Embedding(2, self.n_embd, dtype=dtype)
        self.token_embd_norm = nn.LayerNorm(self.n_embd, eps=self.layer_norm_eps, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                NomicBertEncoderLayer(
                    self.n_embd,
                    self.n_head,
                    self.head_dim,
                    self.ffn_len,
                    self.layer_norm_eps,
                    dtype=dtype,
                )
                for _ in range(self.n_layer)
            ]
        )

    def _rebuild_derived_buffers(self) -> None:
        """self.rope.inv_freq is a real computed value, not a GGUF tensor from_gguf ever
        `.copy_()`'s in - see ModelArchitecture._rebuild_derived_buffers's own docstring for why
        this has to be recreated for real after _construct_without_init."""
        self.rope = RotaryEmbedding(**self._rope_kwargs)

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls, loader: GGUFModelLoader, dtype: torch.dtype = torch.float32
    ) -> "NomicBertArchitecture":
        model = cls._construct_without_init(loader.metadata, dtype=dtype)
        model._rebuild_derived_buffers()
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        stage_started = time.monotonic()
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.token_types.weight.copy_(loader.load_tensor("token_types.weight"))
        self.token_embd_norm.weight.copy_(loader.load_tensor("token_embd_norm.weight"))
        self.token_embd_norm.bias.copy_(loader.load_tensor("token_embd_norm.bias"))
        logger.debug(
            "materializing: token/type embeddings in %.1fs", time.monotonic() - stage_started
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.self_attn.qkv_proj.weight.copy_(loader.load_tensor(prefix + "attn_qkv.weight"))
            layer.self_attn.o_proj.weight.copy_(loader.load_tensor(prefix + "attn_output.weight"))
            layer.attn_output_norm.weight.copy_(
                loader.load_tensor(prefix + "attn_output_norm.weight")
            )
            layer.attn_output_norm.bias.copy_(loader.load_tensor(prefix + "attn_output_norm.bias"))
            layer.mlp.gate_proj.weight.copy_(loader.load_tensor(prefix + "ffn_gate.weight"))
            layer.mlp.up_proj.weight.copy_(loader.load_tensor(prefix + "ffn_up.weight"))
            layer.mlp.down_proj.weight.copy_(loader.load_tensor(prefix + "ffn_down.weight"))
            layer.layer_output_norm.weight.copy_(
                loader.load_tensor(prefix + "layer_output_norm.weight")
            )
            layer.layer_output_norm.bias.copy_(
                loader.load_tensor(prefix + "layer_output_norm.bias")
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
        del image_embeddings  # real vision support is LlamaArchitecture-only - see base.py
        _, seq_len = input_ids.shape
        logger.debug("forward start: seq_len=%d", seq_len)
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
        token_type_ids = torch.zeros_like(input_ids)  # v1: no sentence-pair inputs
        stage_started = time.monotonic()
        cos, sin = self.rope(position_ids)
        logger.debug("rope: %.1fms", (time.monotonic() - stage_started) * 1000)

        stage_started = time.monotonic()
        x = self.token_embd(input_ids) + self.token_types(token_type_ids)
        x = self.token_embd_norm(x)
        logger.debug("embedding lookup + norm: %.1fms", (time.monotonic() - stage_started) * 1000)

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            x = layer(x, cos, sin)
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
        return x
