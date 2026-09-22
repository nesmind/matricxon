import logging
import time
from collections.abc import Callable

import torch
from torch import nn

from app.architectures.base import GenerationCancelledError, ModelArchitecture
from app.architectures.bert_layers import BertEncoderLayer
from app.gguf.loader import GGUFModelLoader
from app.gguf.metadata import GGUFMetadata
from app.runtime.kv_cache import KVCache

logger = logging.getLogger(__name__)


class BertArchitecture(ModelArchitecture):
    """Classic (encoder-only, non-causal) BERT: learned absolute position
    embeddings + token-type embeddings, post-LN encoder layers, GELU FFN.

    Structurally separate from the decoder classes since encoder-only is
    not autoregressive: `forward()` here returns final hidden states
    (batch, seq_len, hidden_dim), never logits or a vocab projection - there
    isn't one in this GGUF (no LM head tensor), and `kv_cache` is always
    unused (a KVCache belongs to one autoregressive generation call; nothing
    here ever generates). `EmbeddingEngine` does the actual mean-pooling
    into a single embedding vector on top of this.
    """

    NAME = "bert"

    def __init__(self, metadata: GGUFMetadata, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        arch = metadata.arch_key
        self.n_embd = metadata.get_u32(arch("embedding_length"))
        self.n_head = metadata.get_u32(arch("attention.head_count"))
        self.n_layer = metadata.get_u32(arch("block_count"))
        self.ffn_len = metadata.get_u32(arch("feed_forward_length"))
        self.layer_norm_eps = metadata.get_f32(arch("attention.layer_norm_epsilon"))
        self.max_position_embeddings = metadata.get_u32(arch("context_length"))
        self.vocab_size = len(metadata.require("tokenizer.ggml.tokens"))

        self.token_embd = nn.Embedding(self.vocab_size, self.n_embd, dtype=dtype)
        self.position_embd = nn.Embedding(self.max_position_embeddings, self.n_embd, dtype=dtype)
        self.token_types = nn.Embedding(2, self.n_embd, dtype=dtype)
        self.token_embd_norm = nn.LayerNorm(self.n_embd, eps=self.layer_norm_eps, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                BertEncoderLayer(
                    self.n_embd, self.n_head, self.ffn_len, self.layer_norm_eps, dtype=dtype
                )
                for _ in range(self.n_layer)
            ]
        )

    @classmethod
    def supports(cls, metadata: GGUFMetadata) -> bool:
        return metadata.architecture == cls.NAME

    @classmethod
    def from_gguf(
        cls, loader: GGUFModelLoader, dtype: torch.dtype = torch.float32
    ) -> "BertArchitecture":
        model = cls._construct_without_init(loader.metadata, dtype=dtype)
        model._rebuild_derived_buffers()  # no-op here (no rope) - see the base class's own contract
        model._defer_materialization(loader)
        return model

    def _materialize_weights(
        self, loader: GGUFModelLoader, stop_check: Callable[[], bool] | None = None
    ) -> None:
        stage_started = time.monotonic()
        self.token_embd.weight.copy_(loader.load_tensor("token_embd.weight"))
        self.position_embd.weight.copy_(loader.load_tensor("position_embd.weight"))
        self.token_types.weight.copy_(loader.load_tensor("token_types.weight"))
        self.token_embd_norm.weight.copy_(loader.load_tensor("token_embd_norm.weight"))
        self.token_embd_norm.bias.copy_(loader.load_tensor("token_embd_norm.bias"))
        logger.debug(
            "materializing: token/position/type embeddings in %.1fs",
            time.monotonic() - stage_started,
        )

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            prefix = f"blk.{i}."
            layer.self_attn.q_proj.weight.copy_(loader.load_tensor(prefix + "attn_q.weight"))
            layer.self_attn.q_proj.bias.copy_(loader.load_tensor(prefix + "attn_q.bias"))
            layer.self_attn.k_proj.weight.copy_(loader.load_tensor(prefix + "attn_k.weight"))
            layer.self_attn.k_proj.bias.copy_(loader.load_tensor(prefix + "attn_k.bias"))
            layer.self_attn.v_proj.weight.copy_(loader.load_tensor(prefix + "attn_v.weight"))
            layer.self_attn.v_proj.bias.copy_(loader.load_tensor(prefix + "attn_v.bias"))
            layer.self_attn.o_proj.weight.copy_(loader.load_tensor(prefix + "attn_output.weight"))
            layer.self_attn.o_proj.bias.copy_(loader.load_tensor(prefix + "attn_output.bias"))
            layer.attn_output_norm.weight.copy_(
                loader.load_tensor(prefix + "attn_output_norm.weight")
            )
            layer.attn_output_norm.bias.copy_(loader.load_tensor(prefix + "attn_output_norm.bias"))
            layer.mlp.up_proj.weight.copy_(loader.load_tensor(prefix + "ffn_up.weight"))
            layer.mlp.up_proj.bias.copy_(loader.load_tensor(prefix + "ffn_up.bias"))
            layer.mlp.down_proj.weight.copy_(loader.load_tensor(prefix + "ffn_down.weight"))
            layer.mlp.down_proj.bias.copy_(loader.load_tensor(prefix + "ffn_down.bias"))
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
        x = (
            self.token_embd(input_ids)
            + self.position_embd(position_ids)
            + self.token_types(token_type_ids)
        )
        x = self.token_embd_norm(x)
        logger.debug("embedding lookup + norm: %.1fms", (time.monotonic() - stage_started) * 1000)

        for i, layer in enumerate(self.layers):
            layer_started = time.monotonic()
            x = layer(x)
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
