"""The memorizing transformer module."""

import torch
from memorizing_transformers_pytorch import MemorizingTransformer
from torch import nn


class MemoryTransformer(nn.Module):
    """The memorizing transformer module."""
    def __init__(
        self,
        dim_self: int,
        num_heads: int,
        num_layers: int,
        batch_size: int,
        memorizing_layers: tuple[int, ...],
        max_knn_memories: int,
        num_retrieved_memories: int
    ) -> None:
        """Initialize the memorizing transformer.

        Args:
            dim_self (int): The dimension of the self-attention.
            num_heads (int): The number of heads.
            num_layers (int): The number of layers.
            batch_size (int): The batch size.
                Each batch keeps track of its own memories.
            memorizing_layers (tuple[int, ...]): The layers which have memories.
            max_knn_memories (int): The maximum amount of memories to keep.
            num_retrieved_memories (int): The number of memories to retrieve.
        """
        super(MemoryTransformer, self).__init__()
        self.model = MemorizingTransformer(
            dim = dim_self,
            dim_head = dim_self // num_heads,
            depth = num_layers,
            memorizing_layers = memorizing_layers,
            max_knn_memories = max_knn_memories,
            num_retrieved_memories = num_retrieved_memories,
        )
        self.knn_memories = self.model.create_knn_memories(batch_size = batch_size)

    def forward(self, x: torch.Tensor, batch_indices: list | None) -> torch.Tensor:
        """The forward pass.

        Args:
            x (torch.Tensor): The input tensor.
            batch_indices (list | None): The batch indices for which the
                memories must be cleared at the end of this forward pass.
                If None, all memories will be cleared.

        Returns:
            torch.Tensor: The output tensor.
        """
        x = self.model(x, self.knn_memories)
        self.knn_memories.clear_memory(batch_indices)
        return x

class TransformerMapper(nn.Module):
    """A transformer mapper module."""

    def __init__(
        self,
        dim_clip: int,
        dim_embedding: int,
        prefix_length: int,
        clip_length: int,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        memorizing_layers: tuple[int, ...],
        max_knn_memories: int,
        num_retrieved_memories: int
    ) -> None:
        """Initialize the transformer mapper.

        Args:
            dim_clip (int): The dimension of the clip embeddings.
            dim_embedding (int): The dimension of the gpt embeddings.
            prefix_length (int): The length of the prefix.
            clip_length (int): The length of the prefix.
            batch_size (int): The number of batches per input.
            num_layers (int, optional): The number of layers.
            num_heads (int): The number of heads.
            memorizing_layers (tuple[int, ...]): The layers which have memories.
            max_knn_memories (int): The maximum amount of memories to keep.
            num_retrieved_memories (int): The number of memories to retrieve.
        """
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = MemoryTransformer(
            dim_embedding, num_heads, num_layers, batch_size,
            memorizing_layers, max_knn_memories, num_retrieved_memories
        )
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(
            torch.randn(prefix_length, dim_embedding), requires_grad=True
        )

    def forward(
        self,
        x: torch.Tensor,
        batch_indices: list | None = None
    ) -> torch.Tensor:
        """The forward pass.

        Args:
            x (torch.Tensor): The input tensor.
            batch_indices (optional, list | None): The batch indices for which the
                memories must be cleared at the end of this forward pass.

        Returns:
            torch.Tensor.
        """
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(
            x.shape[0], *self.prefix_const.shape
        )
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix, batch_indices)[:, self.clip_length :]

        return out
