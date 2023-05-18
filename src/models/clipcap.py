"""The ClipCap model."""

import torch
import torch.nn as nn
from memory_transformer import TransformerMapper
from transformers import (
    GPT2LMHeadModel,
)


class ClipCaptionModel(nn.Module):
    """The ClipCap model."""

    def __init__(
        self,
        prefix_length: int,
        batch_size: int,
        clip_length: int | None,
        prefix_size: int,
        num_layers: int,
        num_heads: int,
        memorizing_layers: tuple[int, ...],
        max_knn_memories: int,
        num_retrieved_memories: int
    ) -> None:
        """Initialize the model.

        Args:
            prefix_length (int): The length of the prefix.
            batch_size (int): The number of batches per input.
            clip_length (int | None): The length of the prefix.
            prefix_size (int): The dimension of the prefix embeddings.
            num_layers (int): The number of layers.
            num_heads (int): The number of heads.
            memorizing_layers (tuple[int, ...]): The layers which have memories.
            max_knn_memories (int): The maximum amount of memories to keep.
            num_retrieved_memories (int): The number of memories to retrieve.
        """
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = TransformerMapper(
            prefix_size,
            self.gpt_embedding_size,
            prefix_length,
            clip_length,
            batch_size,
            num_layers,
            num_heads,
            memorizing_layers,
            max_knn_memories,
            num_retrieved_memories
        )

    # @functools.lru_cache #FIXME
    def get_dummy_token(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Create a dummy token for the start of the caption.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device to use.

        Returns:
            torch.Tensor: The dummy token.
        """
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The forward pass of the ClipCap model.

        Args:
            tokens (torch.Tensor): The tokens to predict.
            prefix (torch.Tensor): The prefix to use.
            mask (torch.Tensor | None, optional): The mask to use. Defaults to
                None.
            labels (torch.Tensor | None, optional): The labels to use.
                Defaults to None.

        Returns:
            torch.Tensor: The output of the model.
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        return self.gpt(
            inputs_embeds=embedding_cat, labels=labels, attention_mask=mask
        )


class ClipCaptionPrefix(ClipCaptionModel):
    """The ClipCap model without fine-tuning gpt."""

    def parameters(self, recurse: bool = True):
        """The parameters of the model.

        Args:
            recurse (bool, optional): Whether to recurse. Defaults to True.

        Returns:
            Iterator[Parameter]: The parameters.
        """
        return self.clip_project.parameters()

    def train(self, mode: bool = True) -> "ClipCaptionPrefix":
        """Train the model.

        Args:
            mode (bool, optional): Whether to train. Defaults to True.

        Returns:
            ClipCaptionPrefix: The model.
        """
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()

        return self
