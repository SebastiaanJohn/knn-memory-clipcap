"""COCO dataset for CLIP."""


import os
import pickle
import sys

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2Tokenizer,
)


class ClipCocoDataset(Dataset):
    """Dataset for COCO captions and CLIP embeddings."""

    def __init__(
        self,
        data_path: str,
        prefix_length: int,
        gpt2_type: str = "gpt2",
        normalize_prefix: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path (str): The path to the data file.
            prefix_length (int): The length of the prefix to be used.
            gpt2_type (str, optional): The type of GPT2 model to use. Defaults
                to "gpt2".
            normalize_prefix (bool, optional): Whether to normalize the
                prefix. Defaults to False.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        with open(data_path, "rb") as f:
            all_data = pickle.load(f)

        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption["caption"] for caption in captions_raw]

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", "rb") as f:
                (
                    self.captions_tokens,
                    self.caption2embedding,
                    self.max_seq_len,
                ) = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(
                    torch.tensor(
                        self.tokenizer.encode(caption["caption"]),
                        dtype=torch.int64,
                    )
                )
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(
                    max_seq_len, self.captions_tokens[-1].shape[0]
                )
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", "wb") as f:
                pickle.dump(
                    [
                        self.captions_tokens,
                        self.caption2embedding,
                        max_seq_len,
                    ],
                    f,
                )
        all_len = torch.tensor(
            [len(self.captions_tokens[i]) for i in range(len(self))]
        ).float()
        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max())
        )

    def pad_tokens(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad tokens to max_seq_len and create mask.

        Args:
            item (int): Index of the item.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tokens and mask respectively.
        """
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.zeros(padding, dtype=torch.int64) - 1)
            )
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat(
            (torch.ones(self.prefix_length), mask), dim=0
        )  # adding prefix mask

        return tokens, mask

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        """Get item from the dataset."""
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return tokens, mask, prefix
