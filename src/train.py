"""Training script for CLIPCAP model."""

import argparse
import json
import os
import pickle
import sys
import torch
import torch.nn as nn

from dataset.activitynet import ActivityNetDataset
from memorizing_transformers_pytorch import MemorizingTransformer

from torch.nn import functional as nnf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AdamW,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
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
        self.bs = batch_size
        self.model = MemorizingTransformer(
            dim = dim_self,
            dim_head = dim_self // num_heads,
            depth = num_layers,
            memorizing_layers = memorizing_layers,
            max_knn_memories = max_knn_memories,
            num_retrieved_memories = num_retrieved_memories,
        )

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
        with self.model.knn_memories_context(batch_size = self.bs) as knn_memories:
            x = self.model(x, knn_memories)
            knn_memories.clear_memory(batch_indices)
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
            x (torch.Tensor).
            batch_indices (optional, list | None). Defaults to None
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


def save_config(args: argparse.Namespace) -> None:
    """Save the config to a file."""
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, "w") as outfile:
        json.dump(config, outfile)


def load_model(
    config_path: str, epoch_or_latest: str | int = "_latest"
) -> tuple[ClipCaptionModel, argparse.ArgumentParser]:
    """Load the model from a config file.

    Args:
        config_path (str): The path to the config file.
        epoch_or_latest (str | int, optional): The epoch to load. Defaults to
            "_latest".

    Returns:
        tuple[ClipCaptionModel, argparse.ArgumentParser]: The model and the
            parser.
    """
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(
        args.out_dir, f"{args.prefix}{epoch_or_latest}.pt"
    )
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device("cpu"))
        )
    else:
        print(f"{model_path} is not exist")

    return model, parser


def train(
    dataset : ClipCocoDataset | ActivityNetDataset,
    model: ClipCaptionModel,
    args: argparse.Namespace,
    lr: float = 2e-5,
    warmup_steps: int = 5000,
    output_dir: str = ".",
    output_prefix: str = "",
) -> ClipCaptionModel:
    """Train the model.

    Args:
        dataset (ClipCocoDataset | ActivityNetDataset): The dataset to use.
        model (ClipCaptionModel): The model to train.
        args (argparse.Namespace): The arguments.
        lr (float, optional): The learning rate. Defaults to 2e-5.
        warmup_steps (int, optional): The warmup steps. Defaults to 5000.
        output_dir (str, optional): The output directory. Defaults to ".".
        output_prefix (str, optional): The output prefix. Defaults to "".

    Returns:
        model: The trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    if args.use_video_dataset:
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
    else:
        train_dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(train_dataloader),
    )

    save_config(args)

    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)
        for i, (tokens, mask, prefix) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, mask, prefix = (
                tokens.to(device),
                mask.to(device),
                prefix.to(device, dtype=torch.float32),
            )

            if args.use_video_dataset:
                # Compute the batch indices of those frames
                # that are the last in a sequence.
                contains_caption = (mask == 1).any(dim=1)
                idx = contains_caption.nonzero(as_tuple=True)[0].tolist()

                # Compute the forward pass of the TransformerMapper
                # on all frames in the batch, thereby storing new memories.
                # Use the batch indices to clear memories of videos
                # after generating a prefix for their last frame.
                prefix_projections = model.clip_project(prefix, idx).view(
                    -1, model.prefix_length, model.gpt_embedding_size
                )

                # Continue the forward and backward pass of the ClipCaptionModel
                # with only those frames which are the last in a sequence
                # and therefore have a caption. If there are none, continue.

                if len(idx) == 0:
                    progress.update()
                    continue

                tokens, mask, prefix, prefix_projections =  \
                    tokens[idx], mask[idx], prefix[idx], prefix_projections[idx]

                embedding_text = model.gpt.transformer.wte(tokens)
                embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
                outputs = model.gpt(inputs_embeds=embedding_cat, attention_mask=mask)
            else:
                outputs = model(tokens, prefix, mask)

            logits = outputs.logits[:, model.prefix_length - 1 : -1]
            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.flatten(),
                ignore_index=0,
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.set_postfix({"loss": loss.item()})
            progress.update()
            if (i + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )
        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
            )

    return model


def main() -> None:
    """Main training routine."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", default="./data/coco/oscar_split_train.pkl")
    parser.add_argument("--out_dir", default="./checkpoints")
    parser.add_argument(
        "--prefix", default="coco_prefix", help="prefix for saved filenames"
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument(
        "--only_prefix", dest="only_prefix", action="store_true"
    )
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    parser.add_argument(
        "--use_video_dataset", dest="use_video_dataset", action="store_true"
    )
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--memorizing_layers", type=tuple, default=(4,5))
    parser.add_argument("--max_knn_memories", type=int, default=64000)
    parser.add_argument("--num_retrieved_memories", type=int, default=32)

    args = parser.parse_args()

    prefix_length = args.prefix_length

    if args.use_video_dataset:
        dataset = ActivityNetDataset(args.data, args.bs, prefix_length)
    else:
        dataset = ClipCocoDataset(
            args.data, prefix_length, normalize_prefix=args.normalize_prefix
        )

    prefix_dim = 512

    if args.only_prefix:
        model = ClipCaptionPrefix(
            prefix_length,
            batch_size = args.bs,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            num_heads = args.num_heads,
            memorizing_layers = args.memorizing_layers,
            max_knn_memories = args.max_knn_memories,
            num_retrieved_memories = args.num_retrieved_memories
        )
        print("Train only prefix")
    else:
        model = ClipCaptionModel(
            prefix_length,
            batch_size = args.bs,
            clip_length=args.prefix_length_clip,
            prefix_size=prefix_dim,
            num_layers=args.num_layers,
            num_heads = args.num_heads,
            memorizing_layers = args.memorizing_layers,
            max_knn_memories = args.max_knn_memories,
            num_retrieved_memories = args.num_retrieved_memories
        )
        print("Train both prefix and GPT")
        sys.stdout.flush()

    train(
        dataset,
        model,
        args,
        output_dir=args.out_dir,
        output_prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
