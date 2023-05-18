"""Training script for CLIPCAP model."""

import argparse
import logging
import os
import sys

import torch
from dataset.activitynet import ActivityNetDataset
from dataset.coco import ClipCocoDataset
from models.clipcap import ClipCaptionModel, ClipCaptionPrefix
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from utils import save_config


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
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.checkpoint:
        model.load_state_dict(
            torch.load(args.checkpoint, map_location=torch.device("cpu"))
        )
        print(f"loading model from {args.checkpoint}")

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

            if args.use_memory:
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


def main(args) -> None:
    """Main training routine."""
    if args.use_video_dataset:
        dataset = ActivityNetDataset(args.data, args.bs, args.prefix_length)
    else:
        dataset = ClipCocoDataset(
            args.data, args.prefix_length, normalize_prefix=args.normalize_prefix
        )

    if args.only_prefix:
        model = ClipCaptionPrefix(
            args.prefix_length,
            batch_size = args.bs,
            clip_length=args.prefix_length_clip,
            prefix_size=args.prefix_dim,
            num_layers=args.num_layers,
            num_heads = args.num_heads,
            memorizing_layers = args.memorizing_layers,
            max_knn_memories = args.max_knn_memories,
            num_retrieved_memories = args.num_retrieved_memories
        )
        logging.info("Train only prefix")
    else:
        model = ClipCaptionModel(
            args.prefix_length,
            batch_size = args.bs,
            clip_length=args.prefix_length_clip,
            prefix_size=args.prefix_dim,
            num_layers=args.num_layers,
            num_heads = args.num_heads,
            memorizing_layers = args.memorizing_layers,
            max_knn_memories = args.max_knn_memories,
            num_retrieved_memories = args.num_retrieved_memories
        )
        logging.info("Train both prefix and GPT")
        sys.stdout.flush()

    train(
        dataset,
        model,
        args,
        output_dir=args.out_dir,
        output_prefix=args.prefix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #  Model configuration
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--memorizing_layers", type=tuple, default=(4,5))
    parser.add_argument("--use_memory", action="store_true")

    # Data and checkpoints
    parser.add_argument("--checkpoint", default=None, help="checkpoint to load")
    parser.add_argument("--data", default="src/data/activitynet_ViT-B_32_train_300.pkl")
    parser.add_argument("--out_dir", default="./checkpoints")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=1)

    # Transformer memory configuration
    parser.add_argument("--max_knn_memories", type=int, default=64000)
    parser.add_argument("--num_retrieved_memories", type=int, default=32)

    # Prefix configuration
    parser.add_argument("--prefix", default="coco_prefix", help="prefix for saved filenames")
    parser.add_argument("--prefix_dim", type=int, default=512)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--prefix_length_clip", type=int, default=10)
    parser.add_argument(
        "--normalize_prefix", dest="normalize_prefix", action="store_true"
    )
    parser.add_argument(
        "--only_prefix", dest="only_prefix", action="store_true"
    )

    # Dataset configuration
    parser.add_argument(
        "--use_video_dataset", dest="use_video_dataset", action="store_true"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Arguments: {args}")

    main(args)
