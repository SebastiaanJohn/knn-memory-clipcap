"""Training script for CLIPCAP model."""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from dataset.activitynet import ActivityNetDataset
from dataset.activitynet_frame import ActivityNetFrameDataset
from dataset.coco import ClipCocoDataset
from models.clipcap import ClipCaptionModel, ClipCaptionPrefix
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import forward_with_memory, save_config, setup_logging
from validate import validation


def train(
    train_dataset : ClipCocoDataset | ActivityNetDataset
            | ActivityNetFrameDataset,
    valid_dataset : ClipCocoDataset | ActivityNetDataset
            | ActivityNetFrameDataset,
    model: ClipCaptionModel,
    args: argparse.Namespace,
    lr: float = 2e-5,
    warmup_steps: int = 5000,
    output_dir: Path = Path("."),
    output_prefix: str = "",
) -> ClipCaptionModel:
    """Train the model.

    Args:
        train_dataset (ClipCocoDataset | ActivityNetDataset
            | ActivityNetLastFrameDataset): The dataset to train on.
        valid_dataset (ClipCocoDataset | ActivityNetDataset
            | ActivityNetLastFrameDataset): The dataset to validate on.
        model (ClipCaptionModel): The model to train.
        args (argparse.Namespace): The arguments.
        lr (float, optional): The learning rate. Defaults to 2e-5.
        warmup_steps (int, optional): The warmup steps. Defaults to 5000.
        output_dir (Path, optional): The output directory. Defaults to Path(".").
        output_prefix (str, optional): The output prefix. Defaults to "".

    Returns:
        model: The trained model.
    """
    if args.use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    batch_size = args.bs
    epochs = args.epochs

    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)

    if args.use_memory:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
    else:
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=epochs * len(train_dataloader),
    )

    save_config(args)

    valid_losses = []
    for epoch in range(epochs):
        logging.info(f">>> Training epoch {epoch}")
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
                outputs, tokens = forward_with_memory(model, tokens, prefix, mask)

                if outputs is None:
                    progress.update()
                    continue
            else:
                outputs = model.forward(tokens, prefix, mask)

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
                    output_dir / f"{output_prefix}_latest.pt"
                )

        progress.close()
        if epoch % args.save_every == 0 or epoch == epochs - 1:
            torch.save(
                model.state_dict(),
                output_dir / f"{output_prefix}-{epoch:03d}.pt",
            )

        valid_loss = validation(valid_dataset, model, args, device)
        valid_losses.append(valid_loss)
        logging.info(f"Validation loss: {valid_loss}")

    # Save the model with the lowest validation loss as best model.
    best_model_idx = np.argmin(valid_losses)
    logging.info(
        f"Best model found at epoch {best_model_idx}, loss: {valid_losses[best_model_idx]}, saving...")
    best_model_path = Path(output_dir) / f"{output_prefix}-best.pt"
    shutil.copy(
        Path(output_dir) / f"{output_prefix}-{best_model_idx:03d}.pt",
        best_model_path,
    )

    logging.info("Valid losses per epoch:")
    for i, loss in enumerate(valid_losses):
        logging.info(f"Epoch {i}: {loss}")

    return model


def main(args) -> None:
    """Main training routine."""
    if args.use_video_dataset:
        if args.use_memory:
            train_dataset = ActivityNetDataset(
                args.train_path, args.bs, args.prefix_length)
            valid_dataset = ActivityNetDataset(
                args.valid_path, args.bs, args.prefix_length)
        else:
            train_dataset = ActivityNetFrameDataset(
                args.train_path, args.prefix_length, args.frame)
            valid_dataset = ActivityNetFrameDataset(
                args.valid_path, args.prefix_length, args.frame)
    else:
        train_dataset = ClipCocoDataset(
            args.train_path, args.prefix_length, normalize_prefix=args.normalize_prefix
        )
        valid_dataset = ClipCocoDataset(
            args.valid_path, args.prefix_length, normalize_prefix=args.normalize_prefix
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

    output_dir = Path(args.out_dir)

    train(
        train_dataset,
        valid_dataset,
        model,
        args,
        lr = args.lr,
        output_dir=output_dir,
        output_prefix=args.prefix,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #  Model configuration
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--memorizing_layers", type=tuple, default=(4,5))

    # Data and checkpoints
    parser.add_argument("--checkpoint", default=None, help="checkpoint to load")
    parser.add_argument("--train_path", default="data/activitynet_ViT-B_32_train_2000.pkl")
    parser.add_argument("--valid_path", default="data/activitynet_ViT-B_32_dev_first_250.pkl")
    parser.add_argument("--out_dir", default="./checkpoints")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--use_mps", dest="use_mps", action="store_true", help="Use GPU on Apple devices")
    parser.add_argument("--lr", type=float, default=2e-5)

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
    parser.add_argument(
        "--use_memory", dest="use_memory", action="store_true"
    )
    parser.add_argument(
        "--frame",
        help="Whether to use the last frame, first, middle, or all the frames of the video clip.",
        choices=("last", "first", "middle", "all"),
        default="last",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(f"logs/{args.prefix}.log")
    logging.info(f"Arguments: {args}")

    main(args)
