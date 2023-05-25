"""Evaluation script for ClipCaptioning."""

import argparse
import logging
import sys

import torch
from dataset.activitynet import ActivityNetDataset
from dataset.activitynet_frame import ActivityNetFrameDataset
from dataset.coco import ClipCocoDataset
from models.clipcap import ClipCaptionModel, ClipCaptionPrefix
from torch.nn import functional as nnf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import forward_with_memory


def fill(tensor: torch.Tensor, num_rows: int, device: torch.device) -> torch.Tensor:
    """Append rows of zeros to a tensor."""
    shape = [num_rows] + list(tensor.shape[1:])
    zeros = torch.zeros(shape, dtype=tensor.dtype)

    return torch.cat((tensor, zeros.to(device)))

def validation(
    dataset: ClipCocoDataset | ActivityNetDataset
            | ActivityNetFrameDataset,
    model: ClipCaptionModel,
    args: argparse.Namespace,
    device: torch.device
) -> float:
    """Evaluate the model.

    Args:
        dataset (ClipCocoDataset | ActivityNetDataset | ActivityNetLastFrameDataset): The dataset to use.
        model (ClipCaptionModel): The model to evaluate.
        args (argparse.Namespace): The arguments.

    Returns:
        float: The average loss over the dataset.
    """
    model = model.to(device)
    model.eval()

    eval_dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, drop_last=False
    )

    total_loss, num_captions = 0.0, 0
    with torch.no_grad():
        for (tokens, mask, prefix) in tqdm(eval_dataloader, desc="Evaluating"):
            tokens, mask, prefix = (
                tokens.to(device),
                mask.to(device),
                prefix.to(device, dtype=torch.float32),
            )

            batch_size = len(tokens)

            if args.use_memory:
                outputs, tokens = forward_with_memory(model, tokens, prefix, mask)

                if outputs is None:
                    continue
            else:
                if batch_size < args.bs:
                    tokens = fill(tokens, args.bs - batch_size, device)
                    prefix = fill(prefix, args.bs - batch_size, device)
                    mask = fill(mask, args.bs - batch_size, device)

                outputs = model.forward(tokens, prefix, mask)

            logits = outputs.logits[:batch_size, model.prefix_length - 1 : -1]
            tokens = tokens[:batch_size]

            loss = nnf.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens.flatten(),
                ignore_index=0,
                reduction = "sum"
            )

            total_loss += loss.item()
            num_captions += len(tokens)

    return total_loss / num_captions

def main(args) -> None:
    """Main function."""
    if args.use_video_dataset:
        if args.use_memory:
            dataset = ActivityNetDataset(args.data, args.bs, args.prefix_length)
        else:
            dataset = ActivityNetFrameDataset(args.data, args.prefix_length, args.frame)
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
        sys.stdout.flush()

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        logging.info("No checkpoint specified")

    if args.use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    loss = validation(dataset, model, args, device)

    logging.info(f"Validation loss: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #  Model configuration
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--memorizing_layers", type=tuple, default=(4,5))

    # Data and checkpoints
    parser.add_argument("--checkpoint", default=None, help="checkpoint to load")
    parser.add_argument("--data", default="src/data/activitynet_ViT-B_32_validation_500.pkl")
    parser.add_argument("--out_dir", default="./checkpoints")

    # Training configuration
    parser.add_argument("--bs", type=int, default=40)
    parser.add_argument("--use_mps", dest="use_mps", action="store_true", help="Use GPU on Apple devices")

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

    logging.basicConfig(level=logging.INFO)

    main(args)
