"""Evaluation script for ClipCaptioning."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from dataset.activitynet import ActivityNetDataset
from dataset.activitynet_frame import ActivityNetFrameDataset
from dataset.coco import ClipCocoDataset
from evaluation.inference.inference import generate_beam
from models.clipcap import ClipCaptionModel, ClipCaptionPrefix
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer


def fill(tensor: torch.Tensor, num_rows: int, device: torch.device) -> torch.Tensor:
    """Append rows of zeros to a tensor."""
    shape = [num_rows] + list(tensor.shape[1:])
    zeros = torch.zeros(shape, dtype=tensor.dtype)

    return torch.cat((tensor, zeros.to(device)))

def generate_captions(
    dataset: ClipCocoDataset | ActivityNetDataset
            | ActivityNetFrameDataset,
    model: ClipCaptionModel,
    args: argparse.Namespace,
    output_dir: Path = Path(".")
) -> None:
    """Evaluate the model.

    Args:
        dataset (ClipCocoDataset | ActivityNetDataset
            | ActivityNetLastFrameDataset): The dataset to use.
        model (ClipCaptionModel): The model to evaluate.
        args (argparse.Namespace): The arguments.
        output_dir (str, optional): The output directory. Defaults to ".".
    """
    if args.use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    eval_dataloader = DataLoader(
        dataset, batch_size=args.bs, shuffle=False, drop_last=False
    )

    ground_truths, generated_captions = [], []
    with torch.no_grad():
        for (tokens, mask, prefix) in tqdm(eval_dataloader, desc="Evaluating"):
            tokens, mask, prefix = (
                tokens.to(device),
                mask.to(device),
                prefix.to(device, dtype=torch.float32),
            )

            batch_size = len(tokens)

            if args.use_memory:
                contains_caption = (mask == 1).any(dim=1)
                idx = contains_caption.nonzero(as_tuple=True)[0].tolist()

                prefix_projections = model.clip_project(prefix, idx).view(
                    -1, 1, model.prefix_length, model.gpt_embedding_size
                )

                if len(idx) == 0:
                    continue

                tokens, mask, prefix_projections =  \
                    tokens[idx], mask[idx], prefix_projections[idx]
            else:
                if batch_size < args.bs:
                    prefix = fill(prefix, args.bs - batch_size, device)

                prefix_projections = model.clip_project(prefix).view(
                    -1, 1, model.prefix_length, model.gpt_embedding_size
                )[:batch_size]

            mask = mask[:, args.prefix_length:]
            tokens = [t[m.bool()] for t, m in zip(tokens, mask)]
            ground_truths += [tokenizer.decode(gt) for gt in tokens]
            generated_captions += [
                generate_beam(model, tokenizer, embed=prefix_embed)[0]
                for prefix_embed in prefix_projections
            ]

    ground_truths = dict(enumerate(ground_truths))
    memory_str = "memory" if args.use_memory else "no_memory"
    filename = output_dir / f"{memory_str}_references.json"

    with filename.open('w') as fp:
        json.dump(ground_truths, fp)

    generated_captions = dict(enumerate(generated_captions))
    filename = output_dir / f"{memory_str}_captions.json"

    with filename.open('w') as fp:
        json.dump(generated_captions, fp)


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

    output_dir = Path(args.out_dir)

    generate_captions(
        dataset,
        model,
        args,
        output_dir=output_dir,
    )


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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.info(f"Arguments: {args}")

    main(args)
