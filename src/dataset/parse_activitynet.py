"""Parse ActivityNet Captions dataset and save clip embeddings."""

import argparse
import logging
import pickle
from pathlib import Path

import clip
import skimage.io as io
import torch
from datasets import (
    Array2D,
    Dataset,
    Features,
    IterableDataset,
    Sequence,
    Value,
    load_dataset,
)
from PIL import Image
from tqdm import tqdm
from transformers import GPT2Tokenizer


def get_device() -> torch.device:
    """Get the device to use for computing CLIP embeddings.

    Returns:
        torch.device: The device to use.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_activitynet(
    dataset: IterableDataset,
    frames_dir: str,
    clip_model_type: str = "ViT-B/32",
    gpt2_type: str = "gpt2",
    batch_size: int = 32,
    use_all_video_clips: bool = False,
) -> Dataset:
    """Pre-process the ActivityNet Captions dataset.

    Converts a dataset of videos to a pre-processed dataset of video clips. In
    the original dataset, each video has multiple captions and the videos are
    stored as frames in a directory. In the new dataset, each video clip has
    one caption, and the video clips are stored as a tensor of frame
    embeddings.

    Args:
        dataset (IterableDataset): The ActivityNet Captions dataset.
        frames_dir (str): Path to the folder containing the frames.
        clip_model_type (str, optional): Name of the clip model to use.
            Defaults to "ViT-B/32".
        gpt2_type (str, optional): Name of the GPT-2 model variant to use.
            Defaults to "gpt2".
        batch_size (int, optional): The batch size to use when computing frame
            embeddings. Defaults to 32.
        use_all_video_clips (bool, optional): Whether to use all video clips
            from the dataset. If False, only the first video clip from each
            video is used. Defaults to False.

    Returns:
        Dataset: Dataset of video clip frame embeddings with their captions.
            Each sample in the dataset contains the following keys:
                video_id (str): The video id. Formatted as "v_xxxxxxxxxx".
                frames (torch.Tensor): The frame embeddings of the video clip.
                    Shape: [num_frames, embedding_size].
                caption (torch.Tensor): Token ids of the video clip's caption.
                    Shape: [caption_length].
            The Dataset object also contains the following attribute:
                pad_token_id (int): The token id of a special padding token
                    that can be used to pad the captions in this dataset.
    """
    device = get_device()
    logging.info(f"Using device: {device}")
    clip_model, preprocess = clip.load(
        clip_model_type, device=device, jit=False
    )
    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type, pad_token="[PAD]")

    prepr_dataset = []
    for entry in tqdm(dataset, desc="Parsing dataset"):
        video_id = entry["video_id"]
        video_frames_dir = Path(frames_dir) / video_id

        if not video_frames_dir.exists():
            logging.warning(f"Video {video_id} does not exist")
            continue

        for start, end, caption in zip(
            entry["captions_starts"],
            entry["captions_ends"],
            entry["en_captions"],
        ):
            # Get the start and end frame numbers.
            start_frame = int(start * 5) + 1
            end_frame = int(end * 5) + 1

            while not (
                Path(video_frames_dir) / f"{end_frame:06d}.jpg"
            ).exists():
                end_frame -= 1

            # Compute the frame embeddings.
            embeddings = []
            for frame_number in range(start_frame, end_frame + 1, batch_size):
                # Load the next batch of frames.
                images = [
                    preprocess(
                        Image.fromarray(
                            io.imread(
                                Path(video_frames_dir)
                                / f"{frame_number + i:06d}.jpg"
                            )
                        )
                    )
                    .unsqueeze(0)
                    .to(device)
                    for i in range(
                        min(batch_size, end_frame - frame_number + 1)
                    )
                ]

                # Compute the frame embeddings for the batch.
                with torch.no_grad():
                    embedding = clip_model.encode_image(
                        torch.cat(images, dim=0)
                    ).cpu()
                    embeddings.append(embedding)

            # Add the video clip to the new dataset.
            prepr_dataset.append(
                {
                    "video_id": video_id,
                    "frames": torch.cat(embeddings, dim=0),
                    "caption": tokenizer.encode(
                        caption, return_tensors="pt"
                    ).squeeze(0),
                }
            )

            # If we only want to use the first video clip from each video,
            # break out of the loop.
            if not use_all_video_clips:
                break

    # Construct a new HuggingFace dataset from the list of video clips.
    prepr_dataset = Dataset.from_list(
        prepr_dataset,
        features=Features(
            {
                "video_id": Value("string"),
                "frames": Array2D(shape=(None, 512), dtype="float32"),
                "caption": Sequence(feature=Value("int64")),
            }
        ),
    )
    prepr_dataset.set_format(
        type="torch", columns=["frames", "caption"], output_all_columns=True
    )
    prepr_dataset.pad_token_id = tokenizer.pad_token_id
    return prepr_dataset


def main(args: argparse.Namespace):
    """Parse ActivityNet Captions dataset and save clip embeddings.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Load the dataset.
    logging.info(f"Loading {args.split} split of ActivityNet Captions...")
    dataset = load_dataset(
        "Leyo/ActivityNet_Captions", split=f"{args.split}[0:{args.subset}]"
    )

    # Pre-process the dataset.
    logging.info(f"Computing CLIP embeddings for {args.split} split...")
    prepr_dataset = parse_activitynet(
        dataset,
        args.frames_dir,
        args.clip_model_type,
        args.gpt2_type,
        args.batch_size,
    )

    # Save pre-processed dataset in parent folder of `args.frames_dir`.
    logging.info("Saving pre-processed dataset...")
    output_dir = Path(args.frames_dir).parent
    clip_model_type = args.clip_model_type.replace("/", "_")
    with Path.open(
        output_dir
        / f"activitynet_{clip_model_type}_{args.split}_{args.subset}.pkl",
        "wb",
    ) as f:
        pickle.dump(prepr_dataset, f)

    logging.info("Done")


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create the argument parser.
    parser = argparse.ArgumentParser()

    # Define command line arguments.
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "validation", "test"),
        help="The dataset split to use.",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=2000,
        help="Number of videos to use from the split.",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="src/data/train_subset_2000",
        help="Path to the directory containing the video frames.",
    )
    parser.add_argument(
        "--clip_model_type",
        type=str,
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
        help="The CLIP model to use.",
    )
    parser.add_argument(
        "--gpt2_type",
        type=str,
        default="gpt2",
        help="The GPT-2 model variant to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use for computing CLIP embeddings.",
    )
    parser.add_argument(
        "--use_all_video_clips",
        action="store_true",
        help="Whether to use all video clips from each video.",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
