"""Parse ActivityNet Captions dataset and save clip embeddings."""

import argparse
import logging
import pickle
from pathlib import Path

import clip
import skimage.io as io
import torch
from datasets import Dataset, IterableDataset, load_dataset
from PIL import Image
from tqdm import tqdm
from transformers import (
    GPT2Tokenizer,
)


def parse_activitynet(
    dataset: IterableDataset, frames_dir: str, clip_model_type: str,  gpt_type: str = "gpt2",
) -> Dataset:
    """Parse ActivityNet Captions dataset.

    We have a dataset of videos (folder of image frames), where each video has multiple captions.
    We want to create a new dataset consititng of the videos seperated by caption,
    and each caption video should have their frames embedded by clip

    Output example:
        [
            {
                "video_id": "v-1",
                "en_caption": torch.tensor([
                    token1_id,
                    token2_id,
                    ...
                ])
                "frames": torch.tensor([
                    frame1_embedding,
                    frame2_embedding,
                    ...
                ])
            },
        ]

    Args:
        dataset (IterableDataset): The ActivityNet Captions dataset
        frames_dir (str): Path to the folder containing the frames
        clip_model_type (str): Name of the clip model to use

    Returns:
        Dataset: Dataset of videos with embedded frames
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps")
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_type, pad_token="[PAD]")

    new_dataset = []
    for entry in tqdm(dataset, desc="Parsing dataset"):
        video_id = entry["video_id"]
        video_frames_dir = Path(frames_dir) / video_id

        if not video_frames_dir.exists():
            logging.warning(f"Video {video_id} does not exist")
            continue

        for start, end, caption in zip(
            entry["captions_starts"], entry["captions_ends"], entry["en_captions"]
        ):
            start_frame = int(start * 5) + 1
            end_frame = int(end * 5) + 1

            while not (Path(video_frames_dir) / f"{end_frame:06d}.jpg").exists():
                end_frame -= 1

            embeddings = []
            for frame_number in range(start_frame, end_frame + 1):
                frame_path = Path(video_frames_dir) / f"{frame_number:06d}.jpg"

                image = io.imread(frame_path)
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = clip_model.encode_image(image).cpu()
                    embeddings.append(embedding)

            new_dataset.append(
                {
                    "video_id": video_id,
                    "en_caption": tokenizer.encode(
                        caption,
                        return_tensors="pt"
                    ),
                    "frames": torch.cat(embeddings, dim=0),
                }
            )
    
    # Return the new dataset.
    new_dataset = Dataset.from_list(new_dataset)
    new_dataset.pad_token_id = tokenizer.pad_token_id
    return new_dataset


def main(args: argparse.Namespace) -> None:
    """Parse ActivityNet Captions dataset and save clip embeddings."""
    logging.info(f"Computing clip embeddings for {args.split} split")

    dataset = load_dataset(
        "Leyo/ActivityNet_Captions",
        split=f"{args.split}[0:{args.subset}]",
    )

    new_dataset = parse_activitynet(dataset, args.frames_dir, args.clip_model_type)

    # save dataset in parent folder of frames_dir
    output_dir = Path(args.frames_dir).parent

    clip_model_name = args.clip_model_type.replace("/", "_")
    with Path.open(
        output_dir / f"activitynet_{args.split}_{clip_model_name}_{args.subset}.pkl",
        "wb",
    ) as f:
        pickle.dump(new_dataset, f)

    logging.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clip_model_type",
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
    )
    parser.add_argument(
        "--frames_dir",
        required=True,
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test"),
    )
    parser.add_argument(
        "--subset",
        default=300,
        type=int,
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]: %(message)s",
    )

    main(args)
