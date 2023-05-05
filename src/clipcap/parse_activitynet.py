"""Parse ActivityNet Captions dataset and save clip embeddings."""

import argparse
import logging
import pickle
from pathlib import Path

import clip
import skimage.io as io
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def parse_activitynet(
    dataset, frames_dir: str, clip_model_type: str
) -> tuple[list, list]:
    """Parse ActivityNet Captions dataset and save clip embeddings."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    all_embeddings = []
    all_captions = []

    for entry in tqdm(dataset, desc="Parsing dataset"):
        video_id = entry["video_id"]
        video_frames_dir = Path(frames_dir) / video_id

        if not video_frames_dir.exists():
            continue

        for start, end in zip(entry["captions_starts"], entry["captions_ends"]):
            start_frame = int(start * 5) + 1
            end_frame = int(end * 5) + 1

            for frame_number in range(start_frame, end_frame + 1):
                frame_path = Path(video_frames_dir) / f"{frame_number:06d}.jpg"

                if not frame_path.exists():
                    continue

                image = io.imread(frame_path)
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = clip_model.encode_image(image).cpu()

                all_embeddings.append(embedding)
                all_captions.append(entry)

    return all_embeddings, all_captions


def main(args: argparse.Namespace) -> None:
    """Parse ActivityNet Captions dataset and save clip embeddings."""
    logging.info(f"Computing clip embeddings for {args.split} split")

    dataset = load_dataset("Leyo/ActivityNet_Captions", split=args.split)
    frames_dir = Path(args.root_dir) / "frames" / args.split

    all_embeddings, all_captions = parse_activitynet(
        dataset, frames_dir, args.clip_model_type
    )

    embeddings_path = Path(args.root_dir) / "embeddings"
    embeddings_path.mkdir(parents=True, exist_ok=True)

    clip_model_name = args.clip_model_type.replace("/", "_")
    with Path.open(
        embeddings_path / f"{clip_model_name}_{args.split}_embeddings.pkl", "wb"
    ) as file:
        pickle.dump(
            {
                "clip_embedding": torch.cat(all_embeddings, dim=0),
                "captions": all_captions,
            },
            file,
        )

    logging.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clip_model_type",
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
    )
    parser.add_argument(
        "--root_dir",
        default="/Users/sebastiaan/Developer/knn-memory-clipcap/data/ActivityNet_Captions",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=("train", "validation", "test"),
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]: %(message)s",
    )

    main(args)
