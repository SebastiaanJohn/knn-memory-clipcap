"""Split the ActivityNet frames directory into train, test, and val directories."""

import argparse
import logging
import shutil
from pathlib import Path

from dataset import IterableDataset, load_dataset
from tqdm import tqdm


def split_frames_dir(dataset: IterableDataset, source_dir: str, split: str) -> None:
    """Split frames directory into train, test, and val directories.

    Args:
        dataset (IterableDataset): The dataset to split the frames directory for.
        source_dir (str): The directory containing the frames.
        split (str): The directory to split the frames into.
    """
    for entry in tqdm(dataset, desc=f"Splitting {split} frames directory"):
        video_id = entry["video_id"]
        video_frames_dir = Path(source_dir) / video_id

        if not video_frames_dir.exists():
            logging.warning(f"Video: ({video_id}) does not exist.")
            continue

        dest_videos_frames_dir = Path(source_dir).parent / split / video_id

        if not dest_videos_frames_dir.exists():
            shutil.move(video_frames_dir, dest_videos_frames_dir)


def main(args: argparse.Namespace) -> None:
    """Main function."""
    logging.info("Splitting frames directory")

    for split in ["train", "validation", "test"]:
        logging.info(f"Splitting {split} frames directory...")
        dataset = load_dataset(
            "Leyo/ActivityNet_Captions",
            split=split,
        )
        split_frames_dir(dataset, args.source_dir, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Split the ActivityNet frames directory into train, test, and val directories."
        )
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The directory containing the frames.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        filename="split_frames_dir.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )

    main(args)
