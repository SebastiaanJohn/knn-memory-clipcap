"""Create a subset of the dataset."""

import argparse
import logging
import shutil
from pathlib import Path

from datasets import IterableDataset, load_dataset
from tqdm import tqdm


def create_subset(
    dataset: IterableDataset, source_dir: str, args: argparse.Namespace
) -> None:
    """Split frames directory into train, test, and val directories.

    Args:
        dataset (IterableDataset): The dataset to split the frames directory for.
        source_dir (str): The directory containing the frames.
        args (argparse.Namespace): The command line arguments.
    """
    output_dir = f"{args.split}_subset_{args.subset_size}"
    Path(source_dir).parent.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(dataset, desc=f"Extracting videos to {output_dir}"):
        video_id = entry["video_id"]
        video_frames_dir = Path(source_dir) / video_id

        if not video_frames_dir.exists():
            logging.warning(f"Video: ({video_id}) does not exist.")
            continue

        dest_videos_frames_dir = Path(source_dir).parent / output_dir / video_id

        if not dest_videos_frames_dir.exists():
            shutil.copytree(
                video_frames_dir, dest_videos_frames_dir, dirs_exist_ok=True
            )

    logging.info(f"Finished extracting videos to {output_dir}.")


def main(args: argparse.Namespace) -> None:
    """Main function."""
    logging.info(f"Extracting {args.subset_size} videos from {args.split} split.")
    dataset = load_dataset(
        "Leyo/ActivityNet_Captions",
        split=f"{args.split}[0:{args.subset_size}]",
    )
    create_subset(dataset, args.source_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Create a subset of the dataset."))

    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The directory containing the frames.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="The split of the dataset to create a subset of.",
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=300,
        help="The size of the subset to create.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]: %(message)s",
    )

    main(args)
