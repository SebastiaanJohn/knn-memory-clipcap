"""Dataset class for loading ActivityNet video clips efficiently."""

import argparse
import logging
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ActivityNetLastFrameDataset(Dataset):
    """Dataset for loading the last frame of ActivityNet video clips."""

    def __init__(
        self, prepr_dataset_path: str, prefix_length: int
    ) -> None:
        """Initialize an ActivityNetDataset object.

        Args:
            prepr_dataset_path (str): Path to the pre-processed ActivityNet
                pickle file. The pickle file should contain a HuggingFace
                `datasets.Dataset` object.
            prefix_length (int): The length of the caption's prefix.
        """
        self.prefix_length = prefix_length

        # Load pre-processed dataset.
        logging.info("Loading pre-processed dataset...")
        with open(prepr_dataset_path, "rb") as f:
            self.prepr_dataset = pickle.load(f)
        self.pad_token_id = 198
        logging.info(
            f"Dataset contains {self.prepr_dataset.num_rows} video clips."
        )

        # We only need the last frame of each video clip.
        self.prepr_dataset = self.prepr_dataset.map(
            lambda x: {
                "frames": x["frames"][-1],
                "caption": x["caption"],
                "video_id": x["video_id"],
            },
            num_proc=1,
        )

        # Get the maximum sequence length.
        self.max_seq_len = max(len(data["caption"]) for data in self.prepr_dataset)

        logging.info(f"Max sequence length: {self.max_seq_len}")

    def pad_tokens(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad tokens to max_seq_len and create mask.

        Args:
            item (int): Index of the item.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tokens and mask respectively.
        """
        tokens = self.prepr_dataset[item]['caption']
        padding = self.max_seq_len - len(tokens)
        if padding > 0:
            tokens = torch.cat((tokens, torch.full((padding,), self.pad_token_id, dtype=torch.long)), dim=0)
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]

        # Create a mask of ones for non-padding tokens and zeros for padding tokens
        mask = [1 if token != self.pad_token_id else 0 for token in tokens]

        # Convert mask to PyTorch tensors
        mask = torch.tensor(mask, dtype=torch.float)

        # Add prefix mask
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)

        return tokens, mask


    def __len__(self) -> int:
        """Return the amount of video clip frames in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.prepr_dataset)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        """Get item from the dataset."""
        tokens, mask = self.pad_tokens(item)
        prefix = self.prepr_dataset[item]['frames']

        return tokens, mask, prefix

def main(args: argparse.Namespace) -> None:
    """Test whether data loading using a DataLoader works correctly.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Create the dataset and dataloader.
    clip_model_type = args.clip_model_type.replace("/", "_")
    dataset = ActivityNetLastFrameDataset(
        Path("src/data")
        / f"activitynet_{clip_model_type}_{args.split}_{args.subset}.pkl",
        args.batch_size,
        args.prefix_length,
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # Print a single batch for testing purposes.
    logging.info("Going through dataloader to ensure no errors occur...")
    for batch_idx, (captions, masks, frames) in enumerate(tqdm(dataloader)):
        if batch_idx == 1:
            logging.info(f"Batch {batch_idx}:")
            logging.info(f"{captions.shape=}")
            logging.info(f"{masks.shape=}")
            logging.info(f"{frames.shape=}")
            logging.info(f"{captions=}")
            logging.info(f"{masks=}")
            logging.info(f"{frames=}")
            logging.info("")


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
        default=300,
        help="Number of videos to use from the split.",
    )
    parser.add_argument(
        "--clip_model_type",
        type=str,
        default="ViT-B/32",
        choices=("RN50", "RN101", "RN50x4", "ViT-B/32"),
        help="The CLIP model to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size to use for data loading.",
    )
    parser.add_argument(
        "--prefix_length",
        type=int,
        default=20,
        help="Number of tokens to use as prefix for the caption.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers to use for data loading.",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
