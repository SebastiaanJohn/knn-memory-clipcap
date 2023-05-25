"""Dataset class for loading ActivityNet video clips efficiently."""

import argparse
import logging
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ActivityNetFrameDataset(Dataset):
    """Dataset for loading specified frames of ActivityNet video clips."""

    def __init__(
        self, prepr_dataset_path: str, prefix_length: int, frame: str = "last"
    ) -> None:
        """Initialize an ActivityNetDataset object.

        Args:
            prepr_dataset_path (str): Path to the pre-processed ActivityNet
                pickle file. The pickle file should contain a HuggingFace
                `datasets.Dataset` object.
            prefix_length (int): The length of the caption's prefix.
            frame (str, optional): Which frame to use from the video clip. Can be 'first', 'last', 'middle', 'all'.
        """
        self.prefix_length = prefix_length
        self.frame = frame  # Save the frame option
        file_path = Path(prepr_dataset_path)

        # Load pre-processed dataset.
        logging.info("Loading pre-processed dataset...")
        with Path.open(file_path, "rb") as f:
            prepr_dataset = pickle.load(f)
        self.pad_token_id = 198

        # Expand the dataset according to the frame option.
        self.prepr_dataset = self._expand_dataset(prepr_dataset)

        logging.info(f"Expanded dataset contains {len(self.prepr_dataset)} frames.")

        # Get the maximum sequence length.
        self.max_seq_len = max(len(data["caption"]) for data in self.prepr_dataset)

        logging.info(f"Max sequence length: {self.max_seq_len}")

    def _expand_dataset(self, prepr_dataset: list[dict]) -> list[dict]:
        """Expands the dataset according to the frame option."""
        expanded_dataset = []
        for data in prepr_dataset:
            frames = self._extract_frames(data["frames"])
            for frame in frames:
                expanded_dataset.append({
                    "frames": frame.unsqueeze(0),
                    "caption": data["caption"],
                    "video_id": data["video_id"],
                })
        return expanded_dataset

    def _extract_frames(self, frames: torch.Tensor) -> list[torch.Tensor] | torch.Tensor:
        """Extracts the frames according to the frame option."""
        if self.frame == "all":
            return frames
        elif self.frame == "middle":
            return [frames[len(frames) // 2]]
        elif self.frame == "first":
            return [frames[0]]
        elif self.frame == "last":
            return [frames[-1]]
        else:
            raise ValueError(f"Invalid frame option: {self.frame}")

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
    dataset = ActivityNetFrameDataset(
        args.dataset_path,
        args.prefix_length,
        args.frame,
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
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the pre-processed dataset.",
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
    parser.add_argument(
        "--frame",
        help="Whether to use the last frame, first, middle, or all the frames of the video clip.",
        choices=("last", "first", "middle", "all"),
        default="last",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
