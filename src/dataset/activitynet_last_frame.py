"""Dataset class for loading ActivityNet video clips efficiently."""

import argparse
import logging
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ActivityNetLastFrameDataset(Dataset):
    """Dataset for COCO captions and CLIP embeddings."""

    def __init__(
        self,
        data_path: str,
        prefix_length: int,
        gpt2_type: str = "gpt2",
        normalize_prefix: bool = False,
    ) -> None:
        """Initialize dataset.

        Args:
            data_path (str): The path to the data file.
            prefix_length (int): The length of the prefix to be used.
            gpt2_type (str, optional): The type of GPT2 model to use. Defaults
                to "gpt2".
            normalize_prefix (bool, optional): Whether to normalize the
                prefix. Defaults to False.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        with open(data_path, "rb") as f:
            all_data = pickle.load(f)

        print("Data size is %0d" % len(all_data["clip_embedding"]))
        sys.stdout.flush()

        self.prefixes = all_data["clip_embedding"]
        captions_raw = all_data["captions"]
        self.image_ids = [caption["image_id"] for caption in captions_raw]
        self.captions = [caption["caption"] for caption in captions_raw]

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", "rb") as f:
                (
                    self.captions_tokens,
                    self.caption2embedding,
                    self.max_seq_len,
                ) = pickle.load(f)
        else:
            self.captions_tokens = []
            self.caption2embedding = []
            max_seq_len = 0
            for caption in captions_raw:
                self.captions_tokens.append(
                    torch.tensor(
                        self.tokenizer.encode(caption["caption"]),
                        dtype=torch.int64,
                    )
                )
                self.caption2embedding.append(caption["clip_embedding"])
                max_seq_len = max(
                    max_seq_len, self.captions_tokens[-1].shape[0]
                )
            # self.max_seq_len = max_seq_len
            with open(f"{data_path[:-4]}_tokens.pkl", "wb") as f:
                pickle.dump(
                    [
                        self.captions_tokens,
                        self.caption2embedding,
                        max_seq_len,
                    ],
                    f,
                )
        all_len = torch.tensor(
            [len(self.captions_tokens[i]) for i in range(len(self))]
        ).float()
        self.max_seq_len = min(
            int(all_len.mean() + all_len.std() * 10), int(all_len.max())
        )

    def pad_tokens(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Pad tokens to max_seq_len and create mask.

        Args:
            item (int): Index of the item.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tokens and mask respectively.
        """
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat(
                (tokens, torch.zeros(padding, dtype=torch.int64) - 1)
            )
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat(
            (torch.ones(self.prefix_length), mask), dim=0
        )  # adding prefix mask

        return tokens, mask

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.captions_tokens)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        """Get item from the dataset."""
        tokens, mask = self.pad_tokens(item)
        prefix = self.prefixes[self.caption2embedding[item]]
        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / prefix.norm(2, -1)

        return tokens, mask, prefix

class ActivityNetLastFrameDataset(Dataset):
    """Dataset for loading the last frame of ActivityNet video clips."""

    def __init__(
        self, prepr_dataset_path: str, batch_size: int, prefix_length: int
    ) -> None:
        """Initialize an ActivityNetDataset object.

        Args:
            prepr_dataset_path (str): Path to the pre-processed ActivityNet
                pickle file. The pickle file should contain a HuggingFace
                `datasets.Dataset` object.
            batch_size (int): The batch size. Must be the same as the batch
                size used for the DataLoader.
            prefix_length (int): The length of the caption's prefix.
        """
        self.batch_size = batch_size
        self.prefix_length = prefix_length

        # Load pre-processed dataset.
        logging.info("Loading pre-processed dataset...")
        with open(prepr_dataset_path, "rb") as f:
            self.prepr_dataset = pickle.load(f)

        print(self.prepr_dataset[0])

        self.pad_token_id = 198
        self.frame_embed_dim = self.prepr_dataset.features["frames"].shape[1]
        logging.info(
            f"Dataset contains {self.prepr_dataset.num_rows} video clips."
        )


    def __len__(self) -> int:
        """Return the amount of video clip frames in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.prepr_dataset)

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the last frame of a video clip and its corresponding caption.

        Args:
            item (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple
                containing the caption, the caption mask, and the frame.
        """
        return (
            self.prepr_dataset[item]["caption"][:self.prefix_length],
            self.prepr_dataset[item]["caption_mask"][:self.prefix_length],
            self.prepr_dataset[item]["frames"][-1],
        )

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
