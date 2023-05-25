"""Dataset class for loading ActivityNet video clips efficiently."""

import argparse
import logging
import pickle

import prtpy
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ActivityNetDataset(Dataset):
    """Dataset for loading ActivityNet video clips and captions."""

    def __init__(
        self, prepr_dataset_path: str, batch_size: int, prefix_length: int
    ) -> None:
        r"""Initialize an ActivityNetDataset object.

        The data is layed out in a horizontal stack of frames, where the frames
        of a video clip are stacked horizontally. The vertical dimension is
        created automatically by the DataLoader, and represents the batch
        dimension. The __getitem__ function will return a single entry from
        the stack.

        Example illustration of a horizontal stack of frames:
                        Step1  Step2  Step3  Step4  Step5  Step6  Step7
                          v      v      v      v      v      v      v
                     / [F1-V1, F2-V1, F3-V1, F4-V1, F5-V1, F1-V4, F2-V4]
        Batch size -+  [F1-V2, F2-V2, F3-V2, F1-V5, F2-V5, F3-V5, F4-V5]
                     \ [F1-V3, F1-V6, F2-V6, F3-V6, empty, empty, empty]
        - Fx-Vy means frame x of video clip y.
        - In this illustration, V1 has 5 frames, V2 has 3 frames, V3 has 1
          frame, V4 has 2 frames, V5 has 4 frames, and V6 has 3 frames.
        - The indices of the frames in the horizontal stack are layed out as
          follows. This is done to ensure that the frames are loaded in the
          correct order by the DataLoader.
                          Step1  Step2  Step3  Step4  Step5  Step6  Step7
                            v      v      v      v      v      v      v
                       / [  0  ,   3  ,   6  ,   9  ,  12  ,  15  ,  18  ]
          Batch size -+  [  1  ,   4  ,   7  ,  13  ,  16  ,  19  ,  22  ]
                       \ [  2  ,  10  ,  17  ,  20  ,  23  ,  24  ,  25  ]
        - We would like the total number of batches to be the least number
          possible. This problem is equivalent to the multiway partitioning
          problem (https://en.wikipedia.org/wiki/Multiway_number_partitioning),
          where we want to minimize the largest sum. This problem is NP-hard,
          but we use the prtpy implementation of the Multifit algorithm, which
          can approximate the optimal solution in paseuo-polynomial time. The
          approximation is guaranteed to be within a factor of 13/11 of the
          optimal solution.

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
        self.pad_token_id = 198
        self.frame_embed_dim = self.prepr_dataset.features["frames"].shape[1]
        logging.info(
            f"Dataset contains {self.prepr_dataset.num_rows} video clips."
        )

        # Count total number of frames in the dataset.
        logging.info("Calculating number of frames and caption lengths...")
        self.num_frames = []
        self.caption_lens = []
        for video_clip in tqdm(self.prepr_dataset):
            self.num_frames.append(video_clip["frames"].shape[0])
            self.caption_lens.append(video_clip["caption"].shape[0])
        total_frames = sum(self.num_frames)
        logging.info(f"Total frames in dataset: {total_frames}")

        # Solve the multiway partitioning problem.
        logging.info("Dividing frames optimally over batches...")
        bins = prtpy.partition(
            algorithm=prtpy.partitioning.multifit,
            numbins=self.batch_size,
            items=dict(enumerate(self.num_frames)),
        )
        self.steps = max(
            sum(
                self.num_frames[video_clip_idx]
                for video_clip_idx in video_clip_idcs
            )
            for video_clip_idcs in bins
        )
        logging.info(f"Number of batches: {self.steps}")
        if len(bins) < self.batch_size:
            logging.warning(
                f"Number of bins ({len(bins)}) is less than batch size "
                f"({self.batch_size}). This means that all batches will "
                "contain some empty frames, which is not efficient. Please "
                f"lower the batch size to {len(bins)}."
            )

        # Create horizontal stack of frames.
        # hor_stack[i][j] is a tuple containing:
        # - int: The index of the video clip in the dataset.
        # - int: The index of the frame within the video clip.
        # If the video clips are exhausted, hor_stack[i][j] is None.
        logging.info("Creating horizontal stack of frames...")
        self.hor_stack = []
        for bin in tqdm(bins):
            layer = []
            for video_clip_idx in bin:
                for frame_idx in range(self.num_frames[video_clip_idx]):
                    layer.append((video_clip_idx, frame_idx))
            layer += [None] * (self.steps - len(layer))
            self.hor_stack.append(layer)

        # Get the longest caption length within the current batch.
        logging.info("Getting max caption length for each step...")
        self.max_caption_len = []
        for j in tqdm(range(self.steps)):
            max_caption_len = 1
            for i in range(self.batch_size):
                if self.hor_stack[i][j] is None:
                    continue
                video_clip_idx, frame_idx = self.hor_stack[i][j]
                if frame_idx == self.num_frames[video_clip_idx] - 1:
                    caption_len = self.caption_lens[video_clip_idx]
                    if caption_len > max_caption_len:
                        max_caption_len = caption_len
            self.max_caption_len.append(max_caption_len)

    def __len__(self) -> int:
        """Return the amount of video clip frames in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return self.steps * self.batch_size

    def __getitem__(
        self, item: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single frame and its corresponding video clip's caption.

        Args:
            item (int): Index of the frame within the horizontal stack of
                frames.

        Returns:
            Tuple containing:
                torch.Tensor: The caption token ids, IF AND ONLY IF the frame
                    is the last one of the video clip. If the frame is not the
                    last one, a tensor filled with padding tokens is returned
                    with the same shape as the longest caption in the current
                    batch. If none of the video clips in the current batch
                    have a caption, a tensor with a single padding token is
                    returned. The padding token id can be retrieved from the
                    `pad_token_id` attribute.
                    Shape: [max_caption_len] or [1].
                torch.Tensor: The caption mask, IF AND ONLY IF the frame is
                    the last one of the video clip. If the frame is not the
                    last one, a tensor filled with zeros is returned with the
                    same shape as the longest caption in the current batch +
                    the prefix length. If none of the video clips in the
                    current batch have a caption, a zero tensor with shape
                    1 + the prefix length is returned. The prefix length can be
                    retrieved from the `prefix_length` attribute. A value in
                    the mask is 1 if the corresponding token in the caption is
                    non-padding, and 0 otherwise.
                    Shape: [max_caption_len + prefix_length]
                        or [1 + prefix_length].
                torch.Tensor: The frame embedding. If no frame was assigned to
                    this item (because the video clip was exhausted), a zero
                    tensor is returned.
                    Shape: [embedding_dim].
        """
        j, i = divmod(item, self.batch_size)

        # Initialize the caption with padding tokens and the mask with zeros.
        caption = torch.full((self.max_caption_len[j],), self.pad_token_id)
        mask = torch.zeros(self.prefix_length + self.max_caption_len[j])

        # Retrieve the frame and caption corresponding to the current item.
        if self.hor_stack[i][j] is None:
            frame = torch.zeros(self.frame_embed_dim)
        else:
            video_clip_idx, frame_idx = self.hor_stack[i][j]
            video_clip = self.prepr_dataset[video_clip_idx]
            frame = video_clip["frames"][frame_idx]
            if frame_idx == self.num_frames[video_clip_idx] - 1:
                caption_len = self.caption_lens[video_clip_idx]
                caption[:caption_len] = video_clip["caption"]
                total_len = self.prefix_length + caption_len
                mask[:total_len] = torch.ones(total_len)

        return caption, mask, frame


def main(args: argparse.Namespace) -> None:
    """Test whether data loading using a DataLoader works correctly.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # Create the dataset and dataloader.
    dataset = ActivityNetDataset(
        args.dataset_path,
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
        default=10,
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
