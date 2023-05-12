"""Dataset class for ActivityNet Captions."""

import logging
import math
import pickle
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ActivityNetDataset(Dataset):
    """Dataset for loading ActivityNet video clips and captions."""

    def __init__(self, data_path: str, batch_size: int) -> None:
        """Initialize ActivityNetDataset.

        Args:
            data_path (str): The path to the pre-processed ActivityNet dataset
                pickle file.
            batch_size (int): The batch size. Must be the same as the batch
                size used for the DataLoader.
        """
        with Path.open(data_path, "rb") as f:
            self.dataset = pickle.load(f)

        logging.info(
            f"There are {len(self.dataset)} video clips in the dataset."
        )

        # * The rest of this function is used for creating data structures that
        # * are used for efficient batching. The data structures are explained
        # * in more detail in the __getitem__ function docstring.

        # Initialize horizontal stack of frames.
        # horizontal_stack[i][j] is a tuple containing:
        # - The index of the video clip in the dataset.
        # - The index of the frame within the video clip.
        total_frames = sum(len(v["frames"]) for v in self.dataset)
        self.batch_size = batch_size
        self.batching_steps = math.ceil(total_frames / self.batch_size)
        self.horizontal_stack = [
            [(None, None) for _ in range(self.batching_steps)]
            for _ in range(self.batch_size)
        ]

        # Create horizontal stack of frames.
        current_video_clip_idx = 0
        for i in range(self.batching_steps):
            for j in range(self.batch_size):
                if self.horizontal_stack[j][i] == (None, None):
                    continue

                # Get next video clip if current one is exhausted.
                video_clip = self.dataset[current_video_clip_idx]
                for k, _frame in enumerate(video_clip["frames"]):
                    self.horizontal_stack[j][i + k] = (
                        current_video_clip_idx,
                        k,
                    )
                current_video_clip_idx += 1

        # Get the longest caption length within the current batch.
        self.max_caption_len = []
        for i in range(self.batching_steps):
            max_caption_len = 1
            for j in range(self.batch_size):
                if self.horizontal_stack[j][i][0] is None:
                    continue
                    
                video_clip = self.dataset[self.horizontal_stack[j][i][0]]
                frame_idx = self.horizontal_stack[j][i][1]
                if frame_idx == len(video_clip["frames"]) - 1:
                    max_caption_len = max(
                        max_caption_len, video_clip["en_caption"].shape[0]
                    )
            self.max_caption_len.append(max_caption_len)

    def __len__(self) -> int:
        """Return the amount of video clips in the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Get a single frame from a video clip from the dataset.

        The data is layed out in a horizontal stack of frames, where the frames
        of a video clip are stacked horizontally. The vertical dimension is
        created automatically by the DataLoader, and represents the batch
        dimension. This function will return a single entry from the stack.
        Example illustration:
            - Fx-Vy means frame x of video clip y.
            - In this illustration, V1 has 5 frames, V2 has 3 frames, V3
              has 1 frame, V4 has 2 frames, V5 has 4 frames, and V6 has 3
              frames.
                        Step1  Step2  Step3  Step4  Step5  Step6  Step7
                          v      v      v      v      v      v      v
                     / [F1-V1, F2-V1, F3-V1, F4-V1, F5-V1, F1-V4, F2-V4]
        Batch size -+  [F1-V2, F2-V2, F3-V2, F1-V5, F2-V5, F3-V5, F4-V5]
                     \ [F1-V3, F1-V6, F2-V6, F3-V6, empty, empty, empty]
            - The indices of the frames in the horizontal stack are layed out
              as follows. This is done to ensure that the frames are loaded in
              the correct order by the DataLoader.
                        Step1  Step2  Step3  Step4  Step5  Step6  Step7
                          v      v      v      v      v      v      v
                     / [  0  ,   3  ,   6  ,   9  ,  12  ,  15  ,  18  ]
        Batch size -+  [  1  ,   4  ,   7  ,  13  ,  16  ,  19  ,  22  ]
                     \ [  2  ,  10  ,  17  ,  20  ,  23  ,  24  ,  25  ]

        Args:
            item (int): Index of the frame within the horizontal stack of
                frames.

        Returns:
            Tuple containing:
                torch.Tensor: The frame embedding. If the frame is an "empty"
                    frame (this only occurs in the last batch), the frame
                    embedding is a zero tensor.
                    Shape: [512]
                torch.Tensor: The caption token ids, if the frame is the last
                    one of the video clip. If the frame is not the last one,
                    the caption token ids is a zero tensor that has the same
                    shape as the longest caption in the current batch. If none
                    of the video clips in the current batch have a caption,
                    the caption token ids is a zero tensor with shape [1].
                    Shape: [max_caption_len]
        """
        # Get the current video clip and frame index in O(1).
        curr_batch = item // self.batch_size
        video_clip_idx, frame_idx = self.horizontal_stack[
            item % self.batch_size
        ][curr_batch]

        # Initialize the caption with pad tokens.
        caption = torch.full(
            [self.max_caption_len[curr_batch]], self.dataset.pad_token_id
        )

        if video_clip_idx is None:
            frame = torch.zeros(512)
        else:
            video_clip = self.dataset[video_clip_idx]
            frame = video_clip["frames"][frame_idx]

            # Check if the frame is the last one of the video clip.
            if frame_idx == len(video_clip["frames"]) - 1:
                caption[: video_clip["en_caption"].shape[0]] = video_clip[
                    "en_caption"
                ]

        return frame, caption
