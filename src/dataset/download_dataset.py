"""Download ActivityNet Captions videos from YouTube."""

import argparse
import logging
from pathlib import Path
from typing import Iterable

from dataset import load_dataset
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


def download_video(video: dict, path: str) -> None:
    """Download video from YouTube.

    Args:
        video (dict): Video information.
        path (str): Path to save video.
    """
    try:
        yt = YouTube(video["video_path"], use_oauth=True, allow_oauth_cache=True)
        yt.streams.filter(progressive=True, file_extension="mp4").filter(
            res="360p"
        ).first().download(path, f"{video['video_id']}.mp4")
    except VideoUnavailable:
        logging.warning(f"Video: ({video['video_id']}) is unavailable.")
    except KeyError:
        logging.warning(f"Video: ({video['video_id']}) gives streamingData error.")


def download_activitynet_videos(
    dataset: Iterable,
    path: str = "./data",
    split: str = "train",
) -> None:
    """Download ActivityNet Captions videos from YouTube.

    Args:
        dataset (Iterable): Dataset to download videos from.
        path (str, optional): Path to save videos. Defaults to "./data".
        split (str, optional): Split to download. Defaults to "train".
    """
    # Create directory if it doesn't exist
    path = f"{path}/ActivityNet_Captions/{split}/videos"
    Path(path).mkdir(parents=True, exist_ok=True)

    for video in tqdm(dataset):
        if Path(f"{path}/{video['video_id']}.mp4").exists():
            continue
        download_video(video, path)


def main(args) -> None:
    """Download ActivityNet Captions videos from YouTube."""
    logging.info(f"Downloading {args.split} videos...")

    # Load dataset
    split_percentage = 100 // args.chunks
    datasets = load_dataset(
        "Leyo/ActivityNet_Captions",
        split=[
            f"{args.split}[{k}%:{k+split_percentage}%]"
            for k in range(0, 100, split_percentage)
        ],  # type: ignore
    )

    thread_map(
        lambda dataset: download_activitynet_videos(
            dataset, path=args.path, split=args.split
        ),
        datasets,
    )

    logging.info(f"Finished downloading {args.split} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Split to download.",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=5,
        help="Number of chunks to split the dataset into for parallel downloading.",
    )
    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Path to save videos.",
    )

    args = parser.parse_args()

    # save logs to file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(f"logs/download_dataset_{args.split}.log")],
    )

    main(args)
