"""Python program to dump frames of many videos."""

# Inspired from: https://github.com/escorciav/video-utils/blob/master/tools/batch_dump_frames.py

import argparse
import logging
import subprocess
from pathlib import Path
from pprint import pformat

from joblib import Parallel, delayed


def dump_frames(
    filename: str,
    output_format: Path = Path("frame-%06d.jpg"),
    filters: str = "-qscale:v 1",
) -> bool:
    """Dump frames of a video-file.

    Args:
        filename (str): full path of video-file
        output_format (Path, optional): output format for frames.
        filters (str, optional): additional filters for ffmpeg, e.g., "-vf scale=320x240".

    Returns:
        success (bool)
    """
    cmd = f"ffmpeg -v error -i {filename} {filters} {output_format}"

    try:
        subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            shell=True,
        )
    except subprocess.CalledProcessError as err:
        logging.debug(f"Impossible to dump video {filename}")
        logging.debug(f"Traceback:\n{err.output}")
        return False
    return True


def dump_wrapper(
    filename: str, dirname: Path, frame_format: str, filters: str, ext: str
) -> tuple[str, bool, int | None]:
    """Wrapper for dump_frames function.

    Args:
        filename (str): The filename of the video.
        dirname (Path): The directory where the frames will be dumped.
        frame_format (str): The format of the frames.
        filters (str): The filters to be applied to the video.
        ext (str): The extension of the frames.

    Returns:
        tuple[str, bool, int | None]: The filename of the video, a boolean
            indicating if the frames were dumped successfully, and the number
            of frames dumped.
    """
    filename_path = Path(filename)
    print(filename_path)
    filename_noext = filename_path.stem
    print(filename_noext)
    frame_dir = dirname / filename_noext
    print(frame_dir)
    # Check if the input file exists
    if not filename_path.is_file():
        logging.debug(f"Unexistent file {filename}")
        return filename_noext, False, None

    # If the frame directory doesn't exist, create it
    if not frame_dir.is_dir():
        Path(frame_dir).mkdir(parents=True, exist_ok=True)
    else:
        # Check if the first frame file already exists, if so, return True
        first_frame = frame_dir / frame_format.format(1)
        if first_frame.is_file():
            return filename_noext, True, None

    output = frame_dir / frame_format
    success = dump_frames(filename, output, filters)
    num_frames = None

    # If dump_frames is successful, count the number of frames
    if success:
        num_frames = len(list(frame_dir.glob(f"*{ext}")))

    return filename_noext, success, num_frames


def main(args: argparse.Namespace) -> None:
    """Main function."""
    logging.info("Dumping frames")
    logging.info(f"Arguments:\n{pformat(vars(args))}")

    if len(args.filters) == 0:
        args.filters = (
            f'-vf "fps={args.fps}, '
            f'scale={args.width}:{args.height}" '
            f"-qscale:v 2"
        )

    video_files = [video for video in args.root.glob("*.mp4")]
    logging.info(f"Loaded {len(video_files)} videos from {args.root}")

    out_dir = args.root.parent / "frames"

    logging.info("Dumping frames...")
    ext = Path(args.frame_format).suffix
    logging.info(ext)

    status = Parallel(n_jobs=args.n_jobs, verbose=args.verbose)(
        delayed(dump_wrapper)(
            video_id, out_dir, args.frame_format, args.filters, ext
        )
        for video_id in video_files
    )

    logging.info("Dumping report")
    logging.info("Creating summary file...")

    with Path.open(out_dir.parent / "summary.csv", "w", newline="") as fid:
        fid.write("path,successful_frame_extraction,num_frames\n")
        for i in status:
            fid.write(f"{i[0]},{i[1]},{i[2]}\n")

    logging.info("Succesful execution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract frames of a bunch of videos. "
            "The file-system organization is preserved if relative-path are used."
        )
    )
    parser.add_argument(
        "-r",
        "--root",
        type=Path,
        required=True,
        help="The root directory containing the videos.",
    )
    parser.add_argument(
        "-f",
        "--frame-format",
        default="%06d.jpg",
        help="Format used for naming frames e.g. %%06d.jpg",
    )
    parser.add_argument("--filters", default="", help="Filters for ffmpeg")

    parser.add_argument(
        "--width", default=320, help="Frame width (only valid for empty filters)"
    )
    parser.add_argument(
        "--height", default=240, help="Frame height (only valid for empty filters)"
    )
    parser.add_argument("--fps", default=5, help="FPS for frame extraction")

    parser.add_argument(
        "-n", "--n-jobs", default=1, type=int, help="Max number of process"
    )
    parser.add_argument("--verbose", type=int, default=0, help="verbosity level")
    parser.add_argument("-log", "--loglevel", default="INFO", help="verbosity level")

    args = parser.parse_args()

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {args.loglevel}")

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d]: %(message)s",
        level=numeric_level,
    )

    main(args)
