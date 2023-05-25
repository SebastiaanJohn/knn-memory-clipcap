"""Python program to dump frames of single video."""

# Inspired from: https://github.com/escorciav/video-utils/blob/master/tools/batch_dump_frames.py

import os

import logging
from tqdm import tqdm
import subprocess
import shutil
from pathlib import Path
import skimage.io as io
from datasets import (
    Array2D,
    Dataset,
    Features,
    Sequence,
    Value,
)
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch
import clip
import pickle
from PIL import Image

from dataset.activitynet import ActivityNetDataset
from evaluation.inference.inference import generate_beam

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
):
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
    filename_noext = filename_path.stem
    frame_dir = dirname / filename_noext

    # Check if the input file exists
    if not filename_path.is_file():
        print(f"Unexistent file {filename}")
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

def extract_frames(video_path: str) -> None:
    """Main function."""

    print(f"Extracting frames for video: {video_path}")

    filters = f'-vf "fps=5, ' f'scale=320:240" ' f"-qscale:v 2"
    video_path = Path(video_path)
    out_dir = video_path.parent / "frames"
    ext = Path("%06d.jpg").suffix

    status = dump_wrapper(
        str(video_path),
        out_dir,
        "frame-%06d.jpg",
        filters,
        ext,
    )

    if status[1]:
        print('Succesfully dumped frames of {}: {} frames'.format(status[0], status[2]))
    else:
        print('unsuccesful...')

def generate_caption(video_name, model, remove_dirs=False):  

    # directory where the frames of the video are stored
    frames_dir = './frames/{}'.format(video_name[:-4])
    video_frames_dir = Path(frames_dir)
    if not video_frames_dir.exists():
        print(f"Video has no frames saved!")
        return None
    
    # directory where the embeddings should be stored
    pickle_dir = './embeddings/embeddings_{}.pkl'.format(video_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load models
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", pad_token="[PAD]")

    prepr_dataset = []
    embeddings = []
    # images = []
    # compute the frame embeddings
    progress = tqdm(total=len(os.listdir(frames_dir)), desc='Creating CLIP embeddings of frames...')
    for frame_number in range(1, len(os.listdir(frames_dir)) + 1):
        images = [preprocess(Image.fromarray(io.imread(Path(video_frames_dir)/ f"frame-{frame_number:06d}.jpg"))).unsqueeze(0).to(device)]

        with torch.no_grad():
            embedding = clip_model.encode_image(torch.cat(images, dim=0)).to(device)
            embeddings.append(embedding)
        
        progress.update()
    progress.close()

    # add the video embeddings to the new dataset
    prepr_dataset.append(
        {
            "video_id": 0,
            "frames": torch.cat(embeddings, dim=0),
            "caption": tokenizer.encode('', return_tensors="pt").squeeze(0),
        }
    )

    # construct a new HuggingFace dataset
    prepr_dataset_class = Dataset.from_list(
        prepr_dataset,
        features=Features(
            {
                "video_id": Value("string"),
                "frames": Array2D(shape=(None, 512), dtype="float32"),
                "caption": Sequence(feature=Value("int64")),
            }
        ),
    )
    prepr_dataset_class.set_format(type="torch", columns=["frames", "caption"], output_all_columns=True)
    prepr_dataset_class.pad_token_id = tokenizer.pad_token_id

    # create embeddings directory
    if not os.path.exists('./embeddings'):
        os.makedirs('./embeddings')

    # overwrite pkl file if it exists already
    with Path.open(Path(pickle_dir), "wb",) as f:
        pickle.dump(prepr_dataset_class, f)

    # load embeddings in correct format
    dataset = ActivityNetDataset(pickle_dir, 1, 10)
    model = model.to(device)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # create and iterate over dataloader
    eval_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    progress = tqdm(total=len(eval_dataloader.dataset), desc='Generating caption...')
    for tokens, mask, prefix in eval_dataloader:
        tokens, mask, prefix = (tokens.to(device),mask.to(device),prefix.to(device, dtype=torch.float32))
        contains_caption = (mask == 1).any(dim=1)
        idx = contains_caption.nonzero(as_tuple=True)[0].tolist()
        with torch.no_grad():
            prefix_projections = model.clip_project(prefix, idx).view(-1, 1, model.prefix_length, model.gpt_embedding_size)

        if len(idx) == 0:
            progress.update()
            continue

        tokens, mask, prefix_projections = tokens[idx], mask[idx], prefix_projections[idx]
        mask = mask[:, 10:]
        tokens = [t[m.bool()] for t, m in zip(tokens, mask)]
        progress.update()

    generated_caption = [generate_beam(model, tokenizer, embed=prefix_embed)[0] for prefix_embed in prefix_projections][-1]
    progress.close()

    if remove_dirs:
        # remove folder with frames
        shutil.rmtree(frames_dir, ignore_errors=True)
        # remove CLIP embeddings
        os.remove(pickle_dir)


    return generated_caption
