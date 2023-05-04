# kNN-Memory with ClipCap: Enhanced Long-Range Dependency Handling

This repository provides the code for our project, which combines kNN-Memory and ClipCap to improve long-range dependency handling. The project builds on the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and [Memorizing Transformers](https://github.com/lucidrains/memorizing-transformers-pytorch) repositories.

## Requirements

The code is written in Python 3.10. Install the required packages using either `pip install -r requirements.txt` or by creating a conda environment with the provided environment.yml file using `conda env create -f environment.yml`.

## Dataset

Our experiments use the [ActivityNet Caption](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset. Download the videos by running:

```bash
python3 src/datasets/download_dataset.py
```

To extract frames from the downloaded videos or your own videos, execute:

```bash
python3 src/datasets/extract_frames.py -r <path_to_videos>
```

This command creates a `frames` folder in the videos' parent directory. By default, frames are extracted at 5 fps. To modify this setting, use the `-fps` flag. The script also generates a summary CSV file in the `frames` folder, containing the video ID, frame extraction success status, and number of frames extracted.

## (Optional) Pretrained model weights

Download the COCO and Conceptual Captions model weights by running the following commands:

```bash
cd src/clipcap/models/

wget "https://drive.google.com/u/0/uc?id=1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX&export=download&confirm=t" -O "coco_weights.pt"

wget "https://drive.google.com/u/0/uc?id=14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT&export=download&confirm=t" -O "conceptual_weights.pt"
```
