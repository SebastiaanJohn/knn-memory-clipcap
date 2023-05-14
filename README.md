# kNN-Memory with ClipCap: Enhanced Long-Range Dependency Handling

This repository provides the code for our project, which combines kNN-Memory and ClipCap to improve long-range dependency handling. The project builds on the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and [Memorizing Transformers](https://github.com/lucidrains/memorizing-transformers-pytorch) repositories. This work is conducted as part of the academic curriculum for the [Deep Learning 2](https://uvadl2c.github.io) course at the University of Amsterdam.


## Requirements

The code is written in Python 3.10. Install the required packages using either `pip install -r requirements.txt` or by creating a conda environment with the provided `environment.yml` file using `conda env create -f environment.yml`. Activate the environment using `conda activate knn-memory-clipcap`.


## Dataset

Our experiments use the [ActivityNet Caption](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset. Use one of the following methods to download the dataset. The first method is recommended, other methods are only provided for full reproducibility.

1. To download the pre-processed video clips, run:
    ```bash
    cd src/data/

    wget "https://drive.google.com/u/0/uc?id=11r0znF5EteRYWoVErI9dJYqb0dxnBzYE&export=download&confirm=t" -O activitynet_ViT-B_32_train_300.pkl
    wget "https://drive.google.com/u/0/uc?id=1Spjqxmv-HGW6dPl-SIgSDYQNbzvKncLb&export=download&confirm=t" -O activitynet_ViT-B_32_validation_100.pkl
    ```
    Instead of `wget`, you can also download the files manually from [here](https://drive.google.com/drive/folders/16HZede6SwJXrhKBcl6Gg2TodsPqUI8Kl).

2. If you want to pre-process the dataset manually, run:
    ```bash
    cd src/data/

    wget "https://drive.google.com/u/0/uc?id=1eIRY9AUTmP_4hKRcUb_3mN6jzP4YxBQu&export=download&confirm=t" -O train_subset_300.zip
    wget "https://drive.google.com/u/0/uc?id=1YCrqpjox0ePmt-aFJ6tvRZx0Mc0oKBCi&export=download&confirm=t" -O validation_subset_100.zip

    unzip train_subset_300.zip
    unzip validation_subset_100.zip

    rm train_subset_300.zip
    rm validation_subset_100.zip

    cd ../..

    python3 src/clipcap/parse_activitynet.py --split train --subset 300 --frames_dir src/data/train_subset_300/
    python3 src/clipcap/parse_activitynet.py --split validation --subset 100 --frames_dir src/data/validation_subset_100/
    ```
    Instead of `wget`, you can also download the files manually from [here](https://drive.google.com/drive/folders/16HZede6SwJXrhKBcl6Gg2TodsPqUI8Kl).

3. If you want to download the entire ActivityNet Caption dataset from scratch, run:
    ```bash
    python3 src/datasets/download_dataset.py
    ```
    *WARNING*: this will download the entire dataset, which is about 200 GB in size.

    To extract frames from the downloaded videos or your own videos, execute:
    ```bash
    python3 src/datasets/extract_frames.py -r <path_to_videos>
    ```
    This command creates a `frames` folder in the videos' parent directory. By default, frames are extracted at 5 fps. To modify this setting, use the `-fps` flag. The script also generates a summary CSV file in the `frames` folder, containing the video ID, frame extraction success status, and number of frames extracted.
