# kNN-Memory with ClipCap: Enhanced Long-Range Dependency Handling

This repository provides the code for our project, which combines kNN-Memory and ClipCap to improve long-range dependency handling. The project builds on the [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and [Memorizing Transformers](https://github.com/lucidrains/memorizing-transformers-pytorch) repositories. This work is conducted as part of the academic curriculum for the [Deep Learning 2](https://uvadl2c.github.io) course at the University of Amsterdam. You can read our comphrehensive report [here](https://github.com/SebastiaanJohn/knn-memory-clipcap/blob/main/blogpost.md).

## Project Structure

The project is structured as follows:

```
├── checkpoints (model checkpoints)
├── demos (demo notebooks)
├── images (images used in the report)
├── logs (training logs)
├── src (source code)
│   ├── dataset (dataset code, including parsers)
│   ├── evaluation (evaluation code for metrics)
│   ├── memorizing_transformers_pytorch (Memorizing Transformers code)
│   ├── models (model code for kNN-Memory and ClipCap)
├── generate_captions.py (generate captions for a dataset)
├── predict.py (predict captions for a video)
├── train.py (train a model)
├── validate.py (validate a model)
├── utils.py (utility functions)
├── environment.yml (conda environment file)
├── requirements.txt (pip requirements file)
├── blogpost.md (report)
├── pyproject.toml (project file)
└── README.md (this file)
```

## Requirements

The code is written in Python 3.10. Install the required packages using either `pip install -r requirements.txt` or by creating a conda environment with the provided `environment.yml` file using `conda env create -f environment.yml`. Activate the environment using `conda activate knn-memory-clipcap`.

## Dataset

Our experiments use the [ActivityNet Caption](https://cs.stanford.edu/people/ranjaykrishna/densevid/) dataset. Use one of the following methods to download the dataset. The first method is recommended, other methods are only provided for full reproducibility.

1. To download the pre-processed video clips, run:

    ```bash
    cd src/data/

    wget "https://drive.google.com/u/0/uc?id=19wiL2M3vMLN40QFUio0qeX0_ylraXxJv&export=download&confirm=t" -O activitynet_ViT-B_32_train_first_2000.pkl
    wget "https://drive.google.com/u/0/uc?id=1m0Q7qzmHpTPk0qvN2aOAJ5pnU9A9HOAJ&export=download&confirm=t" -O activitynet_ViT-B_32_dev_first_250.pkl
    wget "https://drive.google.com/u/0/uc?id=19wiL2M3vMLN40QFUio0qeX0_ylraXxJv&export=download&confirm=t" -O activitynet_ViT-B_32_validation_first_500.pkl
    ```

    Instead of `wget`, you can also download the files manually from [here](https://drive.google.com/drive/folders/1-2Eifr-kgIzHsTiijgvItUrhtl7ePWAm). The files should be placed in the `src/data/` folder. Additionally, the pre-processed COCO dataset can be found there as well.

2. If you want to download the entire ActivityNet Caption dataset from scratch, run:

    ```bash
    python3 src/datasets/download_dataset.py
    ```

    *WARNING*: this will download the entire dataset, which is about 200 GB in size.

    To extract frames from the downloaded videos or your own videos, execute:

    ```bash
    python3 src/datasets/extract_frames.py -r <path_to_videos>
    ```

    This command creates a `frames` folder in the videos' parent directory. By default, frames are extracted at 5 fps. To modify this setting, use the `-fps` flag. The script also generates a summary CSV file in the `frames` folder, containing the video ID, frame extraction success status, and number of frames extracted.

    To pre-process the dataset, run:

    ```bash
    python3 src/dataset/parsers/parse_activitynet.py --split <split> 
    ```

    Other arguments are available, see `python3 src/dataset/parsers/parse_activitynet.py --help` for more information.

## Demo

Generating a caption for a video can be done in the demo notebook found in `notebooks/demo.ipynb`.

## Training

To train a model, run:

```bash
python src/train.py --train_path activitynet_ViT-B_32_train_first_2000.pkl --valid_path activitynet_ViT-B_32_dev_first_250.pkl --checkpoint checkpoints/coco/coco_prefix-best.pt --prefix activitynet_with_memory --only_prefix --use_video_dataset --use_memory
```

Use the `--use_memory` flag to enable kNN-Memory, and the `--use_video_dataset` flag to use the video dataset. Additionally, the `--only_prefix` flag can be used to only train the prefix model. The full argument list is available using `python src/train.py --help`.

## Evaluation

To evaluate a model, run:

```bash
python src/validate.py --data /Users/sebastiaan/Developer/knn-memory-clipcap/src/data/ --checkpoint checkpoints/activitynet_with_memory-best.pt --only_prefix --use_video_dataset --use_memory
```

The full argument list is available using `python src/validate.py --help`.

## Generate Captions

To generate captions for a dataset, run:

```bash
python src/generate_captions.py --data /Users/sebastiaan/Developer/knn-memory-clipcap/src/data/ --checkpoint checkpoints/activitynet_with_memory-best.pt --only_prefix --use_video_dataset --use_memory
```

This will generate two JSON files that can be used to calculate the evaluation metrics. The full argument list is available using `python src/generate_captions.py --help`.

## Evaluation Metrics

To calculate the evaluation metrics, run:

```bash
???
```

## Acknowledgements

This project is conducted as part of the academic curriculum for the [Deep Learning 2](https://uvadl2c.github.io) course at the University of Amsterdam. We would like to thank the course staff for their support and feedback.
