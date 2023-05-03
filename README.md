# kNN-Memory with ClipCap for Improved Long-Range Dependency Handling
This repository contains the code for our project on kNN-Memory with ClipCap for Improved Long-Range Dependency Handling. The code is based on [ClipCap](https://github.com/rmokady/CLIP_prefix_caption) and [Memorizing Transformers](https://github.com/lucidrains/memorizing-transformers-pytorch).


## Requirements
The code is written in Python 3.10. The requirements can be installed using `pip install -r requirements.txt` or with the conda environment file `conda env create -f environment.yml`.


## (Optional) Pretrained model weights
To download the COCO model weightsnd Conceptual Captions weights respectively, run:
```bash
cd src/clipcap/models/

wget "https://drive.google.com/u/0/uc?id=1IdaBtMSvtyzF0ByVaBHtvM0JYSXRExRX&export=download&confirm=t" -O "coco_weights.pt"

wget "https://drive.google.com/u/0/uc?id=14pXWwB4Zm82rsDdvbGguLfx9F8aM7ovT&export=download&confirm=t" -O "conceptual_weights.pt"
```
