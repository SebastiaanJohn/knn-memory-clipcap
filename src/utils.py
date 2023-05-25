"""Utility functions for the project."""

import argparse
import json
import logging
from pathlib import Path

import torch


def save_config(args: argparse.Namespace) -> None:
    """Save the config to a file."""
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_dir = Path(args.out_dir)
    out_path = out_dir / f"{args.prefix}.json"
    with out_path.open("w") as outfile:
        json.dump(config, outfile)

def forward_with_memory(model, tokens, prefix, mask):
    """The forward pass of the Memorizing ClipCap model.

    Args:
        model (ClipCaptionModel): The model to use.
        tokens (torch.Tensor): The tokens to predict.
        prefix (torch.Tensor): The prefix to use.
        mask (torch.Tensor): The mask to use.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The output
        of the model and the corresponding captions.
    """
    # Compute the batch indices of those frames
    # that are the last in a sequence.
    contains_caption = (mask == 1).any(dim=1)
    idx = contains_caption.nonzero(as_tuple=True)[0].tolist()

    # Compute the forward pass of the TransformerMapper
    # on all frames in the batch, thereby storing new memories.
    # Use the batch indices to clear memories of videos
    # after generating a prefix for their last frame.
    prefix_projections = model.clip_project(prefix, idx).view(
        -1, model.prefix_length, model.gpt_embedding_size
    )

    # Continue the forward (and backward) pass of the ClipCaptionModel
    # with only those frames which are the last in a sequence
    # and therefore have a caption. If there are none, continue.

    if len(idx) == 0:
        return None, []

    tokens, mask, prefix, prefix_projections =  \
        tokens[idx], mask[idx], prefix[idx], prefix_projections[idx]

    embedding_text = model.gpt.transformer.wte(tokens)
    embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
    outputs = model.gpt(inputs_embeds=embedding_cat, attention_mask=mask)

    return outputs, tokens

def setup_logging(logfile: str='logs/application.log') -> None:
    """Setup logging to file and console."""
    # Create a logger object.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a file handler.
    file_handler = logging.FileHandler(logfile)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    # Create a console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))

    # Add the handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
