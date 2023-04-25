"""This file defines the prediction function for ClipCap."""


import clip
import numpy as np
import PIL.Image
import skimage.io as io
import torch
import torch.nn.functional as nnf
from torch import nn
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


WEIGHTS_PATHS = {
    "coco": "coco_weights.pt",
    "conceptual-captions": "conceptual_weights.pt",
}

class Predictor:
    """Predictor class for ClipCap."""

    def __init__(self) -> None:
        """Load the model into memory to make running multiple predictions efficient."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, self.preprocess = clip.load(
            "ViT-B/32", device=self.device, jit=False
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.models = {}
        self.prefix_length = 10
        for key, weights_path in WEIGHTS_PATHS.items():
            model = ClipCaptionModel(self.prefix_length)
            model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
            model = model.eval()
            model = model.to(self.device)
            self.models[key] = model

    def predict(self, image: np.ndarray, model_name: str, use_beam_search: bool = True) -> str:
        """Run a single prediction on the model.

        Args:
            image (np.ndarray): The image to caption.
            model_name (str): The name of the model to use.
            use_beam_search (bool): Whether to use beam search or greedy search.

        Returns:
            str: The caption for the image.
        """
        image = io.imread(image)
        model = self.models[model_name]
        pil_image = PIL.Image.fromarray(image)
        image = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prefix = self.clip_model.encode_image(image).to(
                self.device, dtype=torch.float32
            )
            prefix_embed = model.clip_project(prefix).reshape(1, self.prefix_length, -1)
        if use_beam_search:
            return generate_beam(model, self.tokenizer, embed=prefix_embed)[0]
        else:
            return generate2(model, self.tokenizer, embed=prefix_embed)


class MLP(nn.Module):
    """A simple MLP."""

    def __init__(self, sizes: tuple[int, ...], bias: bool = True, act = nn.Tanh):
        """Initialize the MLP.

        Args:
            sizes (Tuple[int, ...]): The sizes of the layers.
            bias (bool, optional): Whether to use bias. Defaults to True.
            act (optional): The activation function to use. Defaults to nn.Tanh.
        """
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)


class ClipCaptionModel(nn.Module):
    """The model for ClipCap."""

    def __init__(self, prefix_length: int, prefix_size: int = 512):
        """Initialize the model.

        Args:
            prefix_length (int): The length of the prefix.
            prefix_size (int, optional): The size of the prefix. Defaults to 512.
        """
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create a dummy token for the start of the caption.

        Args:
            batch_size (int): The batch size.
            device (torch.device): The device to use.

        Returns:
            torch.Tensor: The dummy token.
        """
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The forward pass of the ClipCap model.

        Args:
            tokens (torch.Tensor): The tokens to predict.
            prefix (torch.Tensor): The prefix to use.
            mask (torch.Tensor | None, optional): The mask to use. Defaults to None.
            labels (torch.Tensor | None, optional): The labels to use. Defaults to None.

        Returns:
            torch.Tensor: The output of the model.
        """
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)

        return self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)



class ClipCaptionPrefix(ClipCaptionModel):
    """The ClipCap model with a prefix."""
    def parameters(self, recurse: bool = True):
        """The parameters of the model.

        Args:
            recurse (bool, optional): Whether to recurse. Defaults to True.

        Returns:
            Iterator[Parameter]: The parameters.
        """
        return self.clip_project.parameters()

    def train(self, mode: bool = True) -> "ClipCaptionPrefix":
        """Train the model.

        Args:
            mode (bool, optional): Whether to train. Defaults to True.

        Returns:
            ClipCaptionPrefix: The model.
        """
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()

        return self


def generate_beam(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    beam_size: int = 5,
    prompt: str | None = None,
    embed: torch.Tensor | None = None,
    entry_length: int = 67,
    temperature: float = 1.0,
    stop_token: str = ".",
):
    """Beam search generation.

    Args:
        model (ClipCaptionModel): The model to use.
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        beam_size (int, optional): Beam size. Defaults to 5.
        prompt (str | None, optional): The prompt to use. Defaults to None.
        embed (torch.Tensor | None, optional): The embedding to use. Defaults to None.
        entry_length (int, optional): The length of the entry. Defaults to 67.
        temperature (float, optional): The temperature to use. Defaults to 1.0.
        stop_token (str, optional): The stop token to use. Defaults to ".".

    Returns:
        tuple[list[str], list[float]]: The generated captions and their scores.
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)
    with torch.no_grad():
        if embed is not None:
            generated = embed
        else:
            if tokens is None:
                tokens = torch.tensor(tokenizer.encode(prompt))
                tokens = tokens.unsqueeze(0).to(device)
                generated = model.gpt.transformer.wte(tokens)
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()
            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                generated = generated.expand(beam_size, *generated.shape[1:])
                next_tokens, scores = next_tokens.permute(1, 0), scores.squeeze(0)
                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                next_tokens_source = next_tokens // scores_sum.shape[1]
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]
            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )
            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break
    scores = scores / seq_lengths
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts


def generate2(
    model: ClipCaptionModel,
    tokenizer: GPT2Tokenizer,
    tokens: torch.Tensor | None = None,
    prompt: str | None = None,
    embed: torch.Tensor | None = None,
    entry_count: int = 1,
    entry_length: int = 67,  # maximum number of words
    top_p: float = 0.8,
    temperature: float = 1.0,
    stop_token: str = ".",
):
    """Greeedy generation.

    Args:
        model (ClipCaptionModel): The model to use.
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        tokens (torch.Tensor | None, optional): The tokens to use. Defaults to None.
        prompt (str | None, optional): The prompt to use. Defaults to None.
        embed (torch.Tensor | None, optional): The embedding to use. Defaults to None.
        entry_count (int, optional): The number of entries to generate. Defaults to 1.
        entry_length (int, optional): The length of the entry. Defaults to 67.
        temperature (float, optional): The temperature to use. Defaults to 1.0.
        top_p (float, optional): The top p to use. Defaults to 0.8.
        stop_token (str, optional): The stop token to use. Defaults to ".".

    Returns:
        list[str]: The generated captions.
    """
    model.eval()
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]
