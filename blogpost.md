# Enhancing Long-Range Dependency Management: A Comprehensive Study on the Integration of kNN-Memory and ClipCap Techniques

## Introduction

_Image captioning_ is a multimodal task that involves generating textual descriptions of images. This research investigates a method called [ClipCap](https://arxiv.org/abs/2111.09734), explicitly proposed for this task.

Previous methods proposed for this task relied on encoder-decoder frameworks, which are resource-intensive in terms of time and dataset size. A new model pipeline, ClipCap, was introduced to overcome these challenges. The ClipCap model pipeline utilizes pre-trained models, specifically CLIP and GPT2, to deliver superior performance in a computationally efficient manner. The CLIP model extracts semantic information from visual data, while the GPT2 model serves as a decoder to generate the textual description. In training the ClipCap model pipeline, the GPT2 model is fine-tuned, and the CLIP model is kept static. This approach conserves resources while simultaneously leveraging the expressiveness and adaptability of the GPT2 model to new tasks.
Nonetheless, a bridge between the image and text modalities is still necessary. To create this connection, a mapping network, a multi-layered perceptron (MLP), is trained to link the latent spaces of CLIP and GPT2. The original authors also considered an alternative version of the ClipCap model pipeline, where only the mapping network is trained, and the CLIP and GPT2 models are kept static. This version was found to be comparable to state-of-the-art performance, but it is not explored in this research.

During the training phase, the CLIP encoder (pre-trained on image-text pairs using contrastive loss) extracts embeddings from the visual data, encapsulating semantic information about the image. These embeddings are subsequently processed through the mapping network to obtain prefix embeddings containing salient words associated with the image content. The prefix embeddings are merged with the caption embeddings - the output of the GPT tokenizer - and the combined embeddings are passed through the GPT2 decoder to generate the textual description of the image.

During inference, the ClipCap model pipeline generates captions by predicting one token at a time, guided by the GPT2 model. This model utilizes a greedy approach, which predicts a probability for all tokens in the vocabulary for each token.

The performance of the ClipCap model pipeline was evaluated using three datasets: [Conceptual Captions](https://aclanthology.org/P18-1238.pdf), [NoCaps](https://arxiv.org/abs/1812.08658), and [(Common Objects in Context) COCO](https://arxiv.org/abs/1405.0312). These datasets were curated to represent a wide array of visual concepts. The ClipCap model pipeline demonstrated excellent performance with the Conceptual Captions and NoCaps datasets, thereby establishing itself as an efficient and effective solution for image captioning. Moreover, given its training on diverse images, the ClipCap model pipeline was found to generalize well to randomly chosen new images.

In conclusion, the ClipCap model pipeline is a swift, user-friendly, and efficient solution for image captioning that achieves state-of-the-art performance across multiple datasets.

## Expanding Multimodal Capabilities: Potential and Challenges

One of the key strengths of the pipeline model proposed in the ClipCap paper is its multimodal nature. This model utilizes information from images and text for the caption generation task, capitalizing on a mapping network to facilitate a more comprehensive understanding and exploitation of available data resources. Combining the CLIP model with a pre-trained language model in a multimodal pipeline yields more comprehensive and competent captions. It ensures that training costs (both in terms of training time and required data volume) remain low. The authors have effectively utilized powerful pre-trained models, resulting in a simple method that requires no additional annotations and is quick to train. Moreover, the proposed image-captioning method comprises multiple self-contained components (CLIP model, mapping network, pre-trained language model), allowing for the swift adaptation or replacement of these components by different models. This feature enables the pipeline model to be easily adapted for different tasks or used in ablation studies to understand the model's underlying mechanisms better.

However, the proposed captioning method needs to consider the dependencies' duration between data resources adequately. Visual data, such as video segments, naturally have long-range dependencies between individual frames within a single video (for example, the first frame in a video may not match the last frame, yet some dependency still exists between the two). Ignoring such long-range dependencies in the proposed pipeline model could result in a model incapable of achieving state-of-the-art captioning performances. 

Our research aims to address this issue by modifying the original ClipCap model to account for long-range dependencies in the visual data, achieved by incorporating a memory attribute into the mapping network. Additionally, as previously mentioned, the modular architecture of the ClipCap model allows for the potential of an ablation study, which we plan to utilize in conjunction with our earlier adaptations in the mapping network.

## Utilizing Memory for Enhanced Long-Range Dependency Management

Our research investigates potential performance enhancements in video captioning by integrating long-range dependencies by applying [kNN-memory](https://arxiv.org/abs/2203.08913) in the ClipCap pipeline.

The kNN-memory extends transformer models to memorize internal representations of past inputs, aiming to improve the performance of language modeling tasks. The memory system utilizes an approximate kNN lookup to recall the most recent key-value pairs. This strategy enables the model to harness learned information from previously encountered data for current predictions, thereby accounting for long-range dependencies. The original paper applies this concept to language models, effectively addressing the issue of long-term dependencies. However, in the context of image captioning, this problem is less pertinent. The captions for images are succinct enough that the lack of long-term dependency handling does not significantly impact the outcome.

Nevertheless, we encounter problems associated with long-range dependencies when expanding the captioning task to video. The caption of a video depends on all frames within that video. Using the current ClipCap architecture, frames occurring later in the sequence significantly influence the final caption. To address this issue, we propose to utilize the kNN-memory transformer framework, as proposed by [Wu et al. 2022](https://arxiv.org/abs/2203.08913).

## Datasets

In keeping with the methodology of the ClipCap research, we will use the COCO dataset for the initial pretraining of our mapping network. Renowned for its diversity in everyday scene contexts, the COCO dataset comprises over 300,000 images, each with five associated captions. This dataset enables our model to learn from various objects and scenes, enhancing its ability to generalize and adapt to novel instances.

Following the pretraining, we will employ the [ActivityNet Captions](https://arxiv.org/pdf/1705.00754v1.pdf) dataset for finetuning. The ActivityNet Captions dataset provides a more task-specific data source explicitly designed for the temporal localization and captioning of activities. With 20,000 videos sourced from YouTube, amounting to 849 hours of footage, accompanied by 100,000 detailed descriptions of sequences of actions within the videos, it presents an optimal choice for our research.

### Preprocessing

Videos are converted into image frames at a rate of five frames per second (fps). Since our focus is solely on captioning and not temporal action localization, we extract all frames from the start to end of each caption, treating it as an independent video clip. These frames are individually embedded using the ClipCap model, then concatenated into a single tensor. The captions are tokenized using the GPT2 tokenizer. Given that we are only finetuning the model, we will use a small subset of the dataset. The final preprocessed datasets can be accessed via the links provided in our GitHub repository. The distribution of the dataset across different categories is outlined in the table below.

| **Split**   | **Train** | **Test** |
|-------------|-----------|----------|
| Videos      | 300       | 100      |
| Video Clips | ?         | ?        |

## Results

## Conclusion

## References
Agrawal, Harsh et al. (2019). “Nocaps: Novel object captioning at scale”. In: Proceedings of the IEEE/CVF international conference on computer vision, pp. 8948–8957.
Caba Heilbron, Fabian et al. (2015). “Activitynet: A large-scale video benchmark for human activity understanding”. In: Proceedings of the ieee conference on computer vision and pattern recognition, pp. 961–970.
Lin, Tsung-Yi et al. (2014). “Microsoft coco: Common objects in context”. In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer, pp. 740–755.
Sharma, Piyush et al. (2018). “Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning”. In: Proceedings of the 56th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers), pp. 2556–2565.
Wu, Yuhuai et al. (2022). “Memorizing Transformers”. In: International Conference on Learning Representations.
