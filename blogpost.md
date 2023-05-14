# Enhancing Long-Range Dependency Management: A Comprehensive Study on the Integration of kNN-Memory and ClipCap Techniques

- [Enhancing Long-Range Dependency Management: A Comprehensive Study on the Integration of kNN-Memory and ClipCap Techniques](#enhancing-long-range-dependency-management-a-comprehensive-study-on-the-integration-of-knn-memory-and-clipcap-techniques)
  - [Introduction](#introduction)
    - [ClipCap Summary](#clipcap-summary)
    - [Main results](#main-results)
    - [Additional Results](#additional-results)
    - [Ablation Studies](#ablation-studies)
    - [Related Work](#related-work)
  - [Expanding Multimodal Capabilities: Potential and Challenges](#expanding-multimodal-capabilities-potential-and-challenges)
  - [Utilizing Memory for Enhanced Long-Range Dependency Management](#utilizing-memory-for-enhanced-long-range-dependency-management)
  - [Datasets](#datasets)
    - [Preprocessing](#preprocessing)
  - [Results](#results)
  - [Conclusion](#conclusion)
  - [References](#references)
  - [Contributions](#contributions)


## Introduction

<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well. -->
__Image captioning__ is a multimodal task that involves generating textual descriptions of images. This research investigates a method called [ClipCap](https://arxiv.org/abs/2111.09734), explicitly proposed for this task, and explores the potential of enhancing this method by integrating long-range dependency handling into the model.

### ClipCap Summary

The ClipCap method utilizes a pipeline of pre-trained models to generate captions for images. This pipeline consists of the [CLIP](https://arxiv.org/abs/2103.00020) model, a mapping network, and a pre-trained language model (LM), namely [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf). The CLIP image encoder extracts high-level information from the visual data while the pre-trained LM generates the caption. The mapping network bridges the two, linking the latent spaces of the two models.

More specifically, for a given image, the CLIP image encoder generates an embedding containing high-level information about the image. This embedding is passed through the mapping network to obtain a so-called "prefix," a list of embeddings associated with the image content. Finally, the prefix embeddings are used as input to GPT-2, which will generate the caption autoregressively.

<!-- ### Motivation

Moreover, the ClipCap model pipeline is modular, allowing for the swift adaptation or replacement of the CLIP endoder and GPT-2 decoder models. This feature enables the pipeline model to be easily adapted for different tasks, but this capability was not explored in the original paper. This coud be a nice aspect to review in ablation studies to understand the model's underlying mechanisms better. -->

### Main results

The authors experiment with two different training procedures for the ClipCap model pipeline. In the first approach, the CLIP model is kept static, GPT-2 is fine-tuned, and the mapping network is an MLP that is trained from scratch. In the second approach, the CLIP and GPT-2 models are kept static, and the mapping network is a transformer encoder that is trained from scratch. The authors found that the first approach often yielded better results but required more training time.

Both approaches were evaluated on the [Conceptual Captions](https://aclanthology.org/P18-1238.pdf), [NoCaps](https://arxiv.org/abs/1812.08658), and [COCO](https://arxiv.org/abs/1405.0312) datasets, achieving state-of-the-art performance while requiring significantly less training time and data than previous methods. Additionally, the ClipCap architecture is more straightforward and faster than earlier methods.

### Additional Results

Multiple other experiments were conducted to determine when ClipCap performs well and when not. For example, the authors found that fine-tuning the LM results in a much more expressive model but that this model is more susceptible to overfitting. Additionally, an interpretability study was conducted to understand further the model's inner workings, in which the prefix embeddings are interpreted as a sequence of tokens. It was found that the interpretation is meaningful when both the mapping network and the LM are trained but that it becomes essentially unreadable when only the mapping network is trained. The authors hypothesize this happens because the network also maneuvers the fixed LM.

### Ablation Studies

The authors conducted multiple ablation studies to verify and motivate ClipCap's design choices. They found that the mapping network is crucial for the model's performance and that a Transformer architecture is superior when the LM is frozen, while an MLP is more effective when also fine-tuning the LM. Furthermore, the prefix length was a crucial hyperparameter; a prefix that is too short results in a lack of expressiveness, while a prefix that is too long results in a huge model that is slow to train.

### Related Work

Recent advantages in supervised convolutional models have showed promising performances in image-based recognition tasks ([6](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Sun_Human_Action_Recognition_ICCV_2015_paper.pdf), [7](https://openaccess.thecvf.com/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf), [8](https://openaccess.thecvf.com/content_cvpr_2015/papers/Donahue_Long-Term_Recurrent_Convolutional_2015_CVPR_paper.pdf)).


## Expanding Multimodal Capabilities: Potential and Challenges
<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->
One of the key strengths of the pipeline model proposed in the ClipCap paper is its multimodal nature. This model utilizes information from images and text for the caption generation task, capitalizing on a mapping network to facilitate a more comprehensive understanding and exploitation of available data resources. Combining the CLIP model with a pre-trained language model in a multimodal pipeline yields more comprehensive and competent captions. It ensures that training costs (both in terms of training time and required data volume) remain low. The authors have effectively utilized powerful pre-trained models, resulting in a simple method that requires no additional annotations and is quick to train. Moreover, the proposed image-captioning method comprises multiple self-contained components (CLIP model, mapping network, pre-trained language model), allowing for the swift adaptation or replacement of these components by different models. This feature enables the pipeline model to be easily adapted for different tasks or used in ablation studies to understand the model's underlying mechanisms better.

However, the proposed captioning method needs to consider the dependencies' duration between data resources adequately. Visual data, such as video segments, naturally have long-range dependencies between individual frames within a single video (for example, the first frame in a video may not match the last frame, yet some dependency still exists between the two). Ignoring such long-range dependencies in the proposed pipeline model could result in a model incapable of achieving state-of-the-art captioning performances.

Our research aims to address this issue by modifying the original ClipCap model to account for long-range dependencies in the visual data, achieved by incorporating a memory attribute into the mapping network. Additionally, as previously mentioned, the modular architecture of the ClipCap model allows for the potential of an ablation study, which we plan to utilize in conjunction with our earlier adaptations in the mapping network.


## Utilizing Memory for Enhanced Long-Range Dependency Management
<!-- Describe your novel contribution. -->
Our research investigates potential performance enhancements in video captioning by integrating long-range dependencies by applying [kNN-memory](https://arxiv.org/abs/2203.08913) in the ClipCap pipeline.

The kNN-memory extends transformer models to memorize internal representations of past inputs, aiming to improve the performance of language modeling tasks. The memory system utilizes an approximate kNN lookup to recall the most recent key-value pairs. This strategy enables the model to harness learned information from previously encountered data for current predictions, thereby accounting for long-range dependencies. The original paper applies this concept to language models, effectively addressing the issue of long-term dependencies. However, in the context of image captioning, this problem is less pertinent. The captions for images are succinct enough that the lack of long-term dependency handling does not significantly impact the outcome.

Nevertheless, we encounter problems associated with long-range dependencies when expanding the captioning task to video. The caption of a video depends on all frames within that video. Using the current ClipCap architecture, frames occurring later in the sequence significantly influence the final caption. To address this issue, we propose to utilize the kNN-memory transformer framework, as proposed by [Wu et al. 2022](https://arxiv.org/abs/2203.08913).

## Datasets

In keeping with the methodology of the ClipCap research, we will use the COCO dataset for the initial pretraining of our mapping network. Renowned for its diversity in everyday scene contexts, the COCO dataset comprises over 300,000 images, each with five associated captions. This dataset enables our model to learn from various objects and scenes, enhancing its ability to generalize and adapt to novel instances.

Following the pretraining, we will employ the [ActivityNet Captions](https://arxiv.org/pdf/1705.00754v1.pdf) dataset for finetuning. The ActivityNet Captions dataset provides a more task-specific data source explicitly designed for the temporal localization and captioning of activities. With 20,000 videos sourced from YouTube, amounting to 849 hours of footage, accompanied by 100,000 detailed descriptions of sequences of actions within the videos, it presents an optimal choice for our research.

### Preprocessing

Videos are converted into image frames at a rate of five frames per second (fps). Since our focus is solely on captioning and not temporal action localization, we extract all frames from the start to end of each caption, treating it as an independent video clip. These frames are individually embedded using the ClipCap model, then concatenated into a single tensor. The captions are tokenized using the GPT2 tokenizer. Given that we are only finetuning the model, we will use a small subset of the dataset. The final preprocessed datasets can be accessed via the links provided in our GitHub repository. The distribution of the dataset across different categories is outlined in the table below.

| __Split__   | __Train__ | __Test__ |
|-------------|-----------|----------|
| Videos      | 300       | 100      |
| Video Clips | ?         | ?        |


## Results
<!-- Results of your work (link that part with the code in the jupyter notebook) -->


## Conclusion


## References
1. Agrawal, Harsh et al. (2019). “Nocaps: Novel object captioning at scale”. In: Proceedings of the IEEE/CVF international conference on computer vision, pp. 8948–8957.
2. Caba Heilbron, Fabian et al. (2015). “Activitynet: A large-scale video benchmark for human activity understanding”. In: Proceedings of the ieee conference on computer vision and pattern recognition, pp. 961–970.
3. Lin, Tsung-Yi et al. (2014). “Microsoft coco: Common objects in context”. In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13. Springer, pp. 740–755.
4. Sharma, Piyush et al. (2018). “Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning”. In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2556–2565.
5. Wu, Yuhuai et al. (2022). “Memorizing Transformers”. In: International Conference on Learning Representations.
6. Sun, L., Jia, K., Yeung, D. Y., & Shi, B. E. (2015). Human action recognition using factorized spatio-temporal convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 4597-4605).
7. Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision (pp. 4489-4497).
8. Donahue, J., Anne Hendricks, L., Guadarrama, S., Rohrbach, M., Venugopalan, S., Saenko, K., & Darrell, T. (2015). Long-term recurrent convolutional networks for visual recognition and description. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2625-2634).

## Contributions
<!-- Close the notebook with a description of each student's contribution. -->
