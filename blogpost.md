<!-- omit from toc -->
# MemClipCap: Enhancing ClipCap with long-range dependency handling for video captioning
> Authors: Sebastiaan Dijkstra, Erik Buis, Jan Bakker, Jelke Matthijsse, Dennis Agafonov \
> Date: 29-5-2023 \
> Deep Learning 2 \
> University of Amsterdam

- [Introduction](#introduction)
  - [Related Work](#related-work)
  - [ClipCap Summary](#clipcap-summary)
  - [Main results](#main-results)
  - [Further findings](#further-findings)
- [Exploring ClipCap's Capabilities](#exploring-clipcaps-capabilities)
  - [Strengths](#strengths)
  - [Weaknesses](#weaknesses)
- [Our contribution](#our-contribution)
- [Datasets](#datasets)
  - [Pre-processing](#pre-processing)
  - [Data loading](#data-loading)
- [Experimental details](#experimental-details)
  - [Evaluation](#evaluation)
- [Results](#results)
- [Conclusion \& Discussion](#conclusion--discussion)
- [Contributions](#contributions)
- [References](#references)


# Introduction
<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well. -->
Image captioning is a multimodal task that involves generating textual descriptions of images. In our research project, we investigated a method called ClipCap[^mokady2021clipcap], which was explicitly designed for this task. One of the main advantages of ClipCap is that it is made of several building blocks that can easily be swapped out, meaning that it can be adapted to different datasets and tasks. Additionally, ClipCap requires relatively little training time, since it leverages pre-trained models that can be frozen or fine-tuned.

The key idea behind our research is that the original ClipCap architecture will not be good at captioning videos, since it does not remember information from previously seen input frames. We propose MemClipCap, a method that integrates a Memorizing Transformer[^wu2022memorizing] into ClipCap. By doing this, we hypothesize that it will be able to remember information from previous frames and thus generate better captions for videos than a baseline image captioning model that only looks at individual frames. A further goal is to stay true to ClipCap's design philosophy by keeping the model modular and efficient to train.

We will first provide an overview of relevant previous work, followed by a brief overview of the ClipCap method and its capabilities. Then, we will discuss the strengths and weaknesses of ClipCap, which will motivate our proposed enhancement. Following this, we will discuss how we implemented this enhancement and why we made certain design decisions. Next, we will present our results and conclusions, and finally, we will wrap up with a discussion of our findings.

## Related Work
Where image captioning is a task that has been extensively explored using various methods, video captioning has proven to be more challenging in existing research.

Progress in using long short-term memory (LSTM) models for image captioning has sparked interest in their applicability to video captioning[^gao2017video]. However, traditional LSTM models treat videos as static sequences and do not consider the selection of salient features. To address this limitation, an attention-based LSTM model was proposed[^gao2017video]. This model incorporates the attention mechanism to capture the important temporal structures of videos. However, LSTM-based models including this attention-based variant suffer from long training times. In contrast, our method leverages pre-trained models that can be frozen or fine-tuned to reduce training times.

Recent advancements in Transformer-based models have also provided options for video captioning. Earlier work introduced an encoder-decoder based Transformer model for end-to-end dense video captioning[^zhou2018dense]. While this method takes advantage of the strengths of Transformer-based models, it lacks adaptability in that it can not take advantage of progress in other Transformer-based models. In contrast, our method is modular and thus allows for easy swapping of the model components when better ones emerge in the future.

In the domain of multimodal vision-language modelling, alternative methods leverage pre-training with the BERT architecture[^li2020oscar] [^devlin2018bert] [^zhang2021vinvl] [^zhou2020unified] [^wang2021simvlm]. However, these methods are either limited to specific datasets[^li2020oscar] [^zhang2021vinvl] [^zhou2020unified], compromising their generalizability, or involve computationally intensive pre-training processes[^wang2021simvlm]. In contrast, the modularity of the ClipCap pipeline makes it efficient to train and relatively simple to implement, positioning it as a promising alternative to these methods.

## ClipCap Summary
The ClipCap method utilizes a pipeline of pre-trained models to generate captions for images. This pipeline consists of the CLIP[^radford2021learning] model, a mapping network, and a pre-trained language model (LM), namely GPT-2[^radford2019language] (see [figure 1]). The CLIP image encoder extracts high-level information from the visual data while the pre-trained LM generates the caption. The mapping network serves as a bridge between the two, linking the latent spaces of the two models.

More specifically, for a given image, the CLIP image encoder generates an embedding containing high-level information about the image. This embedding is passed through the mapping network to obtain a so-called "prefix", a list of embeddings associated with the image content. Finally, the prefix embeddings are used as input to GPT-2, which will generate the output caption autoregressively.

[figure 1]: images/ClipCap_approach_B.png "ClipCap pipeline"
![ClipCap pipeline][figure 1]
_[Figure 1]: Overview of the ClipCap pipeline when using training procedure B. In this approach, the CLIP and GPT-2 models are kept frozen, while the mapping network is a Transformer encoder that is trained from scratch._

## Main results
The authors experiment with two different training procedures for the ClipCap model pipeline. In the first approach (A), the CLIP model is kept frozen and GPT-2 is fine-tuned, while the mapping network is an MLP that is trained from scratch. In the second approach (B), the CLIP and GPT-2 models are both kept frozen, while the mapping network is a Transformer[^vaswani2017attention] encoder that is trained from scratch (see [figure 1]). The authors found that the first approach often yielded better results but required more training time. However, seeing as the accuracy decrease for approach B was relatively small, we decided to use this approach for our video captioning experiments.

Both approaches were evaluated on the Conceptual Captions[^sharma2018conceptual], NoCaps[^agrewal2019nocaps], and COCO[^lin2014coco] datasets, achieving state-of-the-art performance while requiring significantly less training time and data than previous methods. Additionally, the ClipCap architecture is more straightforward and faster than earlier methods.

## Further findings
Multiple other experiments were conducted to determine when ClipCap performs well and when it does not. For example, the authors found approach A results in a much more expressive model, but that this model is more susceptible to overfitting. Additionally, an interpretability study was conducted to further understand the model's inner workings, in which the prefix embeddings were interpreted as a sequence of tokens. The authors found that the interpretation is meaningful when both the mapping network and the LM are trained (approach A) but that it becomes essentially unreadable when only the mapping network is trained (approach B). They hypothesize that this happens because in approach B, the mapping network is able to exploit intricacies of the LM that steer it towards generating the correct caption. These can intuitively be seen as "tricks" that are not interpretable to humans.

The authors conducted multiple ablation studies to verify and motivate ClipCap's design choices. They found that the mapping network is crucial for the model to perform well and that a Transformer architecture is superior when the LM is frozen, while an MLP is more effective when the LM is additionally fine-tuned. Furthermore, the prefix length was a crucial hyperparameter; a prefix that is too short results in a lack of expressiveness, while a prefix that is too long results in a very large model that will be slow to train.


# Exploring ClipCap's Capabilities
<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->

## Strengths
One of the key strengths of ClipCap model is its use of pre-trained models. This allows the model to be trained on a small amount of data, which is a significant advantage over other methods that require large amounts of data to achieve state-of-the-art performance. It also ensures that training time remains low, since the number of trainable parameters stays constant when training only the mapping network.

Moreover, the ClipCap pipeline is modular, allowing for the swift adaptation or replacement of the image encoder, mapping network and/or LM component(s). This makes the model future-proof, as a more powerful image encoder or LM can be easily integrated into the pipeline. However, the authors did not explore the potential of using other pre-trained models. It would be useful to review this aspect in the form of an ablation study to understand the pipeline's strengths and weaknesses better.

Other than a dataset of image-text pairs, the model does not require any additional annotations. Additionally, the method is easy to understand and implement, and the authors provide a reference implementation that worked out-of-the-box.

## Weaknesses
Apart from images, other visual data such as video segments naturally have long-range dependencies between individual frames. When interpreted seperately, each frame may not contain enough information to generate a meaningful caption for the whole video. However, when interpreted jointly, emergent patterns may be observed that can be used to generate a more accurate caption. The ClipCap model does not account for these long-range dependencies as the mapping network's Transformer only has a limited context window, and therefore may not be able to generate accurate captions considering entire videos.


# Our contribution
<!-- Describe your novel contribution. -->
Our research aims to explore potential performance enhancements in video captioning by integrating a Memorizing Transformer[^wu2022memorizing] into ClipCap's mapping network. The original paper applies the concept of Memorizing Transformers to language models, aiming to address the challenge of long-term dependencies within textual information. We propose that this approach can be extended to incorporate visual information in the context of video captioning, where long-range dependencies are also prevalent.

The Memorizing Transformer is an extension of the original Transformer[^vaswani2017attention] architecture that incorporates so-called _memory layers_ (see [figure 2]). Typically, a Transformer can only take a fixed amount of tokens into account, called the _context window_, when calculating attention[^vaswani2017attention]. Increasing the size of this window is impractical, as the memory size required scales quadratically with the size of the context window[^kitaev2020reformer]. The Memorizing Transformer aims to improve upon this by adding an external memory component into the conventional self-attention mechanism.

Specifically, each memory layer has an external memory consisting of keys and values generated by the self-attention mechanism for previous tokens. This memory can be attended to by subsequent token chunks to retrieve valuable information. Rather than employing full attention over the weighted sum of all keys and values in the memory, an approximate k-nearest neighbours (kNN) algorithm is used to attend to the memory, identifying the most relevant keys and values (the number of which is a hyperparameter). These selected keys and values are then utilized to compute the so-called _top-k attention_.

[figure 2]: images/Memorizing_Transformer.png "Memorizing Transformer architecture"
![Memorizing Transformer architecture][figure 2]
_[Figure 2]: Overview of the Memorizing Transformer architecture[^wu2022memorizing]. The external memory (left) is updated after each training step, and it is accessed in subsequent steps using approximate kNN lookup._

The approximate kNN algorithm allows the external memory to be scaled quite significantly, as there exist efficient implementations of this algorithm. Additionally, since the external memory does not participate in backpropagation, it functions as a non-learnable parameter, enabling even more efficient scaling of memory size.

Thus, these memory layers incorporate both local self-attention (computed using the context window) and top-k attention (computed using the external memory). To calculate the next token, these two attention mechanisms are combined using a gating mechanism:
$$V_a = V_m \cdot g + V_c \cdot (1-g)$$
Here, the the gating mechanism involves a learnable scalar $g$ called the _gate_ (bounded between 0 and 1 through the sigmoid function), which determines the relative importance of each of the attention mechanisms $`V_m`$ (the top-k attention) and $`V_c`$ (the local attention) to calculate the combined result $`V_a`$ of both types of attention.


# Datasets
In line with the methodology of ClipCap, we will use the COCO dataset[^lin2014coco] for the initial pretraining stage of our mapping network. This allows the model to learn a general understanding of the relationship between images and text.

Following the pretraining, we will employ the ActivityNet Captions dataset[^krishna2017dense] for fine-tuning. The ActivityNet Captions dataset provides a more task-specific data source explicitly designed for captioning temporally spread-out activities in videos. It contains 20k videos with 100k detailed descriptions of sequences of events within them. To the best of our knowledge, this makes it the optimal choice for our research.

## Pre-processing
Videos are converted into image frames at a rate of 5 frames per second (fps). Since our focus is solely on captioning and not temporal action localization, we extract all frames from the start to the end of each captioned segment, treating each such segment as an independent _video clip_. We will denote the set of $C$ video clips by $`\{c_i\}^C_{i=1}`$ where the amount of frames of clip $`c_i`$ is given by $`f(c_i)`$ and the total number of frames is given by $`F = {\sum}_{i=1}^C f(c_i)`$. The frames are individually embedded using the CLIP image encoder, and the captions are tokenized using the GPT-2 tokenizer. Given that we are only finetuning the model, we will use only a subset of ActivityNet Captions. The final pre-processed datasets can be downloaded with the links provided in our GitHub repository[^github]. Some statistics of the dataset splits we used for training and testing are shown below:

|                 | __Train__ | __Test__ |
|-----------------|-----------|----------|
| Videos          | 300       | 100      |
| Video clips $C$ | 1112      | 371      |
| Frames $F$      | 198020    | 69731    |
| Length (hours)  | 275       | 97       |

## Data loading
Since the external memory layers of the Memorizing Transformer need to be updated sequentially, we have to process each video clip's frames one after the other. This would effectively make the batch size equal to 1, making the training process very inefficient. Instead, we parallelize the operation by processing multiple video clips at a time. An illustration of this parallel data loading process is shown in [figure 3]. In this visualization, video clips are layed out horizontally and stacked vertically, where the amount of rows correponds to the batch size $B$ and the amount of columns is the number of batching steps $S$.

[figure 3]: images/Dataloader_ActivityNet.png "Parallel data processing"
![Parallel data processing][figure 3]
_[Figure 3]: Schematic of the parallel data loading process. The video clip indices are just for illustration purposes; they correspond with neither the real video clips nor their lengths. The red blocks with $`\varnothing`$ represent padding frames._

It should be noted that since the video clips may not contain the same amount of frames, the rows in the table may not stop at the same step. When a step contains less then $B$ frames, the rest is filled with padding frames. Now, the more steps we have, the longer the training time of our model will be. Thus, we want to minimize the amount of steps $S$.

Mathematically speaking, each row can be seen as a _bin_, where we want to partition a list of numbers $`f(c_1), \dots, f(c_C)`$ into $B$ bins such that the maximum bin size is minimized. This corresponds to the multiway number parititioning problem[^graham1969bounds], which is a well-known problem in computer science[^wikipedia2023mnp]. Unfortunately, this problem is NP-complete[^garey1979computers], so we cannot find an optimal solution in polynomial time. To evaluate alternative approximation algorithms, the _approximation ratio_ can be used, which is the largest bin returned by such an algorithm divided by the largest sum in the optimal solution. In our code, we use the `prtpy` implementation[^coinor2023prtpy] of the Multifit algorithm[^coffman1978application] [^wikipedia2023multifit], which has a worst-case approximation ratio of $\frac{13}{11}$ in the general $B$-bin case[^yue1990exact]. This implies that the amount of padding frames we add will be bounded by $\frac{13}{11} B S - F$.


# Experimental details
The foundational model was initially pre-trained on the COCO dataset throughout ten epochs. This process was carried out by replicating the parameters used in ClipCap. Specifically, we utilized a batch size of 40, a prefix dimension of 512, and a prefix length of 10. The learning rate for this phase was set at 2e-5, and we implemented an AdamW optimizer to regulate the training process.

Following this pre-training phase, we fine-tuned the baseline and Memorizing Transformer-enhanced models. The fine-tuning was executed on the ActivityNet Captions dataset, again over ten epochs, while maintaining the same parameters as in the pre-training stage. This step ensured that the models effectively adapted to the specific requirements of the video captioning task.

Our model selection was guided by the validation loss, with the best-performing model based on this criterion chosen for further evaluation. The training process for all models was conducted on a single M1 Max GPU to ensure uniform computational resources.
Given the same pre-training conditions, the fine-tuning process allows us to observe how effectively our modification performs relative to the baseline. This strategy offers a fair and accurate comparison of both models' performance on the video captioning task.

Additionally, using the M1 Max GPU across all training processes maintains consistency and prevents computational discrepancies from influencing the results. By doing so, we ensure that any observed performance difference is genuinely a result of the model's structure and not due to external hardware factors.

## Evaluation
In order to evaluate our ClipMemCap model for the given video captioning task, we employed several captioning evaluation metrics. Similar to the original ClipCap paper, we validated our results using widely used evaluation metrics: BLEU[^papineni2002bleu], METEOR[^denkowski2014meteor] and ROUGE-L[^lin2004rouge].

BLEU is an automatic machine translation evaluation metric that can replace expensive human evaluations. It compares machine-generated translations to human-generated ground truths by calculating a similarity score based on the matching n-grams between the generated and ground truth captions. For our model validation, we used an average of BLEU-1, BLEU-2, BLEU-3, and BLEU-4, which consider unigrams, bigrams, trigrams, and fourgrams, respectively.

METEOR evaluates generated captions by aligning them with reference ground truths. This alignment is achieved by exhaustively identifying matches (according to different matchers; exact, stem, synonym, and paraphrase) between both captions. The alignment is then created using the largest subset of matches. The content and functions words within this alignment are identified and counted for every matcher within the matches of this type to calculate the precision and recall for the matched words. Finally, these measures are combined to compute a score that reflects the overall quality of the translation.

ROUGE-L is an evaluation method that measures sentence-to-sentence similarity based on the longest common subsequence (LCS) statistics between a generated sentences and a set of reference ground truths. The underlying idea behind this metric is that higher similarity between two candidate sentences results in a longer LCS. The length of the longest LCS is then used to calculate the LCS-based F-measure, which represents the ROUGE-L score.

The original ClipCap paper also utilized the CIDEr score, which quantifies the similarity based on multiple ground truth captions to assess the generalizability of the proposed model. However, since the ActivityNet Captions dataset does not provide multiple reference captions per video clip, the CIDEr metric is not applicable for our model validation.


# Results
<!-- Results of your work (link that part with the code in the jupyter notebook) -->

The table below shows the results of our ClipMemCap model compared to our three baseline models.

|                 | __ClipMemCap__ | __Baseline #1__ | __Baseline #2__ | __Baseline #3__ |
|-----------------|----------------|-----------------|-----------------|-----------------|
| Bleu_1          | 19.2153        | 20.8486         | 20.2418         | **21.4235**     |
| Bleu_2          | 8.0668         | 8.5534          | 8.8150          | **9.0937**      |
| Bleu_3          | 3.5685         | 3.5515          | 3.6270          | **3.8431**      |
| Bleu_4          | 1.0236         | 0.7883          | 1.2708          | **1.3604**      |
| METEOR          | 9.4595         | 9.8738          | 10.0628         | **10.2781**     |
| ROUGE_L         | 24.6301        | **24.9494**     | 23.7886         | 24.6975         |

The results of our study indicate that our ClipMemCap model did not demonstrate superior performance compared to the three baseline models across the evaluation metrics BLEU-1, BLEU-2, BLEU-3, BLEU-4, METEOR, and Rouge-L. These baseline models specifically consider, respectively, the first, middle, or last frame of a clip to generate captions. Additionally, the three baseline models exhibited similar performance levels.

One possible explanation for these findings is that our model solely focuses on captioned clips, which are initially extracted from the ActivityNet dataset based on the premise that each clip represents a distinct action or event within a specific timeframe. Consequently, the frames within a given clip tend to be highly similar to each other, as their similarities justify their extraction as a single clip. Thus, summarizing all frames of a clip (as our ClipMemCap model does) does not enhance performance compared to the baselines that rely on a single frame. Furthermore, the similarities among frames may account for the limited differences observed between the three baseline models. In essence, since the frames are sufficiently similar, generating accurate captions can be achieved with any one frame, regardless of whether it is the first, middle, or last frame.

Furthermore, when examining each of the four models individually, we observed a descending trend in the scores of the Bleu metrics, namely BLEU-1, BLEU-2, BLEU-3, and BLEU-4. These metrics evaluate the similarity of n-grams (n consecutive words) between the generated caption and the ground truth caption. As the length of these n-grams increases, the probability of finding matching sequences decreases. Consequently, matching single words (unigrams) has the highest probability, while matching longer sequences, such as 4-word sequences (4-grams), has a lower probability. Thus, BLEU-1 yields the highest score, while BLEU-4 yields the lowest score due to the higher difficulty of finding exact matches for longer sequences


# Conclusion & Discussion
> TODO Somewhere in the discussion, we should mention the following:
> While we only train a baseline model that uses the last frame of each video clip to caption the entire clip, we hypothesize that our enhancement would also be able to produce better captions than an image captioning model that would caption any other frame of the video clip. This is because our enhancement is able to integrate information from all frames of the video clip, while the image captioning model would only be able to use information from a single frame.


> TODO Somewhere in the discussion, we should mention the following:
> While we only train a model to caption videos, we hypothesize that our enhancement would also be able to generalize to other modalities such as audio files, because we integrate a general form of long-range dependency handling that is not specific to videos.


# Contributions
<!-- Close the notebook with a description of each student's contribution. -->


# References
[^github]: Our GitHub repository. https://github.com/SebastiaanJohn/knn-memory-clipcap

[^graham1969bounds]: Graham, R. L. (1969). Bounds on multiprocessing timing anomalies. SIAM journal on Applied Mathematics, 17(2), 416-429.

[^garey1979computers]: Garey, M. R., & Johnson, D. S. (1979). Computers and intractability (Vol. 174). San Francisco: freeman, 238.

[^coinor2023prtpy]: The `prtpy` Python library. https://github.com/coin-or/prtpy

[^coffman1978application]: Coffman, Jr, E. G., Garey, M. R., & Johnson, D. S. (1978). An application of bin-packing to multiprocessor scheduling. SIAM Journal on Computing, 7(1), 1-17.

[^wikipedia2023mnp]: The multiway number partitioning problem. https://en.wikipedia.org/wiki/Multiway_number_partitioning

[^wikipedia2023multifit]: The Multifit algorithm. https://en.wikipedia.org/wiki/Multifit_algorithm

[^yue1990exact]: Yue, M. (1990). On the exact upper bound for the multifit processor scheduling algorithm. Annals of Operations Research, 24(1), 233-259.

[^mokady2021clipcap]: Mokady, R., Hertz, A., & Bermano, A. H. (2021). Clipcap: Clip prefix for image captioning. arXiv preprint arXiv:2111.09734.

[^vaswani2017attention]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.

[^radford2021learning]: Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021, July). Learning transferable visual models from natural language supervision. In International conference on machine learning (pp. 8748-8763). PMLR.

[^radford2019language]: Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog, 1(8), 9.

[^krishna2017dense]: Krishna, R., Hata, K., Ren, F., Fei-Fei, L., & Carlos Niebles, J. (2017). Dense-captioning events in videos. In Proceedings of the IEEE international conference on computer vision (pp. 706-715).

[^agrewal2019nocaps]: Agrawal, H. et al. (2019). "Nocaps: Novel object captioning at scale". In: Proceedings of the IEEE/CVF international conference on computer vision (pp. 8948–8957).

[^caba2015activitynet]: Caba Heilbron, F. et al. (2015). "Activitynet: A large-scale video benchmark for human activity understanding". In: Proceedings of the ieee conference on computer vision and pattern recognition (pp. 961–970).

[^lin2014coco]: Lin, T et al. (2014). "Microsoft coco: Common objects in context". In: Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014 (pp. 740–755).

[^sharma2018conceptual]: Sharma, P. et al. (2018). "Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning". In: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (pp. 2556–2565).

[^wu2022memorizing]: Wu, Y. et al. (2022). "Memorizing Transformers". In: International Conference on Learning Representations.

[^kitaev2020reformer]: Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The efficient transformer. arXiv preprint arXiv:2001.04451.

[^sun2015human]: Sun, L. et al. (2015). "Human action recognition using factorized spatio-temporal convolutional networks". In: Proceedings of the IEEE international conference on computer vision (pp. 4597-4605).

[^tran2015learning]: Tran, D. et al. (2015). "Learning Spatiotemporal Features with 3D Convolutional Networks". In: Proceedings of the IEEE international conference on computer vision (pp. 4489-4497).

[^donahue2015long]: Donahue, J.et al. (2015). "Long-term recurrent convolutional networks for visual recognition and description". In: Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2625-2634).

[^gao2017video]: Gao, L. et al. (2017). "Video Captioning With Attention-Based LSTM and Semantic Consistency". In: IEEE Transactions on Multimedia (pp. 2045-2055).

[^li2020oscar]: Li, X. et al. (2020). "Oscar: Object-semantics aligned pre-training for vision-language tasks". In: European Conference on Computer Vision (pp. 121–137).

[^devlin2018bert]: Devlin, J., Chang, M., Lee, K. and Toutanova, K. (2018). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding".

[^zhang2021vinvl]: Pengchuan Zhang et al. (2021). "VinVL: Revisiting visual representations in vision-language models". In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5579–5588).

[^zhou2020unified]: Zhou, L. et al. (2020). "Unified vision-language pretraining for image captioning and VQA". In: Proceedings of the AAAI Conference on Artificial Intelligence (pp. 13041–13049).

[^wang2021simvlm]: Wang, Z. (2021). "SimVLM: Simple Visual Language Model Pretraining with Weak Supervision".

[^zhou2018dense]: Transformer-based dense captioning - Zhou, L. et al. (2018). “End-to-End Dense Video Captioning with Masked Transformer”.

[^papineni2002bleu]: Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). "Bleu: a method for automatic evaluation of machine translation". In Proceedings of the 40th annual meeting of the Association for Computational Linguistics (pp. 311-318).

[^denkowski2014meteor]: Denkowski, M., & Lavie, A. (2014, June). "Meteor universal: Language specific translation evaluation for any target language". In Proceedings of the ninth workshop on statistical machine translation (pp. 376-380).

[^lin2004rouge]: Lin, C. Y., & Och, F. J. (2004, July). Automatic evaluation of machine translation quality using longest common subsequence and skip-bigram statistics. In Proceedings of the 42nd Annual Meeting of the Association for Computational Linguistics (ACL-04) (pp. 605-612).


