<!-- omit from toc -->
# MemClipCap: Enhancing ClipCap with long-range dependency handling for video captioning
> Authors: Sebastiaan Dijkstra, Erik Buis, Jan Bakker, Jelke Matthijsse, Dennis Agafonov \
> Date: 28-5-2023 \
> Deep Learning 2 \
> University of Amsterdam

- [Abstract](#abstract)
- [Introduction](#introduction)
  - [Related Work](#related-work)
  - [ClipCap Summary](#clipcap-summary)
- [Exploring ClipCap's Capabilities](#exploring-clipcaps-capabilities)
  - [Strengths](#strengths)
  - [Weaknesses](#weaknesses)
- [Methodology](#methodology)
  - [Memorizing Transformer](#memorizing-transformer)
  - [Datasets](#datasets)
    - [Pre-processing](#pre-processing)
    - [Data loading](#data-loading)
  - [Training](#training)
- [Experiments](#experiments)
  - [Training and evaluation procedure](#training-and-evaluation-procedure)
  - [Conducted experiments](#conducted-experiments)
  - [Evaluation](#evaluation)
- [Results](#results)
  - [Main results](#main-results)
    - [Quantitative analysis](#quantitative-analysis)
      - [Effect of clip length on performance](#effect-of-clip-length-on-performance)
    - [Qualitative analysis](#qualitative-analysis)
  - [Ablation studies](#ablation-studies)
    - [Batch size](#batch-size)
      - [Analysis of accuracy measurements](#analysis-of-accuracy-measurements)
    - [Initial clip vs all clips](#initial-clip-vs-all-clips)
- [Conclusion](#conclusion)
- [Contributions](#contributions)
- [References](#references)


# Abstract
This study aims to enhance the ClipCap method for video captioning tasks by incorporating a Memorizing Transformer into the existing architecture. Our proposed model, MemClipCap, exploits long-range dependencies within video clips to generate more accurate textual descriptions. We conducted experiments on the ActivityNet Captions dataset and compared the performance of MemClipCap with three baseline models that use a single frame to generate captions. Despite the theoretical potential of integrating memory layers, our results demonstrated that MemClipCap did not consistently outperform the baseline models across the chosen evaluation metrics. However, our MemClipCap model exhibits flexibility in processing videos of arbitrary lengths, indicating potential applications involving diverse video content. Future research directions involve more extensive hyperparameter searches, exploring the integration of multiple modalities, and evaluating the performance of MemClipCap on longer videos.


# Introduction
<!-- Introduction: An analysis of the paper and its key components. Think about it as a nicely formatted review as you would see on OpenReview.net. It should contain one paragraph of related work as well. -->
Image captioning is a multimodal task that involves generating textual descriptions of images. In our research project, we investigated a method called ClipCap[^mokady2021clipcap], which was explicitly designed for this task. One of the main advantages of ClipCap is that it is made of several building blocks that can easily be swapped out, meaning that it can be adapted to different datasets and tasks. Additionally, ClipCap requires relatively little training time since it leverages pre-trained models that can be frozen or fine-tuned.

The key idea behind our research is that the original ClipCap architecture will not be good at captioning videos since it does not remember information from previously seen input frames. We propose MemClipCap, a method that integrates a Memorizing Transformer[^wu2022memorizing] into ClipCap. Our method is illustrated in [figure 1]. By doing this, we hypothesize that it will be able to remember information from previous frames and thus generate better captions for videos than a baseline image captioning model that only looks at individual frames. A further goal is to stay true to ClipCap's design philosophy by keeping the model modular and efficient to train.

We will first provide an overview of the relevant previous work, followed by a summary of the ClipCap method and its capabilities. Next, we will discuss the strengths and weaknesses of ClipCap, motivating our proposed enhancement, MemClipCap. We will then describe the implementation of this enhancement and the rationale behind our design choices. Following this, we will present our experimental results and discuss the conclusions drawn from these findings. Next, we will provide multiple ablation studies to further investigate the MemClipCap model's performance. Finally, we will wrap up with a comprehensive analysis of our research and its implications.

[figure 1]: images/MemClipCap.png "MemClipCap model"
![MemClipCap model][figure 1] \
_[Figure 1]: Overview of the MemClipCap model._


## Related Work
Where image captioning is a task that has been extensively explored using various methods, video captioning has proven to be more challenging in existing research.

Progress in using long short-term memory (LSTM) models for image captioning has sparked interest in their applicability to video captioning[^gao2017video]. However, traditional LSTM models treat videos as static sequences and do not consider the selection of salient features. To address this limitation, an attention-based LSTM model was proposed[^gao2017video]. This model incorporates the attention mechanism to capture the essential temporal structures of videos. However, LSTM-based models, including this attention-based variant, have long training times. In contrast, our method leverages pre-trained models that can be frozen or fine-tuned to reduce training times.

Recent advancements in Transformer-based models have also provided options for video captioning. Earlier work introduced an encoder-decoder based Transformer model for end-to-end dense video captioning[^zhou2018dense]. While this method takes advantage of the strengths of Transformer-based models, it lacks adaptability because it can not take advantage of progress in other Transformer-based models. In contrast, our method is modular and thus allows for easy swapping of the model components when better ones emerge in the future.

In the domain of multimodal vision-language modelling, alternative methods leverage pre-training with the BERT architecture[^li2020oscar] [^devlin2018bert] [^zhang2021vinvl] [^zhou2020unified] [^wang2021simvlm]. However, these methods are either limited to specific datasets[^li2020oscar] [^zhang2021vinvl] [^zhou2020unified], compromising their generalizability, or involve computationally intensive pre-training processes[^wang2021simvlm]. In contrast, the modularity of the ClipCap pipeline makes it efficient to train and relatively simple to implement, positioning it as a promising alternative to these methods.

## ClipCap Summary
The ClipCap method utilizes a pipeline of pre-trained models to generate captions for images. This pipeline consists of the CLIP[^radford2021learning] model, a mapping network, and a pre-trained language model (LM), namely GPT-2[^radford2019language] (see [figure 2]). The CLIP image encoder extracts high-level information from the visual data while the pre-trained LM generates the caption. The mapping network bridges the two, linking the latent spaces of the two models.

More specifically, for a given image, the CLIP image encoder generates an embedding containing high-level information about the image. This embedding is passed through the mapping network to obtain a so-called "prefix", a list of embeddings associated with the image content. Finally, the prefix embeddings are used as input to GPT-2, which will generate the output caption autoregressively.

[figure 2]: images/ClipCap_approach_B.png "ClipCap pipeline"
![ClipCap pipeline][figure 2] \
_[Figure 2]: Overview of the ClipCap pipeline when using training procedure B. In this approach, the CLIP and GPT-2 models are kept frozen, while the mapping network is a Transformer encoder that is trained from scratch._

The authors experiment with two different training procedures for the ClipCap model pipeline. In the first approach (A), the CLIP model is kept frozen and GPT-2 is fine-tuned, while the mapping network is an MLP that is trained from scratch. In the second approach (B), the CLIP and GPT-2 models are both kept frozen, while the mapping network is a Transformer[^vaswani2017attention] encoder that is trained from scratch (see [figure 2]). The authors found that the first approach often yielded better results but required more training time. However, seeing as the accuracy decrease for approach B was relatively small, we decided to use this approach for our video captioning experiments.

Both approaches were evaluated on the Conceptual Captions[^sharma2018conceptual], NoCaps[^agrewal2019nocaps], and COCO[^lin2014coco] datasets, achieving state-of-the-art performance while requiring significantly less training time and data than previous methods. Additionally, the ClipCap architecture is more straightforward and faster than earlier methods.


# Exploring ClipCap's Capabilities
<!-- Exposition of its weaknesses/strengths/potential which triggered your group to come up with a response. -->

## Strengths
One key strength of the ClipCap model is its use of pre-trained models. This allows the model to be trained on a small amount of data, which is a significant advantage over other methods that require large amounts of data to achieve state-of-the-art performance. It also ensures that training time remains low since the number of trainable parameters stays constant when training only the mapping network.

Moreover, the ClipCap pipeline is modular, allowing for the swift adaptation or replacement of the image encoder, mapping network and LM component(s). This makes the model future-proof, as a more powerful image encoder or LM can be easily integrated into the pipeline. However, the authors did not explore the potential of using other pre-trained models. It would be helpful to review this aspect in the form of an ablation study to understand the pipeline's strengths and weaknesses better.

Besides a dataset of image-text pairs, the model does not require additional annotations. Additionally, the method is easy to understand and implement, and the authors provide a reference implementation that worked out of the box.

## Weaknesses
Apart from images, other visual data, such as video segments, naturally have long-range dependencies between individual frames. When interpreted separately, each frame may not contain enough information to generate a meaningful caption for the whole video. However, when interpreted jointly, emergent patterns may be observed that can be used to generate a more accurate caption. The ClipCap model does not account for these long-range dependencies as the mapping network's Transformer only has a limited context window and, therefore, may be unable to generate accurate captions considering entire videos.


# Methodology

## Memorizing Transformer
<!-- Describe your novel contribution. -->
Our research aims to explore potential performance enhancements in video captioning by integrating a Memorizing Transformer[^wu2022memorizing] into ClipCap's mapping network. The original paper applies the concept of Memorizing Transformers to language models, aiming to address the phenomenon of long-term dependencies within textual information. We propose extending this approach to incorporate visual information in the context of video captioning, where long-range dependencies are also prevalent.

The Memorizing Transformer is an extension of the original Transformer[^vaswani2017attention] architecture that incorporates so-called _memory layers_ (see [figure 3]). Typically, a Transformer can only account for a fixed amount of tokens, called the _context window_, when calculating attention[^vaswani2017attention]. Increasing the size of this window is impractical, as the memory size required scales quadratically with the context window size[^kitaev2020reformer]. The Memorizing Transformer aims to improve this by adding an external memory component to the conventional self-attention mechanism.

Specifically, each memory layer has an external memory consisting of keys and values generated by the self-attention mechanism for previous tokens. Subsequent token chunks can attend to this memory to retrieve valuable information. Rather than employing full attention over the weighted sum of all keys and values in the memory, an approximate k-nearest neighbours (kNN) algorithm is used to attend to the memory, identifying the most relevant keys and values (the number of which is a hyperparameter). These selected keys and values are then utilized to compute the so-called _top-k attention_.

[figure 3]: images/Memorizing_Transformer.png "Memorizing Transformer architecture"
![Memorizing Transformer architecture][figure 3] \
_[Figure 3]: Overview of the Memorizing Transformer architecture[^wu2022memorizing]. The external memory (left) is updated after each training step, and it is accessed in subsequent steps using approximate kNN lookup._

The approximate kNN algorithm allows the external memory to be scaled quite significantly, as efficient implementations exist. Additionally, since the external memory does not participate in backpropagation, it functions as a non-learnable parameter, enabling even more efficient scaling of memory size.

Thus, these memory layers incorporate local self-attention (computed using the context window) and top-k attention (computed using the external memory). These two attention mechanisms are combined using a gating mechanism to calculate the next token:
$$V_a = V_m \cdot g + V_c \cdot (1-g)$$
Here, the gating mechanism involves a learnable scalar $g$ called the _gate_ (bounded between 0 and 1 through the sigmoid function), which determines the relative importance of each of the attention mechanisms $`V_m`$ (the top-k attention) and $`V_c`$ (the local attention) to calculate the combined result $`V_a`$ of both types of attention.

## Datasets
In line with the methodology of ClipCap, we use the COCO dataset[^lin2014coco] for the initial pre-training stage of our mapping network. This allows the model to learn a general understanding of the relationship between images and text.

Following the pre-training, we will employ the ActivityNet Captions dataset[^krishna2017dense] for fine-tuning. The ActivityNet Captions dataset provides a more task-specific data source explicitly designed for captioning temporally spread-out activities shown in videos. It contains 20k videos with 100k detailed descriptions of sequences of events. To the best of our knowledge, this makes it the optimal choice for our research.

### Pre-processing
Videos are converted into image frames at a rate of 5 frames per second (fps). We extract all frames from the start to the end of each captioned segment, treating each segment as an independent _video clip_. We will denote the set of video clips by $`\{c_i\}^C_{i=1}`$, where the amount of frames of clip $`c_i`$ is given by $`f(c_i)`$. The frames are individually embedded using the CLIP image encoder, and the captions are tokenized using the GPT-2 tokenizer. Given that we are only fine-tuning the model, we will use only a subset of ActivityNet Captions. The final pre-processed datasets can be downloaded using the links in our GitHub repository[^github].

A visualization of an ActivityNet Captions video clip is shown in [figure 4]. Each of the captions in this figure is associated with a specific time interval within the video clip. It should be noted that even though we treat the video clips as independent, they may not be, as references to previously described activities or objects may be made (e.g. "Eventually", "Another", "The woman").

[figure 4]: images/ActivityNet_Captions.png "ActivityNet Captions video clip example"
![ActivityNet Captions video clip example][figure 4] \
_[Figure 4]: Visualization of an ActivityNet Captions video clip[^krishna2017dense]. The video clip is shown on the left, and the corresponding caption is shown on the right._

### Data loading
Since the external memories of the Memorizing Transformer need to be updated sequentially, we have to process each video clip's frames one after the other. This effectively makes the batch size equal to 1, making the training process very inefficient. Instead, we parallelize the operation by processing multiple video clips simultaneously. This parallel data loading process is illustrated in [figure 5]. In this visualization, video clips are laid out horizontally and stacked vertically, where the number of rows corresponds to the batch size $B$, and the number of columns is the number of batching steps $S$.

[figure 5]: images/Dataloader_ActivityNet_Captions.png "Parallel data processing"
![Parallel data processing][figure 5] \
_[Figure 5]: Schematic of the parallel data loading process. The video clip indices are just for illustration purposes; they correspond with neither the real video clips nor their lengths._

It should be noted that since the video clips may not contain the same amount of frames, the rows in the table may not stop at the same step. When a step contains less than $B$ frames, the rest is filled with padding frames (represented in [figure 5] by red $`\varnothing`$ blocks). Note that the more steps we have, the longer the training time of our model will be. Thus, we want to minimize the number of steps $S$.

Mathematically, each row can be seen as a _bin_ $b$ in which we can put a set of video clips $`C^{(b)}`$. The _bin size_ is the total amount of frames that the bin contains and is given by $`f(b) = {\sum}_{c_j \in C^{(b)}} f(c_j)`$. Now, to minimize $S$, we want to partition the video clips into $B$ bins such that the maximum bin size is minimized. Thus, we are looking for the set of bins $`\{C^{(b)}\}_{b=1}^B`$ that minimizes $`\max(f(b) \mid b \in \{1, \dots, B\})`$. This corresponds to the multiway number partitioning problem[^graham1969bounds], a well-known computer science problem[^wikipedia2023mnp]. Unfortunately, this problem is NP-complete[^garey1979computers], so we cannot find an optimal solution in polynomial time. The _approximation ratio_ can be used to evaluate alternative approximation algorithms, which is the largest bin size returned by such an algorithm divided by the largest bin size in the optimal solution. In our code, we use the `prtpy` implementation[^coinor2023prtpy] of the Multifit algorithm[^coffman1978application] [^wikipedia2023multifit], which has a worst-case approximation ratio of $\frac{13}{11}$ in the general $B$-bin case[^yue1990exact]. This implies that the extra training time of our model is at most 18.2% larger than the optimal configuration (note that this is an upper bound since empty frames will not prolong the training time by much).

## Training
Regarding model architecture, MemClipCap is simply ClipCap with a Memorizing Transformer as its mapping network. This Memorizing Transformer incorporates one or more memory layers and operates on batches of single frames instead of the original paper's usage of token sequences[^wu2022memorizing]. Only the mapping network of MemClipCap is trained, while both the image encoder (CLIP) and the textual decoder (GPT-2) are kept frozen.

The training data is loaded in batches. As described in the [data loading](#data-loading) section, each batch index corresponds to a different series of video clips; for example, in [figure 5], clips $`c_1, c_{11}, \dots, c_{91}`$ are all laid out one frame after another at index 0. Thus, each batch index requires a separate external memory at each memorizing layer, which is cleared at the start of a new video clip.

The forward pass of the MemClipCap model begins with the computation of the CLIP embeddings for all frames in the batch. Each CLIP embedding is then mapped to the LM's embedding space by the Memorizing Transformer (see left side of [figure 1]). In doing so, the self-attention mechanism of each Transformer layer generates a (key, value) pair for every input. The (key, value) pair generated at layer $l$ for an input at batch index $i$ is appended to the external memory $M_{l, i}$ (see also [figure 3]). Before appending, the top-k nearest (key, value) pairs for that input are first retrieved. This allows us to incorporate information from previous frames using the gating mechanism described in the [Memorizing Transformer](#memorizing-transformer) section.

The remainder of the forward pass and the full backward pass are executed only with those frames in the batch that are the last in a video clip (see right side of [figure 1]). This is because, during inference, a caption should only be generated once the model has seen all frames; for other frames, it is sufficient to update the external memories. During training, the forward pass continues by concatenating the prefix generated by the mapping network to the ground truth caption embedding and feeding the result to the LM. Like with ClipCap, the training objective is to predict the caption tokens conditioned on the prefix in an autoregressive fashion. A cross-entropy loss is therefore used to update the mapping network's parameters in the backward pass. In contrast, during inference, beam search is used to generate a caption based on the LM's outputs given the prefix that the mapping network generated.

We pre-trained a _base model_ and fine-tuned the baselines with MemClipCap's architecture, which is shown in [figure 1]. However, in this process, we regarded each input image as a video clip of length 1, meaning all memories were cleared at the end of each forward pass, which effectively disabled the memory mechanism and made the model equivalent to ClipCap's architecture.


# Experiments

## Training and evaluation procedure
We initially pre-trained a _base model_ on the full COCO dataset, which we use for the baselines and our proposed model. This allows us to observe how effectively our modification performs relative to the baselines, which offers a fair and accurate comparison of both models' performance on the video captioning task. The pre-training phase was executed for 10 epochs, with a batch size of 40, a prefix embedding dimension of 512, and a prefix length of 10. We used the AdamW[^loshchilov2017decoupled] optimizer with a learning rate of 2e-5.

Following this pre-training phase, we fine-tuned the base model using a subset of the ActivityNet Captions dataset called the __train__ set. The fine-tuning process was executed over 10 epochs while maintaining the same parameters as in the pre-training stage. For the Memorizing Transformer, we selected layers 4 and 5 as memory layers with 32 retrieved memories each and set the maximum amount of kNN memories to 64000.

We used a __validation__ set to select the best model out of the 10 fine-tuning epochs based on the validation loss and then evaluated the selected model on the __test__ set.

Furthermore, we discriminate between models that were only trained and evaluated on the initial video clip of each video and models that were trained and evaluated on all video clips. We refer to these models as the _initial clip_ and _all clips_ models, respectively. Note that we keep the total amount of video clips constant in each case. Please refer to [table 1](#table-dataset-splits) for an overview of the dataset splits.

<!-- I use HTML here because default Markdown can't render multicolumn/multirow cells. Also, I wanted to align the cells inidividually, which is not possible with the standard Markdown syntax. Also also, I wanted to give the table an ID so I could reference it, which is not possible with standard Markdown. -->
<table id="table-dataset-splits">
  <tr>
    <th colspan=2></th>
    <th style="text-align: center;">Videos</th>
    <th style="text-align: center;">Video clips</th>
    <th style="text-align: center;">Frames</th>
    <th style="text-align: center;">Length (hh:mm:ss)</th>
  </tr>
  <tr>
    <td rowspan=3 style="text-align: center;"><b>Initial clip</b></th>
    <td style="text-align: center;"><b>Train</b></td>
    <td style="text-align: right;">2 000</td>
    <td style="text-align: right;">2 000</td>
    <td style="text-align: right;">396 712</td>
    <td style="text-align: center;">22:02:22</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Validation</b></td>
    <td style="text-align: right;">250</td>
    <td style="text-align: right;">250</td>
    <td style="text-align: right;">52 954</td>
    <td style="text-align: center;">02:56:30</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Test</b></td>
    <td style="text-align: right;">500</td>
    <td style="text-align: right;">500</td>
    <td style="text-align: right;">113 500</td>
    <td style="text-align: center;">06:18:20</td>
  </tr>
  <tr>
    <td rowspan=3 style="text-align: center;"><b>All clips</b></th>
    <td style="text-align: center;"><b>Train</b></td>
    <td style="text-align: right;">540</td>
    <td style="text-align: right;">2 043</td>
    <td style="text-align: right;">364 657</td>
    <td style="text-align: center;">20:15:31</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Validation</b></td>
    <td style="text-align: right;">67</td>
    <td style="text-align: right;">232</td>
    <td style="text-align: right;">41 787</td>
    <td style="text-align: center;">02:19:17</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Test</b></td>
    <td style="text-align: right;">133</td>
    <td style="text-align: right;">477</td>
    <td style="text-align: right;">94 956</td>
    <td style="text-align: center;">05:16:31</td>
  </tr>
</table>

_[Table 1](#table-dataset-splits): ActivityNet Captions dataset splits we used for training, validation and testing._

The training process for all models was conducted on a single M1 Max GPU. By keeping the hardware consistent across all training sessions, we ensure that any observed performance difference is genuinely a result of the model's structure, which allows us to directly compare the models' training time.

## Conducted experiments
While the main results were produced by training and evaluating our MemClipCap model, we also introduced three baseline models to obtain an in-depth understanding of MemClipCap's performance. The baselines generate captions by only looking at each clip's first, middle, or last frame. By comparing our proposed model with these baseline models, we can assess the effectiveness of incorporating external memory in the context of video captioning tasks.

The choice of first, middle, and last frames was motivated by the idea that these frames represent different stages of a given clip. Comparing these baselines provides a better understanding of MemClipCap's ability to capture relevant information across different stages of the video and whether retaining the memory of previous frames causes the model to generate more accurate captions.

The "last frame" baseline in particular is useful to evaluate whether MemClipCap effectively utilizes its memory given that it can use information from the clip's earlier stages, while the "last frame" baseline cannot. This would help identify whether long-range dependencies in the MemClipCap model were actually being utilized.

Additionally, for all MemClipCap models and baselines, we conducted ablation studies with different batch sizes, comparing the original batch size of 40 to a smaller batch size of 5. Furthermore, we also experimented with comparing the performance of the models when using only the initial clip of each video to when using all clips of each video.

## Evaluation
In order to evaluate our MemClipCap model for the given video captioning task, we employed several captioning evaluation metrics. Similar to the original ClipCap paper, we validated our results with the widely used evaluation metrics BLEU[^papineni2002bleu], METEOR[^denkowski2014meteor], and ROUGE-L[^lin2004rouge]. Concerning BLEU, we used the BLEU-1, BLEU-2, BLEU-3, and BLEU-4 variants, which measure the n-gram precision of the generated captions compared to the ground truth captions.

The original ClipCap paper also utilized the CIDEr score. However, this metric cannot be used since it requires multiple ground truth captions per video clip, which the ActivityNet Captions dataset does not provide.


# Results
<!-- Results of your work (link that part with the code in the jupyter notebook) -->

## Main results

### Quantitative analysis
[Table 2](#table-main-results) below shows the main results of our MemClipCap model compared to our three baseline models. The table only shows the best performing models for each category. All best models used the initial clip of each video. For our best MemClipCap model, a batch size of 40 performed best, and for our best baseline models, we found that a batch size of 5 exhibited the best performance. We discuss all other results in the [ablation studies](#ablation-studies) section.

<table id="table-main-results">
  <tr>
    <th style="text-align: center;"></th>
    <th style="text-align: center;">BLEU-1</th>
    <th style="text-align: center;">BLEU-2</th>
    <th style="text-align: center;">BLEU-3</th>
    <th style="text-align: center;">BLEU-4</th>
    <th style="text-align: center;">METEOR</th>
    <th style="text-align: center;">ROUGE-L</th>
  </tr>
  <tr>
    <td style="text-align: center;"><b>MemClipCap</b><br>(bs 40, initial clip)</td>
    <td>19.2153</td>
    <td>8.0668</td>
    <td>3.5685</td>
    <td>1.0236</td>
    <td>9.4595</td>
    <td>24.6301</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #1</b><br>(first frame, bs 5, initial clip)</td>
    <td>20.8486</td>
    <td>8.5534</td>
    <td>3.5515</td>
    <td>0.7883</td>
    <td>9.8738</td>
    <td><b><i>24.9494<b></i></td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #2</b><br>(middle frame, bs 5, initial clip)</td>
    <td>20.2418</td>
    <td>8.8150</td>
    <td>3.6270</td>
    <td>1.2708</td>
    <td>10.0628</td>
    <td>23.7886</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #3</b><br>(last frame, bs 5, initial clip)</td>
    <td><b><i>21.4235</i></b></td>
    <td><b><i>9.0937</i></b></td>
    <td><b><i>3.8431</i></b></td>
    <td><b><i>1.3604</i></b></td>
    <td><b><i>10.2781</i></b></td>
    <td>24.6975</td>
  </tr>
</table>

_[Table 2](#table-main-results): Main results of our best MemClipCap model compared to our best three baseline models. The best model for each metric is shown in __bold italics__._

The results of our study indicate that MemClipCap did not demonstrate superior performance compared to the three baseline models across all evaluation metrics. This suggests that one frame may already provide sufficient information for generating accurate captions.

One possible explanation for these findings is that each clip in the ActivityNet Captions dataset is designed to represent only a single action or event. As a result, the frames within a clip are highly similar, which could justify why MemClipCap did not demonstrate superior performance compared to the baseline models that rely on just a single frame.

One more interesting finding is that the baseline model's performance was generally ordered as last frame > middle frame > first frame. This suggests that the last frame of a clip may contain more information or be more representative than the first frame. However, as stated earlier, the similarity among frames is high, which may explain the limited differences observed between the three baseline models. Since the frames are sufficiently similar, generating accurate captions can be achieved with any frame, regardless of its position within the clip.


#### Effect of clip length on performance
One intriguing area we wished to explore was the effect of clip length on the performance of our models. To this end, we plotted the performance of each of the best models against the clip lengths. [Figure 6] shows these results for MemClipCap and the three baseline models.

Upon closer examination, it appears that MemClipCap and the middle frame baseline perform better as the number of frames in a clip increases. On the other hand, the last frame baseline demonstrates better performance on short clips while performing increasingly worse on longer ones. This trend indicates that MemClipCap is actually remembering content from previous frames and that not only the last frame is taken into account.

However, the majority of video clips in our test set consist of a small number of frames, as illustrated by the histogram in [figure 7]. This clearly shows that the test results are biased towards the performance of models on shorter clips. As a result, even though MemClipCap's performance on a wide range of clip lengths appears to be relatively robust, its overall score is lower when compared to the last frame baseline, even though this baseline is only slightly better on short clips.

[figure 6]: images/plots/nframes_vs_metrics.png "Clip length vs. performance for all best models"
![Clip length vs. performance for all best models][figure 6] \
_[Figure 6]: Performance of our best MemClipCap and baseline models on the test set as a function of the number of frames in the clip. A window size of 25 frames was used to smooth the curves._

[figure 7]: images/plots/Initial_clip_nframes_frequency.png "Clip length histogram"
![Clip length histogram][figure 7] \
_[Figure 7]: Histogram of the clip length (number of frames) when using the initial clip of each video._


### Qualitative analysis
For the qualitative analysis, two video clips were selected which provide interesting insights into the generated captions by our models. [Figure 8] shows an overview of these video clips.

[figure 8]: images/Qualitative_analysis.png "Qualitative analysis"
![Qualitative analysis][figure 8] \
_[Figure 8]: Reference and generated captions with their corresponding METEOR scores for two video clips. For each clip, the ground truth caption as provided by ActivityNet is shown, followed by the caption generated by the MemClipCap model (batch size 40, initial clip), the caption generated by the third baseline model (last frame, batch size 5, initial clip), and two random frames from that clip. For both generated captions, the METEOR score is provided._

For the video on the left, the MemClipCap model exhibits a relatively high score when evaluated with the METEOR metric. This can also be seen from the fact that the MemClipCap caption is almost identical to the ground truth caption, missing only one detail regarding the chair. Conversely, the baseline model generates a more elaborate caption, albeit containing potentially inaccurate information (i.e. the uncertainty of the man's location), but it is assigned a lower score. Interestingly, we observed the same phenomenon for many other captions, which suggests that providing incorrect information in the caption is more detrimental than omitting certain details.

The video clip shown on the right also provides interesting insights. Although the caption generated by MemClipCap is not wrong, it lacks the detailed description which the ground truth caption provides. The METEOR metric punishes this behaviour, which results in a very low score. Looking at the baseline model, it provides a more detailed caption which is correct, but the details mentioned do not align directly with those in the ground truth caption. This results in a lower score, even though the caption is more detailed and correct.

In summary, while the quantitative results show that that last frame baseline model outperforms MemClipCap, we observed that for the two clips in [figure 8] and many other clips, a single frame is often informatory enough to generate a solid caption. This could be the main reason why the baseline models perform similarly to or even better than the MemClipCap model. However, to firmly ground this claim, a more thorough analysis of the captions generated by the baseline models is required.

## Ablation studies

### Batch size
To assess the impact of the batch size on the performance of MemClipCap and the baseline models, we conducted an ablation study comparing models trained with different batch sizes. More specifically, in [table 3](#table-batch-size), we evaluate MemClipCap with a batch size of 5 and the baselines with a batch size of 40.

MemClipCap with a batch size of 5 showed higher scores for all evaluation metrics than when a batch size of 40 was used, indicating that the model benefits from a smaller batch size when incorporating long-range dependencies. This improved performance may be attributed to the increased attentiveness of the model towards individual video clips, as smaller batch sizes can provide more focused learning. In contrast, the baseline models exhibited substantially lower scores for all metrics when trained with a batch size of 40. This performance drop can be explained by the fact that batches often contain clips from the same video, which can lead to gradients that update the model strongly in a certain direction, resulting in the parameters jumping around the loss function's minimum, never converging to a stable solution.

<table id="table-batch-size">
  <tr>
    <th></th>
    <th style="text-align: center;">BLEU-1</th>
    <th style="text-align: center;">BLEU-2</th>
    <th style="text-align: center;">BLEU-3</th>
    <th style="text-align: center;">BLEU-4</th>
    <th style="text-align: center;">METEOR</th>
    <th style="text-align: center;">ROUGE-L</th>
  </tr>
  <tr>
    <td style="text-align: center;"><b>MemClipCap</b><br>(bs 5, initial clips)</td>
    <td>19.5892</td>
    <td>8.2377</td>
    <td>4.3551</td>
    <td>1.5735</td>
    <td>9.5871</td>
    <td>25.1340</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #1</b><br>(first frame, bs 40, initial clips)</td>
    <td>0.6649</td>
    <td>0.0195</td>
    <td>0.0000</td>
    <td>0.0000</td>
    <td>0.9341</td>
    <td>1.0959</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #2</b><br>(first frame, bs 40, initial clips)</td>
    <td>4.1829</td>
    <td>1.0198</td>
    <td>0.1629</td>
    <td>0.0801</td>
    <td>3.9912</td>
    <td>7.0548</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #3</b><br>(first frame, bs 40, initial clips)</td>
    <td>2.9117</td>
    <td>0.6578</td>
    <td>0.0846</td>
    <td>0.0595</td>
    <td>3.2522</td>
    <td>4.7583</td>
  </tr>
</table>

_[Table 3](#table-batch-size): Results of our MemClipCap model with batch size 5 and our three baseline models with batch size 40._

#### Analysis of accuracy measurements
The quantitative evaluation in the [batch size](#batch-size) section demonstrates that MemClipCap with batch size 5 yields slightly better results than with batch size 40. However, a manual analysis of the generated captions reveals that this improvement may lack significance: numerous clips feature "introductory credits", accompanied by the ground truth caption "The credits of the clip are shown". Naturally, both model variants generate various renditions of this caption. However, when these captions match the ground truth exactly, they form outliers that pull the average performance up significantly. The model with batch size 5 contains more of such outliers compared to the batch size 40 model, and consequently, its average performance is higher, but it is not necessarily better at generating good captions. We accounted for this by selecting the batch size 40 model as our "best" variant.

### Initial clip vs all clips
[Table 2](#table-main-results) showed the results for the best performing models, which were obtained by training and evaluating only on the initial clip of each video. For comparison, [table 4](#table-initial-clip-vs-all-clips) also shows the results of training and evaluating on all clips in a video, while keeping the total number of clips used for training constant.

<table id="table-initial-clip-vs-all-clips">
  <tr>
    <th></th>
    <th style="text-align: center;">BLEU-1</th>
    <th style="text-align: center;">BLEU-2</th>
    <th style="text-align: center;">BLEU-3</th>
    <th style="text-align: center;">BLEU-4</th>
    <th style="text-align: center;">METEOR</th>
    <th style="text-align: center;">ROUGE-L</th>
  </tr>
  <tr>
    <td style="text-align: center;"><b>MemClipCap</b><br>(bs 40, all clips)</td>
    <td><b><i>14.4053</i></b></td>
    <td>4.0560</td>
    <td>1.1455</td>
    <td>0.1057</td>
    <td>6.9259</td>
    <td><b><i>17.0485</i></b></td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #1</b><br>(first frame, bs 5, all clips)</td>
    <td>13.8661</td>
    <td><b><i>4.2128</i></b></td>
    <td>1.1045</td>
    <td><b><i>0.2402</i></b></td>
    <td>6.9834</td>
    <td>16.3011</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #2</b><br>(middle frame, bs 5, all clips)</td>
    <td>12.7060</td>
    <td>3.3036</td>
    <td>1.0532</td>
    <td>0.1370</td>
    <td>6.9234</td>
    <td>15.5418</td>
  </tr>
  <tr>
    <td style="text-align: center;"><b>Baseline #3</b><br>(last frame, bs 5, all clips)</td>
    <td>12.7000</td>
    <td>3.7467</td>
    <td><b><i>1.1650</i></b></td>
    <td>0.2127</td>
    <td><b><i>7.1890</i></b></td>
    <td>15.0981</td>
  </tr>
</table>

_[Table 4](#table-initial-clip-vs-all-clips): Results of our best MemClipCap model compared to our best three baseline models, when trained on all clips of each video instead of just the initial clip. The best model for each metric is shown in __bold italics__._

Comparing both tables reveals that the _initial clip_ models consistently outperform the _all clips_ models on all metrics. We identify two possible reasons for this. First, as mentioned in the [pre-processing](#pre-processing) section, our model treats all clips as independent, while the ground truth captions in the data may reference earlier captions from the same video. This effect was shown in [figure 4]. Second, as the total number of clips is kept constant, having multiple clips coming from the same video reduces the amount of diversity in the training data. Therefore, when constrained to a maximum number of clips for training, it is more efficient to use only one clip per video.


# Conclusion
Our research aimed to enhance the ClipCap method specifically for video captioning by integrating a Memorizing Transformer into the existing architecture. Despite the theoretical underpinnings suggesting that understanding long-range dependencies within a video is crucial for video captioning, our results demonstrated that the proposed MemClipCap model did not outperform the baseline models across the chosen evaluation metrics.

One advantage of our MemClipCap model is its ability to provide a consistently accurate performance for videos of arbitrary lengths. This flexibility allows the model to process a wide array of video clips without being restricted by a predefined length. As a result, MemClipCap holds potential for applications involving diverse video content in various settings.

Our model's performance might improve with a more extensive hyperparameter search, as various hyperparameters (e.g. memorizing layers, number of retrieved memories, batch size, learning rate) could be further tuned for optimal results. However, we were not able to perform these extensive optimizations due to resource constraints, so we leave this as a recommendation for future work.

Additionally, we recommend further investigation into the generalizability of the MemClipCap model to other modalities. An exploration of utilizing multiple modalities simultaneously, like combining audio and visual data, could also be a valuable extension of the current model. Additionally, it would be interesting to examine the effectiveness of the MemClipCap model when applied to video clips that feature more than one distinct activity, where the disparity between earlier and later frames is more pronounced, potentially allowing our model to capitalize on the long-range dependencies and enhance its captioning performance.


# Contributions
<!-- Close the notebook with a description of each student's contribution. -->
- Dennis Agafonov: Coding: COCO parsing, evaluation code, demo code and notebook, bugfixes.  Writing: draft version, introduction, related work, memorizing transformer, results.
- Jelke Matthijsse: evaluation code, demo code, bugfixes. Writing: draft, Clipcap's capabilities, results. Evaluating models.
- Sebastiaan Dijkstra: coding including: dataset, parsers, validation, clipcap, general improvements + bugfixes. Model training/evaluation. Writing including: abstract, datasets, experiments, results, ablation, discussion, conclusion, and general grammar/spelling improvements.
- Erik Buis: coding including: ActivityNet parsing + dataloading (Multifit algorithm), general improvements + bugfixes. Sections I wrote fully or mostly: first part of "introduction", "ClipCap summary", ClipCap "strengths" and "weaknesses", "datasets", "effect of clip length on performance". Sections I (re)wrote partially: All sections. Other writing activities in order of time spent: improving flow/cohesiveness of text, conciseness of text, reorganizing structure, grammar fixes. Other activities: creation of all tables and figures (except figure 3 & 8) in the blogpost, including all plots and code necessary to create the plots.
- Jan Bakker: Coding: integration of the memorizing transformer with ClipCap, training script, validation loss and caption generation script. Writing: Training section and Initial clip vs all clips subsection


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

[^loshchilov2017decoupled]: Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.
