# Long-Form Video Understanding: VideoBERT, MERLOT, and a Contrastive Look at VideoPoet

## What This Report Teaches

This report explains two core long-form video understanding papers — **VideoBERT** and **MERLOT** — and then uses **VideoPoet** as a contrastive third source, because it is **not** actually a video-understanding paper. It is a **video generation** paper about using a large language model to synthesize video from multimodal inputs. Two of the URLs you provided also do not match their titles: `2104.00285` is **CUPID**, not VideoBERT, and `2106.13230` is not MERLOT. The correct papers matching your titles are **VideoBERT** at arXiv `1904.01766`, **MERLOT** at arXiv `2106.02636`, and **VideoPoet** at arXiv `2312.14125`. ([arXiv][1])

By the end, you should understand how long-form video understanding evolved from **tokenizing video into “visual words” and jointly modeling it with language** in VideoBERT, to **learning multimodal script knowledge from millions of videos and transcripts** in MERLOT, and why VideoPoet is better understood as a related **multimodal sequence model** that shares some representation ideas but solves a different problem. 

---

## Key Takeaways

* **VideoBERT’s core idea is to treat video like language.** It converts short video clips into discrete visual tokens and jointly models them with text using a BERT-style objective. This matters because it brings language-model machinery into video representation learning. The practical implication is that video can be handled as a sequence of high-level discrete events rather than only as raw pixels or short clip features. 

* **MERLOT’s core idea is to learn “script knowledge” from long videos.** Script knowledge means understanding what usually happens before, during, and after an event. This matters because long-form video understanding is not just recognizing a frame; it is understanding event structure over time. The practical implication is that models can learn commonsense temporal reasoning from large collections of videos with transcripts, even without manual labels. 

* **MERLOT is more explicitly built for temporal commonsense than VideoBERT.** VideoBERT learns high-level joint video-language structure, but MERLOT directly targets temporal ordering, cross-segment context, and event-level reasoning. This matters because “what happened before or after?” is central to long-form video understanding. The practical implication is that MERLOT is the stronger paper for interview discussions about long-range temporal reasoning. 

* **VideoPoet does not belong naturally in a long-form video understanding trio.** It is a multimodal **video generation** model, not a long-form video understanding model. This matters because generation and understanding require different outputs, training goals, and evaluation. The practical implication is that you should present VideoPoet in an interview as a useful contrast: it shows how LLM-style multimodal token modeling can generate video, but it is not a direct successor to VideoBERT or MERLOT on understanding tasks. ([arXiv][2])

* **All three papers rely on turning video into sequences.** VideoBERT turns clips into visual tokens, MERLOT turns frames and transcript segments into multimodal sequences, and VideoPoet turns images, video, and audio into discrete tokens for an autoregressive model. This matters because sequence modeling is the conceptual bridge between language models and video. The practical implication is that tokenization is one of the most important ideas connecting language-model progress to video research. 

* **Data scale is central in all three papers, but used differently.** VideoBERT uses 312K cooking videos, MERLOT uses 6 million YouTube videos forming YT-Temporal-180M, and VideoPoet uses a mixture of paired and unpaired multimodal data plus task adaptation. This matters because long-form temporal understanding does not emerge well from small curated datasets alone. The practical implication is that strong video systems increasingly depend on large weakly supervised or self-supervised corpora. 

* **The field moves from local clip semantics toward richer event reasoning.** VideoBERT is already interested in longer semantic structure than short action clips, but MERLOT pushes much harder on contextual event reasoning over time. The practical implication is that interview answers should distinguish “recognizing content in a clip” from “understanding a multi-step event unfolding over time.” 

---

## Background and Foundations

Long-form video understanding is the problem of making a model understand **events that unfold over many seconds or minutes**, rather than only recognizing a short action clip. A short-clip action classifier might recognize “cutting,” “stirring,” or “opening a door.” A long-form video model should instead be able to reason about sequences like “prepare ingredients, cook, plate the dish,” or infer what likely happened before and after a visible moment. VideoBERT states this explicitly: it wants high-level semantic features corresponding to actions and events that unfold over longer time scales, not only low-level motion patterns or textures. 

A key difficulty is that long videos are expensive to label, and their meaning is often distributed across time. One frame rarely tells the full story. These papers therefore rely heavily on **weak supervision** or **self-supervision**. VideoBERT uses automatic speech recognition (ASR) transcripts from cooking videos. MERLOT uses millions of YouTube videos with transcripts and no manual labels. VideoPoet, while generative, also emphasizes large-scale multimodal pretraining and reuse of unpaired videos across multiple tasks. 

Three background ideas matter before diving into the papers.

| Idea                  | Plain-English meaning                                                 | Why it matters here                                            |
| --------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------- |
| Tokenization          | Turn video or other modalities into discrete sequence elements        | Lets language-model architectures operate on video-like inputs |
| Weak/self-supervision | Learn from transcripts, timing, or structure instead of manual labels | Makes large-scale video learning practical                     |
| Temporal reasoning    | Understand order, cause, and event progression over time              | This is the core challenge in long-form video understanding    |

These are shared ideas across the papers, even though VideoPoet uses them for generation rather than understanding. 

---

## Big Picture First

The easiest way to connect these papers is to ask: **What does the model think a video is?**

* **VideoBERT** treats a video as a sequence of **visual words** aligned with text. 
* **MERLOT** treats a video as a sequence of **frames plus transcript segments** that should be jointly understood over time. 
* **VideoPoet** treats video as **multimodal discrete tokens** to be generated autoregressively. ([arXiv][2])

That gives a simple progression:

| Paper     | Main problem                                                              | Main output                                               | Best mental model                          |
| --------- | ------------------------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------ |
| VideoBERT | Learn joint video-language representations from long instructional videos | Representations for classification/captioning/forecasting | “BERT over video tokens and text”          |
| MERLOT    | Learn multimodal script knowledge and temporal commonsense from videos    | Representations for QA and reasoning                      | “Contextual event understanding over time” |
| VideoPoet | Generate video from multimodal conditions with an LLM-like model          | Video and audio tokens                                    | “LLM-style multimodal video generator”     |

This is the cleanest high-level picture. The first two papers are about **understanding**. The third is about **generation**, but still useful as a contrastive example of multimodal sequence modeling over video. 

---

## Core Concepts Explained

### 1. Visual tokens

A **visual token** is a discrete symbol representing a chunk of video. VideoBERT creates these tokens by extracting clip features with a pretrained S3D network and then vector-quantizing them with hierarchical k-means into a vocabulary of 20,736 clusters. In plain English, the model stops looking at raw frames and instead reads a sentence-like sequence of learned “visual words.” This exists because BERT expects discrete tokens, not continuous videos. It matters because it is the key step that lets VideoBERT reuse language-model training ideas. 

### 2. Joint video-language modeling

VideoBERT learns a joint model over video tokens and linguistic tokens. It uses text-only, video-only, and video-text training regimes, with a linguistic-visual alignment objective for cross-modal training. At a high level, this teaches the model both the structure of video sequences and how they relate to spoken language. This matters because many long-form videos come with narration or ASR that describes the event. 

### 3. Script knowledge

**Script knowledge** means knowledge of how events typically unfold over time: what usually happens before, during, and after a situation. MERLOT is explicitly designed to learn this. It is not only matching frames to nearby words; it is learning the dynamic context behind scenes. This matters because long-form video understanding often requires inference, not just recognition. A restaurant scene may imply ordering, serving, eating, and leaving, even if not all steps are visible. 

### 4. Temporal ordering

**Temporal ordering** means the model should understand the correct sequence of events. MERLOT uses reordering scrambled video frames as one of its training objectives. This exists because a model that truly understands an event should know which moments likely come earlier or later. This matters for long-form reasoning, forecasting, and QA. 

### 5. Contrastive frame-transcript matching

MERLOT uses a contrastive objective to match contextualized transcript representations to their corresponding frames. In plain English, the model learns which language belongs with which moment in the video. The paper’s ablations say this loss is crucial to downstream performance. This matters because transcripts are noisy and weakly aligned, so explicit cross-modal matching helps the model ground language in what is visible. 

### 6. Long-context multimodal sequence modeling

All three papers rely on long sequence processing, but in different ways. VideoBERT uses BERT over text and visual words. MERLOT uses a joint vision-language transformer over multiple segments and contextualized transcript spans. VideoPoet uses a decoder-only prefix language model over text embeddings plus visual and audio tokens. This matters because the core engineering problem is not just vision; it is how to handle long multimodal sequences. 

### 7. Understanding versus generation

Video understanding means producing representations, answers, labels, or forecasts about a video. Video generation means synthesizing new video. VideoPoet is useful here because it looks superficially similar — tokenization, multimodal sequences, large pretrained transformer — but it solves a different problem. This matters in interviews because many modern multimodal papers share architectural language while doing different tasks. ([arXiv][2])

---

## Step-by-Step Technical Walkthrough

## 1. VideoBERT

1. **Collect long instructional videos with speech.** VideoBERT builds a large cooking-video corpus from YouTube, filtering to 312K videos under 15 minutes, totaling 23,186 hours. Of these, 180K have ASR available and 120K English-ASR videos are used for text-involving objectives. This exists because the method needs large weakly supervised video-language data. 

2. **Turn video into discrete visual words.** The paper samples 30-frame, 1.5-second clips, extracts S3D features, and vector-quantizes them with hierarchical k-means into 20,736 visual tokens. This lets video be processed like text. 

3. **Turn ASR into text tokens.** ASR transcripts are segmented into sentences and tokenized with BERT WordPieces. This provides the language side of the sequence. 

4. **Train a BERT-style joint model.** The model uses text-only and video-only mask completion, plus a video-text alignment objective. The point is to learn the structure of each modality and their correspondence. 

5. **Use the learned model for downstream tasks.** The paper demonstrates zero-shot action classification and feature extraction for dense video captioning. On YouCook II action classification, VideoBERT’s cross-modal setup reaches 43.3 top-5 verb accuracy and 33.7 top-5 object accuracy, competitive with a supervised S3D baseline in top-5 settings. On captioning, VideoBERT+S3D reaches BLEU-4 4.33, METEOR 11.94, ROUGE-L 28.80, and CIDEr 0.55 on YouCook II validation. 

**Purpose:** learn high-level semantic and temporally extended representations from weak supervision.
**Trade-off:** the model is elegant, but domain-specific. It is trained mainly on cooking videos, and the paper itself notes future work should cover broader domains and more spatially fine-grained representations. 

---

## 2. MERLOT

1. **Build a large and diverse pretraining corpus.** MERLOT introduces YT-Temporal-180M, derived from 6 million public YouTube videos spanning many domains, not just instructional videos. The paper says this diversity is important and that limiting pretraining to instructional data hurts downstream performance. 

2. **Segment video and transcript jointly.** The model aligns transcript BPE tokens with timestamps and forms 180 million multimodal segments. This gives it event slices that combine frames and language. 

3. **Encode frames and words together.** MERLOT uses an image encoder, word embeddings, and a joint vision-language transformer so that frames and transcript segments are contextualized together. 

4. **Use multiple self-supervised objectives.** The paper combines contrastive frame-transcript matching, masked word recovery, and temporal ordering of scrambled frames. This forces the model to learn both local grounding and longer-range event structure. 

5. **Transfer to reasoning tasks.** MERLOT is then fine-tuned on many downstream benchmarks. The paper reports state of the art on 12 video QA datasets and 80.6% accuracy on VCR Q→A, with 65.1 on the joint Q→AR metric. 

**Purpose:** learn temporal commonsense and script knowledge from long videos.
**Trade-off:** the model is more powerful for long-range reasoning, but also more complex than VideoBERT and more dependent on large-scale diverse pretraining. The paper’s ablations show the contrastive cross-modal loss is crucial, and performance improves with more context and more diverse data. 

---

## 3. VideoPoet

1. **Tokenize all modalities.** VideoPoet converts images, video, and audio into discrete tokens using MAGVIT-v2 for visual data and SoundStream for audio, while text is represented with frozen T5-XL embeddings. ([arXiv][2])

2. **Use a decoder-only LLM backbone.** The model is a prefix language model with a decoder-only architecture that conditions on text embeddings, visual tokens, and audio tokens, then autoregressively predicts visual and audio tokens. ([arXiv][2])

3. **Pretrain on multiple generative tasks.** The paper uses a mixture of multimodal pretraining tasks and paired plus unpaired videos, then performs task adaptation for specific generation settings. It explicitly frames this as a unified LLM-style generative setup rather than a specialized video-understanding pipeline. ([arXiv][2])

4. **Add super-resolution and task chaining.** Generated token sequences are refined with a super-resolution module, and the system can chain tasks such as image-to-video, stylization, outpainting, and video-to-audio. ([arXiv][2])

5. **Evaluate on generative benchmarks.** The paper reports, for example, VideoPoet (Pretrain) at CLIPSIM 0.3049 and FVD 213 on MSR-VTT, and FVD 355 / IS 38.44 on UCF-101; it also reports strong human preference on motion interestingness, realism, and temporal consistency versus several contemporaneous generators. ([arXiv][2])

**Purpose:** generate video, not understand it.
**Trade-off:** useful as a multimodal token-modeling example, but it does not directly answer the long-form video understanding question. It belongs in a different subfield: multimodal generative video modeling. ([arXiv][2])

---

## Paper-by-Paper Explanation

## VideoBERT: *A Joint Model for Video and Language Representation Learning*

### Problem addressed

VideoBERT asks whether language-model ideas can learn **high-level, temporally extended semantic video representations** from large amounts of weakly supervised video and language, especially for instructional videos. The paper explicitly contrasts this with prior work focused on low-level motion patterns and short time scales. 

### Method used

It extracts S3D clip features, vector-quantizes them into visual tokens, pairs them with ASR-derived text tokens, and trains a BERT-large style model with text-only, video-only, and video-text objectives. 

### Main innovation

The main innovation is treating video as a sequence of discrete visual words so that **joint masked video-text modeling** can be handled with BERT-style training. 

### Main findings

The model shows competitive zero-shot action classification on YouCook II and improves captioning performance when used as a feature extractor, with the best reported setup being VideoBERT+S3D. It also shows monotonic gains with larger pretraining datasets, up to 300K videos in the paper’s scaling experiment. 

### Limitations

The paper is mainly focused on cooking videos, uses coarse clip-level tokenization, and notes that future work should incorporate finer spatial detail and multiple temporal scales. So while it is influential, it is not a complete solution to general long-form video understanding. 

### What changed compared with earlier work

Instead of using labels or only short-clip objectives, VideoBERT brings in **language-style pretraining** and learns joint visual-linguistic structure over longer event horizons. 

---

## MERLOT: *Multimodal Neural Script Knowledge Models*

### Problem addressed

MERLOT asks how to learn **temporal commonsense** and **script knowledge** from videos, so a model can reason about what happened before, what is happening now, and what is likely to happen next. 

### Method used

It pretrains on YT-Temporal-180M from 6 million YouTube videos, combines frame encoding with transcript modeling, and uses multiple self-supervised objectives including contrastive frame-transcript matching, masked word prediction, and temporal ordering. 

### Main innovation

The main innovation is **explicit multimodal script learning over time**, not just joint representation learning. The model is built to learn event structure and temporal commonsense from weak supervision. 

### Main findings

The paper reports state of the art on 12 video QA datasets and strong transfer to static-image commonsense tasks, including 80.6% on VCR Q→A and 65.1 on Q→AR. It also shows that diverse large-scale video pretraining outperforms still-image pretraining and narrower instructional-video corpora. 

### Limitations

MERLOT is substantially larger in ambition and pretraining scale than VideoBERT, which makes it harder to reproduce. It also relies on transcripts and large-scale video corpora, which may not exist for every domain. Information about efficient deployment or compact variants is not provided in the paper sections reviewed. 

### What changed compared with earlier work

Compared with VideoBERT-like joint token models, MERLOT pushes more directly on **event reasoning** and **temporal commonsense**, using richer self-supervised objectives and more diverse data. 

---

## VideoPoet: *A Large Language Model for Zero-Shot Video Generation*

### Problem addressed

VideoPoet asks how an LLM-style multimodal model can generate high-quality video from varied conditioning signals, including text, images, videos, and audio. That is a generation problem, not an understanding problem. ([arXiv][2])

### Method used

It tokenizes modalities into a shared discrete space, uses a decoder-only language model backbone, trains on multiple generative tasks, and adds super-resolution for higher-quality video output. ([arXiv][2])

### Main innovation

Its main innovation is a **unified LLM-style multimodal generative framework** for many video-creation tasks. ([arXiv][2])

### Main findings

The paper reports strong zero-shot text-to-video performance and favorable human evaluation against several other generators, especially on motion-related dimensions. ([arXiv][2])

### Limitations

For this topic, the biggest limitation is conceptual: it is not really a long-form video understanding paper. It helps mainly as a contrastive example of what happens when video is treated as a token sequence for generation rather than for reasoning or QA. ([arXiv][2])

### What changed compared with earlier work

Compared with older specialized video generators, VideoPoet pushes toward a more unified multimodal LLM-style generative foundation model. But that is a different branch of the literature than VideoBERT and MERLOT. ([arXiv][2])

---

## Comparison Across Papers or Methods

| Dimension            | VideoBERT                                        | MERLOT                                                  | VideoPoet                             |
| -------------------- | ------------------------------------------------ | ------------------------------------------------------- | ------------------------------------- |
| Main goal            | Joint video-language representation learning     | Multimodal script knowledge and temporal commonsense    | Zero-shot video generation            |
| Core modality setup  | Visual tokens + text tokens                      | Frames + transcripts + joint transformer                | Visual/audio tokens + text embeddings |
| Main training signal | Masked modeling + video-text alignment           | Contrastive matching + masked words + temporal ordering | Multitask autoregressive generation   |
| Best fit for topic?  | Yes                                              | Yes, especially strongly                                | Only as a contrastive source          |
| Main strength        | Simple and influential bridge from BERT to video | Stronger long-range event reasoning                     | Unified multimodal token generation   |
| Main weakness        | Domain-limited and relatively coarse             | Data- and compute-heavy                                 | Not actually an understanding paper   |

This comparison is the cleanest interview summary: VideoBERT is the early bridge, MERLOT is the stronger long-form reasoning model, and VideoPoet is a related but different generative line. 

---

## Real-World System and Application

A practical long-form video understanding system based on the first two papers would likely include four stages:

1. **Video segmentation and tokenization**, turning long videos into manageable sequence elements. VideoBERT does this with clip-level vector quantization; MERLOT does this with multimodal segments aligned to transcripts. 
2. **Cross-modal grounding**, connecting frames to transcript segments or spoken words. Both papers rely heavily on this, though MERLOT does so more explicitly with contrastive frame-transcript matching. 
3. **Long-context temporal modeling**, so the model can reason about what earlier and later parts of the video imply. MERLOT is the stronger example here. 
4. **Task adaptation**, such as QA, captioning, event forecasting, or commonsense inference. VideoBERT shows captioning and zero-shot classification; MERLOT shows broad QA transfer. 

In real applications, these ideas could support instructional-video indexing, long-form QA over tutorials or lectures, event summarization, step forecasting, or multimodal search over archived video. The papers do not provide complete production architectures, but they clearly show how long-form understanding systems benefit from combining temporal segmentation, language grounding, and large-scale weak supervision. 

**Information not provided:** detailed production serving systems, memory management for very long videos, retrieval-augmented video pipelines, and enterprise-scale deployment practices are not described in these sources. 

---

## Limitations and Trade-offs

The biggest trade-off between VideoBERT and MERLOT is **simplicity versus temporal richness**. VideoBERT is easier to explain and more historically foundational, but it uses relatively coarse tokenization and a narrower domain. MERLOT is much stronger for long-form temporal reasoning, but it needs more data, more pretraining complexity, and more varied objectives. 

Another key trade-off is **weak supervision versus precise grounding**. Both papers benefit from transcripts and video timing, but ASR and narration are noisy. VideoBERT explicitly notes that the spoken words may refer to things not visually present. MERLOT combats this with stronger objectives and more context, but it still depends on weakly aligned multimodal signals rather than dense human annotation. 

A third trade-off is **understanding versus generation**. VideoPoet shows that multimodal token models can be very powerful, but success in generation does not automatically imply success in understanding. A generator may synthesize coherent motion without being the right tool for long-form QA or commonsense reasoning over real videos. ([arXiv][2])

Finally, there is a **domain and evaluation trade-off**. VideoBERT evaluates mainly on cooking tasks, while MERLOT evaluates on broader QA and commonsense tasks. This means MERLOT provides stronger evidence for general long-form understanding, while VideoBERT is better read as a foundational early step. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain that **VideoBERT** turns video into discrete visual tokens and jointly models them with language using BERT-style objectives, **MERLOT** learns temporal commonsense and script knowledge from millions of transcripted videos using multiple self-supervised objectives, and **VideoPoet** is a video-generation model that shares the token-sequence mindset but is not a direct long-form video understanding paper. 

### Likely interview questions

#### 1. What is the core idea of VideoBERT?

VideoBERT converts short video clips into discrete visual words and jointly models those visual tokens with text using BERT-style training, so video can be processed like a language sequence. 

#### 2. Why was VideoBERT important?

It was one of the clearest early demonstrations that language-model pretraining ideas could be transferred to video understanding by discretizing video into tokens and using weak video-language supervision at scale. 

#### 3. What does MERLOT add beyond VideoBERT?

MERLOT is more explicitly focused on long-range event understanding. It uses richer self-supervised objectives, more diverse data, and is built to learn temporal commonsense and script knowledge, not just joint token representations. 

#### 4. What is “script knowledge” in MERLOT?

It is knowledge of how events typically unfold over time: what usually happens before, during, and after a situation. MERLOT tries to learn this from videos and transcripts. 

#### 5. Why is MERLOT better for long-form video understanding?

Because its objectives directly encourage temporal reasoning across segments, including frame-transcript matching and temporal ordering, and its evaluation is broader and more reasoning-heavy. 

#### 6. Why does VideoPoet not fit cleanly here?

Because it is a video generation model, not a long-form video understanding model. It is useful as a contrastive multimodal token model, but its output is generated video rather than answers, labels, or representations for reasoning tasks. ([arXiv][2])

#### 7. What is the shared idea across all three papers?

All three treat multimodal video data as sequences that can be modeled with transformer-style architectures, often after tokenization or discretization. 

#### 8. How would you summarize the field progression?

A good summary is: **VideoBERT** brought BERT-style joint token modeling to video; **MERLOT** pushed toward temporal commonsense and script knowledge; **VideoPoet** shows how similar multimodal sequence ideas later support large-scale video generation rather than understanding. This is a synthesis across the sources. 

---

## Glossary

| Term                          | Beginner-friendly definition                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| Long-form video understanding | Understanding events and meaning that unfold over many seconds or minutes, not just short clips |
| Visual token / visual word    | A discrete symbol representing a chunk of video                                                 |
| Vector quantization           | Turning continuous features into discrete codebook entries                                      |
| ASR transcript                | Automatically generated speech-to-text from a video’s audio                                     |
| Joint video-language model    | A model trained to understand video and text together                                           |
| Script knowledge              | Knowledge of the typical sequence of events in a situation                                      |
| Temporal commonsense          | Common-sense reasoning about what likely happened before or after a visible event               |
| Contrastive matching          | Training a model to bring matching pairs together and push mismatched pairs apart               |
| Temporal ordering objective   | A training task where the model must recover correct frame order                                |
| Decoder-only transformer      | A transformer that predicts the next token autoregressively                                     |
| Prefix language model         | A decoder-style model that conditions on a prefix and generates continuation tokens             |
| Multimodal sequence model     | A model that processes multiple modalities, such as text and video, in one token sequence       |

These definitions are beginner-friendly paraphrases of ideas used across the papers. 

---

## Recap

You should now have a clear map of this topic. **VideoBERT** is the foundational paper that shows how to turn video into discrete tokens and jointly model it with language. **MERLOT** is the stronger long-form video understanding paper because it explicitly learns event structure, temporal commonsense, and script knowledge from large-scale transcripted video. **VideoPoet** is not really part of the same task family, but it is a useful contrast showing how similar multimodal token ideas later feed into video generation. 

For interviews, the most important thing is to say clearly that **VideoBERT and MERLOT are about understanding**, while **VideoPoet is about generation**. Then explain the real progression inside understanding: from **joint video-language token modeling** to **multimodal temporal script knowledge**. That is the strongest and most defensible way to present this paper set. 

---

## Key Citations

* [VideoBERT: A Joint Model for Video and Language Representation Learning](https://arxiv.org/abs/1904.01766)

* [MERLOT: Multimodal Neural Script Knowledge Models](https://arxiv.org/abs/2106.02636)

* [VideoPoet: A Large Language Model for Zero-Shot Video Generation](https://arxiv.org/abs/2312.14125)

* [Source note: provided `2104.00285` points to CUPID, not VideoBERT](https://arxiv.org/abs/2104.00285)

[1]: https://arxiv.org/abs/2104.00285?utm_source=chatgpt.com "CUPID: Adaptive Curation of Pre-training Data for Video-and-Language ..."
[2]: https://arxiv.org/pdf/2312.14125 "VideoPoet: A Large Language Model for Zero-Shot Video Generation"


---
---
---

