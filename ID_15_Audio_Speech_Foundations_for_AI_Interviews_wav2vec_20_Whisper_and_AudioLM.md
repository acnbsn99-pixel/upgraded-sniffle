# Audio & Speech Foundations for AI Interviews: wav2vec 2.0, Whisper, and AudioLM

## What This Report Teaches

This report explains three important papers that together cover a large part of modern speech and audio AI:

1. **wav2vec 2.0**: how to learn useful speech representations from **unlabeled audio**
2. **Whisper**: how to build a robust speech recognition and translation system from **massive weakly supervised audio-text data**
3. **AudioLM**: how to generate realistic audio by treating audio as a **language modeling problem over discrete tokens**

By the end, you should understand:

* the difference between **self-supervised learning**, **weak supervision**, and standard supervised learning in speech
* how modern speech systems turn raw waveforms into internal representations or tokens
* why **Transformers**, **masking**, **quantization**, and **autoregressive generation** matter in audio
* how these papers fit together conceptually
* what trade-offs matter in real systems
* how to explain these ideas clearly in an interview

All factual content below is drawn from the three cited papers. When I go beyond the papers to connect ideas across them, I label that as **Reasoned interpretation**. When the papers do not supply a detail, I write **Information not provided**.

---

## Key Takeaways

* **Speech models do not always need labeled transcripts to learn useful structure.**
  wav2vec 2.0 shows that a model can first learn from raw audio alone, then use a small amount of labeled data later.
  **Why it matters:** labeled speech is expensive.
  **Practical implication:** this is especially valuable for low-resource languages and domains.

* **Robust speech recognition can come from scale and diversity, not just cleaner labels.**
  Whisper trains on 680,000 hours of weakly supervised audio and becomes strong in zero-shot settings.
  **Why it matters:** a model trained on many environments and languages can generalize better.
  **Practical implication:** strong “works out of the box” behavior can reduce task-specific fine-tuning.

* **Audio generation needs both long-term structure and short-term sound quality.**
  AudioLM splits these into **semantic tokens** and **acoustic tokens**.
  **Why it matters:** one representation alone did not give both coherence and fidelity.
  **Practical implication:** hierarchical generation is a key design pattern for realistic audio.

* **Discrete tokens are a major bridge between audio and language modeling.**
  Both wav2vec 2.0 and AudioLM use discrete units, but for different reasons.
  **Why it matters:** tokens let models reason over long sequences more efficiently and more abstractly.
  **Practical implication:** tokenization design strongly affects what the model learns.

* **The training objective shapes what the representation keeps and what it ignores.**
  wav2vec 2.0 uses a contrastive task over masked spans; Whisper predicts text and task tokens; AudioLM predicts future audio tokens.
  **Why it matters:** the objective determines whether a model becomes good at recognition, robustness, or generation.
  **Practical implication:** architecture alone is not enough; target design is crucial.

* **Evaluation in speech can be misleading if you only test in-distribution.**
  Whisper argues that many systems look excellent on the benchmark they were trained on, but break under distribution shift.
  **Why it matters:** interviewers often care about robustness, not only benchmark scores.
  **Practical implication:** production systems should be judged on out-of-distribution behavior and long-form reliability.

* **These papers are related, but they solve different problems.**
  wav2vec 2.0 is mainly about learning speech representations, Whisper is about robust speech-to-text and translation, and AudioLM is about generation.
  **Why it matters:** they are not direct replacements for one another.
  **Practical implication:** choosing the right method depends on whether you need encoding, transcription, or generation.

---

## Background and Foundations

### Why speech is hard

Speech is not just “text with sound.” A speech signal contains multiple layers at once:

* **linguistic content**: the words and meaning
* **speaker identity**: who is talking
* **prosody**: rhythm, stress, and intonation
* **recording conditions**: microphone, room, noise, compression
* **timing structure**: pauses, speaking rate, overlaps

A good speech system has to decide which parts matter for its task.

* For **speech recognition**, the main goal is to recover the words.
* For **speech translation**, the system must recover meaning and express it in another language.
* For **audio generation**, the system must produce realistic sound over time, not just correct words.

### Why labels are a bottleneck

For text, it is easy to collect huge corpora. For speech, accurate transcripts are costly. This creates an imbalance:

| Data type                   | Easy to collect? | Typical use                                             |
| --------------------------- | ---------------: | ------------------------------------------------------- |
| Raw audio                   |              Yes | Self-supervised or weakly supervised learning           |
| Audio with transcripts      |               No | Supervised ASR and speech translation                   |
| Audio with rich annotations |      Even harder | Speaker tasks, diarization, alignment, prosody analysis |

This is why the papers matter:

* **wav2vec 2.0** asks: can we learn from raw audio before seeing labels?
* **Whisper** asks: can massive noisy labels still produce robust systems?
* **AudioLM** asks: can we model audio like text once audio is turned into tokens?

### Three learning setups you must distinguish

| Learning setup               | What the model gets                                        | Example in these papers        | Main benefit          | Main risk                          |
| ---------------------------- | ---------------------------------------------------------- | ------------------------------ | --------------------- | ---------------------------------- |
| **Supervised learning**      | input + correct target label                               | Whisper predicting transcripts | clear training signal | labels are expensive               |
| **Self-supervised learning** | raw input only; model creates its own prediction task      | wav2vec 2.0 pre-training       | uses unlabeled data   | task may not match downstream need |
| **Weak supervision**         | labels exist but are noisy, incomplete, or loosely aligned | Whisper internet transcripts   | scales cheaply        | noisy targets can mislead training |

### Important prerequisite concepts

#### Transformer

A **Transformer** is a neural network architecture that models relationships across a sequence. In speech, the sequence might be time steps from audio features or tokens.

Why it matters here:

* wav2vec 2.0 uses a Transformer to build **contextualized speech representations**
* Whisper uses an **encoder-decoder Transformer**
* AudioLM uses **decoder-only Transformers** to generate tokens autoregressively

#### Token

A **token** is a discrete unit the model predicts or reads.

In text, tokens are usually subwords or words.
In audio, tokens can be learned codebook entries, semantic units, or codec symbols.

#### Quantization

**Quantization** means mapping continuous vectors to discrete choices from a codebook.

Why this matters:

* wav2vec 2.0 uses quantized targets during self-supervised training
* AudioLM relies on discrete semantic and acoustic tokens for generation

#### CTC

**Connectionist Temporal Classification (CTC)** is a training loss used when input and output lengths differ and exact alignment is unknown.

wav2vec 2.0 uses CTC during fine-tuning for speech recognition.

#### WER

**Word Error Rate (WER)** measures transcription mistakes using insertions, deletions, and substitutions.

Useful, but not perfect: Whisper explicitly argues that WER can unfairly penalize harmless formatting differences.

### How the three papers relate

Historically and conceptually, you can think of them as three moves:

1. **Learn better audio representations from unlabeled speech**
   wav2vec 2.0

2. **Scale speech-to-text supervision and push for robust zero-shot behavior**
   Whisper

3. **Use discrete audio representations to generate coherent audio directly**
   AudioLM

They are not one linear upgrade path, but they do share a core theme:
**learn structured internal representations of audio, then use them for powerful downstream behavior.**

---

## Big Picture First

### A simple mental model

All three papers follow some version of this pattern:

1. **Compress or represent audio**
2. **Model context over time**
3. **Predict something useful**
4. **Decode that prediction into text or audio**

The main difference is **what gets predicted**.

| Paper           | Input               | Internal representation                  | What is predicted     | Final output                                            |
| --------------- | ------------------- | ---------------------------------------- | --------------------- | ------------------------------------------------------- |
| **wav2vec 2.0** | raw waveform        | latent speech features + context vectors | masked latent targets | speech representation, then text after fine-tuning      |
| **Whisper**     | log-Mel spectrogram | encoder states                           | text and task tokens  | transcript, translation, timestamps, language/task info |
| **AudioLM**     | audio waveform      | semantic + acoustic tokens               | next audio tokens     | generated audio waveform                                |

### The overall problem each paper is solving

* **wav2vec 2.0**: “How do we learn speech structure without transcripts?”
* **Whisper**: “How do we make speech recognition robust across many datasets, languages, and domains?”
* **AudioLM**: “How do we generate realistic audio that stays coherent over time?”

### What changed across the papers

| Dimension        | wav2vec 2.0                                           | Whisper                      | AudioLM                                         |
| ---------------- | ----------------------------------------------------- | ---------------------------- | ----------------------------------------------- |
| Main task        | representation learning for ASR                       | robust ASR and translation   | audio generation                                |
| Supervision type | self-supervised pre-training + supervised fine-tuning | large-scale weak supervision | unsupervised tokenization + generative modeling |
| Audio form       | raw waveform                                          | spectrogram                  | discrete audio tokens                           |
| Output target    | masked latent target, later text                      | text tokens + task tokens    | future audio tokens                             |
| Key strength     | low-label efficiency                                  | zero-shot robustness         | long-term coherent audio generation             |

### The most important conceptual split

There are two big families here:

* **Recognition-oriented systems**

  * wav2vec 2.0
  * Whisper

* **Generation-oriented systems**

  * AudioLM

Recognition asks: “What was said?”
Generation asks: “What should sound come next?”

That difference changes the objective, architecture, and evaluation.

---

## Core Concepts Explained

### 1. Speech representation learning

**What it is:**
Learning internal features of speech that capture useful structure.

**Why it exists:**
Raw waveforms are too low-level. Good internal features make downstream tasks easier.

**How it works at a high level:**
A model converts audio into embeddings that summarize local sound and broader context.

**Where it appears:**
Central to wav2vec 2.0; also indirectly inside AudioLM through w2v-BERT-derived semantic tokens.

**Why it matters:**
If representations are strong, you need less labeled data later.

---

### 2. Masked prediction for audio

**What it is:**
Hide part of the signal and train the model to identify or predict what belongs there.

**Why it exists:**
It forces the model to use surrounding context, not just memorize local details.

**How it works at a high level:**
wav2vec 2.0 masks spans in latent space, then asks the context model to find the correct latent target among distractors.

**Where it appears:**
wav2vec 2.0 pre-training. AudioLM also benefits from prior masked-language-model-style pre-training inside w2v-BERT, though AudioLM itself is not trained with masked prediction.

**Why it matters:**
This is one of the main mechanisms that lets speech models learn from unlabeled audio.

---

### 3. Contrastive learning

**What it is:**
A training method where the model must score the correct target higher than incorrect alternatives.

**Why it exists:**
In speech, there is often no direct label available during pre-training.

**How it works at a high level:**
For each masked time step, wav2vec 2.0 compares the context vector against one true quantized latent target and several distractors from the same utterance.

**Where it appears:**
wav2vec 2.0 pre-training.

**Why it matters:**
It teaches the model which contextual clues correspond to the right hidden speech content.

---

### 4. Quantization and codebooks

**What it is:**
Turning continuous vectors into discrete symbols chosen from a learned vocabulary.

**Why it exists:**
Discrete symbols can be easier to compare, predict, and model over long time scales.

**How it works at a high level:**
A codebook stores candidate vectors. The model picks entries from one or more codebooks and uses them as token-like targets.

**Where it appears:**
wav2vec 2.0 uses product quantization and a Gumbel-softmax-based differentiable selection process.
AudioLM uses codec tokens from SoundStream and semantic tokens from clustered w2v-BERT features.

**Why it matters:**
Quantization is the bridge from raw signal to token-style modeling.

---

### 5. Encoder-decoder vs decoder-only modeling

| Style                   | Plain-English idea                                                    | Used in                 | Best for                       |
| ----------------------- | --------------------------------------------------------------------- | ----------------------- | ------------------------------ |
| **Encoder-decoder**     | first understand the input, then generate an output conditioned on it | Whisper                 | audio-to-text tasks            |
| **Decoder-only**        | generate the next token from previous tokens                          | AudioLM                 | autoregressive generation      |
| **Encoder + task head** | encode the input, then add a small task-specific output layer         | wav2vec 2.0 fine-tuning | recognition after pre-training |

This is an important interview distinction.

---

### 6. CTC for speech recognition

**What it is:**
A loss that allows training speech-to-text systems without knowing exact frame-to-character alignment.

**Why it exists:**
Audio is long and token timing is not pre-aligned.

**How it works at a high level:**
The model outputs a probability distribution over labels at each time step, and CTC sums over many valid alignments.

**Where it appears:**
wav2vec 2.0 fine-tuning.

**Why it matters:**
It is a classic way to convert frame-like speech representations into text.

---

### 7. Multitask token formatting

**What it is:**
Representing different tasks with special tokens inside one text-generation format.

**Why it exists:**
A single model can handle transcription, translation, language identification, timestamps, and no-speech detection if the task is specified clearly.

**How it works at a high level:**
Whisper prepends or includes special tokens that tell the decoder what to do.

**Where it appears:**
Whisper.

**Why it matters:**
It simplifies the overall speech pipeline and supports one model for many tasks.

---

### 8. Semantic tokens vs acoustic tokens

**What it is:**
Two different levels of audio abstraction.

* **Semantic tokens** capture higher-level structure such as linguistic content or musical pattern
* **Acoustic tokens** capture sound quality, speaker identity, and local details

**Why it exists:**
One representation alone did not give both long-term coherence and high-fidelity sound.

**How it works at a high level:**
AudioLM first predicts semantic tokens for structure, then acoustic tokens for detailed rendering.

**Where it appears:**
AudioLM.

**Why it matters:**
This is the key design idea that makes AudioLM work.

---

## Step-by-Step Technical Walkthrough

### 1. wav2vec 2.0

#### High-level goal

Learn useful speech representations from raw audio without transcripts, then fine-tune with a small amount of labeled speech.

#### Pipeline

1. **Input: raw waveform**

   * The model takes the speech waveform directly.
   * A convolutional feature encoder converts it into a shorter sequence of latent speech representations.

2. **Create latent features**

   * The convolutional network extracts local sound patterns.
   * Output: a sequence of latent vectors over time.

3. **Mask spans in latent space**

   * The model randomly chooses starting positions and masks consecutive latent time steps.
   * Important detail: masking happens in latent space, not on the raw waveform.

4. **Build context with a Transformer**

   * The masked latent sequence goes into a Transformer.
   * Output: contextualized representations that can use information from the whole utterance.

5. **Quantize latent targets**

   * Separately, the feature encoder outputs are discretized through product quantization.
   * These quantized vectors serve as training targets.

6. **Apply the contrastive objective**

   * For each masked position, the model sees one true quantized target and multiple distractors.
   * It must score the true one highest.
   * Practical meaning: “from the surrounding audio, identify which hidden speech unit should be here.”

7. **Apply diversity loss**

   * A second loss encourages the model to use codebook entries more evenly.
   * Why it exists: without it, the model might collapse to using only a small part of the codebook.

8. **Fine-tune on labeled speech**

   * After pre-training, a linear projection layer is added on top.
   * The system is trained with CTC on transcribed speech.
   * Output: text labels for speech recognition.

#### Why each step exists

| Stage                       | Purpose                                      | Output                  | Main trade-off                            |
| --------------------------- | -------------------------------------------- | ----------------------- | ----------------------------------------- |
| CNN feature encoder         | compress raw audio into useful latent frames | latent speech vectors   | may lose detail if too aggressive         |
| Masking                     | force contextual reasoning                   | masked latent sequence  | too much masking can harm learning        |
| Transformer context network | capture long-range dependencies              | context vectors         | higher compute cost                       |
| Quantization                | define discrete targets                      | codebook-based targets  | discretization can throw away detail      |
| Contrastive loss            | learn which target fits the context          | trained representations | needs careful negative sampling           |
| Diversity loss              | avoid codebook collapse                      | broader codebook usage  | adds another tuning knob                  |
| CTC fine-tuning             | map learned representations to text          | ASR output              | optimized for recognition, not generation |

#### Plain-English meaning of the losses

* **Contrastive loss:**
  “Pick the correct hidden speech unit from a small candidate set.”

* **Diversity loss:**
  “Do not keep using the same few discrete symbols over and over.”

#### Why this matters in practice

wav2vec 2.0 is powerful because the expensive labeled phase comes **after** the model has already learned a lot from unlabeled audio.

---

### 2. Whisper

#### High-level goal

Train one robust model for speech recognition and related tasks using very large, noisy, multilingual audio-text data.

#### Pipeline

1. **Input: internet audio paired with transcripts**

   * Data is collected from the web.
   * The labels are weakly supervised, so filtering matters.

2. **Filter and segment the data**

   * The paper describes automated filtering to remove low-quality or machine-generated transcripts.
   * Audio is broken into **30-second segments** paired with the transcript portion in that window.
   * Segments with no speech are also included for voice activity detection.

3. **Convert audio to log-Mel spectrograms**

   * Audio is resampled to 16 kHz.
   * An 80-channel log-Mel spectrogram is computed.
   * This is the model input rather than raw waveform.

4. **Encode the audio**

   * A Transformer encoder processes the spectrogram features.
   * Output: audio-conditioned hidden states.

5. **Specify the task with special tokens**

   * The decoder receives tokens that indicate language, task, timestamps, no-timestamps mode, and other formatting details.

6. **Decode text autoregressively**

   * The decoder predicts the output sequence token by token.
   * Output may be:

     * transcription
     * translation into English
     * timestamps
     * language-related outputs
     * no-speech behavior

7. **Use one multitask format**

   * Instead of separate models for separate tasks, Whisper maps them into one decoder format.

8. **Long-form inference**

   * Because the model is trained on 30-second chunks, long audio must be handled by transcribing chunk after chunk.
   * The paper discusses timestamp-based buffering and decoding heuristics for reliability.

#### Why each step exists

| Stage                       | Purpose                               | Output                           | Main trade-off                                                   |
| --------------------------- | ------------------------------------- | -------------------------------- | ---------------------------------------------------------------- |
| web-scale data collection   | maximize diversity and scale          | large training set               | labels are noisy                                                 |
| filtering                   | improve transcript quality            | cleaner weak supervision         | filtering can still miss errors                                  |
| spectrogram conversion      | stable audio features for Transformer | 2D time-frequency representation | not raw-audio end-to-end                                         |
| encoder-decoder Transformer | map audio to text                     | text sequence                    | expensive at large scale                                         |
| multitask special tokens    | unify many speech tasks               | shared interface                 | task formatting becomes part of model design                     |
| zero-shot evaluation        | test broad generalization             | cross-dataset results            | may underperform heavily fine-tuned specialists on some settings |
| chunked long-form decoding  | transcribe longer audio               | practical transcription pipeline | windowing errors can propagate                                   |

#### Why Whisper is a big deal

The key idea is not a brand-new architecture. The paper explicitly says it uses an off-the-shelf encoder-decoder Transformer. The innovation is largely in:

* **dataset scale**
* **multilingual + multitask training**
* **simple unified decoding format**
* **zero-shot robustness emphasis**

#### Practical meaning of the model design

Whisper is trying to act less like a benchmark-specialist model and more like a general speech system.

---

### 3. AudioLM

#### High-level goal

Generate high-quality audio with long-term consistency by turning audio generation into token prediction.

#### Core challenge

If you model only very detailed audio tokens, the sound may be high quality but the long-term content becomes inconsistent.
If you model only high-level semantic tokens, the structure may be coherent but the sound quality is poor.

AudioLM solves this by splitting the problem.

#### Pipeline

1. **Input: waveform**

   * Audio is processed into two token streams.

2. **Create acoustic tokens**

   * A neural audio codec called **SoundStream** compresses audio into discrete codec tokens.
   * These tokens are good for reconstruction quality.

3. **Create semantic tokens**

   * A pre-trained self-supervised speech model, **w2v-BERT**, produces representations.
   * A k-means clustering step turns selected intermediate features into discrete semantic tokens.
   * These tokens capture longer-range structure better.

4. **Stage 1: semantic language model**

   * A decoder-only Transformer predicts semantic tokens autoregressively.
   * Purpose: model long-term structure such as linguistic content or musical pattern.

5. **Stage 2: coarse acoustic language model**

   * Another decoder-only Transformer predicts coarse acoustic tokens conditioned on semantic tokens.
   * Purpose: add speaker identity, recording conditions, and major acoustic properties.

6. **Stage 3: fine acoustic language model**

   * A third decoder-only Transformer predicts fine acoustic tokens conditioned on coarse acoustic tokens.
   * Purpose: improve local detail and remove remaining compression artifacts.

7. **Detokenize back to waveform**

   * The SoundStream decoder reconstructs the final waveform from acoustic tokens.

#### Why hierarchical generation is necessary

The paper shows a useful trade-off:

| Token type          | Good at                           | Bad at                           |
| ------------------- | --------------------------------- | -------------------------------- |
| **Semantic tokens** | phonetic and long-range structure | reconstruction quality           |
| **Acoustic tokens** | reconstruction quality            | long-term linguistic consistency |

AudioLM combines both rather than forcing one representation to do everything.

#### Why the three-stage design exists

| Stage                    | Input                                              | Output                 | Why it exists                      | Failure mode if missing                        |
| ------------------------ | -------------------------------------------------- | ---------------------- | ---------------------------------- | ---------------------------------------------- |
| Semantic modeling        | past semantic tokens                               | future semantic tokens | long-term coherence                | speech can sound like babbling                 |
| Coarse acoustic modeling | semantic tokens + past coarse acoustic tokens      | coarse acoustic tokens | render major sound properties      | voice and environment may be weakly controlled |
| Fine acoustic modeling   | coarse acoustic tokens + past fine acoustic tokens | fine acoustic tokens   | improve fidelity and local details | audio remains noticeably codec-like            |

#### Plain-English interpretation of the probabilities

When the paper writes things like “predict the next token given previous tokens,” it is doing the same basic thing as a language model for text:

* first decide the **content plan**
* then render the **sound details**

That is why AudioLM is called a language modeling approach to audio generation.

---

## Paper-by-Paper Explanation

### Paper 1: Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

#### Problem addressed

Speech recognition usually needs large amounts of transcribed audio, but unlabeled audio is much easier to collect.

#### Method used

The paper pre-trains on raw audio using masked latent prediction with a contrastive objective over quantized targets, then fine-tunes with CTC on labeled speech.

#### Main innovation

The model jointly learns:

* contextualized speech representations
* discrete speech units used as targets

This avoids a two-step pipeline where quantization is learned separately.

#### Main findings

* Strong results even with very small labeled datasets
* Good gains from more unlabeled pre-training data
* Strong performance even when using a relatively simple Transformer + CTC setup

#### Limitations

* The fine-tuned recognition system still depends on labeled data
* The paper itself notes that a seq2seq architecture and word-piece vocabulary could improve performance further
* The paper focuses on speech recognition, not a full multitask speech system

#### What changed compared with earlier work

Earlier work often learned quantized units separately or reconstructed lower-level input features. wav2vec 2.0 made the representation learning more end-to-end and more effective.

#### Directly stated facts

* The model masks latent representations and solves a contrastive task over quantized speech representations.
* It uses a convolutional encoder, a Transformer context network, and a quantization module.
* With pre-training on 53.2k hours of unlabeled LibriVox audio and only 10 minutes of labeled data, it reports very strong Librispeech performance.
* On full Librispeech labeled training, it reports 1.8/3.3 WER on test-clean/test-other.

#### Reasoned interpretation

This paper is best understood as a foundation-model-style encoder for speech before the term “foundation model” became common in speech. Its strongest idea is not just masking, but choosing a target design that makes the model learn useful abstractions instead of memorizing surface details.

#### Information not provided

* A production deployment recipe for real-time systems
* A multilingual system in this paper
* Detailed latency analysis for inference

---

### Paper 2: Whisper: Robust Speech Recognition via Large-Scale Weak Supervision

#### Problem addressed

Many speech systems perform well only after dataset-specific fine-tuning and can be brittle when the distribution changes.

#### Method used

Train an encoder-decoder Transformer on 680,000 hours of weakly supervised, multilingual, multitask audio-text data, using a unified token-based format for multiple speech tasks.

#### Main innovation

The main innovation is the combination of:

* very large weakly supervised training data
* multilingual and multitask supervision
* one unified decoder interface using special tokens
* strong emphasis on zero-shot transfer

#### Main findings

* Strong zero-shot generalization across many datasets
* Good multilingual and translation behavior at scale
* The model family scales from 39M to 1.55B parameters
* The paper argues that zero-shot Whisper models close much of the robustness gap to human behavior compared with models trained narrowly on benchmark data

#### Limitations

* The training data is noisy and requires filtering
* The model can produce plausible but incorrect speaker-name guesses
* It is trained on 30-second chunks, so long-form transcription needs decoding strategies
* The paper reports weaker performance on some settings such as VoxPopuli and non-overlapping-language language identification cases

#### What changed compared with earlier work

Earlier work often relied on smaller clean datasets, self-supervised encoders plus fine-tuning, or narrower supervised mixtures. Whisper pushes much harder on weak supervision scale and task unification.

#### Directly stated facts

* Whisper uses 680,000 hours of labeled audio data.
* Of that, 117,000 hours cover 96 non-English languages, and 125,000 hours are X-to-English translation data.
* Audio is segmented into 30-second chunks.
* The model uses an encoder-decoder Transformer over 80-channel log-Mel spectrograms.
* Tasks such as transcription, translation, language identification, and voice activity detection are represented as token prediction problems.
* The paper evaluates zero-shot across many datasets rather than relying only on in-distribution fine-tuning.

#### Reasoned interpretation

Whisper is less about inventing a brand-new architecture and more about showing that broad, diverse, weak supervision can produce a practical speech foundation model that is useful without much adaptation.

#### Information not provided

* A full public breakdown of every data source in the training mix
* A complete production serving design
* A claim that Whisper solves all speech tasks equally well; in fact, the paper shows uneven performance across tasks and languages

---

### Paper 3: AudioLM: a Language Modeling Approach to Audio Generation

#### Problem addressed

How can a model generate audio that is both high quality and coherent over long time spans?

#### Method used

Turn audio into discrete token streams, then generate those tokens hierarchically:

1. semantic tokens
2. coarse acoustic tokens
3. fine acoustic tokens

#### Main innovation

The hybrid tokenization and hierarchical modeling design:

* semantic tokens for long-range structure
* acoustic tokens for high-fidelity rendering

#### Main findings

* The model can generate plausible speech continuations without transcripts
* It preserves speaker identity strongly when prompted with short speech from unseen speakers
* Human raters struggled to distinguish short generated continuations from real speech in the reported test
* The framework also works for piano continuation

#### Limitations

* The system carries misuse risk because it can continue a speaker’s voice convincingly
* The paper notes possible inconsistency for underrepresented accents and dialects
* The work is about continuation and generation, not a controllable speech assistant or full text-to-speech system
* It requires multiple components, including tokenizers and staged models

#### What changed compared with earlier work

Earlier speech generation methods either relied on text conditioning or struggled to combine long-term structure with high acoustic quality. AudioLM explicitly separates those jobs.

#### Directly stated facts

* AudioLM maps audio to discrete token sequences and frames generation as language modeling.
* It uses semantic tokens from w2v-BERT representations and acoustic tokens from SoundStream.
* The system uses three stages of decoder-only Transformer modeling.
* For speech continuation, the paper reports strong speaker preservation and realistic generations.
* For a subjective test distinguishing real vs generated short speech samples, raters were near chance in the reported setup.
* The paper also trains a detector that achieves high accuracy at distinguishing AudioLM-generated speech from original speech in its evaluation.
* The framework extends to piano continuation.

#### Reasoned interpretation

AudioLM is an audio-generation analogue of a multi-level planning-and-rendering system. First it decides what should happen in a coarse, semantic sense, then it decides how it should sound in detail.

#### Information not provided

* A general controllable interface for arbitrary text-conditioned speech synthesis
* A full end-user product design
* A public deployment recipe for safeguards beyond the detector studied in the paper

---

## Comparison Across Papers or Methods

### Side-by-side comparison

| Dimension              | wav2vec 2.0                                               | Whisper                                              | AudioLM                                                |
| ---------------------- | --------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------ |
| Primary goal           | learn speech representations for downstream ASR           | robust speech recognition and translation            | realistic audio generation                             |
| Main supervision       | self-supervised pre-training, then supervised fine-tuning | weakly supervised audio-text training                | token modeling over discrete audio representations     |
| Input form             | raw waveform                                              | log-Mel spectrogram                                  | raw waveform converted to tokens                       |
| Core architecture      | CNN + Transformer + quantizer; later CTC head             | encoder-decoder Transformer                          | three decoder-only Transformers plus tokenizers/codecs |
| Main prediction target | masked quantized latent units                             | text tokens and task tokens                          | future semantic/acoustic tokens                        |
| Output                 | speech embeddings, then text                              | transcript / translation / timestamps / task outputs | generated audio waveform                               |
| Biggest strength       | label efficiency                                          | zero-shot robustness                                 | coherence + quality in generation                      |
| Biggest weakness       | still needs fine-tuning for ASR                           | noisy supervision and chunked long-form decoding     | misuse risk and system complexity                      |
| Interview label        | self-supervised speech encoder                            | large-scale weakly supervised speech-to-text model   | hierarchical token-based audio generator               |

### Training-method comparison

| Question                                          | wav2vec 2.0                                          | Whisper                                        | AudioLM                                          |
| ------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- | ------------------------------------------------ |
| What is the model learning to do during training? | identify true masked latent target among distractors | predict text and task-format tokens from audio | predict the next audio tokens                    |
| Why does that objective make sense?               | it teaches speech structure without labels           | it directly optimizes broad speech tasks       | it turns audio generation into sequence modeling |
| What is the hidden representation trying to keep? | phonetic and contextual speech information           | information needed for text and speech tasks   | both long-range content and local acoustics      |
| What should it ignore?                            | speaker/background details when not helpful for ASR  | dataset-specific quirks when possible          | irrelevant randomness that harms coherence       |

### How they fit in practice

| Use case                                                     | Best paper to discuss |
| ------------------------------------------------------------ | --------------------- |
| Low-resource ASR with little labeled data                    | wav2vec 2.0           |
| A robust multilingual ASR baseline for real applications     | Whisper               |
| Generating speech/audio continuations                        | AudioLM               |
| Explaining the difference between recognition and generation | all three together    |

---

## Real-World System and Application

### Directly stated by the papers

* **wav2vec 2.0** supports speech recognition with limited labeled data after pre-training.
* **Whisper** is designed as a broad speech processing system covering transcription, translation, language identification, voice activity detection, and timestamped decoding within one model format.
* **AudioLM** supports realistic speech continuation and piano continuation, and its conclusion mentions possible future use in conditioned tasks such as text-to-speech or speech-to-speech translation.

### Reasoned interpretation

The papers do **not** describe one integrated production system together. But a practical AI organization could use their ideas in different layers:

1. **Representation layer**

   * Use wav2vec-style self-supervised pre-training when labeled speech is scarce.

2. **Recognition layer**

   * Use a Whisper-like multitask encoder-decoder model when the main goal is robust transcription or translation across varied real-world audio.

3. **Generation layer**

   * Use an AudioLM-like hierarchical token generator when the goal is audio continuation or high-fidelity synthesis.

4. **Safety and monitoring**

   * AudioLM highlights the need for synthetic-audio detection.
   * Whisper highlights the need for robust long-form decoding and out-of-distribution evaluation.

### Information not provided

* A full unified architecture combining all three approaches into one deployment stack
* Cost, latency, and serving benchmarks for such an integrated system
* Security or compliance requirements for regulated industries

---

## Limitations and Trade-offs

### Cross-paper trade-offs

| Trade-off                 | wav2vec 2.0                                                 | Whisper                                         | AudioLM                                |
| ------------------------- | ----------------------------------------------------------- | ----------------------------------------------- | -------------------------------------- |
| Label efficiency          | very strong                                                 | less central; relies on lots of weak labels     | not about labels for text tasks        |
| Out-of-the-box usability  | lower; needs fine-tuning                                    | high relative to many prior systems             | generation-focused, not ASR-focused    |
| Robustness across domains | improved via pre-training, but not the paper’s main framing | central goal                                    | not the main evaluation target         |
| Generation quality        | not relevant                                                | not relevant                                    | central strength                       |
| System simplicity         | moderate                                                    | conceptually simple interface, huge data effort | complex multistage pipeline            |
| Safety concerns           | mainly deployment/data equity                               | hallucination and robustness concerns           | strong impersonation/spoofing concerns |

### Concrete limitations by paper

#### wav2vec 2.0

* Excellent for recognition pre-training, but it is not itself a full general speech assistant.
* Needs labeled fine-tuning for ASR.
* The paper itself suggests further gains were likely possible with seq2seq decoding and word-piece vocabulary.

#### Whisper

* Weak supervision scales well, but weak labels are noisy.
* The paper reports that transcript formatting strongly affects WER, which complicates evaluation.
* Long-form transcription is not native; it requires chunking and decoding heuristics.
* Performance is not uniformly strongest on every task and language.

#### AudioLM

* Realistic continuation raises misuse risks.
* High-quality generation requires multiple tokenizers and staged models.
* The model is not presented as a general controllable assistant.
* Continuation quality may vary across underrepresented accents or dialects.

### A very important interview point

A common mistake is to say “Whisper replaced wav2vec 2.0” or “AudioLM is just Whisper for generation.”

That is not right.

* **wav2vec 2.0** solves representation learning for speech recognition efficiency
* **Whisper** solves robust speech-to-text and translation with web-scale weak supervision
* **AudioLM** solves high-quality coherent audio generation

They share ideas, but their goals are different.

---

## Interview-Ready Understanding

### What you should be able to explain clearly

You should be able to explain:

1. why unlabeled audio is valuable in speech AI
2. the difference between self-supervised learning and weak supervision
3. how wav2vec 2.0 learns from masked latent audio
4. why Whisper’s multitask token format matters
5. why zero-shot robustness is different from in-distribution benchmark performance
6. why AudioLM separates semantic and acoustic tokens
7. why recognition systems and generation systems use different objectives
8. what trade-offs you would consider in a real product

### Likely interview questions and plain-English answers

| Question                                                         | Concise model answer                                                                                                                                                                          |
| ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **What problem does wav2vec 2.0 solve?**                         | It reduces dependence on transcripts by learning speech representations from unlabeled audio first, then fine-tuning with a smaller labeled set.                                              |
| **Why does wav2vec 2.0 use masking and contrastive learning?**   | Masking forces the model to use surrounding context, and contrastive learning gives it a way to learn without explicit labels by choosing the correct hidden target among distractors.        |
| **What is the role of quantization in wav2vec 2.0?**             | It turns continuous speech features into discrete target-like units, which makes the contrastive task more robust and helps the model learn more abstract speech structure.                   |
| **How is Whisper different from wav2vec 2.0?**                   | Whisper is trained directly on large audio-text data for speech tasks, while wav2vec 2.0 first learns an encoder from unlabeled audio and then needs fine-tuning for recognition.             |
| **Why is Whisper considered robust?**                            | Because it is trained on a very large and diverse dataset and evaluated zero-shot across many domains, so it is less tied to one benchmark distribution.                                      |
| **Why does Whisper use special tokens?**                         | They let one decoder handle many tasks, such as transcription, translation, timestamps, and language-related behavior, through a shared token format.                                         |
| **What is the core idea of AudioLM?**                            | AudioLM treats audio generation like language modeling, but uses two token levels: semantic tokens for long-term structure and acoustic tokens for sound quality.                             |
| **Why not generate only acoustic tokens in AudioLM?**            | Because acoustic tokens alone can give good sound quality but weak long-term consistency, which can sound like realistic babbling rather than coherent content.                               |
| **What is the difference between semantic and acoustic tokens?** | Semantic tokens capture higher-level content, while acoustic tokens capture local sound details such as speaker identity and recording characteristics.                                       |
| **When would you choose each method?**                           | Use wav2vec 2.0 when labels are scarce and you need a strong speech encoder, Whisper when you need robust transcription or translation, and AudioLM when you need realistic audio generation. |

### Stronger interview version: one-minute synthesis

A good one-minute answer could sound like this:

> wav2vec 2.0, Whisper, and AudioLM represent three major directions in speech AI. wav2vec 2.0 shows that you can learn strong speech representations from unlabeled audio using masked prediction and contrastive learning, which is great for low-resource ASR. Whisper shows that if you scale weakly supervised audio-text data enough, an encoder-decoder Transformer can become a robust zero-shot speech system across many tasks and languages. AudioLM tackles a different problem: generation. It turns audio into discrete semantic and acoustic tokens, then models them hierarchically so it can keep long-term coherence while still sounding realistic. Together, they show how modern speech systems move between raw audio, learned representations, text outputs, and generated audio.

---

## Glossary

| Term                                            | Beginner-friendly definition                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **ASR (Automatic Speech Recognition)**          | Technology that converts spoken audio into text.                                                                   |
| **Waveform**                                    | The raw audio signal over time.                                                                                    |
| **Log-Mel spectrogram**                         | A time-frequency representation of audio that is easier for many models to process than the raw waveform.          |
| **Representation**                              | An internal feature vector the model uses to summarize input information.                                          |
| **Latent representation**                       | A hidden internal representation, not directly visible to the user.                                                |
| **Self-supervised learning**                    | Learning from unlabeled data by creating a prediction task from the input itself.                                  |
| **Weak supervision**                            | Training with labels that exist but may be noisy, incomplete, or imperfect.                                        |
| **Transformer**                                 | A sequence model that uses attention to connect information across time steps or tokens.                           |
| **Encoder-decoder Transformer**                 | A model where one network encodes the input and another generates the output conditioned on that encoding.         |
| **Decoder-only Transformer**                    | A model that predicts the next token from previous tokens.                                                         |
| **Masking**                                     | Hiding parts of the input or latent sequence during training so the model must infer them from context.            |
| **Contrastive loss**                            | A loss that makes the correct target score higher than incorrect alternatives.                                     |
| **Quantization**                                | Converting continuous vectors into discrete codebook choices.                                                      |
| **Codebook**                                    | A learned set of discrete entries a model can choose from during quantization.                                     |
| **Product quantization**                        | A quantization method that picks entries from multiple codebooks and combines them.                                |
| **Gumbel softmax**                              | A technique that lets a model make approximately discrete choices while still being trainable by gradient descent. |
| **CTC (Connectionist Temporal Classification)** | A speech-recognition loss that handles unknown alignment between audio frames and text labels.                     |
| **Token**                                       | A discrete unit in a sequence, such as a subword in text or a learned symbol in audio.                             |
| **Autoregressive generation**                   | Generating one token at a time, always conditioning on previous generated tokens.                                  |
| **WER (Word Error Rate)**                       | A transcription error metric based on insertions, deletions, and substitutions of words.                           |
| **BLEU**                                        | A metric used to evaluate translation quality by overlap with reference translations.                              |
| **VAD (Voice Activity Detection)**              | Detecting whether speech is present in an audio segment.                                                           |
| **Zero-shot**                                   | Using a model on a task or dataset without additional task-specific training on that dataset.                      |
| **Out-of-distribution**                         | Data that differs from the training distribution in domain, noise, style, speakers, or other ways.                 |
| **Prosody**                                     | The rhythm, stress, and intonation of speech.                                                                      |
| **Codec**                                       | A system that compresses and reconstructs audio.                                                                   |
| **Residual vector quantizer**                   | A stack of quantizers where later quantizers encode the remaining detail not captured by earlier ones.             |
| **Semantic tokens**                             | Discrete units meant to capture high-level content and long-term structure.                                        |
| **Acoustic tokens**                             | Discrete units meant to capture fine sound details and reconstruction quality.                                     |
| **Prompt**                                      | A short initial piece of input used to condition a generation model.                                               |

---

## Recap

You should now understand the main idea behind three major speech/audio research directions:

* **wav2vec 2.0** teaches speech models from unlabeled audio by masking latent spans and learning to identify the right hidden target. It matters because it dramatically reduces reliance on transcripts.
* **Whisper** shows that large, diverse, weakly supervised audio-text training can produce a robust, practical speech model that handles multiple tasks with one token-based interface.
* **AudioLM** shows that realistic audio generation benefits from splitting long-term meaning from local sound detail, then generating those levels hierarchically.

The most important big-picture lesson is this:

> Modern speech systems are defined less by “audio in, text out” and more by how they **represent**, **tokenize**, and **predict structure over time**.

What remains limited or uncertain:

* The papers do not give a single unified production blueprint combining all three approaches.
* Real-world deployment costs, latency, and safety controls are only partially discussed.
* Performance and fairness across all languages, accents, and domains remain incomplete.

---

## Key Citations

[Wav2Vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/pdf/2006.11477)

[Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/pdf/2212.04356)

[AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/pdf/2209.03143)

---
---
---



