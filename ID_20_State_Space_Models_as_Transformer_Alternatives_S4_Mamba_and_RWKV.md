# State Space Models as Transformer Alternatives: S4, Mamba, and RWKV

## What This Report Teaches

This report explains a family of sequence models that aim to keep the strongest advantage of recurrent models - efficient scaling with long sequences - while recovering as much of Transformer-level quality as possible. The three papers do this in different ways. **S4** builds a principled deep-learning version of a classical **state space model (SSM)** and makes it computationally practical. **Mamba** keeps the SSM foundation but adds **selectivity**, meaning the model can decide, based on the current token, what to keep and what to forget. **RWKV** comes from a slightly different angle: it is not presented as a classical SSM paper, but as an architecture that combines Transformer-style parallel training with RNN-style recurrent inference, making it a useful comparison point in the broader “Transformer alternatives” story. 

By the end, you should understand what a state space model is in plain English, why S4 mattered, why Mamba was seen as a breakthrough for language modeling, how RWKV differs from both, and what trade-offs these models make compared with Transformers. You should also be able to explain them in an AI engineer or AI architect interview without relying on dense math notation. 

One source note matters here: the provided RWKV URL points to arXiv **2401.06118**, which is actually a different paper about additive quantization, while the supplied title **“RWKV: Reinventing RNNs for the Transformer Era”** matches arXiv **2305.13048**. This report therefore uses the paper that matches the title. ([arXiv][1])

---

## Key Takeaways

* **S4 made state space models practical for deep sequence modeling.** Before S4, earlier SSM-based approaches had the right theory for long-range memory but were too expensive or unstable to use broadly. S4’s contribution was a parameterization that reduced the core computation to a stable **Cauchy kernel** problem with near-linear complexity in sequence length and state size. The practical implication is that SSMs became a serious alternative backbone rather than only a mathematical curiosity. 

* **Mamba’s big idea is selectivity.** Earlier structured SSMs were mostly **linear time-invariant (LTI)**, meaning their dynamics stayed the same at every position. Mamba argues that this is exactly why they struggle on language: they cannot do enough **content-based reasoning**. By making some SSM parameters depend on the current input token, Mamba can selectively remember, update, or forget information. The practical implication is much stronger performance on discrete data such as language. 

* **Mamba is not just “S4 but bigger.”** It changes both the model and the implementation. Because input-dependent parameters break the old convolution trick, Mamba introduces a **hardware-aware parallel scan** algorithm and a simplified homogeneous block design without attention or even separate MLP blocks. The practical implication is linear-time sequence modeling that is still fast enough to compete in real language-model settings. 

* **RWKV tries to get the best of both Transformers and RNNs.** Its claim is that the model can be trained in a Transformer-like parallel form, but run at inference in an RNN-like recurrent form with constant memory and compute per token. The practical implication is efficient long-context inference without KV-cache growth like a standard Transformer. 

* **These three papers are solving different bottlenecks.** S4 solves “can SSMs be computed efficiently and stably?” Mamba solves “can SSM-style models handle language-quality content selection?” RWKV solves “can we build a scalable recurrent model that trains like a Transformer and infers like an RNN?” The practical implication is that they are best seen as steps in a design space, not as identical competitors. 

* **Transformers are still the reference point in all three papers.** S4 emphasizes long-range efficiency and speed, Mamba reports best-in-class open-model performance at comparable sizes and claims 5x higher inference throughput than similarly sized Transformers, and RWKV reports performance on par with similarly sized Transformers while scaling to 14B parameters. The practical implication is that these papers matter because they try to match Transformer quality without paying full attention cost. 

* **A major trade-off remains information access.** Transformers can revisit all earlier tokens through attention. Recurrent and SSM-style models compress history into a state. That makes them efficient, but it also risks losing fine-grained details. RWKV states this explicitly as a limitation, and Mamba’s selectivity is best understood as a direct attempt to soften exactly this weakness. 

---

## Background and Foundations

A **sequence model** is a model that reads ordered data - words, audio samples, DNA bases, time-series measurements, pixels in a scan order - and predicts something about that sequence. The central difficulty is that useful information may be far apart. A model may need to remember something from thousands or even millions of steps ago. S4 frames this as the problem of **long-range dependencies**, and notes that standard model families such as RNNs, CNNs, and Transformers all have special variants designed to handle them, but still struggle on very long sequences in practice. 

A **state space model (SSM)** is a classical model from control theory and signal processing. In the paper’s basic form, it has a hidden state (x(t)), an input (u(t)), and an output (y(t)), related by
(x'(t) = Ax(t) + Bu(t)) and (y(t) = Cx(t) + Du(t)). In plain English, the hidden state is the model’s running memory, the input updates that memory, and the output reads from it. The matrix (A) controls how memory evolves, (B) controls how new input enters memory, (C) controls how memory is read out, and (D) is a direct skip path from input to output. 

Why is this attractive for deep learning? Because an SSM gives a built-in notion of **state**. Unlike attention, it does not need to compare every token with every previous token. Instead, it compresses the past into a running internal memory. That suggests linear scaling with sequence length. The problem, as S4 explains, is that naïve SSMs do not work well enough, and earlier promising versions were too slow or numerically unstable. 

A key term in S4 and Mamba is **linear time invariance (LTI)**. An LTI system behaves the same way at every timestep. In other words, its update rule does not depend on what token is currently being processed. This is good for efficiency, because it lets you turn recurrence into a convolution and parallelize training. But it is also restrictive: the model cannot easily say “this token is important, keep it” or “this token is irrelevant, ignore it.” Mamba’s entire motivation is that this weakness hurts language modeling. 

RWKV belongs in this conversation because it tries to preserve the RNN advantage - constant-memory recurrent inference - while keeping Transformer-like training parallelism. It is not presented as a formal SSM in the same way S4 and Mamba are. Instead, it reformulates a linear-attention-style mechanism into a recurrent update and combines it with two custom blocks called **time-mixing** and **channel-mixing**. That makes it a close conceptual neighbor, even if its derivation is different. 

---

## Big Picture First

The cleanest mental model across the three papers is this:

1. **S4** says: “Use a principled continuous-time memory model, but reparameterize it so it can be trained efficiently.” 
2. **Mamba** says: “That memory model is still too rigid for language, so make the memory update depend on the current token.” 
3. **RWKV** says: “You can also attack the same efficiency problem by building a recurrent alternative that behaves like a Transformer during training and like an RNN during inference.” 

The comparison below summarizes the high-level design differences. It is a synthesis of the three papers’ method sections and architecture descriptions. 

| Paper | Core idea                                       | Main computational trick                                                                | Why it matters                                                      |
| ----- | ----------------------------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| S4    | Structured state space model for long sequences | Reparameterize the SSM so recurrence and convolution become efficient and stable        | Makes deep SSMs practical                                           |
| Mamba | Selective state space model                     | Make SSM parameters input-dependent, then recover efficiency with a hardware-aware scan | Gives SSM-style models content-based behavior needed for language   |
| RWKV  | Transformer/RNN hybrid                          | Train in parallel, infer recurrently with linear-attention-style updates                | Offers constant-memory inference and large-scale recurrent modeling |

Another useful high-level view is where each model stores history.

* **S4** stores history in a mathematically structured latent state with fixed dynamics. 
* **Mamba** stores history in a latent state too, but can change how that state is updated based on the current input. 
* **RWKV** stores history in recurrent summaries built by its time-mixing mechanism and explicitly notes that this compression can limit recovery of tiny details over very long contexts. 

---

## Core Concepts Explained

### State Space Model

A state space model is a way to describe sequence processing through a hidden memory state. Instead of directly comparing all past tokens to the current one, the model updates a compact internal state and then reads from it. In deep-learning terms, that makes it closer to an RNN than to attention, but with a more principled continuous-time interpretation. S4 uses this as the foundation for its entire architecture. 

### Discretization

The original SSM equations are continuous-time equations. But token sequences are discrete. So the model must be **discretized**, meaning converted from continuous-time dynamics into step-by-step update rules for token positions. Mamba explicitly describes this as turning continuous parameters ((\Delta, A, B)) into discrete parameters ((\bar A, \bar B)). In plain English, discretization answers the question: “How do I update memory once per token?” 

### Recurrence and Convolution

One important insight in S4 is that the same SSM can be viewed in two ways. It can be run as a **recurrence**, which is natural for autoregressive inference, or as a **convolution**, which is natural for parallel training when the whole sequence is known. This duality is one reason SSMs are attractive. S4’s achievement is making both views practical. Mamba inherits the recurrence idea, but loses the simple convolution trick once parameters become input-dependent. 

### HiPPO

**HiPPO** is a theory of continuous-time memorization used by S4 to choose special state matrices (A) that can preserve long-range information. The beginner-friendly way to think about it is: not all memory update rules are equal. A random update matrix forgets badly. HiPPO gives a structured update rule designed for long-range memory. S4 then figures out how to make that structure efficient. 

### Normal Plus Low-Rank (NPLR)

S4’s main technical trick is to show that the special HiPPO matrix can be rewritten as **normal plus low-rank**. “Low-rank” means the hard part of the matrix can be expressed with only a few important directions instead of a full dense correction. This matters because it lets S4 stably diagonalize the easy part and correct the rest with the **Woodbury identity**, eventually reducing the computation to a **Cauchy kernel**. You do not need to memorize the algebra for interviews. The practical point is: S4 found a way to keep the good long-memory structure while turning the expensive computation into a fast, stable one. 

### Linear Time Invariance (LTI)

An LTI model uses the same dynamics at every position. This makes it elegant and efficient, but also rigid. If the update rule is the same everywhere, the model cannot naturally adapt its behavior based on token content. Mamba identifies this as the key weakness of earlier SSMs on language tasks. 

### Selectivity

**Selectivity** means the model can decide, based on the current token, how strongly to preserve, reset, or update its hidden state. Mamba makes parameters like (\Delta), (B), and (C) functions of the input, which makes the state update content-aware. The paper explicitly interprets (\Delta) as similar to an RNN gate: a large value resets and focuses on the current input, while a small value preserves the existing state. This is one of the most interview-important ideas in the Mamba paper. 

### Selective Scan

Once parameters depend on the current input, you can no longer use the old efficient convolution approach. Mamba’s answer is a **hardware-aware parallel scan** algorithm that keeps the model efficient in recurrent mode by carefully using the GPU memory hierarchy. The plain-English lesson is that Mamba is as much a systems paper as an architecture paper. The math idea alone would not have been enough without the implementation trick. 

### Time-Mixing and Channel-Mixing

RWKV is built from two recurrent-style sub-blocks. **Time-mixing** blends the current token with the previous token and computes a recurrent weighted key-value update across time. **Channel-mixing** is a feature-wise transformation with its own gating behavior. The paper presents these as the core building blocks that let RWKV act partly like attention and partly like an RNN. 

---

## Step-by-Step Technical Walkthrough

### 1. S4: how the model works

1. Start from the classical continuous-time SSM equations. The model has an input, a hidden state, and an output. 
2. Choose a special structured state matrix (A) using the HiPPO framework so the hidden state can preserve long-range information. 
3. Discretize the continuous model so it can operate on token sequences. 
4. Observe that the model can be computed either as a recurrence or as a convolution. Training prefers the parallel convolution-style view; autoregressive generation prefers the recurrent view. 
5. Reparameterize the difficult structured matrix into a normal-plus-low-rank form. This makes the algebra tractable and stable. 
6. Reduce the convolution computation to a small number of Cauchy kernel multiplies. That is the step that makes the model efficient. 

**What the formulas are trying to do:** They are describing how memory evolves and how the full sequence transformation can be computed without explicitly storing huge intermediate states. The important practical fact is that S4 gets near-linear complexity and makes both training and inference usable. 

**Key trade-off:** S4 is elegant and efficient, but still fundamentally time-invariant. That becomes the opening for Mamba. 

### 2. Mamba: how the model works

1. Begin with the SSM setup used in earlier models like S4: a hidden state evolves through parameters such as (\Delta, A, B, C). 
2. Make some of those parameters functions of the current input. This turns a fixed memory update rule into a content-dependent one. 
3. Lose the simple convolution shortcut, because time invariance is now broken. 
4. Recover efficiency with a **selective scan** implementation that avoids materializing huge expanded states in slow memory. 
5. Use a simplified repeated **Mamba block**, which combines ideas from prior SSM blocks and MLP-style blocks into one homogeneous module. 
6. Run training and inference in linear time with strong throughput and long-context scaling. 

**Why this matters:** Mamba’s synthetic tasks make the motivation concrete. On **Selective Copying**, the model must remember only the relevant tokens and ignore the rest. On **Induction Heads**, it must learn a copying behavior similar to what is often discussed in Transformer in-context learning. The paper reports that Mamba’s selective SSM solves these tasks and generalizes perfectly to million-length sequences in the induction experiment, while non-selective alternatives do not. 

**Key trade-off:** Mamba fixes the rigidity problem, but it needs a more complex implementation than S4 because the simple convolution route is gone. 

### 3. RWKV: how the model works

1. Process the sequence with stacked residual blocks. Each block has a **time-mixing** sub-block and a **channel-mixing** sub-block. 
2. In time-mixing, blend the current token and previous token through learned interpolation, then compute a weighted key-value accumulation across time. 
3. In channel-mixing, apply a gated feature transformation that behaves more like an MLP-style update over channels. 
4. Train the model in a parallelized form, similar in spirit to Transformers. 
5. Re-express the time-mixing update as a recurrent computation for inference, which gives constant memory and compute per token. The paper explicitly shows a recursive form for the WKV computation. 

**What the formulas are trying to do:** They are building a running summary of the past using learned decay and gating, rather than explicitly attending to all previous tokens. In plain English, RWKV tries to keep the “one-state-summary” efficiency of an RNN while preserving enough expressiveness to scale like modern language models. 

**Key trade-off:** Because information is funneled through recurrent summaries, the paper explicitly warns that very fine-grained information may be harder to recover than with full self-attention over long contexts. 

---

## Paper-by-Paper Explanation

## S4: Efficiently Modeling Long Sequences with Structured State Spaces

### Problem addressed

S4 addresses a bottleneck in earlier state-space-based long-sequence models: they had promising theory for long-range memory, but were computationally impractical and numerically unstable as general-purpose deep models. 

### Method used

The paper starts from a continuous-time SSM, uses HiPPO-structured state matrices for long memory, discretizes the model, and then reparameterizes the state matrix into normal-plus-low-rank form so the key computation reduces to a Cauchy kernel. This supports both an efficient recurrent view and an efficient convolutional view. 

### Main innovation

The core innovation is not merely “use an SSM.” It is the parameterization and algorithmic reduction that make structured SSMs stable and efficient enough for deep learning. In practical terms, S4 converts a beautiful but unusable idea into something trainable. 

### Main findings

S4 reports 91% accuracy on sequential CIFAR-10, closes the WikiText-103 gap to Transformers to within 0.8 perplexity, performs autoregressive generation about 60x faster than standard autoregressive models on CIFAR-10 and WikiText-103, and is the first model to solve Long Range Arena Path-X of length 16,384 with 88% accuracy after prior work was at random guessing. 

### Limitations

The paper is technically heavy and depends on specialized numerical structure. It also solves the efficiency/stability problem without solving the later “content selection in language” problem that Mamba highlights. 

### What changed compared with earlier work

Compared with earlier linear state space layers, S4 makes structured SSMs fast and stable enough to be broadly useful. This is why S4 is the foundation paper for the later SSM revival. 

---

## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

### Problem addressed

Mamba starts from the observation that previous subquadratic models, including structured SSMs, had not matched attention on important discrete modalities like language. The paper argues that the missing ingredient is **content-based reasoning**. 

### Method used

The paper makes SSM parameters input-dependent, creating a **selective** SSM. Because this breaks time invariance and the efficient convolution route, it introduces a hardware-aware recurrent scan algorithm and places the resulting layer inside a simplified repeated Mamba block. 

### Main innovation

The main innovation is selectivity itself: the model can choose what to propagate or forget based on the current token. A second innovation is the systems work needed to make that selective model efficient. 

### Main findings

The paper reports that Mamba has 5x higher inference throughput than Transformers, improves on real data up to million-length sequences, and that Mamba-3B outperforms same-size Transformers while matching Transformers roughly twice its size on language modeling. In zero-shot downstream evaluation, Table 3 states that Mamba is best-in-class for each size shown and generally matches baselines at twice the model size. 

### Limitations

The paper acknowledges that selectivity comes with extra implementation complexity. Information not provided: the paper does not present a full production serving stack or broader safety/governance issues because its focus is architecture and efficiency, not deployment policy. 

### What changed compared with earlier work

Compared with S4, Mamba relaxes the time-invariant assumption and turns SSMs into content-aware sequence models. This is the central reason it performed much better on language. 

---

## RWKV: Reinventing RNNs for the Transformer Era

### Problem addressed

RWKV addresses the Transformer trade-off directly: strong performance and parallel training, but quadratic scaling and growing memory cost with context length. It asks whether a recurrent alternative can keep more of the Transformer’s strengths than earlier RNN-style models did. 

### Method used

RWKV builds residual blocks with time-mixing and channel-mixing, reformulates its linear-attention-style weighted key-value computation into a recurrent form, and uses parallelization during training with constant-memory recurrent inference. 

### Main innovation

Its main innovation is architectural: it tries to unify Transformer-style training and RNN-style inference in a single model. The paper also frames its linear-attention reformulation as a way to remove quadratic complexity without approximation. 

### Main findings

The paper states that RWKV scales from 169M to 14B parameters trained on the Pile and performs on par with similarly sized Transformers. It also reports lower Pile test loss with increased context length and shows competitive zero-shot results against major open-source Transformer baselines on several reasoning tasks. 

### Limitations

The paper explicitly states that its linear-attention recurrent compression may limit tasks requiring recall of fine details over very long contexts, because information is funneled through a single vector-like summary rather than preserved as full pairwise interactions as in quadratic attention. 

### What changed compared with earlier work

Compared with classical RNNs, RWKV is much more parallelizable and scalable. Compared with S4 and Mamba, it is less grounded in formal continuous-time SSM theory and more grounded in an RNN/linear-attention hybrid design. That makes it a useful “third branch” in the Transformer-alternative landscape. 

---

## Comparison Across Papers or Methods

The table below compares the approaches on the dimensions that matter most for understanding the design space. It synthesizes the methods and stated findings across the three papers. 

| Aspect                  | S4                                    | Mamba                                                | RWKV                                               |
| ----------------------- | ------------------------------------- | ---------------------------------------------------- | -------------------------------------------------- |
| Core family             | Structured state space model          | Selective state space model                          | RNN / linear-attention hybrid                      |
| Main weakness addressed | SSM efficiency and stability          | Lack of content-aware reasoning in earlier SSMs      | Transformer quadratic cost and RNN scalability gap |
| Time invariance         | Yes, mostly LTI                       | No, parameters depend on input                       | Recurrent with learned decay/gating                |
| Training view           | Can use convolutional parallelization | Hardware-aware scan in recurrent mode                | Transformer-like parallelization                   |
| Inference view          | Recurrent latent-state generation     | Recurrent linear-time inference                      | Recurrent constant-memory inference                |
| Main strength           | Principled long-memory backbone       | Strong language performance with linear-time scaling | Large-scale recurrent LM with efficient inference  |
| Main limitation         | Too rigid for content selection       | More complex implementation                          | Compressed recurrent state may lose fine detail    |

Another useful comparison is what changed historically from paper to paper. 

| Historical step | What changed                                                                   | Why it mattered                                                         |
| --------------- | ------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| S4              | Made structured SSMs usable                                                    | Opened the door for serious non-attention long-sequence models          |
| Mamba           | Added input-dependent selection                                                | Closed much of the quality gap on language                              |
| RWKV            | Scaled a recurrent alternative to 14B and emphasized Transformer-like training | Showed that recurrent models could still be competitive at modern scale |

---

## Real-World System and Application

In a practical LLM system, these models matter because their cost profile is different from Transformers. A standard Transformer stores a growing **KV cache** during autoregressive generation. A recurrent or SSM-style model instead keeps a fixed-size hidden state. That can make long-context inference much cheaper in memory, especially when generating many tokens. S4 emphasizes fast recurrent generation, Mamba emphasizes throughput gains at inference, and RWKV emphasizes constant memory and computation during inference. 

A practical system-level interpretation looks like this:

1. **Use S4-style ideas** when you want a principled long-memory sequence layer and are comfortable with more mathematical structure. 
2. **Use Mamba-style ideas** when language quality is central and you want a strong linear-time backbone with better content selection. 
3. **Use RWKV-style ideas** when recurrent inference efficiency is the main attraction and you want something closer in engineering spirit to an RNN-backed LLM. 

Information not provided: none of these papers gives a full production deployment design covering serving infrastructure, fault tolerance, model routing, security, or observability. They are architecture papers, not end-to-end platform designs. 

---

## Limitations and Trade-offs

The key trade-offs are easier to understand in plain English than in formulas. The summary below is grounded in the papers’ method sections and explicit limitations. 

| Limitation or trade-off         | Concrete meaning                                                    | Why it matters                                                                        |
| ------------------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| State compression               | History is compressed into a state instead of stored token-by-token | Efficient, but may lose fine details compared with full attention                     |
| Time invariance in earlier SSMs | Same dynamics at every timestep                                     | Great for efficiency, weak for content-aware language behavior                        |
| Implementation complexity       | Mamba needs special scan kernels; S4 needs specialized math         | Good theory alone is not enough for practical training/inference                      |
| Long-context reasoning style    | These models summarize history rather than revisiting all tokens    | Different failure mode from Transformer attention                                     |
| Hardware dependence             | Speed claims often depend on optimized kernels and memory hierarchy | Practical gains depend on implementation quality                                      |
| Evaluation scope                | Results are strong, but not universal proof of superiority          | Architecture choice still depends on task, context length, and deployment constraints |

A mature interview answer should make one point very clearly: these models are not “free improvements over Transformers.” They trade **explicit token-to-token access** for **compact recurrent memory**. That is why they can scale linearly, and also why they need design ideas like HiPPO structure, selection, decay, or gating to avoid forgetting the wrong things. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain that **state space models** are sequence models with a running latent memory state. S4 makes them efficient and stable, Mamba makes them selective and content-aware, and RWKV offers a related recurrent alternative that trains in parallel and infers recurrently. You should also be able to explain the core trade-off with Transformers: lower time and memory growth with sequence length, but weaker direct access to every past token unless the recurrent state is designed very carefully. 

### Likely interview questions

#### 1. What is a state space model in plain English?

It is a sequence model that keeps a running hidden state summarizing the past. New inputs update that state, and outputs are read from it. Unlike attention, it does not compare the current token to every previous token directly. 

#### 2. Why was S4 important?

Because earlier SSM-based models had the right long-memory idea but were too slow or unstable. S4 found a parameterization that made them practical with near-linear complexity and strong long-sequence performance. 

#### 3. What is the main weakness of pre-Mamba SSMs?

They were mostly linear time-invariant, so they could not adapt their memory behavior based on token content. Mamba argues this is why they underperformed on language. 

#### 4. What does “selective” mean in Mamba?

It means the model’s memory update depends on the current token. The model can choose to keep, ignore, or reset information based on what it is currently reading. 

#### 5. Why did Mamba need a new algorithm?

Because once the SSM becomes input-dependent, you lose the simple convolution trick used by earlier time-invariant SSMs. Mamba had to recover efficiency with a hardware-aware scan implementation. 

#### 6. How is RWKV different from S4 and Mamba?

RWKV is not built as a classical structured SSM. It is a recurrent/linear-attention hybrid with time-mixing and channel-mixing blocks, designed to train in parallel and infer recurrently. 

#### 7. What is the big advantage of RWKV at inference?

The paper says it maintains constant computational and memory complexity during inference, which is attractive compared with Transformer KV-cache growth. 

#### 8. What is the main weakness of recurrent alternatives compared with Transformers?

They compress past information into a hidden state rather than keeping all pairwise token interactions explicitly available. That can hurt tasks needing precise recall of small details far back in context. 

#### 9. Why does S4 use both recurrence and convolution?

They are two views of the same model. Convolution is good for parallel training; recurrence is good for step-by-step generation. 

#### 10. How would you summarize the historical progression?

S4 made structured SSMs usable, Mamba made them content-aware enough for stronger language modeling, and RWKV showed that large recurrent alternatives could still be scaled and remain competitive with Transformers. 

---

## Glossary

| Term                         | Beginner-friendly definition                                                                                |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Sequence model               | A model that processes ordered data such as tokens, audio, or time-series                                   |
| Long-range dependency        | A case where useful information lies far back in the sequence                                               |
| State space model (SSM)      | A model that updates a hidden memory state over time and reads outputs from it                              |
| Hidden state                 | The running internal memory of the model                                                                    |
| Discretization               | Turning a continuous-time update rule into a step-by-step token update rule                                 |
| Linear time invariance (LTI) | The system uses the same update rule at every timestep                                                      |
| HiPPO                        | A structured way to design memory matrices for long-range continuous-time memorization                      |
| Normal plus low-rank (NPLR)  | A matrix form that is mostly easy to handle plus a small corrective part                                    |
| Cauchy kernel                | The stable kernel computation S4 reduces its main convolution problem to                                    |
| Recurrence                   | Processing one timestep after another while carrying state forward                                          |
| Convolutional view           | A parallel sequence computation view that can replace explicit recurrence when the system is time-invariant |
| Selectivity                  | Letting the memory update depend on the current input token                                                 |
| Selective scan               | Mamba’s efficient implementation for input-dependent recurrent updates                                      |
| Time-mixing                  | RWKV’s block for combining current and previous-token information over time                                 |
| Channel-mixing               | RWKV’s block for feature-wise transformation with gating                                                    |
| KV cache                     | The stored key/value tensors Transformers keep during autoregressive inference                              |
| Throughput                   | How many tokens or examples can be processed per unit time                                                  |

The definitions above are derived from how the three papers frame their models and computations. 

---

## Recap

You should now understand the main story. S4 shows how to turn classical state space ideas into an efficient deep-learning sequence model. Mamba shows that efficiency alone is not enough for language, and adds content-dependent selectivity plus a specialized scan implementation to recover quality. RWKV shows a related but distinct path: build a large recurrent model that behaves more like a Transformer during training and more like an RNN during inference. 

The most important conceptual difference from Transformers is that these models do not keep full token-to-token interaction available in the same way. They replace that with compact state updates. That is why they scale better with long sequences, and also why the central design question becomes: **how do we keep the right information in state without forgetting what matters?** S4 answers with structure, Mamba answers with selection, and RWKV answers with time-decay plus gating. 

What remains uncertain is also important. These papers show strong progress, but they do not prove that one universal Transformer replacement has already won. The right choice still depends on the task, the sequence length regime, the hardware stack, and whether you care more about training simplicity, inference memory, or exact long-context recall. 

---

## Key Citations

* Efficiently Modeling Long Sequences with Structured State Spaces. 

* Mamba: Linear-Time Sequence Modeling with Selective State Spaces. 

* RWKV: Reinventing RNNs for the Transformer Era. 

[1]: https://arxiv.org/abs/2401.06118?utm_source=chatgpt.com "Extreme Compression of Large Language Models via Additive Quantization"

---
---
---

