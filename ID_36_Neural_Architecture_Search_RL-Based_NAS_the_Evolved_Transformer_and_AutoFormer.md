# Neural Architecture Search: RL-Based NAS, the Evolved Transformer, and AutoFormer

## What This Report Teaches

This report explains **Neural Architecture Search (NAS)** from first principles and then uses three papers to show how the field changed over time. The first paper, **Neural Architecture Search with Reinforcement Learning**, treats architecture design as a sequential decision problem and uses a controller RNN trained with policy gradients. The second, **The Evolved Transformer**, moves to **evolutionary search** over Transformer-like encoder-decoder cells and adds a practical resource-allocation method called **Progressive Dynamic Hurdles (PDH)** so the search can run on a much more expensive translation task. The third, **AutoFormer**, moves again toward **one-shot NAS** with a shared **supernet**, so thousands of candidate vision transformers can be evaluated using inherited weights instead of training each one from scratch. 

There is one important source note. Your third entry has a title/URL mismatch: the URL `2008.06808` points to **Finding Fast Transformers: One-Shot Neural Architecture Search by Component Composition**, while **AutoFormer: Searching Transformers for Visual Recognition** is a different paper at `2107.00651`. Because your title clearly specifies AutoFormer, I based the report on the actual AutoFormer paper and I call out the mismatch explicitly here. ([arXiv][1])

By the end, you should understand what a **search space** is, how NAS algorithms propose and score architectures, why evaluation cost dominates NAS in practice, how **reinforcement learning**, **evolution**, and **weight-sharing supernets** differ, and why NAS gradually shifted from “train every candidate almost independently” toward “reuse weights so search becomes affordable.” 

---

## Key Takeaways

* **NAS turns model design into an optimization problem over architectures.** Instead of hand-writing the network, you define a search space, a search algorithm, and a score such as validation accuracy. **Why it matters:** it changes architecture design from manual intuition to automated search. **Practical implication:** the quality of NAS depends as much on the search space and evaluation loop as on the search algorithm itself. 

* **The 2016 RL NAS paper is conceptually simple but computationally expensive.** A controller RNN samples an architecture, that child model is trained, and its validation accuracy becomes the reward for REINFORCE. **Why it matters:** this paper made NAS famous, but it also exposed the central bottleneck of NAS: candidate evaluation is very costly. **Practical implication:** early NAS was powerful but hard to reproduce at scale. 

* **The Evolved Transformer keeps the black-box search idea but makes it more practical for sequence models.** It uses evolutionary search, warm-starts the population with the human-designed Transformer, and uses PDH to train promising candidates longer while discarding weak ones early. **Why it matters:** it shows NAS can be adapted to expensive tasks like machine translation, not just smaller vision benchmarks. **Practical implication:** search efficiency often comes from smarter evaluation policy, not only from a different optimizer. 

* **Search-space design is a major part of NAS.** The Evolved Transformer builds a huge structured search space over encoder and decoder cells, while AutoFormer searches depth, embedding dimension, Q-K-V dimension, number of heads, MLP ratio, and model size ranges. **Why it matters:** NAS cannot discover architectures outside the space you give it. **Practical implication:** good NAS often depends on a carefully engineered search space that already contains strong human ideas. 

* **AutoFormer represents the one-shot NAS shift.** It encodes a very large transformer search space into a supernet and trains candidate subnets by weight sharing, then uses evolution only after the supernet is trained. **Why it matters:** it reduces the cost of evaluating many candidates. **Practical implication:** modern NAS often separates “train a reusable supernet once” from “search inside it cheaply afterward.” 

* **Weight sharing solves one problem but introduces another.** AutoFormer argues that its “weight entanglement” makes inherited subnet performance close to retraining from scratch, but this is exactly the core risk in one-shot NAS: the supernet’s ranking of candidates may not perfectly match full retraining. **Why it matters:** one-shot NAS trades evaluation fidelity for efficiency. **Practical implication:** fast NAS methods are often only as good as the reliability of inherited-weight evaluation. 

* **Across these papers, NAS evolves from expensive direct evaluation to more resource-aware and reusable search.** RL NAS trains sampled children directly, the Evolved Transformer uses staged evaluation with hurdles, and AutoFormer relies on a trained supernet plus constrained evolutionary search. **Why it matters:** this is the main historical arc the interviewer usually wants to hear. **Practical implication:** when discussing NAS, always mention the trade-off between search quality and search cost. 

---

## Background and Foundations

At a high level, NAS asks a simple question: **can we automate architecture design instead of relying entirely on human trial and error?** In all three papers, the answer is yes, but each paper chooses a different way to search. The first paper uses reinforcement learning, the second uses evolution, and the third uses a one-shot supernet plus evolution. 

To understand NAS, you need four basic ingredients:

1. **Search space**: what architectures are allowed
2. **Search strategy**: how candidates are proposed
3. **Evaluation strategy**: how candidates are scored
4. **Selection rule**: how better candidates influence future search 

A beginner-friendly way to think about NAS is that it works like hyperparameter tuning, except the “hyperparameters” can define actual structural choices such as layer type, connectivity, depth, attention heads, or cell composition. That makes NAS much more expressive than simple tuning, but also much more expensive, because the model structure itself changes. 

The main historical problem in NAS is **candidate evaluation cost**. If every sampled architecture has to be built and trained before you know whether it is good, search becomes very expensive. The 2016 RL paper does exactly that. The later two papers are largely about making search more practical under that constraint. 

---

## Big Picture First

A useful mental model is that these three papers represent three stages of NAS maturity:

| Stage                        | Paper               | Core idea                                                           | Main weakness it tries to solve          |
| ---------------------------- | ------------------- | ------------------------------------------------------------------- | ---------------------------------------- |
| Early black-box NAS          | RL-based NAS        | Sample an architecture, train it, use reward to improve the sampler | Manual architecture design               |
| Resource-aware black-box NAS | Evolved Transformer | Use evolutionary search with warm start and staged evaluation       | Search cost on expensive tasks           |
| One-shot NAS                 | AutoFormer          | Train a supernet once, then search many subnets cheaply             | Re-training every candidate from scratch |

This table synthesizes the progression across the three papers. 

The most important conceptual shift is this:

* In **RL NAS**, the search algorithm learns **which architectures to sample**
* In **The Evolved Transformer**, the search algorithm learns **which mutations survive**
* In **AutoFormer**, much of the work moves into **how the supernet is trained**, because the search itself becomes cheaper after that 

So the field does not merely switch optimizers. It changes **where the computational burden lives**. Early NAS spends compute on many fully trained child models. One-shot NAS spends more effort training a reusable supernet so later search is cheaper. 

---

## Core Concepts Explained

### Search Space

A **search space** is the set of architectures the algorithm is allowed to consider. In the RL NAS paper, the controller emits a sequence of architectural tokens. In the Evolved Transformer, the search space consists of encoder and decoder cells made of NASNet-style blocks with branch-level, block-level, and cell-level choices. In AutoFormer, the search space varies transformer dimensions such as depth, embedding dimension, Q-K-V dimension, MLP ratio, and number of heads. 

Why it matters: NAS can only find good models **inside** the space you define. If the space is too small, search is limited. If it is too large, search becomes expensive and noisy. 

### Controller RNN and REINFORCE

In the RL NAS paper, a recurrent neural network acts as a **controller** that emits one decision token after another to define an architecture. Once the child network is trained, its validation accuracy becomes the reward. The controller is updated with **REINFORCE**, a policy-gradient method, because the reward is not differentiable with respect to the controller’s sampling decisions. 

Plain-English meaning: the controller is learning a policy for “what kind of architecture should I try next?” It gets rewarded when sampled architectures perform well. 

### Evolutionary Search

In evolutionary NAS, you keep a population of candidate architectures, select stronger ones as parents, mutate them, evaluate the children, and replace weaker population members. The Evolved Transformer uses tournament selection and mutates a gene encoding of architecture choices. AutoFormer also uses evolution, but only after a supernet has already been trained. 

Plain-English meaning: instead of learning a probability distribution over architectures like RL NAS, evolution repeatedly asks, “Which candidates survive and reproduce?” 

### Warm Start

A **warm start** means beginning search from a known strong architecture instead of from completely random samples. The Evolved Transformer explicitly seeds the initial population with the Transformer so search is anchored around a proven baseline. 

Why it matters: in a huge search space, starting near a strong design can make search more stable and more useful. It also reflects a recurring truth about NAS: many strong NAS systems build directly on human-designed priors rather than replacing them entirely. 

### Progressive Dynamic Hurdles (PDH)

PDH is the Evolved Transformer’s method for saving compute during search. Every child model is first trained only a small amount. After enough models have been seen, the algorithm computes a hurdle based on current population fitness. Only candidates above the hurdle get extra training; weak candidates are discarded early. 

Plain-English meaning: “Do not spend equal training time on every candidate. Spend more on candidates that already look promising.” 

### Supernet and Weight Sharing

A **supernet** is one large network that contains many smaller candidate architectures as subnets. AutoFormer encodes its search space into a supernet so all candidate subnets share weights on overlapping parts. During supernet training, sampled subnets update only the corresponding shared weights. 

Why it matters: this is what makes one-shot NAS much cheaper than training each candidate independently. But it introduces a new risk: inherited weights may not rank candidates exactly the same way as full retraining would. 

### Weight Entanglement

AutoFormer’s special idea is **weight entanglement**. Instead of giving different candidate blocks completely separate weights, it tries to maximally share common parts so updates in one block benefit others. The paper argues this helps many subnets become well trained enough that inherited performance is comparable to retraining from scratch. 

Why it matters: one-shot NAS often fails when supernet training is a poor proxy for real candidate quality. AutoFormer is trying to improve that proxy. 

---

## Step-by-Step Technical Walkthrough

### 1. Neural Architecture Search with Reinforcement Learning

#### High-level goal

Automatically generate neural architectures by treating architecture design as a sequence of actions chosen by a controller RNN and optimized for validation performance. 

#### Pipeline

1. **Define the architecture as a sequence of tokens**
   For example, the controller can emit choices such as filter size, stride, number of filters, or recurrent-cell operations. Each prediction is made by a softmax and fed into the next time step. 

2. **Sample a child architecture**
   The controller finishes generating an architecture description. 

3. **Build and train the child model**
   The sampled architecture is instantiated as a neural network and trained. When training converges, its validation accuracy is measured. 

4. **Use validation accuracy as the reward**
   The reward (R) is the child model’s validation performance. The controller’s objective is to maximize expected reward over sampled architectures. 

5. **Update the controller with REINFORCE**
   Because the reward is not differentiable through the sampling process, the paper uses policy gradients. It also uses an exponential moving-average baseline to reduce variance. 

6. **Repeat and bias the controller toward better architectures**
   Over time, the controller should sample better candidates more often. The paper shows policy-gradient search improving over random search as training progresses. 

#### What the formula means in plain English

The paper’s objective (J(\theta_c)) is the expected reward under the controller’s sampling distribution. In simple language, that means:

> “Choose controller parameters so the architectures it tends to sample will get high validation accuracy.”

The REINFORCE update increases the probability of actions that appeared in high-reward architectures and decreases the probability of actions that appeared in low-reward ones. The moving-average baseline does not change the target; it just makes training less noisy. 

#### Why this paper mattered

The paper showed that search could find a CIFAR-10 architecture with 3.65% test error and a recurrent cell that reached 62.4 test perplexity on Penn Treebank, outperforming strong hand-designed baselines reported in the paper. But the method is expensive because every sampled child needs its own substantial training run before the controller knows whether it was good. 

---

### 2. The Evolved Transformer

#### High-level goal

Search for a better **feed-forward sequence model** than the original Transformer, directly on a computationally expensive machine-translation task. 

#### Pipeline

1. **Define a Transformer-compatible search space**
   The search space contains two stackable cells, one for the encoder and one for the decoder. Each cell consists of NASNet-style blocks with two branches. The search fields cover branch inputs, normalization, layer type, relative output dimension, activation, combiner function, and number of cells. The space is enormous: about (7.30 \times 10^{115}) models before constraints. 

2. **Warm-start the population with Transformer**
   Instead of starting only from random individuals, the search seeds the initial population with the standard Transformer. 

3. **Use tournament-selection evolution**
   A population of candidate architectures is maintained. The highest-fitness individual in a sampled subpopulation becomes the parent. Its gene encoding is mutated to create a child. The weakest individual in the sampled subpopulation is removed. 

4. **Evaluate fitness on the target task**
   Fitness is based on negative log perplexity on the WMT’14 English-German validation set. This is important: the search is run on the real task, not on a weak proxy benchmark. 

5. **Use Progressive Dynamic Hurdles**
   Because fully training every candidate is too costly, each child is first trained for a small number of steps. Only candidates that clear the current hurdle get extra training. Later hurdles become stricter and grant more additional steps only to stronger models. 

6. **Select the final architecture and fully train it**
   The best architecture found through search becomes the Evolved Transformer. 

#### Why each step exists

| Stage                                   | Purpose                                     | Main benefit                                       | Main trade-off                                  |
| --------------------------------------- | ------------------------------------------- | -------------------------------------------------- | ----------------------------------------------- |
| Search space with encoder/decoder cells | Keep search expressive but Transformer-like | Can represent and improve Transformer-style models | Strong human bias remains in the search space   |
| Warm start with Transformer             | Anchor search at a strong baseline          | Better starting population                         | Less open-ended discovery                       |
| Tournament evolution                    | Explore by mutation and selection           | Simpler and resource-aware                         | Still expensive evaluation                      |
| PDH                                     | Avoid overtraining weak candidates          | Saves compute on large tasks                       | Fitness estimates become staged and approximate |

This summary table is synthesized from the paper’s method section. 

#### Why this paper mattered

The Evolved Transformer is important because it showed NAS could improve a strong, human-designed sequence model rather than only searching small vision cells. The paper reports consistent improvement over Transformer on WMT’14 English-German, English-French, English-Czech, and LM1B; at big size it reports 29.8 BLEU on WMT’14 En-De, and at smaller sizes it matches Transformer quality with 37.6% fewer parameters and beats Transformer by 0.7 BLEU at around 7M parameters. 

---

### 3. AutoFormer

#### High-level goal

Automatically search for strong **vision transformer** architectures while avoiding the cost of training each candidate from scratch. 

#### Pipeline

1. **Show that transformer dimensions matter a lot**
   The paper first demonstrates that choices like depth, embedding dimension, MLP ratio, and number of heads strongly affect ImageNet top-1 accuracy, and that simple manual scaling is not enough. 

2. **Define a transformer search space**
   AutoFormer searches five variable factors: embedding dimension, Q-K-V dimension, number of heads, MLP ratio, and network depth. It partitions the space into three supernets for different model-size ranges: tiny, small, and base. Overall the supernets cover more than (1.7 \times 10^{16}) candidate architectures. 

3. **Encode the space into a supernet**
   Every candidate model is a subnet of a larger supernet. Subnets share weights in their common parts. 

4. **Train the supernet with weight entanglement**
   Each iteration samples a subnet and updates its corresponding weights while freezing the rest. The paper argues this “entangles” shared parts so many subnets become well trained at once. 

5. **Run evolutionary search under resource constraints**
   After supernet training, the method searches for the best subnet under model-size constraints. The objective is to maximize validation accuracy while minimizing model size. 

6. **Select searched models for different scales**
   The resulting AutoFormer-tiny, small, and base models report 74.7%, 81.7%, and 82.4% ImageNet top-1 accuracy with 5.7M, 22.9M, and 53.7M parameters, respectively. ([arXiv][2])

#### What the two-phase design means

AutoFormer’s search pipeline has two phases:

1. **Train once**: make a supernet that contains many candidate transformers
2. **Search cheaply**: evaluate many subnets using inherited weights, then evolve promising candidates under size constraints 

That is the central one-shot NAS idea. The whole point is to avoid retraining every candidate independently. 

#### Why this paper mattered

AutoFormer is important because it brought NAS fully into the vision-transformer era and emphasized that **searching transformer dimensions is itself an architecture problem**. It also shows a more modern NAS mindset: the real method is not just the final evolution loop, but the supernet training scheme that makes the search loop reliable enough to use. 

---

## Paper-by-Paper Explanation

### Paper 1: *Neural Architecture Search with Reinforcement Learning*

**Problem addressed:** Neural architectures worked well, but designing them still required expert knowledge and substantial manual effort. The paper asks whether an RNN can generate architectures automatically and improve them via reinforcement learning. 

**Method used:** A controller RNN samples an architecture token by token, each sampled child model is trained, its validation accuracy becomes the reward, and the controller is updated with REINFORCE plus a moving-average baseline. 

**Main innovation:** Treat architecture design as a sequential decision process with policy-gradient learning. 

**Main findings:** On CIFAR-10 the searched model reports 3.65% test error, and on Penn Treebank the searched recurrent cell reports 62.4 test perplexity, both better than the cited baselines in the paper. 

**Limitations:** Search is costly because each child architecture must be trained to get a reward signal. The method is also highly dependent on the architecture encoding and reward design. The first limitation is directly stated by the paper’s train-the-child loop; the second is a reasoned interpretation of the method. 

**What changed compared with earlier work:** Instead of hand-designing architectures, the paper introduced the now-classic NAS pipeline: sample, train, evaluate, update the search policy. 

### Paper 2: *The Evolved Transformer*

**Problem addressed:** NAS had mostly focused on vision or recurrent sequence models. This paper asks whether NAS can find a better **feed-forward seq2seq** architecture than the Transformer itself. 

**Method used:** It defines a Transformer-compatible search space over encoder/decoder cells, seeds the population with Transformer, uses tournament-selection evolution, and applies PDH to spend more training steps on more promising candidates. 

**Main innovation:** The main innovations are the Transformer-aware search space, warm starting from Transformer, and PDH for searching directly on a costly translation task. 

**Main findings:** The searched architecture improves over Transformer on four sequence tasks, reaches 29.8 BLEU on WMT’14 En-De at big size, and achieves the same quality as big Transformer with 37.6% fewer parameters at smaller size. 

**Limitations:** Search still remains expensive, the search space is heavily structured by human design, and the discovered model is still close enough to Transformer that it should be seen as an evolved variant rather than a wholly alien architecture. The first two are directly supported; the last is a reasoned interpretation of the search-space design. 

**What changed compared with earlier work:** It applied NAS to strong Transformer-like sequence models rather than only to small cells or RNNs, and it made search more practical by changing the evaluation policy. 

### Paper 3: *AutoFormer: Searching Transformers for Visual Recognition*

*(used because the provided title clearly points to AutoFormer, although the URL provided in the prompt points to a different paper)*

**Problem addressed:** Vision transformers are sensitive to design choices like depth, embedding size, head count, and hidden dimensions, but these choices were often made manually. The paper asks how to search transformer architectures efficiently for vision. 

**Method used:** AutoFormer builds a large transformer search space, encodes it into one or more supernets, trains them with weight entanglement, and then runs evolutionary search under model-size constraints to extract strong subnets. 

**Main innovation:** The main innovations are the transformer-specific one-shot search space and the weight-entanglement supernet training strategy. 

**Main findings:** The paper reports AutoFormer-tiny/small/base at 74.7%/81.7%/82.4% ImageNet top-1 with 5.7M/22.9M/53.7M parameters, and argues that many inherited-weight subnets are comparable to retraining from scratch. ([arXiv][2])

**Limitations:** One-shot NAS depends on supernet quality; if inherited weights are a poor proxy, ranking can be misleading. The method also depends on a highly engineered search space and supernet setup. The first point is a method-level interpretation; the second is directly supported by the paper’s design. 

**What changed compared with earlier work:** The method shifts NAS from direct child-model training toward reusable weight-sharing search, and adapts NAS specifically to the design sensitivities of vision transformers. 

---

## Comparison Across Papers or Methods

| Dimension             | RL NAS                                                | The Evolved Transformer                                 | AutoFormer                                                     |
| --------------------- | ----------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------------------- |
| Main search algorithm | Reinforcement learning                                | Evolutionary search                                     | One-shot supernet + evolutionary search                        |
| Candidate evaluation  | Train sampled child model and use validation accuracy | Train/evaluate children with staged resource allocation | Evaluate inherited-weight subnets after supernet training      |
| Main search object    | ConvNets and recurrent cells                          | Transformer-like seq2seq cells                          | Vision transformers                                            |
| Main efficiency idea  | Parallel child training and controller learning       | Progressive Dynamic Hurdles                             | Weight sharing through a supernet                              |
| Biggest strength      | Simple and general formulation                        | Strong Transformer-aware search on real seq2seq task    | Much cheaper large-scale search over many transformer variants |
| Biggest weakness      | Very expensive evaluation                             | Still expensive and search-space-heavy                  | Proxy quality depends on supernet training                     |

This comparison table is synthesized from the three papers’ method sections and abstracts. 

A second useful comparison is how each paper handles **search cost**:

| Paper               | Where most compute goes                        | Main trick to reduce cost                          |
| ------------------- | ---------------------------------------------- | -------------------------------------------------- |
| RL NAS              | Training many child models                     | None fundamentally; search remains direct          |
| Evolved Transformer | Training seq2seq child models on a costly task | Train weak models briefly, promising models longer |
| AutoFormer          | Training the supernet once                     | Reuse inherited weights for many candidates        |

This table captures the main historical shift in NAS. 

---

## Real-World System and Application

In a real ML system, NAS usually fits into a broader workflow:

1. **Choose the target constraint**
   Accuracy alone is rarely enough. You may care about parameter count, memory, or a size budget. The Evolved Transformer explicitly cares about quality at smaller sizes, and AutoFormer searches under model-size constraints. 

2. **Design a search space that reflects real deployment choices**
   AutoFormer’s space includes depth, Q-K-V dimension, head count, MLP ratio, and size ranges because these are practical design knobs for vision transformers. 

3. **Choose how expensive evaluation can be**
   If you can afford near-direct evaluation, black-box NAS is possible. If not, one-shot NAS or staged evaluation becomes more attractive. 

4. **Retrain or validate the winning architecture carefully**
   Even when one-shot search is used, the final chosen model is typically treated as a candidate to validate more seriously. This is a reasoned system interpretation based on the papers’ evaluation logic. 

The papers do **not** provide a full production recipe for a NAS platform. Information not provided includes experiment orchestration systems, distributed search infrastructure, budget governance, or how to integrate NAS with deployment monitoring and rollback. 

---

## Limitations and Trade-offs

The biggest trade-off in NAS is **search quality versus search cost**. RL NAS and the Evolved Transformer use more faithful evaluation of candidates, but pay more computation per candidate. AutoFormer makes search much cheaper, but depends more heavily on the quality of the supernet as a proxy. 

Another major limitation is **search-space bias**. NAS sounds automatic, but the search space still reflects strong human priors. The Evolved Transformer’s search space is explicitly designed so Transformer can be represented exactly, and AutoFormer’s search space focuses on a carefully chosen set of transformer dimensions. 

A third limitation is **objective mismatch**. The architecture you find depends on what you optimize. The Evolved Transformer uses translation fitness on WMT’14 En-De validation, while AutoFormer explicitly balances classification accuracy and model size. A different objective could yield a different “best” model. 

A fourth limitation is **task dependence**. The Evolved Transformer is about sequence-to-sequence language tasks. AutoFormer is about pure vision transformers. Good architectures may not transfer cleanly across domains just because they were discovered by NAS. This is partly directly supported by task-specific setups and partly a reasoned interpretation. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain:

* what NAS is and why it exists
* the difference between a **search space** and a **search strategy**
* why early NAS was so expensive
* how RL-based NAS works in one sentence
* why the Evolved Transformer uses evolution and PDH
* what a supernet is in one-shot NAS
* why weight sharing helps and why it can also be risky 

### Likely interview questions

| Question                                                     | Plain-English answer                                                                                                                                                                               |
| ------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What is NAS?                                                 | NAS is a method for automatically searching over neural-network architectures instead of designing them completely by hand.                                                                        |
| How does the 2016 RL NAS paper work?                         | A controller RNN samples architecture decisions, each sampled child model is trained and scored on validation data, and REINFORCE updates the controller to sample better architectures over time. |
| Why was early NAS expensive?                                 | Because the reward for an architecture usually required building and training that child model before knowing whether it was good.                                                                 |
| Why did the Evolved Transformer use evolution instead of RL? | The paper says evolution is simple and had been shown to be more efficient than reinforcement learning when resources are limited, and it also fit well with warm-starting from Transformer.       |
| What is Progressive Dynamic Hurdles?                         | It is a staged evaluation method that gives more training steps only to candidates that clear performance thresholds, so compute is focused on promising models.                                   |
| What is a supernet in AutoFormer?                            | It is one large network that contains many candidate subnets, letting them share weights so search can evaluate many candidates cheaply.                                                           |
| What is the main risk of one-shot NAS?                       | The inherited performance of a subnet may not perfectly reflect how well that architecture would perform if retrained independently.                                                               |
| How did NAS evolve across these papers?                      | It moved from direct, expensive black-box search to more resource-aware evaluation and then to weight-sharing one-shot search for much cheaper candidate assessment.                               |

This question-and-answer set is a teaching synthesis grounded in the papers. 

### Strong one-minute interview answer

A strong answer would sound like this:

> Neural architecture search automates model design by defining a search space of possible architectures, a method for proposing candidates, and a way to score them. The 2016 RL NAS paper used a controller RNN trained with REINFORCE: it sampled an architecture, trained the child model, and used validation accuracy as reward. That worked, but it was expensive because every candidate needed real training. The Evolved Transformer kept black-box search but switched to evolution, warm-started from the Transformer, and introduced Progressive Dynamic Hurdles so weak candidates were discarded early and strong ones got more training. AutoFormer reflects the next shift: one-shot NAS. It trains a supernet once with weight sharing and then searches many vision-transformer subnets cheaply, which makes NAS much more practical but introduces the risk that inherited-weight evaluation is only a proxy for real performance. 

---

## Glossary

| Term                              | Beginner-friendly definition                                                                            |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Neural Architecture Search (NAS)  | Automatic search for good neural-network architectures.                                                 |
| Search space                      | The set of architectures the search is allowed to consider.                                             |
| Search strategy                   | The algorithm used to propose and refine candidates, such as RL or evolution.                           |
| Controller RNN                    | In RL NAS, the recurrent network that emits architecture decisions token by token.                      |
| Child model                       | One sampled candidate architecture that is built and evaluated during search.                           |
| REINFORCE                         | A policy-gradient algorithm for learning from non-differentiable rewards.                               |
| Reward                            | The score used to tell the search method whether a sampled architecture was good.                       |
| Evolutionary search               | Search that maintains a population, selects strong parents, mutates them, and keeps better descendants. |
| Tournament selection              | A way to choose parents by comparing a small sampled subset of the population.                          |
| Warm start                        | Starting search from a strong known architecture rather than from random architectures only.            |
| Progressive Dynamic Hurdles (PDH) | A staged evaluation method that trains promising candidates longer and weak candidates less.            |
| Supernet                          | A large network that contains many candidate architectures as subnets with shared weights.              |
| One-shot NAS                      | NAS where a single supernet is trained once, then many candidates are evaluated inside it.              |
| Weight sharing                    | Reusing parameters across multiple candidate architectures.                                             |
| Weight entanglement               | AutoFormer’s way of maximizing shared learning across transformer blocks in the supernet.               |
| Q-K-V dimension                   | The size of the query, key, and value projections in attention.                                         |
| MLP ratio                         | The ratio between hidden dimension and embedding dimension in the transformer feed-forward block.       |

This glossary synthesizes terminology used across the three papers. 

---

## Recap

You should now understand the main arc of NAS in these papers:

* **RL NAS** made architecture search a learning problem, but paid a high cost by training many child models directly.
* **The Evolved Transformer** adapted NAS to stronger sequence models and improved practicality with warm starts and staged evaluation.
* **AutoFormer** pushed NAS into the vision-transformer era with supernets, weight sharing, and much cheaper candidate evaluation. 

The most important interview-ready lesson is that NAS is not just “use RL to design networks.” It is a design triangle of:

1. **what architectures are allowed**,
2. **how candidates are proposed**, and
3. **how expensively candidates are evaluated**. 

That last part, evaluation cost, is why the field shifted from direct child training to staged evaluation and then to one-shot supernets. If you can explain that progression clearly, you already understand the most important technical story across these papers. 

---

## Key Citations

[Neural Architecture Search with Reinforcement Learning](https://arxiv.org/pdf/1611.01578)

[The Evolved Transformer](https://arxiv.org/pdf/1901.11117)

[AutoFormer: Searching Transformers for Visual Recognition](https://arxiv.org/pdf/2107.00651)

[Finding Fast Transformers: One-Shot Neural Architecture Search by Component Composition](https://arxiv.org/pdf/2008.06808)

[1]: https://arxiv.org/abs/2008.06808?utm_source=chatgpt.com "[2008.06808] Finding Fast Transformers: One-Shot Neural Architecture ..."
[2]: https://arxiv.org/abs/2107.00651?utm_source=chatgpt.com "AutoFormer: Searching Transformers for Visual Recognition"

---
---
---

