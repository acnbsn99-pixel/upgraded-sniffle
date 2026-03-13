# Speculative Decoding for Faster LLM Inference: From Exact Draft-and-Verify to Medusa Heads

## What This Report Teaches

This report explains three papers about making autoregressive language model decoding faster. They all target the same bottleneck: a large language model normally generates text one token at a time, and each new token requires another expensive forward pass through the model. The papers ask whether we can generate or verify several future tokens per large-model call instead of only one. 

The three papers form a clear progression. The first paper, **Fast Inference from Transformers via Speculative Decoding**, introduces the core exact draft-and-verify idea and proves that the output distribution can stay identical to the target model. The second paper, **Accelerating Large Language Model Decoding with Speculative Sampling**, develops the same core idea independently and focuses more on very large distributed serving with Chinchilla. The third paper, **Medusa**, keeps the same high-level goal but changes the mechanism: instead of using a separate draft model, it adds extra decoding heads to the original model so it can predict several future tokens in parallel. 

By the end, you should understand what speculative decoding is, why it can be exact, what determines whether it speeds things up, why the first two papers are closely related but not identical in emphasis, and why Medusa is both similar to and importantly different from classical speculative decoding. 

---

## Key Takeaways

* **Speculative decoding speeds up generation by proposing several tokens cheaply and then verifying them with the expensive model in parallel.** This matters because the large model is the latency bottleneck. In practice, one expensive target-model pass can sometimes advance generation by several tokens instead of only one. 

* **The first two papers are mainly about exact acceleration, not approximate acceleration.** Their rejection-sampling-style acceptance rule is designed so the final output distribution matches the target model’s distribution, up to implementation numerics. In practice, this is attractive when you want speed without changing model behavior. 

* **Speedup depends on two things: draft quality and draft cost.** If the cheap draft model often proposes tokens the target model would also like, and if the draft model is much cheaper than the target, speculative decoding helps. The first paper formalizes this with acceptance-rate and cost-ratio analysis. In practice, a bad draft model can erase the gains. 

* **The DeepMind paper is not a fundamentally different idea from the first paper; it is an independently developed version focused more on large distributed serving.** It emphasizes Chinchilla, distributed setups, and latency engineering details. In practice, it is especially relevant for very large production-style deployments. 

* **Medusa removes the separate draft model, but it is no longer “free” in the same way.** It adds extra decoding heads and requires fine-tuning, which changes the system design trade-off. In practice, Medusa is easier to integrate when you control the model and can train extra heads, but it is not a drop-in wrapper around an untouched target model like the first two methods. ([arXiv][1])

* **Medusa also introduces an important trade-off between exactness and extra speed.** It can use rejection sampling for output consistency with the original model, but it also proposes a “typical acceptance” scheme that gives more speed while no longer insisting on exact distribution matching. In practice, Medusa lets you choose between stricter fidelity and more aggressive acceleration. ([arXiv][1])

* **All three papers are motivated by the same systems fact: LLM decoding is often memory-bandwidth-bound, not purely compute-bound.** This matters because the accelerator may have spare arithmetic capacity even when decoding is slow. In practice, the goal is to use that idle capacity to verify or score more tokens per step. 

---

## Background and Foundations

### Why autoregressive decoding is slow

A standard autoregressive language model generates text left to right. To produce token number `t+1`, it must first know tokens `1` through `t`. That means generation is inherently sequential: one more token usually means one more forward pass through the full model. Large models make this especially expensive because each step moves a large amount of model state and parameters through memory and often across devices. 

### What “memory-bandwidth-bound” means

A beginner-friendly explanation is this: sometimes the main cost is not the math itself, but moving the model’s weights and caches through memory fast enough. The papers repeatedly argue that large-model decoding often hits this limit. That means the hardware may still have unused arithmetic capability, and speculative methods try to exploit that by doing more useful parallel work per expensive large-model call. 

### Draft model and target model

The first two papers use two models:

* a **draft model**, which is smaller and faster,
* and a **target model**, which is larger, slower, and defines the distribution you actually want to sample from.

The draft proposes likely next tokens. The target then checks those proposals in parallel. 

### Acceptance and verification

The basic speculative pattern is:

1. propose several next tokens cheaply,
2. ask the large model what it thinks about those proposed tokens,
3. keep the proposals that are compatible with exact target-model sampling,
4. and fix the first incompatible spot by resampling correctly.

That is the core idea behind both speculative decoding and speculative sampling. 

### Why this can still be exact

This is the most interview-worthy conceptual point. These methods do **not** simply trust the draft model. They use a carefully designed acceptance-and-resampling rule so that, even though the draft makes guesses, the final accepted output still has the target model’s distribution. In other words, the speedup comes from parallel verification, not from pretending the small model is good enough on its own. 

### How Medusa changes the setup

Medusa keeps the “generate multiple future possibilities, then verify” structure, but it does not use a separate draft model. Instead, it adds extra prediction heads on top of the backbone model’s hidden states so the same model can predict multiple future positions in parallel. Those predictions are organized into candidate continuations and checked with a tree-based attention mechanism. ([arXiv][1])

---

## Big Picture First

A useful mental model is that the three papers improve the same bottleneck in three slightly different ways.

### Paper 1: exact draft-and-verify as a general algorithm

The first paper introduces the general speculative decoding framework. It is model-agnostic in the sense that you do not need to retrain the target model or change its architecture. You wrap it with a faster approximation model and a correctness-preserving acceptance rule. 

### Paper 2: exact draft-and-verify for very large distributed LLM serving

The second paper uses essentially the same core idea, but with stronger emphasis on large-scale serving realities: model parallelism, communication overhead, and careful draft-model design for Chinchilla. The paper explicitly says the work was undertaken concurrently and independently of the first paper, and that the core underlying idea is the same. 

### Paper 3: internal multi-head drafting instead of a separate draft model

Medusa changes the design choice. Rather than maintaining a second smaller model, it adds multiple decoding heads to the original model and fine-tunes them. This makes integration easier in some settings and avoids the operational burden of a separate draft model, but it introduces training requirements that the first two papers did not need. ([arXiv][1])

### One-line summary of the progression

* **Leviathan et al.**: exact speculative decoding without retraining.
* **Chen et al.**: same exact idea, pushed toward large distributed serving.
* **Medusa**: no separate draft model, but extra heads and fine-tuning. 

---

## Core Concepts Explained

### Speculative decoding

Speculative decoding means using a cheap system to guess several future tokens, then using the expensive target model to verify them in parallel. If the cheap guesses are good, one target-model pass can advance the sequence by multiple tokens. The first paper says a single speculative step can generate between 1 and `γ + 1` tokens; the second paper says between 1 and `K + 1` tokens, using slightly different notation for the same basic idea. 

### Rejection sampling, in plain English

Both exact speculative papers use a rejection-sampling-style correction rule. The intuition is:

* if the draft model did not overstate a token relative to the target model, you can safely keep it;
* if the draft model over-favored a token, you sometimes reject it;
* when rejection happens, you resample from the “leftover probability mass” so the final token still follows the target model.

The first paper states this directly with an adjusted distribution proportional to `max(0, p - q)`. The second paper presents the same preservation idea with its own notation and says the target distribution is recovered within hardware numerics. 

### Acceptance rate

The first paper defines an acceptance-rate concept, then uses its expected value `α` to reason about speed. In simple language, `α` measures how often the draft model’s guesses survive verification. High acceptance means the target model frequently agrees with the cheap proposals, which is exactly when speculative decoding becomes valuable. 

### Draft cost ratio

The first paper also formalizes the relative draft-model cost, called `c`. In plain English, this is how expensive the draft is compared with the target. A draft model can have great acceptance but still be a poor choice if it is too slow. The paper’s analysis shows that you need the acceptance behavior to be good enough relative to the draft cost for the method to help. 

### Distribution-preserving versus quality-preserving

This distinction is essential:

* **Distribution-preserving** means outputs are sampled from the same distribution as the target model would have produced.
* **Quality-preserving** means outputs look similarly good, even if the exact probability distribution changes.

The first two papers are strongly about exact distribution preservation. Medusa can operate in that mode with rejection sampling, but its typical-acceptance option is more about maintaining generation quality while accepting that exact distribution matching is not always necessary. 

### Multiple decoding heads

In Medusa, a **decoding head** is an extra prediction head attached to the backbone model’s hidden states. Each head predicts a different future position. This lets the model propose several possible future tokens in parallel without running a separate draft model. ([arXiv][1])

### Tree attention

Medusa’s **tree-based attention** is the mechanism that lets many candidate continuations be processed in parallel. Instead of verifying one linear speculative path, Medusa can assemble multiple candidate branches and evaluate them together. This is one of the main reasons Medusa is not just “speculative decoding with a different name.” ([arXiv][1])

### Typical acceptance

Medusa argues that exact rejection sampling can reduce efficiency, especially with sampling and temperature. So it proposes **typical acceptance**, which keeps candidates that look plausible under the original model rather than insisting on exact distribution matching. That trades exactness for more aggressive acceleration while aiming to keep output quality similar. ([arXiv][1])

---

## Step-by-Step Technical Walkthrough

## 1. Fast Inference from Transformers via Speculative Decoding

### Goal

Speed up a large autoregressive model without changing the target model’s architecture, training procedure, or output distribution. 

### Workflow

1. **Choose two models**
   Use a large target model `Mp` and a smaller approximation model `Mq`. The target is the model you really want to sample from. The approximation model only proposes guesses. 

2. **Draft several guesses autoregressively with the small model**
   The approximation model generates `γ` candidate next tokens one after another. 

3. **Run the large model in parallel on the current prefix and the speculative prefixes**
   The target model evaluates the base prefix and each longer prefix that includes the guessed tokens. This parallel target-model pass is what lets a single expensive call potentially validate several future tokens. 

4. **Accept draft tokens from left to right until the first mismatch event**
   If a guessed token is compatible under the acceptance test, it is kept. Once a token fails, the algorithm stops accepting further speculative guesses for that step. 

5. **Resample correctly at the first rejected position**
   If rejection occurs, the algorithm samples from the adjusted residual distribution based on the target and draft probabilities. This is the correction step that preserves the target distribution. 

6. **Return at least one token, and sometimes many**
   The paper states that each target-model pass produces at least one new token and potentially as many as `γ + 1` tokens. 

### Why each step exists

The cheap draft model is there to guess easy tokens. The large model is there to keep the process exact. The correction rule is there so the draft never biases the final sample distribution. The overall point is to replace many serial target-model steps with fewer serial target-model steps, each doing more useful work. 

### Practical meaning of the formula

The paper’s acceptance formula is saying: “Keep a draft token when it is not overrepresented relative to the target, and if it is overrepresented, reject it with exactly the right probability so the total probability mass still comes out correct.” The residual distribution `max(0, p - q)` is just the leftover portion the target wanted that the draft did not already cover. 

### Trade-offs

This method is exact and requires no retraining, which is a major advantage. But it still needs a suitable draft model, and the gains depend on acceptance rate and systems conditions such as memory bandwidth. The paper’s theory and experiments both show that a good balance between draft cost and acceptance is crucial. 

---

## 2. Accelerating Large Language Model Decoding with Speculative Sampling

### Goal

Accelerate large transformer decoding in latency-critical settings, especially distributed serving, while preserving the target distribution within hardware numerics and avoiding target-model modification. 

### Workflow

1. **Generate a short draft of length `K`**
   A faster draft model proposes a short continuation. The paper emphasizes the draft model can be another autoregressive model or another form of fast proposer, but it focuses on the autoregressive draft case. 

2. **Score the draft with the target model**
   The large target model evaluates the drafted continuation. 

3. **Accept a left-to-right prefix of the draft**
   Using modified rejection sampling, accept as many tokens as possible from the draft prefix. 

4. **Correct the first rejected token**
   If a proposed token is rejected, resample from the appropriate corrected distribution so the target distribution is recovered. The paper states the target distribution is recovered within hardware numerics. 

5. **Possibly generate `K + 1` tokens in one loop**
   If all drafted tokens are accepted, the final target logits can also be used to sample one additional token, so one speculative loop can advance by up to `K + 1` tokens. 

### Why this paper matters separately

Conceptually, this is very close to the first paper. What makes it distinct is emphasis and engineering context. The paper explicitly says the work was done concurrently and independently, and that its main difference is heavier focus on distributed serving and some incremental optimizations. It also spends more attention on how draft-model design interacts with large-model serving topologies. 

### Practical meaning of the method

This paper is very useful for understanding why “just use a tiny draft model” is not always enough. It points out that for distributed systems, a naive small model can be suboptimal because model shape and hardware topology matter. Their draft model for Chinchilla is not just “small”; it is deliberately designed for sampling latency, with 4B parameters and only 8 layers, and is served on the same TPU-v4 hardware family. 

### Trade-offs

The strengths are exactness, large-model relevance, and strong systems realism. The limitation is the same one faced by speculative decoding generally: you still need a good draft model and careful serving design. The benefit is large when acceptance is high enough to offset drafting overhead. 

---

## 3. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

### Goal

Reduce autoregressive decoding steps by having the model itself propose multiple future tokens in parallel, avoiding the need for a separate draft model. ([arXiv][1])

### Workflow

1. **Add multiple decoding heads on top of the backbone model**
   These heads predict several subsequent token positions from the last hidden states. The heads are parameter-efficient and can be added to an existing model. ([arXiv][1])

2. **Generate multiple top predictions from each head**
   Instead of one single proposed continuation, each head contributes likely candidates for its assigned future position. ([arXiv][1])

3. **Assemble candidates into a tree**
   Medusa combines these predictions into multiple candidate continuations. ([arXiv][1])

4. **Process candidates with tree-based attention**
   The backbone model verifies many candidates in parallel through a modified attention structure. ([arXiv][1])

5. **Accept the longest good candidate prefix**
   The system selects the longest accepted candidate prefix to continue generation. ([arXiv][1])

6. **Choose exact or approximate acceptance**
   Medusa can use rejection sampling for original-distribution consistency, or “typical acceptance” for more speed when exact matching is not necessary. ([arXiv][1])

### Training variants

Medusa introduces two training modes:

* **MEDUSA-1**: fine-tune only the extra heads on a frozen backbone. The paper presents this as enabling lossless inference acceleration.
* **MEDUSA-2**: jointly train the heads with the backbone using a special recipe designed to preserve model capability while improving head accuracy and speedup. ([arXiv][1])

It also proposes **self-distillation** so the extra heads can still be trained when the original supervised fine-tuning dataset is unavailable, including RLHF-style cases. ([arXiv][1])

### Why this is different from classical speculative decoding

The first two papers are “wrapper” methods: add a draft model around an unchanged target model. Medusa is more like “change the model slightly so it can speculate internally.” That changes deployment trade-offs:

* no separate draft model to serve,
* easier integration into some existing stacks,
* but additional training and model modification are required. ([arXiv][1])

### Typical acceptance, in plain English

Medusa argues that exact rejection sampling can reject good candidate prefixes too often during non-greedy sampling, which wastes potential speed. So it uses a plausibility-style rule: keep candidates that are typical enough under the original model. This is faster, but it no longer means strict distribution identity. ([arXiv][1])

### Trade-offs

Medusa can be faster and operationally simpler than serving a second draft model, especially if you own the model and can fine-tune it. But it is not a zero-training technique, and some of its best speedups rely on non-exact acceptance choices. The paper also focuses mainly on batch size 1, and later simulation results indicate gains can decrease as batch size grows too large and decoding becomes more compute-bound. ([arXiv][1])

---

## Paper-by-Paper Explanation

## 1. Fast Inference from Transformers via Speculative Decoding

### Problem addressed

Large autoregressive transformer decoding is slow because generating `K` tokens normally needs `K` serial target-model calls. The paper asks whether speculative execution, familiar from processors, can be generalized to stochastic language-model sampling. 

### Method used

Use a smaller approximation model to draft `γ` tokens, run the target model in parallel over the corresponding speculative prefixes, accept a left-to-right prefix of those guesses, and correct the first rejected location with an adjusted sampling distribution so the final output distribution remains unchanged. 

### Main innovation

The paper’s main innovation is not just “use a small model first.” It is the **exact** draft-and-verify acceptance rule plus the systems framing that this can speed up decoding without architecture changes, retraining, or changed outputs. 

### Main findings

The paper reports out-of-the-box **2X-3X** latency improvement versus the T5X baseline for T5-XXL with no change to outputs. On T5-XXL, it reports empirical speedups such as **2.6X** and **3.4X** on translation and **2.3X** and **3.1X** on summarization under different sampling temperatures. 

### Limitations

The method depends on having a good approximation model and enough unused compute to exploit parallel verification. The paper’s own theory makes clear that acceptance rate must be strong enough relative to the draft-model cost for speedup to appear. 

### What changed compared with earlier work

Compared with earlier faster-decoding methods that often required retraining, changed outputs, or greedy-only behavior, this paper’s contribution is exact stochastic acceleration for existing autoregressive transformers. 

### Reasoned interpretation

This paper is best understood as the “foundational exact algorithm” paper for modern speculative decoding. The most important thing it contributes is the guarantee that speedup does not have to mean distribution drift. 

### Information not provided

The paper does not provide a universal recipe for choosing the best draft model in every deployment environment. It gives theory and examples, but draft-model design remains context-dependent. 

---

## 2. Accelerating Large Language Model Decoding with Speculative Sampling

### Problem addressed

Very large language models are hard to serve because decoding is memory-bandwidth-bound and often requires model parallelism, which adds communication overhead. The paper asks whether speculative sampling can reduce latency in this large distributed regime. 

### Method used

Generate a short draft with a faster model, score it with the target model, and use modified rejection sampling to accept a left-to-right subset while preserving the target distribution within hardware numerics. 

### Main innovation

The core idea is the same as the first paper, but the distinctive contribution is stronger engineering focus on large distributed serving, especially with Chinchilla, and on the interaction between draft-model design and deployment topology. 

### Main findings

The paper benchmarks Chinchilla 70B and reports **2-2.5×** decoding speedups without compromising sample quality or changing the model. On XSum and HumanEval, its table shows comparable benchmark performance while reducing mean token time from **14.1 ms/token** to **7.52 ms/token**, **7.00 ms/token**, or **5.73 ms/token** depending on setup, with HumanEval reaching **2.46×** speedup. 

### Limitations

This method still needs a suitable draft model and careful deployment tuning. The paper explicitly shows that naive draft-model choices can underuse hardware or suffer extra communication overhead in distributed setups. 

### What changed compared with earlier work

Compared with the first paper, this work is less about introducing the core idea and more about showing that the idea scales to very large distributed LLM serving. 

### Reasoned interpretation

This paper is best viewed as the “serving-oriented speculative decoding” paper. Interview-wise, it is the one to cite when discussing speculative decoding as a systems technique for huge model deployments. 

### Information not provided

The paper does not claim a fundamentally new distribution-preserving principle beyond the same core speculative idea. It mainly adapts, validates, and engineers it for a different operating regime. 

---

## 3. Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads

### Problem addressed

Classical speculative decoding is powerful, but maintaining a separate draft model is operationally awkward and can be hard to integrate into distributed systems. Medusa asks whether the model can speculate internally instead. ([arXiv][1])

### Method used

Add multiple decoding heads to the model, have them predict future tokens in parallel, organize those predictions into candidates, verify them with tree-based attention, and accept the longest suitable prefix using either exact rejection sampling or faster typical acceptance. ([arXiv][1])

### Main innovation

The core innovation is replacing the external draft model with **internal speculative heads** plus tree-based verification. This makes Medusa operationally simpler in some settings and opens a new design space between exact wrapper methods and model modification. ([arXiv][1])

### Main findings

The paper reports that **MEDUSA-1** can achieve over **2.2×** speedup without compromising generation quality, while **MEDUSA-2** improves this to roughly **2.3-2.8×** overall. For Vicuna-7B specifically, Table 2 reports quality scores of **6.23** for MEDUSA-1 and **6.18** for MEDUSA-2 versus **6.17** baseline, with speedups of **2.18×** and **2.83×** respectively. It also reports component-level gains from adding tree attention and optimized tree configuration. ([arXiv][1])

### Limitations

Medusa requires fine-tuning and backbone access, unlike the first two papers. The paper also mainly studies batch size 1 and later shows via simulation that gains can shrink at larger batch sizes as the system becomes compute-bound rather than memory-bandwidth-bound. ([arXiv][1])

### What changed compared with earlier work

Medusa shifts speculative acceleration from “serve a second model” to “modify the first model so it can propose multiple continuations itself.” That is a substantial architectural and deployment shift. ([arXiv][1])

### Reasoned interpretation

Medusa is best understood as a practical compromise: it gives up the elegance of “no retraining, no architecture change” in return for easier integration and often strong speed. It is especially appealing when you control the model weights and deployment stack. ([arXiv][1])

### Information not provided

The paper does not show that Medusa universally beats classical speculative decoding in every serving environment. It argues for easier integration and strong speedups, but the right choice still depends on whether you can fine-tune and how much exact output fidelity you require. ([arXiv][1])

---

## Comparison Across Papers or Methods

The table below summarizes the conceptual differences. It synthesizes the three papers’ designs and reported claims. 

| Aspect                       | Fast Inference via Speculative Decoding    | Accelerating LLM Decoding with Speculative Sampling                      | Medusa                                                               |
| ---------------------------- | ------------------------------------------ | ------------------------------------------------------------------------ | -------------------------------------------------------------------- |
| Main idea                    | Separate small draft model proposes tokens | Same core draft-and-verify idea, optimized for large distributed serving | Extra heads on the same model predict multiple future tokens         |
| Need a separate draft model? | Yes                                        | Yes                                                                      | No                                                                   |
| Need retraining?             | No target-model retraining required        | No target-model retraining required                                      | Yes, extra heads must be trained                                     |
| Exact output distribution?   | Yes                                        | Yes, within hardware numerics                                            | Yes with rejection sampling; not necessarily with typical acceptance |
| Main systems emphasis        | General exact algorithm                    | Large distributed serving, Chinchilla                                    | Easy integration, no draft model, tree verification                  |
| Reported speedups            | About 2X-3X on T5-XXL                      | About 2-2.5X on Chinchilla tasks                                         | About 2.2X to 2.8X depending on setup                                |

A second comparison is the interview-critical one: what changes in the inference stack. 

| Question                              | Paper 1                      | Paper 2                                                | Paper 3                               |
| ------------------------------------- | ---------------------------- | ------------------------------------------------------ | ------------------------------------- |
| Is the target model unchanged?        | Yes                          | Yes                                                    | No, extra heads are added and trained |
| Is the method exact by design?        | Yes                          | Yes                                                    | Sometimes                             |
| Where does the speculation come from? | Separate approximation model | Separate draft model                                   | Internal decoding heads               |
| Main trade-off                        | Need a good draft model      | Need a good draft model and distributed serving tuning | Need training and model access        |

---

## Real-World System and Application

A practical LLM serving system can think about these papers as three deployment options rather than one winner. This section includes some direct facts from the papers and some clearly marked interpretation. 

### Directly supported by the papers

If you need **strict output parity** with an existing model and cannot retrain or modify it, the first two speculative papers are the natural fit. Both are explicitly designed to leave the target model untouched while preserving its distribution, and both report speedups around the 2X range or higher in their tested settings. 

If you control the model weights and can fine-tune light additional components, Medusa offers another path. It avoids the operational complexity of a separate draft model and is positioned as easy to integrate into existing LLM systems, including distributed environments. ([arXiv][1])

### Reasoned interpretation

In a real production stack, the choice likely depends on organizational constraints more than on speed numbers alone:

* choose **exact speculative decoding/sampling** when behavioral equivalence matters,
* choose **Medusa** when serving simplicity and owning the model matter more,
* and choose **approximate Medusa acceptance** only when you can tolerate some distribution drift in exchange for lower latency.

That interpretation follows directly from the papers’ stated design goals and trade-offs, but the papers do not provide a single universal deployment rule. 

### Information not provided

The papers do not present one end-to-end production architecture that jointly optimizes speculative acceleration, batching, quantization, KV-cache engineering, and tool or RAG serving. They focus specifically on speculative acceleration mechanisms. 

---

## Limitations and Trade-offs

### Exactness versus speed

The first two papers strongly optimize for exactness. Medusa exposes the trade-off more explicitly: rejection sampling can preserve the original distribution, but typical acceptance aims for practical speed and quality instead of exact matching. 

### No retraining versus easier integration

Classical speculative decoding is attractive because it does not require retraining the target model, but it needs a separate draft model that must be trained, served, and maintained. Medusa removes that operational burden at the price of modifying and training the target model. 

### Draft quality versus draft cost

A speculative method only helps when the cheap proposer is both cheap enough and good enough. This is the central speed condition in the first paper’s analysis and a major engineering concern in the second paper’s distributed Chinchilla setup. 

### Batch-size sensitivity

Medusa’s reported experiments focus on batch size 1, and its simulations show gains can decline when batch size becomes too large because the workload shifts away from the memory-bandwidth-bound regime that made speculation attractive in the first place. That is a useful reminder that inference acceleration methods are hardware-regime-dependent. ([arXiv][1])

### Extra-system complexity

All three methods add some complexity:

* draft-model management for the first two,
* extra heads and training for Medusa,
* acceptance logic and verification for all of them.

So these papers are not “free speed.” They are better understood as latency-for-complexity trade systems. That conclusion is a reasoned synthesis of the papers’ designs and engineering discussion. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain:

1. why autoregressive decoding is slow,
2. what it means for decoding to be memory-bandwidth-bound,
3. how speculative decoding uses a cheap draft plus expensive verification,
4. why the first two papers can be exact,
5. what determines whether speculative decoding actually speeds things up,
6. how Medusa avoids a separate draft model,
7. and why Medusa trades away some of the simplicity of “no retraining.” 

### Likely interview questions

**What problem does speculative decoding solve?**
It reduces the number of serial large-model decoding steps by letting one target-model pass validate several future tokens instead of only one. 

**Why can speculative decoding be exact even though it uses a small draft model?**
Because the draft model only proposes candidates. A rejection-and-resampling rule corrects any overconfident draft suggestions so the final accepted tokens still follow the target model’s distribution. 

**What determines the speedup?**
Mainly how often draft tokens are accepted and how cheap the draft process is compared with the target model. High acceptance and low draft cost are the winning combination. 

**How are the first two papers different?**
They are very close in core idea. The second paper says it was developed concurrently and independently, and it emphasizes distributed serving and Chinchilla-scale deployment more heavily. 

**What is Medusa in one sentence?**
Medusa adds multiple decoding heads to the backbone model so it can predict and verify several future tokens internally, instead of relying on a separate draft model. ([arXiv][1])

**Why isn’t Medusa just the same as speculative decoding?**
Because classical speculative decoding wraps an unchanged target model with a separate draft model, while Medusa modifies the target model by adding trainable heads and uses tree-based attention to verify multiple candidate continuations. 

**What is the trade-off in Medusa’s typical acceptance?**
It can improve speed by accepting plausible candidates instead of insisting on exact rejection-sampling behavior, but that means it is no longer strictly matching the original model’s distribution. ([arXiv][1])

**Why does batch size matter?**
These methods are most useful when decoding is memory-bandwidth-bound. If large batch sizes push the system into a more compute-bound regime, speedups can shrink. ([arXiv][1])

### Concise model answers

| Question                          | Good plain-English answer                                                                   |
| --------------------------------- | ------------------------------------------------------------------------------------------- |
| Why is normal LLM decoding slow?  | Because each new token usually requires another full forward pass through a huge model.     |
| What is the key speculative idea? | Guess several next tokens cheaply, then verify them with the expensive model in parallel.   |
| Why does it help?                 | One expensive call can sometimes advance the sequence by several tokens.                    |
| What is the main exactness trick? | Reject and correct draft proposals in a way that preserves the target model’s distribution. |
| What is Medusa’s main twist?      | Internal speculative heads instead of a separate draft model.                               |

This table is a teaching-oriented synthesis of the three papers. 

---

## Glossary

* **Acceptance rate:** How often draft proposals are kept after target-model verification. High acceptance usually means better speculative speedup. 
* **Autoregressive decoding:** Generating one token at a time, where each next token depends on all previous ones. 
* **Draft model:** A faster, usually smaller model that proposes candidate next tokens for speculative decoding. 
* **Exact distribution preservation:** The guarantee that the accelerated method samples from the same distribution as the original target model. 
* **Memory-bandwidth-bound:** A regime where moving data through memory is the bottleneck more than raw arithmetic throughput. 
* **MEDUSA-1:** Medusa variant where extra heads are trained on top of a frozen backbone. ([arXiv][1])
* **MEDUSA-2:** Medusa variant where heads and backbone are trained together with a special recipe to preserve model quality while improving speed. ([arXiv][1])
* **Rejection sampling:** A sampling method that accepts or rejects proposals according to a rule designed to preserve a desired distribution. In the first two papers, this is the key exactness mechanism. 
* **Speculative decoding / speculative sampling:** Acceleration methods that use cheap proposals plus expensive verification to generate multiple tokens per target-model call. 
* **Target model:** The large model whose behavior you actually want to preserve and accelerate. 
* **Tree attention:** Medusa’s attention-mask mechanism for processing multiple candidate continuations in parallel. ([arXiv][1])
* **Typical acceptance:** Medusa’s approximate acceptance rule that favors plausible candidates for more speed, without insisting on exact output-distribution matching. ([arXiv][1])

---

## Recap

These three papers are all about the same systems insight: a large language model often wastes latency generating only one token per expensive decoding step, even though hardware may have enough parallel capacity to verify more than one future token at once. Speculative methods try to exploit that gap. 

The first paper establishes the core exact speculative decoding algorithm. The second paper validates the same core idea in large distributed LLM serving and shows strong Chinchilla results. Medusa changes the design by moving speculation inside the model with multiple decoding heads and tree attention. 

The most important interview-level lesson is this:

**Speculative decoding is not mainly about approximating the large model more cheaply. It is about using cheap proposals plus exact or near-exact verification so the large model does fewer serial steps.** 

What remains limited or uncertain is also important. None of these papers says there is one universally best speculative method. The right choice depends on whether you need exact output parity, whether you can retrain or modify the model, what your hardware regime looks like, and whether managing a separate draft model is acceptable. 

---

## Key Citations

[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/pdf/2211.17192)

[Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/pdf/2302.01318)

[MEDUSA: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/pdf/2401.10774)

[1]: https://arxiv.org/pdf/2401.10774 "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"


---
---
---

