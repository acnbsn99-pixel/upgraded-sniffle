# Efficient Attention and Memory for Transformers: FlashAttention, PagedAttention/vLLM, and Multi-Query Attention

## What This Report Teaches

This report explains three important ideas for making Transformer-based systems faster and more memory-efficient:

1. **FlashAttention**: an exact attention algorithm that is redesigned around GPU memory movement, not just arithmetic count.
2. **PagedAttention / vLLM**: a serving-time memory manager for large language models (LLMs) that treats the key-value cache like paged virtual memory.
3. **Multi-Query Attention (MQA)**: an architectural change that keeps separate query heads but shares keys and values across heads to reduce decoding-time memory bandwidth. 

Together, these papers show that Transformer efficiency is not one problem. It is at least three different problems: the cost of the attention kernel itself, the cost of storing and moving KV cache during generation, and the cost of organizing memory across many concurrent requests in a production serving system. By the end, you should understand how these optimizations differ, how they connect, and how to explain them clearly in an AI engineer or AI architect interview. 

---

## Key Takeaways

* **FlashAttention is about memory traffic, not approximate attention.** It keeps attention exact, but avoids writing the large (N \times N) attention matrix to slow GPU memory by using tiling and fused computation. This matters because wall-clock speed is often limited by memory movement, not just floating-point operations. In practice, it makes exact attention faster and more memory-efficient without changing the model’s semantics. 

* **Multi-Query Attention attacks the decoding bottleneck at the model-architecture level.** Standard multi-head attention uses separate keys and values for every head; MQA shares one set of keys and values across heads. This matters because autoregressive decoding repeatedly reloads KV tensors, and that process is often memory-bandwidth-bound. In practice, MQA can make decoding much faster with only minor quality degradation. 

* **PagedAttention attacks memory waste in serving systems, not just within one attention call.** It stores KV cache in fixed-size blocks that do not need to be contiguous in memory. This matters because real serving workloads have variable-length prompts and outputs, which create fragmentation and wasted space. In practice, vLLM uses this idea to fit more requests into memory and increase throughput. ([arXiv][1])

* **These three papers solve different layers of the stack.** FlashAttention is mainly a kernel-level optimization, MQA is mainly a model-design optimization, and PagedAttention is mainly a runtime/serving optimization. This matters because people often treat “LLM efficiency” as one topic. In practice, strong systems usually need improvements at several layers at once. 

* **Training and serving have different bottlenecks.** FlashAttention helps training and long-sequence attention computation; MQA mostly helps incremental decoding; PagedAttention helps high-throughput serving with many requests. This matters because the best optimization depends on where the system is slow. In practice, a method that is excellent for training may not be the main fix for production serving, and vice versa. 

* **Memory efficiency is often more valuable than reducing nominal compute.** FlashAttention explicitly argues for IO-awareness, and MQA argues that decoding is limited by memory bandwidth. PagedAttention shows that memory fragmentation can cap throughput even when compute is available. In practice, “fewer FLOPs” does not automatically mean “faster system.” 

* **There are trade-offs.** FlashAttention still requires careful GPU-kernel engineering; MQA may slightly hurt model quality; PagedAttention introduces block-size and preemption trade-offs. In practice, optimization usually means choosing what cost to reduce and what complexity to accept. 

---

## Background and Foundations

### Why attention becomes expensive

A Transformer uses **attention** to decide which earlier tokens matter for the current token. In plain English, each token forms a **query** (“what am I looking for?”), compares itself against **keys** (“what information is available?”), and then uses those comparison scores to combine **values** (“what content should I read?”). The standard compact formula is:

[
\text{Attention}(Q,K,V) = \text{softmax}(QK^\top)V
]

What this means in practice:

1. Compute similarity scores between queries and keys.
2. Turn those scores into probabilities with softmax.
3. Use those probabilities to mix the values into the output. 

The expensive part is that for a sequence of length (N), the score matrix can be (N \times N). That becomes costly in both time and memory as sequence length grows. FlashAttention focuses directly on this problem. 

### Why decoding is a different problem from training

During **training**, many token positions can often be processed in parallel. During **autoregressive decoding**, the model generates one token at a time, and each new token depends on the previously generated tokens. That means you keep reusing and extending the stored keys and values from earlier steps. Those stored tensors are often called the **KV cache**. MQA and PagedAttention both focus on this generation-time setting. 

### Why hardware details matter

The FlashAttention paper emphasizes that modern GPUs have a **memory hierarchy**:

* **SRAM**: very fast, very small, on-chip memory.
* **HBM**: slower than SRAM, but much larger GPU memory.
* **CPU DRAM**: even larger and much slower from the GPU’s point of view.

Its central argument is that many attention implementations are bottlenecked by moving data between HBM and SRAM, not by pure arithmetic. This is why the paper uses the term **IO-aware**: it optimizes reads and writes across memory levels. 

### Why serving many requests creates a separate systems problem

The PagedAttention/vLLM paper studies online LLM serving, where many requests with different prompt and output lengths share GPU memory. The paper argues that the KV cache is large, dynamic, and hard to manage efficiently if every request requires one contiguous memory region. This creates **internal fragmentation** and **external fragmentation**, both of which waste memory and reduce throughput. ([arXiv][1])

---

## Big Picture First

A good mental model is that these papers optimize three different layers of the Transformer stack. This layered view is a reasoned interpretation based on the papers’ stated goals and methods. 

| Layer                    | Paper                 | Main Question                                                   | Main Bottleneck                                        |
| ------------------------ | --------------------- | --------------------------------------------------------------- | ------------------------------------------------------ |
| Model architecture       | Multi-Query Attention | How can we reduce decoding-time KV bandwidth?                   | Repeated loading of large per-head K/V tensors         |
| Kernel / operator        | FlashAttention        | How can we compute exact attention with less memory traffic?    | HBM reads/writes for the attention matrix              |
| Runtime / serving system | PagedAttention / vLLM | How can we serve many requests without wasting KV-cache memory? | Fragmentation, duplication, and poor memory allocation |

The most important insight is that these are **complementary**, not interchangeable. If your system is slow because one attention call is wasteful, FlashAttention helps. If your model architecture creates large per-head KV cache during decoding, MQA helps. If your serving engine wastes cache memory across many requests, PagedAttention helps. 

---

## Core Concepts Explained

### Attention, heads, and KV cache

**Multi-head attention** means the model uses several attention heads in parallel. Each head can learn a different pattern of dependency. In standard multi-head attention, each head has its own query, key, and value projections. That increases expressive power, but it also increases the size of stored keys and values during decoding. 

The **KV cache** is the stored set of keys and values for previous tokens, so the model does not recompute them from scratch at every generation step. This makes decoding possible at practical speed, but it creates a large, growing memory footprint. PagedAttention/vLLM shows that for an OPT-13B model, a single token’s KV cache can require about 800 KB, which is why cache management becomes central to serving throughput. ([arXiv][1])

### IO-awareness

**IO** means input/output in the systems sense: reading from and writing to memory. FlashAttention argues that attention performance is often limited by data movement between slow HBM and fast on-chip SRAM. A method can look efficient on paper in FLOPs yet still be slow if it keeps moving large intermediate tensors through HBM. 

Why this matters: many ML explanations focus on arithmetic complexity, but real GPU performance often depends on memory traffic. FlashAttention’s contribution is to redesign exact attention around this hardware reality. 

### Tiling

**Tiling** means splitting a large computation into smaller blocks that fit in fast memory. FlashAttention loads blocks of (K) and (V) into SRAM, then iterates over blocks of (Q), computing partial attention results on-chip instead of materializing the full attention matrix in HBM. 

Why it exists: the full attention matrix is very large. If you avoid storing it in HBM, you reduce slow memory traffic dramatically. Why it matters: FlashAttention reports that this leads to major practical speedups even though it may do some extra recomputation. 

### Recomputation

**Recomputation** means deliberately recomputing certain intermediate values later instead of storing them now. FlashAttention stores only the output and softmax normalization statistics needed to reconstruct the required pieces in the backward pass. The paper argues that this is faster than storing and reloading the full attention matrix from HBM. 

Why it exists: storing every intermediate can consume too much memory and IO bandwidth. Why it matters: this is a good example of a broader systems principle—sometimes doing a bit more compute is faster than doing more memory traffic. 

### Multi-Query Attention

MQA keeps multiple **query heads** but shares **one set of keys and values** across heads. So the model still lets different heads ask different questions, but those heads all read from the same stored memory representation. 

Why it exists: in incremental decoding, the large K/V tensors dominate memory-bandwidth cost. Sharing K/V reduces that cost. The paper’s analysis says this reduces the “offensive” memory-access term by a factor of the number of heads (h). 

Why it matters: this is not just a kernel trick. It changes the model architecture itself in a way that especially helps autoregressive generation. 

### Paging, logical blocks, and physical blocks

PagedAttention divides the KV cache into fixed-size **blocks**. Each request sees a logical sequence of blocks, but those blocks do not need to live next to each other in physical GPU memory. A block table maps logical blocks to physical blocks, just as an operating system maps virtual pages to physical memory. ([arXiv][1])

Why it exists: requiring contiguity causes fragmentation and wasted memory. Why it matters: non-contiguous paged storage lets the system grow KV cache dynamically, reduce waste, and share blocks across requests or outputs when appropriate. ([arXiv][1])

### Copy-on-write

When multiple outputs share the same prompt prefix, they can initially share the same physical KV blocks. If one branch needs to modify a shared block, vLLM uses **copy-on-write**: only then does it create a new block. ([arXiv][1])

Why it exists: beam search and parallel sampling often share a long common prefix. Why it matters: sharing common prefix blocks saves memory and improves throughput without changing model outputs. ([arXiv][1])

---

## Step-by-Step Technical Walkthrough

### 1. Standard attention, step by step

1. Project input hidden states into queries, keys, and values.
2. Compute query-key similarity scores.
3. Apply softmax to turn scores into weights.
4. Use those weights to combine values into outputs.
5. In standard implementations, materialize large intermediate tensors, including the attention matrix. 

**Purpose:** let each token read information from other tokens.

**Trade-off:** expressive and powerful, but expensive in sequence length and memory movement. ([arXiv][2])

### 2. FlashAttention, step by step

1. **Input:** (Q), (K), and (V) are stored in HBM.
2. **Block the inputs:** split (Q), (K), and (V) into tiles.
3. **Load one (K/V) tile into SRAM:** keep it in fast on-chip memory.
4. **Loop over (Q) tiles:** compute partial attention scores and partial outputs against the resident (K/V) tile.
5. **Maintain running softmax statistics:** instead of needing the whole score matrix at once, combine partial results in a numerically stable way.
6. **Write only the final output blocks to HBM:** do not materialize the full (N \times N) attention matrix in HBM.
7. **Backward pass:** recompute needed attention pieces on-chip using saved normalization statistics rather than reading a stored attention matrix. 

**Purpose of each step:** keep as much work as possible in fast SRAM and avoid HBM traffic for huge intermediates.

**Output:** the same exact attention result as standard attention.

**Trade-offs:** more implementation complexity and some recomputation, but much less HBM traffic and much lower memory use. The paper reports up to a 7.6× speedup on the attention computation relative to a PyTorch implementation, linear memory in sequence length, and many fewer HBM accesses than standard attention. 

### 3. Multi-Query Attention, step by step

1. **Input:** the decoder has a current query and a growing cache of previous tokens’ keys and values.
2. **Keep multiple query heads:** each head still computes its own query representation.
3. **Share one set of keys and values across heads:** instead of storing separate K/V per head.
4. **At each decoding step:** all heads attend using their own queries, but they read the same K/V cache.
5. **Produce the output:** combine per-head outputs as usual. 

**Purpose:** reduce decoding-time memory bandwidth by shrinking the K/V tensors that must be reloaded repeatedly.

**Output:** an architectural variant of attention with similar behavior but different parameter sharing.

**Trade-offs:** much faster decoding, little change in training speed, and a small quality trade-off. On the WMT14 EN-DE setup in the paper, training cost stayed nearly the same, but decoder inference dropped from 46 microseconds per token to 3.8 microseconds per token, while dev BLEU changed only slightly from 26.7 to 26.5 and beam-4 test BLEU was 28.4 for the baseline versus 28.5 for MQA. 

### 4. PagedAttention / vLLM, step by step

1. **Input:** multiple requests arrive, each with different prompt lengths and future output lengths.
2. **Represent each request’s KV cache as logical blocks:** each block holds a fixed number of tokens.
3. **Map logical blocks to physical GPU blocks:** blocks do not need to be contiguous.
4. **Grow the cache on demand:** allocate new blocks only when new tokens appear.
5. **Share blocks where possible:** for parallel sampling, beam search, or shared prefixes.
6. **Use copy-on-write when shared blocks diverge:** only duplicate blocks that must change.
7. **Preempt when necessary:** either swap blocks to CPU RAM or recompute KV cache later when the request resumes. ([arXiv][1])

**Purpose:** reduce memory waste, fit more active requests, and improve serving throughput.

**Output:** more efficient online LLM serving with the same model behavior.

**Trade-offs:** the system introduces block-size tuning and preemption-policy choices. The paper reports that block size 16 works well in practice, and that recomputation is more efficient for small block sizes while swapping is more efficient for large block sizes; for medium block sizes 16 to 64, the two methods are comparable. ([arXiv][1])

---

## Paper-by-Paper Explanation

### Paper 1: FlashAttention

#### Problem addressed

Standard attention is slow and memory-hungry for long sequences because it creates large intermediate tensors and moves them repeatedly between HBM and SRAM. The paper argues that many prior approaches focused too much on reducing FLOPs and not enough on reducing IO. 

#### Method used

FlashAttention uses tiling, kernel fusion, and recomputation to compute **exact** attention while avoiding materialization of the large attention matrix in HBM. It also analyzes HBM access complexity and claims optimality over a range of SRAM sizes. 

#### Main innovation

The main innovation is not a new approximate attention pattern. It is a hardware-aware reformulation of exact attention around memory hierarchy and IO cost. ([arXiv][2])

#### Main findings

The paper reports faster end-to-end training, including 15% speedup on BERT-large at sequence length 512, 3× speedup on GPT-2 at sequence length 1K, and 2.4× speedup on Long Range Arena tasks. It also reports up to 7.6× speedup on the attention computation itself and linear memory in sequence length. 

#### Limitations

The method requires sophisticated CUDA implementation and is centered on reducing memory traffic, not eliminating the basic dependence of exact attention on sequence interactions. The paper also notes recomputation in backward, which is a deliberate trade-off. 

#### What changed compared with earlier work

Compared with approximate attention methods, FlashAttention keeps exact attention and focuses on practical wall-clock speed through IO-awareness. Compared with standard fused multi-head attention implementations, it avoids storing the attention matrix and uses on-chip recomputation. 

### Paper 2: PagedAttention / vLLM

#### Problem addressed

Serving LLMs efficiently is hard because KV cache is large, grows dynamically, and is poorly handled by contiguous-memory allocation strategies. Existing systems waste memory through reserved space, internal fragmentation, and external fragmentation. The paper states that actual effective memory in prior systems can be as low as 20.4%. ([arXiv][1])

#### Method used

PagedAttention stores KV cache in fixed-size non-contiguous blocks and manages them with page-table-like mappings. vLLM adds a scheduler, block manager, block sharing, copy-on-write, and preemption via swapping or recomputation. ([arXiv][1])

#### Main innovation

The main innovation is to bring operating-system ideas—virtual memory, paging, and copy-on-write—into the KV-cache management problem for online LLM serving. ([arXiv][1])

#### Main findings

The paper reports near-zero waste in KV-cache memory and 2–4× serving-throughput improvements over prior state-of-the-art systems, with bigger gains on longer sequences, larger models, and more complex decoding algorithms. It also reports KV-block sharing savings of 6.1% to 9.8% for parallel sampling and 37.6% to 55.2% for beam search in one evaluated setting. ([arXiv][1])

#### Limitations

The system introduces engineering complexity, including block-size tuning and preemption-policy choices. Swapping can be expensive for small blocks because of many small CPU-GPU transfers; recomputation and swapping each win in different regimes. ([arXiv][1])

#### What changed compared with earlier work

Unlike approaches that optimize attention kernel IO or reduce peak attention memory, this paper targets **online serving memory management** for dynamic KV cache. It explicitly distinguishes its contribution from FlashAttention by saying FlashAttention reduces attention-computation IO, whereas this paper introduces block-level memory management for online serving. ([arXiv][1])

### Paper 3: Multi-Query Attention

#### Problem addressed

Incremental Transformer decoding is slow because each new decoding step repeatedly reloads large key and value tensors, and this process is limited by memory bandwidth. 

#### Method used

MQA shares one set of keys and values across all heads while keeping separate query heads. That reduces the size of the K/V tensors that must be accessed during incremental decoding. 

#### Main innovation

The innovation is architectural: keep the benefit of multiple query heads while removing the need for head-specific K/V storage. 

#### Main findings

The paper reports much faster decoding with only minor quality degradation. In the WMT14 EN-DE experiment, the decoder inference cost drops from 46 microseconds per token to 3.8 microseconds per token, while training cost stays nearly unchanged. Translation quality is very close to the baseline, and beam-4 test BLEU is slightly higher for MQA in that experiment. 

#### Limitations

The paper reports small quality degradation in some metrics and evaluates mainly on translation and language modeling benchmarks from that period. Information about later large-scale LLM behavior is not provided in this source. 

#### What changed compared with earlier work

Compared with standard multi-head attention, MQA reduces memory-bandwidth requirements in the incremental setting by sharing K/V across heads. The paper’s analysis says this reduces the dominant memory-access term by a factor of the number of heads. 

---

## Comparison Across Papers or Methods

The following comparison summarizes the role of each method based on the papers. 

| Method                | Primary Goal                                          | Main Component Changed                     | Training or Serving?                                                 | Exactness / Quality Effect         | Main Strength                                                    | Main Weakness                                              |
| --------------------- | ----------------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| FlashAttention        | Reduce attention IO and memory footprint              | Attention kernel / operator implementation | Strongly relevant to training and long-context attention computation | Exact attention                    | Faster exact attention, lower memory use, no need to approximate | Requires specialized kernel engineering                    |
| PagedAttention / vLLM | Reduce KV-cache waste and increase serving throughput | Runtime memory manager and scheduler       | Serving                                                              | No model-accuracy change reported  | Fits more requests, shares cache blocks, improves throughput     | More systems complexity, block/preemption trade-offs       |
| Multi-Query Attention | Reduce decoding-time K/V bandwidth                    | Model architecture                         | Primarily decoding / inference                                       | Minor quality degradation reported | Large decoding speedup                                           | Changes model architecture and may slightly affect quality |

A second comparison that is useful in interviews is this: **what resource does each paper save most directly?** This framing is a reasoned interpretation from the papers’ stated bottlenecks. FlashAttention saves HBM traffic inside attention computation, MQA saves K/V bandwidth during incremental decoding, and PagedAttention saves KV-cache memory capacity and reduces waste across many active requests. 

---

## Real-World System and Application

### Directly stated facts

The papers do **not** describe one single integrated production stack that combines all three methods in one experiment. Information about a joint end-to-end deployment is not provided. 

### Reasoned interpretation

A practical LLM system could combine these ideas at different layers:

1. **Model design:** use MQA so the decoder’s KV cache is smaller and cheaper to reload.
2. **Attention kernel:** use FlashAttention to make exact attention calls more IO-efficient.
3. **Serving engine:** use a PagedAttention-like runtime to store and manage KV cache for many requests efficiently. 

This layered combination makes sense because the three methods target different bottlenecks rather than duplicating the same optimization. That is the most important systems insight to carry into interviews. 

---

## Limitations and Trade-offs

### FlashAttention

* It improves exact attention mainly by reducing IO, not by changing the model’s basic computation into a cheap approximation. 
* It uses recomputation in backward, which is a deliberate compute-versus-memory trade-off. 
* It depends on low-level GPU-kernel design, so the implementation burden is higher than a simple framework-level rewrite. ([arXiv][2])

### PagedAttention / vLLM

* The system becomes more complex because it now has logical blocks, physical blocks, block tables, copy-on-write, and preemption policies. ([arXiv][1])
* Block size matters: too large increases internal fragmentation and reduces sharing; too small can hurt some overheads. The paper sets 16 as the default practical choice. ([arXiv][1])
* Swapping versus recomputation is workload-dependent. Small blocks favor recomputation; large blocks favor swapping; medium ranges are comparable. ([arXiv][1])

### Multi-Query Attention

* It changes the architecture rather than merely optimizing an implementation, so it may affect model quality. The paper reports only minor degradation, but it is still a trade-off. 
* Its main advantage is in incremental decoding, not necessarily in training. The paper’s own numbers show almost no training-speed change compared with much larger decoder-speed gains. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain:

1. Why standard attention is often memory-bound on GPUs.
2. Why long-sequence training and autoregressive serving have different bottlenecks.
3. Why FlashAttention is called **IO-aware**.
4. Why MQA helps decoding specifically.
5. Why PagedAttention uses an operating-system paging analogy.
6. Why these methods are complementary rather than substitutes. 

### Likely interview questions and concise model answers

#### 1. What is the core idea of FlashAttention?

FlashAttention keeps attention exact but reorganizes the computation so the large attention matrix is never written to slow GPU memory. It uses tiling, fused computation, and recomputation to reduce HBM traffic, which is often the real bottleneck. 

#### 2. Why is reducing FLOPs not enough?

Because on modern accelerators, many attention workloads are limited by memory movement rather than arithmetic. A method can have fewer theoretical operations and still be slower if it performs too many reads and writes to slow memory. 

#### 3. What problem does MQA solve?

MQA solves the memory-bandwidth bottleneck in incremental decoding. Standard multi-head attention stores separate keys and values per head; MQA shares keys and values across heads, so each decoding step reloads much less data. 

#### 4. Why does MQA mainly help inference instead of training?

Because training can process many positions in parallel, while autoregressive decoding generates one token at a time and repeatedly reads the growing KV cache. The paper’s results show very small training-time change but large decoder-speed gains. 

#### 5. What is PagedAttention in one sentence?

PagedAttention is a KV-cache storage method for LLM serving that breaks cache into fixed-size non-contiguous blocks, similar to virtual memory pages, so the system can reduce fragmentation and share memory more efficiently. ([arXiv][1])

#### 6. How does vLLM increase throughput?

By reducing KV-cache waste, allowing more requests to fit into GPU memory, enabling block sharing across related sequences, and scheduling requests around this paged memory model. ([arXiv][1])

#### 7. Are FlashAttention and PagedAttention the same kind of idea?

No. FlashAttention optimizes how one attention computation is executed on GPU memory hierarchy. PagedAttention optimizes how KV cache is stored and managed across many requests in a serving system. ([arXiv][1])

#### 8. What is the cleanest way to compare these three papers?

FlashAttention is a kernel-level optimization, MQA is an architecture-level optimization, and PagedAttention is a serving-system optimization. They target different bottlenecks and can be combined. The combination point is a reasoned systems interpretation, not a direct experiment from the papers. 

---

## Glossary

| Term                    | Beginner-friendly meaning                                                                                |
| ----------------------- | -------------------------------------------------------------------------------------------------------- |
| Attention               | A mechanism that lets one token look up relevant information from other tokens.                          |
| Query (Q)               | The representation of “what this token is looking for.”                                                  |
| Key (K)                 | The representation used to decide whether another token is relevant.                                     |
| Value (V)               | The information content that gets read if a token is judged relevant.                                    |
| Softmax                 | A function that turns raw scores into positive weights that sum to 1.                                    |
| Multi-head attention    | Several attention heads running in parallel so the model can learn different dependency patterns.        |
| Autoregressive decoding | Generating one token at a time, where each new token depends on previous ones.                           |
| KV cache                | Stored keys and values from previous tokens so the model does not recompute them every decoding step.    |
| Memory bandwidth        | How fast data can be moved between memory and compute units.                                             |
| HBM                     | High Bandwidth Memory on the GPU; large but slower than on-chip SRAM.                                    |
| SRAM                    | Small, very fast on-chip memory on the GPU.                                                              |
| IO-aware                | Designed to reduce costly reads and writes across memory levels.                                         |
| Tiling                  | Breaking a large computation into blocks that fit into fast memory.                                      |
| Recomputation           | Recomputing some intermediates later instead of storing them now.                                        |
| Fragmentation           | Wasted memory caused by awkward allocation patterns.                                                     |
| Internal fragmentation  | Wasted space inside allocated regions because the reserved region is bigger than what is currently used. |
| External fragmentation  | Free memory exists, but it is split into pieces that are hard to use efficiently.                        |
| Paging                  | Managing memory in fixed-size blocks that do not need to be contiguous physically.                       |
| Copy-on-write           | Let multiple users share memory until one needs to modify it, then copy only the modified part.          |
| MQA                     | Multi-Query Attention; separate query heads, shared keys and values.                                     |
| FlashAttention          | An exact attention algorithm optimized around GPU memory movement.                                       |
| PagedAttention          | A paged KV-cache storage method for LLM serving.                                                         |
| vLLM                    | A serving engine built on top of PagedAttention.                                                         |

---

## Recap

These three papers teach a very important lesson: Transformer efficiency is not one problem and does not have one solution. FlashAttention says the attention kernel should be redesigned around memory hierarchy. Multi-Query Attention says decoding can be sped up by shrinking K/V bandwidth at the architectural level. PagedAttention says production serving needs a smarter KV-cache memory manager, not just a faster kernel. 

What matters most for interviews is that you can explain the bottleneck each method targets, why that bottleneck exists, and why the methods are complementary. What remains limited in these sources is a joint end-to-end evaluation of all three ideas together in one unified production stack. Information about that combined deployment is not provided in the papers here. 

---

## Key Citations

* *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (arXiv:2205.14135). ([arXiv][2])

* *Efficient Memory Management for Large Language Model Serving with PagedAttention* (arXiv:2309.06180). ([arXiv][1])

* *Fast Transformer Decoding: One Write-Head is All You Need* (Multi-Query Attention) (arXiv:1911.02150). 

[1]: https://arxiv.org/pdf/2309.06180 "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[2]: https://arxiv.org/pdf/2205.14135?utm_source=chatgpt.com "https://arxiv.org/pdf/2205.14135"


---
---
---

