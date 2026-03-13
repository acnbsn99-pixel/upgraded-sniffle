# Reasoning and Chain-of-Thought: From Zero-Shot-CoT to Self-Consistency to Tree of Thoughts

## What This Report Teaches

This report explains three important papers about improving reasoning at **inference time**, which means improving how a language model is prompted and how its outputs are searched or selected, without retraining the model.

The three papers form a clear progression:

1. **Zero-Shot-CoT** asks: can we get a model to produce step-by-step reasoning without giving worked examples?
2. **Self-Consistency** asks: once the model can produce reasoning paths, should we trust the first one, or should we sample many and aggregate them?
3. **Tree of Thoughts (ToT)** asks: instead of generating one chain from left to right, can we search over many intermediate reasoning states the way classical AI search algorithms do?

By the end, you should understand:

* what **chain-of-thought (CoT)** prompting is,
* why the phrase **“Let’s think step by step”** mattered,
* how **self-consistency** changes decoding,
* how **Tree of Thoughts** turns reasoning into a search problem,
* what each method improves,
* what each method costs,
* and how to explain the trade-offs in an AI engineer or AI architect interview.

---

## Key Takeaways

* **These papers are about better inference, not better training.**
  Core idea: they change prompting, decoding, or search during model use.
  Why it matters: they improve reasoning without changing model weights.
  Practical implication: you can think of them as system-level reasoning strategies layered on top of an existing model.

* **Zero-Shot-CoT showed that a simple trigger phrase can often unlock multi-step reasoning.**
  Core idea: adding “Let’s think step by step” can make a model produce intermediate reasoning.
  Why it matters: it reduced dependence on carefully crafted few-shot reasoning examples.
  Practical implication: prompt design alone can materially change reasoning performance.

* **Self-consistency improves CoT by not trusting a single reasoning path.**
  Core idea: sample many reasoning paths, then choose the most common final answer.
  Why it matters: one greedy chain may be wrong even when several alternative chains would converge on the correct answer.
  Practical implication: better accuracy often comes from answer aggregation, not just a better prompt.

* **Tree of Thoughts generalizes chain-of-thought into a search process.**
  Core idea: reason over a tree of intermediate thoughts instead of one left-to-right chain.
  Why it matters: some tasks require exploration, lookahead, pruning, and backtracking.
  Practical implication: for planning-style problems, search can outperform plain CoT.

* **The three papers improve different parts of the reasoning pipeline.**
  Core idea: Zero-Shot-CoT changes the prompt, Self-Consistency changes decoding, ToT changes the search procedure.
  Why it matters: they are complementary ideas, not just competing slogans.
  Practical implication: in interviews, explain them as improvements at different layers of the inference stack.

* **More reasoning power usually means more inference cost.**
  Core idea: one chain is cheaper than many sampled chains, and many sampled chains are usually cheaper than tree search.
  Why it matters: better reasoning often trades off against latency, tokens, and system complexity.
  Practical implication: choosing a method depends on whether you optimize for speed, cost, or hard-problem accuracy.

* **These methods do not guarantee true reasoning in a philosophical sense.**
  Core idea: they improve performance by eliciting, sampling, or searching over intermediate text.
  Why it matters: better-looking reasoning traces do not automatically mean the model “understands” in a human way.
  Practical implication: use these methods as practical reasoning tools, not as proof of human-like cognition.

---

## Background and Foundations

To understand these papers, you need a few basic ideas.

### What is a language model?

A **language model (LM)** predicts text one token at a time. A **token** is a small text unit, often a word piece rather than a full word.

At inference time, the model receives a prompt and then generates output token by token from left to right.

### What is prompting?

A **prompt** is the text you give the model to shape its behavior.

Common prompting setups are:

* **Zero-shot prompting**: ask the model to do a task using only instructions, with no examples.
* **Few-shot prompting**: include a small number of worked examples inside the prompt.

### What is chain-of-thought prompting?

**Chain-of-thought (CoT)** prompting asks the model to generate intermediate reasoning steps before giving the final answer.

Instead of:

* “Answer: 24”

the model may produce:

* “First do this, then this, then this, so the answer is 24.”

The key intuition is that difficult tasks often become easier if the model is encouraged to decompose them into smaller steps.

### What is decoding?

**Decoding** is how we choose output text from the model’s probability distribution.

Two important decoding styles in these papers are:

* **Greedy decoding**: at each step, choose the highest-probability next token.
* **Sampling**: randomly sample next tokens from the model’s distribution, often with parameters like temperature or top-k filtering.

Greedy decoding is simpler and cheaper, but it may lock the model into one flawed reasoning path. Sampling can produce diverse reasoning paths.

### What is a reasoning path?

A **reasoning path** is one complete intermediate explanation leading to an answer.

For the same question, a model may generate:

* one path that is wrong,
* another that is clumsy but right,
* and another that is different in wording but reaches the same correct answer.

This idea is central to self-consistency.

### What is search?

In classical AI, **search** means exploring multiple candidate states instead of committing immediately to one.

A search method often needs:

* a **state**: the current partial solution,
* a way to **generate next candidates**,
* a way to **evaluate candidates**,
* and a way to **continue, prune, or backtrack**.

Tree of Thoughts imports that logic into LLM inference.

### How the three papers relate

| Paper            | Main question                                               | Main change                          |
| ---------------- | ----------------------------------------------------------- | ------------------------------------ |
| Zero-Shot-CoT    | Can we trigger reasoning without examples?                  | Change the prompt                    |
| Self-Consistency | Should we trust one chain or aggregate many?                | Change decoding and answer selection |
| Tree of Thoughts | Can reasoning be a search process over intermediate states? | Change the whole inference procedure |

Historically and conceptually, they move from:

1. **eliciting one chain**,
2. to **sampling many chains**,
3. to **searching over partial chains**.

---

## Big Picture First

A useful mental model is this:

### Stage 1: One prompted chain

You ask the model to think step by step and produce one reasoning trace.

This is the world of **Zero-Shot-CoT**.

### Stage 2: Many sampled chains

You ask for many reasoning traces, then pick the final answer that appears most consistently.

This is the world of **Self-Consistency**.

### Stage 3: A searched tree of thoughts

You do not force the model into one full chain immediately. Instead, you let it propose intermediate thoughts, evaluate them, keep promising ones, discard weak ones, and possibly backtrack.

This is the world of **Tree of Thoughts**.

### A simple hierarchy

| Method           | Unit of reasoning           | How many paths are explored? | How answer is chosen                      |
| ---------------- | --------------------------- | ---------------------------- | ----------------------------------------- |
| Zero-Shot-CoT    | One chain of thought        | Usually one                  | Final answer from that one chain          |
| Self-Consistency | Multiple full chains        | Many sampled chains          | Aggregate by most consistent final answer |
| Tree of Thoughts | Partial thoughts and states | A searched tree              | Search procedure plus state evaluation    |

### The core shift across the papers

The shift is not only “better prompts.”

It is a deeper progression:

* **Prompting**: tell the model to reason
* **Decoding**: do not trust one chain
* **Search**: reason over alternatives before committing

That is the big picture.

---

## Core Concepts Explained

## Chain-of-Thought (CoT)

### What it is

A prompt style that gets the model to generate intermediate reasoning steps before the answer.

### Why it exists

Many tasks are hard when asked as direct question-to-answer mappings, but easier when broken into smaller steps.

### How it works at a high level

The prompt encourages the model to output a reasoning trace. The model then generates text that looks like a step-by-step explanation.

### Where it appears

All three papers build on CoT as the starting point.

### Why it matters

It creates the intermediate text that later methods can sample, compare, vote over, or search through.

---

## Zero-Shot-CoT

### What it is

A zero-shot method that uses a generic trigger phrase such as **“Let’s think step by step”** to elicit reasoning without giving worked examples.

### Why it exists

Earlier CoT results were closely associated with few-shot examples. This paper asked whether the model already had latent reasoning ability that could be unlocked by a simple prompt.

### How it works

The paper’s method is actually **two-stage prompting**, which is easy to miss:

1. **Reasoning extraction**
   Prompt the model with the question plus a trigger like “Let’s think step by step.”
2. **Answer extraction**
   Feed the generated reasoning back into a second prompt that asks for the final answer in the required format.

### Where it appears

This is the main contribution of the first paper.

### Why it matters

It showed that reasoning improvements did not always require manually crafted few-shot CoT examples.

---

## Greedy decoding vs. sampling

### What they are

* **Greedy decoding** chooses the highest-probability next token each time.
* **Sampling** draws tokens from the model’s probability distribution, allowing diversity.

### Why they exist

Greedy decoding is efficient and deterministic. Sampling is useful when there may be multiple good reasoning paths.

### How they work at a high level

Greedy decoding follows one path. Sampling explores multiple possible paths.

### Where they appear

* Zero-Shot-CoT used greedy decoding for simplicity.
* Self-Consistency relies on sampling.
* ToT may use sampling or proposal prompts to generate candidate thoughts.

### Why they matter

This is the bridge from “one chain” to “many alternatives.”

---

## Self-Consistency

### What it is

A decoding strategy that samples multiple CoT reasoning paths and chooses the final answer that appears most consistently.

### Why it exists

A model may know how to solve a problem, but a single greedy chain may be unlucky or flawed. If several diverse chains independently reach the same answer, that answer is more trustworthy.

### How it works at a high level

1. Prompt the model with CoT.
2. Sample many reasoning paths.
3. Extract the final answer from each path.
4. Aggregate answers and pick the most frequent one.

The paper describes this as **sample-and-marginalize**.

### Where it appears

This is the main contribution of the second paper.

### Why it matters

It improves reasoning accuracy without retraining, extra labels, or external verifiers.

---

## Thoughts in Tree of Thoughts

### What they are

A **thought** is a coherent chunk of text used as an intermediate step toward solving a problem.

A thought can be:

* an intermediate equation,
* a short plan,
* a candidate word fill in a crossword,
* or another meaningful partial solution.

### Why they exist

Single-token generation is often too small a unit for deliberate reasoning. A larger semantic unit can be evaluated more meaningfully.

### How they work

Instead of generating one long uninterrupted sequence, the model works with intermediate thought chunks.

### Where they appear

This is central to Tree of Thoughts.

### Why they matter

They make search possible. Search needs states and candidate moves, and thoughts provide those moves.

---

## State, generator, evaluator, search

Tree of Thoughts can be understood through four pieces.

### 1. State

A **state** is the current partial solution: the input plus the thoughts generated so far.

### 2. Thought generator

The **generator** proposes the next candidate thoughts.

The paper uses two broad strategies:

* **independent sampling**, useful when thought space is rich,
* **proposal prompting**, useful when the next step is more constrained.

### 3. State evaluator

The **evaluator** estimates which states are promising.

It can do this by:

* **valuing** each state independently, such as scoring or classifying it,
* or **voting** across several states to choose the best one.

### 4. Search algorithm

The search procedure decides how to explore the tree:

* **BFS (breadth-first search)** keeps several promising states per depth level.
* **DFS (depth-first search)** follows one promising path deeply, then backtracks when needed.

These parts together define ToT.

---

## Step-by-Step Technical Walkthrough

## 1. Zero-Shot-CoT pipeline

### Goal

Get the model to produce step-by-step reasoning without few-shot examples.

### Workflow

1. **Input**

   * A task question, such as a math or logic problem.

2. **Build the first prompt**

   * Format the question as something like:
     `Q: [question] A: Let’s think step by step.`

3. **Generate reasoning**

   * The model produces a chain of thought.

4. **Build the second prompt**

   * Take the original question and the generated reasoning, then append an answer-extraction trigger such as:

     * “Therefore, the answer is …”
     * or a task-specific answer-format cue.

5. **Generate final answer**

   * The model outputs the answer in the desired format.

### Output

A reasoning trace plus a final answer.

### Why this exists

The first generation elicits reasoning; the second makes answer formatting more reliable.

### Trade-offs

* **Strengths**: simple, cheap, task-agnostic trigger, no worked examples needed.
* **Weaknesses**: still only one path; if the reasoning goes wrong, the answer often goes wrong.
* **Failure mode**: the model may generate plausible but incorrect reasoning, or continue reasoning after reaching the right answer and drift into a wrong final answer.

---

## 2. Self-Consistency pipeline

### Goal

Improve CoT reasoning by aggregating across many sampled reasoning paths.

### Workflow

1. **Input**

   * A problem plus a CoT prompt, usually few-shot CoT in the main experiments.

2. **Generate multiple chains**

   * Sample many reasoning paths instead of using greedy decoding.

3. **Extract answers**

   * Parse the final answer from each sampled chain.

4. **Aggregate**

   * Count how often each answer appears.

5. **Select final answer**

   * Return the most consistent answer.

### Output

One final answer chosen from many sampled chains.

### Why this exists

Correct reasoning can be expressed in multiple ways. Wrong reasoning tends to be less stable and less likely to converge on the same final answer.

### Trade-offs

* **Strengths**: large accuracy gains, no additional training, works off-the-shelf.
* **Weaknesses**: higher inference cost because many paths must be sampled.
* **Failure mode**: if the model’s sampled paths are mostly bad, aggregation does not rescue the answer.

### Practical meaning of the method

Self-consistency is not “just sample more.” It is specifically:

* sample diverse **reasoning paths**,
* then choose the answer with the strongest agreement.

That is why diversity matters so much in this paper.

---

## 3. Tree of Thoughts pipeline

### Goal

Handle tasks where one left-to-right chain is too restrictive because the model needs exploration, evaluation, pruning, and possibly backtracking.

### Workflow

1. **Input**

   * A problem such as Game of 24, creative writing with constraints, or mini crosswords.

2. **Represent the current state**

   * The state is the input plus the thoughts generated so far.

3. **Generate candidate next thoughts**

   * The model proposes several next-step thoughts.

4. **Evaluate candidate states**

   * The model judges which resulting states are promising.
   * This may use:

     * value judgments like “sure / maybe / impossible”
     * or voting among alternatives

5. **Search**

   * Use BFS or DFS to continue exploring promising states.
   * Prune weak branches.
   * Backtrack when needed.

6. **Stop and produce final output**

   * Once a good full solution is found, render it as the final answer.

### Output

A final answer found through search over intermediate thoughts.

### Why this exists

Some problems are not well served by one uninterrupted chain. They require trying alternatives and correcting earlier decisions.

### Trade-offs

* **Strengths**: better for planning-like or search-heavy tasks.
* **Weaknesses**: much more computationally expensive and more task-specific.
* **Failure mode**: weak state evaluation can prune good branches or pursue bad ones.

---

## Paper-by-Paper Explanation

## 1. Large Language Models are Zero-Shot Reasoners

### Note on title

The first URL points to the paper commonly remembered by the trigger phrase **“Let’s think step by step.”** Its actual title is **Large Language Models are Zero-Shot Reasoners**.

### Problem addressed

Few-shot CoT had shown strong reasoning gains, but it depended on task-specific examples. This paper asks whether large language models can perform multi-step reasoning in zero-shot form with a generic trigger.

### Method used

The method, called **Zero-shot-CoT**, adds a trigger sentence such as “Let’s think step by step” and uses a second prompt to extract the answer cleanly.

### Main innovation

The key innovation is not just the trigger phrase. It is the idea that:

* one generic trigger can work across many reasoning tasks,
* and zero-shot reasoning can be elicited without handcrafted CoT exemplars.

### Main findings

The paper reports large gains on several reasoning benchmarks, including:

* MultiArith from 17.7% to 78.7%
* GSM8K from 10.4% to 40.7%

It also reports similar improvement patterns on PaLM 540B and shows better scaling behavior when CoT reasoning is elicited.

### Limitations

* It still relies on a single reasoning path.
* It used greedy decoding in the main setup.
* Gains were strong on arithmetic, symbolic, and logical tasks, but not uniformly strong on commonsense tasks.
* Larger models benefited more; smaller models often did not.

### What changed compared with earlier work

Earlier CoT prompting emphasized few-shot demonstrations. This paper moved the field toward the idea that zero-shot reasoning can be elicited with a simple general trigger.

### Directly stated facts

* The method uses a generic reasoning trigger.
* It uses two-stage prompting.
* It shows large improvements on arithmetic and symbolic reasoning tasks.
* The reasoning benefit becomes clearer at larger model scales.

### Reasoned interpretation

This paper is best understood as an **elicitation paper**. It argues that some reasoning ability was already present in the model and could be unlocked by prompt design.

### Information not provided

A deeper mechanistic explanation of why the trigger works internally is not provided.

---

## 2. Self-Consistency Improves Chain of Thought Reasoning in Language Models

### Problem addressed

CoT prompting helps, but greedy decoding still commits to one reasoning chain. This paper asks whether reasoning quality improves if the model samples multiple chains and then aggregates them.

### Method used

The paper proposes **self-consistency**, a decoding strategy that:

1. samples multiple reasoning paths,
2. extracts final answers,
3. and returns the most consistent answer.

### Main innovation

The main innovation is shifting the improvement target from prompt design to **decoding and answer selection**.

### Main findings

The paper reports strong gains across reasoning benchmarks, including:

* GSM8K: +17.9%
* SVAMP: +11.0%
* AQuA: +12.2%
* StrategyQA: +6.4%
* ARC-challenge: +3.9%

It also shows:

* robustness to different sampling strategies,
* better performance than beam search,
* better performance than several ensemble baselines,
* robustness to imperfect prompts,
* and compatibility with zero-shot CoT.

### Limitations

* It costs more because many reasoning paths must be generated.
* It still aggregates at the **final answer** level rather than searching within intermediate states.
* Its main strength depends on diversity among sampled chains.

### What changed compared with earlier work

This paper changed the field’s view from:

* “Get one good chain”
  to
* “Generate many plausible chains and trust agreement.”

### Directly stated facts

* The paper samples 40 outputs per run in main results.
* It requires no additional training, fine-tuning, or auxiliary model.
* It outperforms beam search and sample-and-rank in the reported comparisons.
* It also improves zero-shot CoT on GSM8K in the paper’s additional study.

### Reasoned interpretation

This paper is best understood as a **test-time ensembling method for reasoning**. It makes CoT more reliable by treating answer agreement as a signal.

### Information not provided

A universal aggregation strategy for open-ended outputs beyond the paper’s settings is not fully developed here.

---

## 3. Tree of Thoughts: Deliberate Problem Solving with Large Language Models

### Problem addressed

CoT and self-consistency still operate on whole chains. They do not explicitly support local branching, lookahead, pruning, or backtracking. This paper asks whether LLM inference should be treated as a search problem over intermediate states.

### Method used

Tree of Thoughts builds a search tree where:

* nodes are partial reasoning states,
* edges are candidate next thoughts,
* the model generates candidates,
* the model also evaluates candidates,
* and a search algorithm such as BFS or DFS decides what to explore next.

### Main innovation

The main innovation is turning reasoning into **deliberate search over thought states**, not just sequence generation.

### Main findings

The paper reports strong improvements on three tasks:

* **Game of 24**
  CoT: 4%
  CoT self-consistency: 9%
  ToT with breadth 1: 45%
  ToT with breadth 5: 74%

* **Creative Writing**
  ToT produces more coherent passages than IO and CoT in the paper’s GPT-4 and human evaluations.

* **Mini Crosswords**
  ToT substantially improves letter-level, word-level, and full-game success over IO and CoT.

### Limitations

* ToT is much more expensive than plain CoT.
* The paper only studies three tasks.
* Performance depends on how thoughts are defined, how states are evaluated, and which search algorithm is used.
* State evaluation can be imperfect and may prune good branches.

### What changed compared with earlier work

This paper moved from **sampling full reasoning paths** to **searching over partial reasoning states**.

### Directly stated facts

* Thoughts are coherent intermediate language units.
* The paper uses both BFS and DFS.
* It uses different generation and evaluation strategies depending on task.
* It reports higher computational cost than CoT and IO prompting.

### Reasoned interpretation

This paper is best understood as the transition from **reasoning as text generation** to **reasoning as search and control**.

### Information not provided

A general best-practice recipe for all reasoning tasks is not provided. The paper shows a framework plus task-specific instantiations.

---

## Comparison Across Papers or Methods

## High-level comparison

| Aspect                     | Zero-Shot-CoT                            | Self-Consistency                                                    | Tree of Thoughts                          |
| -------------------------- | ---------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------- |
| Main target of improvement | Prompting                                | Decoding and aggregation                                            | Search procedure                          |
| Uses worked examples?      | No, in the main method                   | Usually yes in main experiments, but also tested with zero-shot CoT | Depends on task setup                     |
| Number of reasoning paths  | One                                      | Many full chains                                                    | Many partial paths in a tree              |
| Selection logic            | Trust the generated chain                | Choose most consistent final answer                                 | Evaluate states and search                |
| Lookahead / backtracking   | No                                       | No                                                                  | Yes                                       |
| Cost                       | Low                                      | Medium to high                                                      | High                                      |
| Best for                   | Simple reasoning boost with low overhead | Better accuracy when multiple reasoning paths exist                 | Problems needing exploration and planning |

## What each paper adds to the pipeline

| Pipeline layer | Zero-Shot-CoT              | Self-Consistency                  | Tree of Thoughts                   |
| -------------- | -------------------------- | --------------------------------- | ---------------------------------- |
| Prompt design  | Strong contribution        | Uses CoT prompts                  | Uses task-specific thought prompts |
| Decoding       | Mostly greedy in the paper | Sampling-based                    | Sampling/proposal inside search    |
| Aggregation    | Minimal                    | Majority-style answer aggregation | Search-level selection and pruning |
| Search         | None                       | None                              | Central idea                       |

## How they relate conceptually

| Question                              | Zero-Shot-CoT answer | Self-Consistency answer       | ToT answer                         |
| ------------------------------------- | -------------------- | ----------------------------- | ---------------------------------- |
| Should the model reason step by step? | Yes                  | Yes                           | Yes                                |
| Is one chain enough?                  | Usually one chain    | No, sample many               | No, search among partial chains    |
| Where is uncertainty handled?         | Barely               | At answer aggregation         | During intermediate search         |
| What if early decisions are bad?      | Usually stuck        | Try a different sampled chain | Backtrack and explore alternatives |

---

## Real-World System and Application

These papers suggest a practical reasoning stack for LLM systems.

## A simple deployment view

### Option 1: Prompt-only reasoning

Use Zero-Shot-CoT when:

* you want a cheap improvement,
* you do not want to maintain few-shot examples,
* and latency matters.

### Option 2: Multi-sample reasoning

Use Self-Consistency when:

* accuracy matters more than single-call latency,
* the task has a clear final answer that can be aggregated,
* and multiple valid reasoning paths may reach the same answer.

### Option 3: Search-based reasoning

Use Tree of Thoughts when:

* the task requires planning,
* early mistakes are costly,
* backtracking is valuable,
* and the problem naturally supports intermediate states.

## A practical system pipeline

1. **Receive user task**
2. **Classify task difficulty**

   * direct answer,
   * CoT-friendly,
   * multi-sample reasoning,
   * search-heavy planning
3. **Choose inference strategy**

   * direct prompt,
   * Zero-Shot-CoT,
   * Self-Consistency,
   * ToT-style search
4. **Run model calls**
5. **Aggregate or search**
6. **Return final answer**
7. **Optional confidence signal**

   * especially natural with self-consistency

## What the papers support directly

Supported by the papers:

* prompt-triggered reasoning,
* answer aggregation over multiple chains,
* search over intermediate thought states,
* task-specific reasoning control.

Not established as a general result in these papers:

* one universal best method for all tasks,
* one universal latency-accuracy frontier,
* or one universal state-evaluation strategy.

Information not provided: a full production architecture for online serving at scale.

---

## Limitations and Trade-offs

| Issue                        | Zero-Shot-CoT                    | Self-Consistency                      | Tree of Thoughts                          |
| ---------------------------- | -------------------------------- | ------------------------------------- | ----------------------------------------- |
| Inference cost               | Low                              | Higher because of many samples        | Highest because of search                 |
| Reliance on one path         | High                             | Lower                                 | Lowest of the three                       |
| Need for task-specific setup | Low                              | Moderate                              | Often higher                              |
| Handling early mistakes      | Poor                             | Better across samples                 | Best, because it can search and backtrack |
| Ease of implementation       | Easiest                          | Moderate                              | Hardest                                   |
| Best output type             | Structured answers and reasoning | Tasks with aggregatable final answers | Search-heavy and planning-heavy tasks     |

### Concrete trade-offs

* **Zero-Shot-CoT**

  * cheapest,
  * easiest to deploy,
  * but least robust if the one generated chain is flawed.

* **Self-Consistency**

  * often much more accurate,
  * but more expensive because it needs multiple generations.

* **Tree of Thoughts**

  * most powerful on some search tasks,
  * but the most complex and costly,
  * and more dependent on task-specific thought design and evaluation prompts.

### Important conceptual limitation

These methods improve performance by operating on generated text:

* eliciting it,
* sampling more of it,
* or searching over it.

They do not by themselves prove that the model is performing human-like reasoning internally.

---

## Interview-Ready Understanding

## What you should be able to explain

You should be able to explain:

1. what chain-of-thought prompting is,
2. why “Let’s think step by step” mattered,
3. why Zero-Shot-CoT is more than just one phrase because it also uses answer extraction,
4. why greedy decoding can fail even when the model “knows” the answer,
5. how self-consistency works and why diversity matters,
6. how Tree of Thoughts differs from self-consistency,
7. when BFS versus DFS makes sense,
8. and why these methods trade accuracy for inference cost.

## Likely interview questions with concise model answers

### 1. What is the difference between chain-of-thought and self-consistency?

Chain-of-thought generates one reasoning trace. Self-consistency generates many reasoning traces and chooses the final answer with the strongest agreement.

### 2. What did Zero-Shot-CoT contribute?

It showed that step-by-step reasoning can often be elicited in zero-shot form with a generic trigger instead of requiring few-shot reasoning examples.

### 3. Why is self-consistency better than greedy CoT?

Because one greedy chain can be wrong. Sampling multiple chains lets the system use agreement across different reasoning paths as a signal for the final answer.

### 4. Why is beam search not the same as self-consistency?

Beam search tends to favor high-probability but similar outputs. Self-consistency benefits from diverse reasoning paths, and that diversity is one reason it works better.

### 5. What problem does Tree of Thoughts solve that self-consistency does not?

Self-consistency compares complete chains after they are generated. Tree of Thoughts explores and evaluates partial reasoning states during the process, with pruning and backtracking.

### 6. When would you use Tree of Thoughts instead of self-consistency?

When the task requires planning, lookahead, or correction of intermediate decisions, such as puzzle solving or constrained generation.

### 7. What are the main costs of these methods?

Mostly inference-time costs:

* more tokens,
* more model calls,
* more latency,
* and more system complexity.

### 8. How would you summarize the progression across the three papers?

Zero-Shot-CoT elicits one chain, Self-Consistency aggregates many chains, and Tree of Thoughts searches over partial chains.

### 9. Is Tree of Thoughts just a bigger version of self-consistency?

No. Self-consistency samples full chains independently and votes on final answers. ToT explicitly represents intermediate states and uses search decisions during reasoning.

### 10. What is the most interview-worthy insight from this set of papers?

Reasoning quality can often be improved substantially at inference time by changing how the model is prompted, decoded, and searched, even without retraining.

---

## Glossary

* **Answer extraction**: A second prompt or procedure used to convert generated reasoning into a clean final answer.
* **Backtracking**: Returning to an earlier state in search after discovering that the current path is not promising.
* **BFS (Breadth-First Search)**: A search strategy that explores several promising states level by level.
* **Chain-of-Thought (CoT)**: Prompting a model to produce intermediate reasoning steps before the final answer.
* **Consistency**: In the self-consistency paper, agreement among final answers from different sampled reasoning paths.
* **Decoding**: The procedure used to generate output text from model probabilities.
* **DFS (Depth-First Search)**: A search strategy that explores one promising path deeply before trying alternatives.
* **Few-shot prompting**: Prompting with a small number of examples inside the prompt.
* **Greedy decoding**: Always selecting the highest-probability next token.
* **Inference time**: The time when a trained model is used to answer a query.
* **Language model (LM)**: A model that generates or scores text token by token.
* **Marginalize**: In this context, combine evidence across many sampled reasoning paths instead of trusting one path.
* **Prompt**: The text input used to condition the model’s behavior.
* **Reasoning path**: One complete sequence of intermediate steps leading to an answer.
* **Sampling**: Generating outputs by drawing from the model’s probability distribution rather than always taking the top choice.
* **Search state**: A partial solution being considered during search.
* **Self-Consistency**: A decoding method that samples multiple CoT paths and chooses the most consistent final answer.
* **Thought**: In Tree of Thoughts, a coherent chunk of intermediate text that serves as a reasoning step.
* **Tree of Thoughts (ToT)**: A framework that treats reasoning as search over intermediate thought states.
* **Zero-shot prompting**: Prompting without task examples.
* **Zero-Shot-CoT**: A zero-shot prompting method that uses a reasoning trigger to elicit step-by-step thought.

---

## Recap

These three papers are best understood as a sequence of increasingly powerful inference-time reasoning methods.

* **Zero-Shot-CoT** showed that a generic prompt can elicit step-by-step reasoning without examples.
* **Self-Consistency** showed that one chain is often not enough, and that answer agreement across many sampled chains can greatly improve accuracy.
* **Tree of Thoughts** showed that reasoning can be treated as a search problem over intermediate states, with generation, evaluation, pruning, and backtracking.

The biggest idea connecting them is this:

**Reasoning performance is not only a property of the model weights. It is also a property of how you drive the model at inference time.**

For interviews, the most important mental model is:

* prompt the model to reason,
* do not overtrust one chain,
* and when the problem needs planning, search over intermediate states instead of forcing one left-to-right answer path.

What remains limited:

* these methods cost more inference compute,
* they do not solve every reasoning problem,
* and the papers do not provide one universal reasoning recipe for all tasks.

But together, they define a very important part of the modern reasoning toolkit.

---

## Key Citations

[Large Language Models are Zero-Shot Reasoners](https://arxiv.org/pdf/2205.11916)

[Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/pdf/2203.11171)

[Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/pdf/2305.10601)

---
---
---

