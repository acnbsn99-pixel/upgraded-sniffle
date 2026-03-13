# Mathematical Reasoning: From GSM8K Verifiers to Minerva to LeanDojo

## What This Report Teaches

This report explains three different styles of mathematical reasoning with language models:

1. **GSM8K / Training Verifiers to Solve Math Word Problems** focuses on informal school-style math word problems in natural language.
2. **Minerva** scales that idea up to broader quantitative reasoning, including competition math and undergraduate STEM problems.
3. **LeanDojo** moves into **formal theorem proving**, where proofs are checked by a proof assistant and every proof step must be valid. 

These papers belong together because they show a progression from “generate a correct final number” to “generate a long technical solution” to “generate a proof that a formal system will accept.” That progression matters because “math reasoning” is not one single task. Solving a grade-school word problem, solving a college physics calculation, and proving a formal theorem require different kinds of representations, search, feedback, and correctness checking. 

There is one important source note. The third URL you supplied, **arXiv:2303.05398**, points to **MathPrompter**, not LeanDojo. Because the title specifies **LeanDojo: Theorem Proving with Retrieval-Augmented Language Models**, I used the actual LeanDojo paper at **arXiv:2306.15626**. 

By the end, you should understand how mathematical reasoning systems can be built in three increasingly structured ways: **generate and verify natural-language solutions**, **scale domain-specific pretraining plus majority voting**, and **use retrieval plus a formal proof assistant for machine-checkable reasoning**. 

---

## Key Takeaways

* **Mathematical reasoning is not one problem.**
  Core idea: these papers cover informal word problems, broad quantitative reasoning, and formal theorem proving.
  Why it matters: each setting needs different supervision, search, and validation.
  Practical implication: in interviews, do not talk about “math reasoning” as if GSM8K, Minerva, and LeanDojo are the same task. 

* **GSM8K showed that generation alone is often not enough, and that verification can help more than just scaling the generator.**
  Core idea: generate many candidate solutions, then use a learned verifier to pick the best one.
  Why it matters: it treats correctness checking as a simpler problem than full solution generation.
  Practical implication: sample-and-rank can beat naive single-pass generation on multi-step math. 

* **Minerva showed that better technical pretraining data plus inference-time majority voting can strongly improve math and STEM reasoning.**
  Core idea: continue training large base models on scientific and mathematical text, then sample many solutions and vote on the final answer.
  Why it matters: it demonstrated strong quantitative reasoning without external tools.
  Practical implication: domain-specific data and test-time aggregation can matter as much as architecture changes. ([arXiv][1])

* **LeanDojo is fundamentally different because correctness is checked by Lean, not guessed by a heuristic evaluator.**
  Core idea: theorem proving happens inside a proof assistant, where generated tactics either advance the proof state or fail.
  Why it matters: this gives rigorous verification rather than approximate answer checking.
  Practical implication: formal reasoning systems can use LLMs, but the environment constrains and validates every step. 

* **Retrieval is central in formal theorem proving because the model must find useful lemmas and definitions from a huge library.**
  Core idea: LeanDojo’s ReProver retrieves relevant premises before generating the next tactic.
  Why it matters: many proofs fail not because the model cannot write a tactic, but because it does not know which prior theorem to use.
  Practical implication: theorem proving is often a retrieval-and-search problem as much as a generation problem. 

* **Verification means different things in GSM8K and LeanDojo.**
  Core idea: GSM8K trains a neural verifier that predicts whether a natural-language solution is correct, while LeanDojo uses Lean itself to verify proof steps.
  Why it matters: one is learned judgment; the other is formal checking.
  Practical implication: you should clearly distinguish “learned scoring” from “symbolic correctness checking” in interviews. 

* **The papers show a clear progression toward stronger correctness guarantees.**
  Core idea: GSM8K checks final answers, Minerva mostly checks end answers and uses majority voting, LeanDojo checks entire formal proof trajectories.
  Why it matters: stronger guarantees usually require more structure and more constrained problem formats.
  Practical implication: higher reliability often comes from moving from free-form text toward tool-grounded reasoning. 

---

## Background and Foundations

To understand these papers, you need one core distinction: **informal mathematical reasoning** versus **formal mathematical reasoning**. Informal reasoning is what you see in normal school and college problems: the model reads a word problem or equation, explains the steps in plain language, and produces a final answer. Formal reasoning is different: the theorem statement, intermediate states, and proof steps all live inside a precise symbolic system where there is no ambiguity about whether a step is valid. 

### Informal mathematical reasoning

In GSM8K and Minerva, the model is mostly operating in natural language. It may write equations, but it is still producing text as its reasoning trace. This is powerful because the input is natural and flexible, but it also means mistakes can appear anywhere in the chain of reasoning. A fluent explanation can still hide a bad arithmetic step or a wrong assumption. 

### Formal theorem proving

In LeanDojo, the model works with the **Lean** proof assistant. Instead of writing a general essay-like solution, it generates **tactics**, which are commands that transform the current proof state into simpler proof states. Lean checks whether those commands are valid. If they are not valid, the step fails. This is a much tighter feedback loop than ordinary natural-language math reasoning. 

### Why these papers fit together historically

These papers can be read as a progression:

1. **GSM8K** asks how to improve multi-step arithmetic reasoning on a clean grade-school benchmark.
2. **Minerva** asks how far large language models can go on broad quantitative reasoning when trained on more technical content.
3. **LeanDojo** asks how to move from approximate natural-language reasoning to rigorously checked formal proof generation. 

### Important beginner terms

| Term                          | Plain-English meaning                                                    | Why it matters here                     |
| ----------------------------- | ------------------------------------------------------------------------ | --------------------------------------- |
| **Math word problem**         | A story problem written in natural language                              | Core task in GSM8K                      |
| **Verifier**                  | A model that scores whether a candidate solution looks correct           | Central to GSM8K                        |
| **Majority voting**           | Generate many answers and pick the most common final answer              | Central to Minerva                      |
| **Proof assistant**           | Software that checks whether a formal proof is valid                     | Central to LeanDojo                     |
| **Theorem**                   | A statement to be proved                                                 | Central to LeanDojo                     |
| **Tactic**                    | A proof step command used in Lean                                        | Central to LeanDojo                     |
| **Premise**                   | A previously known theorem, lemma, or definition that may help the proof | Central to LeanDojo retrieval           |
| **Retrieval-augmented**       | The model looks up helpful external items before generating              | Central to LeanDojo                     |
| **Pass@1**                    | Success rate on the first attempt                                        | Important evaluation metric in LeanDojo |
| **Natural-language solution** | A step-by-step explanation written as text                               | Central to GSM8K and Minerva            |

The meanings in this table are synthesized directly from the papers’ problem settings and method descriptions. 

---

## Big Picture First

A good mental model is that the three papers differ mainly in **how they handle correctness**.

### GSM8K: correctness by learned ranking

The model generates many candidate solutions. Another model, the verifier, scores them. The system keeps the candidate judged most likely to be correct. This helps because evaluating a completed solution is often easier than writing the full solution from scratch. 

### Minerva: correctness by better domain knowledge plus answer aggregation

Minerva does not introduce a formal verifier. Instead, it strengthens the model through technical pretraining and then improves inference by sampling many solutions and using majority voting on final answers. This works when correct reasoning tends to converge on the same answer more often than wrong reasoning does. ([arXiv][1])

### LeanDojo: correctness by proof assistant feedback

LeanDojo changes the game entirely. The system is no longer just hoping the answer is right. It is interacting with Lean, which can tell it whether a tactic is valid and whether the proof state has improved. That makes the environment part of the reasoning process. 

### The overall progression

| Paper             | Reasoning medium                 | Main improvement lever                  | How correctness is handled                    |
| ----------------- | -------------------------------- | --------------------------------------- | --------------------------------------------- |
| GSM8K / Verifiers | Natural language                 | Generate many, then verify              | Learned verifier scores candidates            |
| Minerva           | Natural language + math notation | Technical pretraining + majority voting | Final-answer agreement and benchmark checking |
| LeanDojo          | Formal proof states + tactics    | Retrieval + proof assistant interaction | Lean formally checks proof steps              |

This comparison is synthesized from the three papers. 

---

## Core Concepts Explained

### Grade-school math reasoning

GSM8K consists of **8.5K** human-written grade-school math word problems, split into **7.5K training** and **1K test** examples. Problems usually take **2 to 8 steps** and use elementary arithmetic. The dataset was designed to be high-quality and diverse rather than scraped and templated. This matters because the paper is trying to test real multi-step reasoning, not shallow pattern matching on repeated templates. 

### Why verifiers help

In GSM8K, the authors argue that generation is fragile because one early mistake can ruin the whole reasoning chain. Their solution is to generate many candidates and train a verifier to judge whether a completed solution is correct. The key idea is that “is this solution correct?” can be easier to learn than “write the whole solution from scratch.” 

### Candidate generation versus ranking

GSM8K separates the system into two roles:

1. the **generator**, which proposes candidate solutions, and
2. the **verifier**, which scores them.

This is important because it introduces a basic architecture pattern that shows up repeatedly in reasoning systems: **generate multiple candidates, then use a simpler signal to pick the best**. 

### Token-level versus solution-level verification

The GSM8K paper compares two verifier styles. A **solution-level verifier** makes one judgment after seeing the entire solution. A **token-level verifier** predicts correctness throughout the solution, effectively acting like a value estimate at each token. The token-level verifier ultimately outperforms the solution-level verifier and overfits less, according to the paper. 

### Majority voting

Minerva relies heavily on **majority voting**, which means sampling many solutions and selecting the most common final answer. The paper reports that majority voting clearly beats greedy decoding and also beats log-likelihood reranking in its comparisons. This matters because it treats the model’s output distribution as useful information rather than trusting a single sample. ([arXiv][1])

### Technical-content pretraining

Minerva starts from PaLM models and continues training them on a high-quality dataset of scientific and mathematical text. The paper says the main data sources are **arXiv papers** and **mathematical web pages**. This matters because ordinary web-scale pretraining does not necessarily give the model the notation, style, and dense technical structure needed for strong quantitative reasoning. ([arXiv][1])

### Formal theorem proving

LeanDojo deals with theorems in Lean rather than natural-language answers. A theorem starts as a formal goal. The model proposes a **tactic**. Lean applies it if valid and returns a new proof state. This repeats until all goals are solved or the search fails. The crucial difference from natural-language reasoning is that the environment can check each step exactly. 

### Premise selection

In formal proof systems, a proof step often depends on the right previously known lemma or definition. LeanDojo treats this as a retrieval problem called **premise selection**. Instead of asking the model to memorize the whole math library, it retrieves relevant premises and concatenates them with the current proof state before generating the next tactic. This is a central idea of the paper, not a small implementation detail. 

### Automatic verification

This is one of the most interview-worthy contrasts across the papers:

* **GSM8K** has weak supervision from the final answer and a learned verifier.
* **Minerva** has benchmark-time correctness checks, but the model itself has no built-in proof checker.
* **LeanDojo** has intrinsic correctness checking because Lean itself validates each step. 

---

## Step-by-Step Technical Walkthrough

## 1. GSM8K / Training Verifiers to Solve Math Word Problems

### Stage 1: Build the dataset

Humans write grade-school word problems and natural-language solutions. The resulting dataset has 8.5K problems with strong quality control and high diversity. The goal is to make held-out test performance meaningful rather than letting models exploit repeated templates. 

### Stage 2: Train a generator

A GPT-3-family model is fine-tuned to produce natural-language solutions. The baseline uses standard language-model fine-tuning and then generates a single low-temperature solution at test time. 

### Stage 3: Generate many candidate solutions

For verification, the system first fine-tunes a generator for two epochs, then samples **100 completions per training problem** and labels each one as correct or incorrect based only on whether the final answer matches the gold answer. This is simple, but it introduces noise: a flawed reasoning chain can still get labeled correct if it lands on the right final answer. 

### Stage 4: Train a verifier

The verifier sees the original problem and a candidate solution and predicts the probability that the solution is correct. By default, the paper uses token-level verification, not just one final judgment after the whole solution. 

### Stage 5: Test-time ranking

At test time, the system samples **100 completions** for each test problem, scores them with the verifier, and returns the top-ranked one. The paper also studies letting the top-ranked candidates vote on the final answer, which can improve performance further as test-time compute increases. 

### Stage 6: Why each step exists

* The generator explores multiple reasoning paths.
* The verifier filters them.
* The final selection step turns generation into search-and-ranking rather than one-shot prediction.

This exists because one bad arithmetic or reasoning step can derail a single generated chain. 

### Main trade-offs

The method improves performance, but it costs more inference because many candidate solutions must be generated and scored. It also relies on weak correctness labels based only on final answers, which can reward accidental or flawed reasoning that happens to end correctly. 

---

## 2. Minerva

### Stage 1: Start from a strong general LM

Minerva begins from PaLM models with **8B, 62B, and 540B parameters**. These are already capable language models before any technical specialization. ([arXiv][1])

### Stage 2: Continue training on technical content

The models are further trained on a high-quality technical corpus built mainly from **arXiv papers** and **mathematical web pages**. The purpose is to expose the model to the notation, language, and structure of technical reasoning. ([arXiv][1])

### Stage 3: Prompt with worked examples

For evaluation, Minerva uses few-shot prompting. On MATH, for example, it uses a fixed **4-shot** prompt. The model then generates step-by-step reasoning in natural language plus mathematical notation, including valid LaTeX in many cases. ([arXiv][1])

### Stage 4: Sample multiple solutions

Rather than taking one greedy answer, Minerva samples many solutions and groups them by final answer. It then picks the most common one, which the paper calls **maj1@k**. The paper reports that majority voting saturates faster than pass@k and performs better than log-likelihood reranking in its comparisons. ([arXiv][1])

### Stage 5: Evaluate on several benchmark families

Minerva is evaluated on:

* **MATH**, a competition-style mathematics benchmark,
* **GSM8K**, school-style word problems,
* **MMLU-STEM**, a STEM-focused subset of MMLU,
* and **OCWCourses**, a curated set of **272 undergraduate-level STEM problems** from MIT OpenCourseWare. ([arXiv][1])

### Stage 6: What makes Minerva different

The paper is not mainly about a new verifier or a proof assistant. Its main idea is that **technical pretraining plus strong prompting plus test-time majority voting** can produce much better quantitative reasoning than a general language model baseline. ([arXiv][1])

### Main trade-offs

Minerva does not use external tools such as a calculator or Python interpreter, and the paper explicitly says it has **no automatic way to verify correctness**. That means it can still make subtle reasoning mistakes even when its explanations look strong. ([arXiv][1])

---

## 3. LeanDojo / ReProver

### Stage 1: Extract formal proof data

LeanDojo processes Lean code to extract things that are not directly obvious from raw source, such as proof trees, intermediate states, premises used in proofs, and other structured proof artifacts. It also enables programmatic interaction with Lean. 

### Stage 2: Represent the current proof state

A theorem proving episode starts from a formal theorem statement. Lean exposes the current goals and local context. This state is the input to the prover. 

### Stage 3: Retrieve relevant premises

Instead of searching the full math library blindly, ReProver retrieves likely useful premises. LeanDojo computes which premises are **accessible** to the current theorem, shrinking the candidate space substantially. The paper reports that while the benchmark has over 130K premises total, the average number of accessible premises is about **33,160**. 

### Stage 4: Concatenate state plus retrieved premises

The current proof state and retrieved premises are concatenated and fed to an encoder-decoder model, which then generates the next tactic. This turns theorem proving into a retrieval-augmented sequence generation problem. 

### Stage 5: Execute the tactic in Lean

Lean checks whether the tactic is valid and updates the proof state. If it fails, the tactic is rejected. If it succeeds, the proof may move closer to completion. This is the main difference from ordinary natural-language math reasoning: the environment itself checks each move. 

### Stage 6: Continue search until solved

The prover repeats this process until the theorem is proved or the search budget is exhausted. Because proof search is combinatorial, retrieval quality matters a lot. Premise selection is one of the major bottlenecks. 

### Main trade-offs

Formal theorem proving gives stronger correctness guarantees, but it is much more constrained than natural-language reasoning. Problems must be formalized in Lean, the model must work with tactics and proof states, and the search space is large. Retrieval helps, but theorem proving remains hard. 

---

## Paper-by-Paper Explanation

## 1. Training Verifiers to Solve Math Word Problems

### Problem addressed

Large language models were still weak on robust multi-step mathematical reasoning, even when the underlying concepts were simple grade-school arithmetic. The paper wanted both a better benchmark and a better method. 

### Method used

The paper introduces **GSM8K**, a dataset of 8.5K grade-school math word problems with natural-language solutions, and proposes **verification**: generate many candidate solutions and train a verifier to rank them. 

### Main innovation

The innovation is the combination of a carefully designed dataset and a generate-then-verify pipeline. The paper’s key claim is that verification scales better than pure fine-tuning and is more compute-efficient than simply making the generator much larger. 

### Main findings

The paper reports that verification gives a significant performance boost over fine-tuning alone, and in the conclusion says that on the full dataset **6B verification slightly outperforms a finetuned 175B model**, which the paper frames as roughly equivalent to a **30x** model size increase. It also finds that token-level verifiers overfit less than solution-level verifiers, and that residual dropout is an important regularizer. 

### Limitations

The verifier is trained from labels based only on whether the final answer is correct, so it can assign positive labels to flawed reasoning that happens to end at the right answer. The method also requires extra test-time compute because many candidate solutions must be generated and ranked. 

### What changed compared with earlier work

Instead of relying only on better generators, the paper argues for separating **generation** from **judgment**. That is a major conceptual move that influenced later reasoning systems. 

---

## 2. Minerva: Solving Quantitative Reasoning Problems with Language Models

### Problem addressed

General language models had improved a lot on language tasks but still struggled on mathematics, science, and engineering problems requiring quantitative reasoning. ([arXiv][1])

### Method used

Minerva starts from PaLM models and continues training them on a large, high-quality technical corpus built from arXiv and mathematical web pages. At inference time it uses worked prompts, sampling, and majority voting rather than greedy decoding. ([arXiv][1])

### Main innovation

The paper’s main contribution is showing that technical-data specialization plus simple but powerful inference-time techniques can turn a large LM into a much stronger quantitative reasoner without external tools. ([arXiv][1])

### Main findings

On the paper’s main table, **Minerva 540B with majority voting** reaches **50.3% on MATH**, **30.8% on OCWCourses**, **78.5% on GSM8K**, and **75.0% on MMLU-STEM**. The paper also says the model can answer nearly a third of over two hundred undergraduate-level STEM problems, and the curated OCWCourses set contains **272 problems**. ([arXiv][1])

### Limitations

The paper explicitly notes that it has **no automatic way to verify answers**, no access to external tools such as a calculator or Python interpreter, and limited direct control over exactly what capabilities the model acquires from broad training. ([arXiv][1])

### What changed compared with earlier work

Compared with GSM8K-style word-problem work, Minerva broadens the domain to more advanced mathematics and STEM reasoning and emphasizes domain-specific technical pretraining. ([arXiv][1])

---

## 3. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models

### Problem addressed

LLM-based theorem proving research had high barriers to entry because data, code, and training setups were often private and hard to reproduce. Also, premise selection remained a major bottleneck in proof generation. 

### Method used

LeanDojo provides an open-source toolkit, benchmark, and models for interaction with Lean. On top of that, the paper builds **ReProver**, a retrieval-augmented prover that retrieves relevant premises and then generates tactics conditioned on the current proof state plus those premises. 

### Main innovation

The paper’s innovations are broader than one model. It provides:

* a reliable Lean interaction environment,
* extracted proof data and premise annotations,
* a benchmark with challenging splits,
* and a retrieval-augmented prover that is cheap enough to train in about **one GPU week**. 

### Main findings

The benchmark contains **98,734 theorems and proofs**, **217,776 tactics**, and **129,243 premises**. On the LeanDojo Benchmark, ReProver achieves **51.2% Pass@1** on the random split and **26.3%** on the harder **novel_premises** split, compared with **29.0% / 7.4%** for GPT-4 and **47.6% / 23.2%** for the no-retrieval baseline. On MiniF2F it reaches **26.5% Pass@1**, and on ProofNet **13.8%**, which the paper says is the first reported result there. 

### Limitations

The task is still difficult, especially on challenging splits requiring generalization to novel premises. Formalization is also a major constraint: the theorem must already live inside Lean, and theorem proving is not directly comparable to free-form natural-language math solving. 

### What changed compared with earlier work

This paper pushes mathematical reasoning into a more rigorous and reproducible regime. It also shows that retrieval can matter as much in theorem proving as it does in question answering, because proofs depend heavily on the right library facts. 

---

## Comparison Across Papers or Methods

The comparison below synthesizes the three papers’ reported setups and results. 

| Aspect             | GSM8K / Verifiers                            | Minerva                                                      | LeanDojo                                                     |
| ------------------ | -------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Main task type     | Grade-school math word problems              | Broad quantitative reasoning                                 | Formal theorem proving                                       |
| Input format       | Natural-language word problem                | Natural-language technical problem                           | Formal theorem in Lean                                       |
| Output format      | Natural-language solution + final answer     | Natural-language reasoning + math notation + final answer    | Tactics / proof steps                                        |
| Core method        | Generate many candidates, rank with verifier | Technical pretraining + few-shot prompting + majority voting | Retrieval-augmented tactic generation inside Lean            |
| Correctness signal | Final-answer labels, learned verifier        | Final-answer evaluation, no built-in verifier                | Lean proof assistant checks each step                        |
| External tool use  | No formal tool; learned verifier only        | No external calculator/Python in main setup                  | Yes, Lean proof assistant                                    |
| Main strength      | Strong sample-and-rank reasoning baseline    | Strong broad quantitative reasoning without tools            | Rigorous, machine-checkable reasoning                        |
| Main weakness      | Weak labels can reward flawed reasoning      | No automatic verification                                    | Requires formalized problems and proof search infrastructure |

A second comparison that interviewers often like is this one: **what kind of “checking” does each system perform?** 

| Paper    | What is being checked?                     | How strong is the check? |
| -------- | ------------------------------------------ | ------------------------ |
| GSM8K    | Whether a candidate solution seems correct | Learned, approximate     |
| Minerva  | Whether the final benchmark answer matches | Benchmark-time only      |
| LeanDojo | Whether each proof step is formally valid  | Exact, symbolic          |

---

## Real-World System and Application

A practical AI system for mathematics could combine ideas from all three papers rather than choosing only one.

### Natural-language tutoring or homework support

For school math or ordinary word problems, a GSM8K-style approach makes sense: let a model generate multiple candidate explanations, then use a verifier or answer-consistency mechanism to choose the best one. This is useful when the input is free-form language and you want readable explanations. 

### Broader STEM assistant

For science, engineering, and technical quantitative reasoning, Minerva suggests a different recipe: train on much more technical content, preserve notation, and use test-time majority voting. This is useful when problems are still in natural language but much more advanced than grade-school arithmetic. ([arXiv][1])

### Formal verification or theorem proving tools

If the goal is not just “get a plausible answer” but “produce a proof the machine can certify,” then LeanDojo is closer to the right architecture. In that setting, the proof assistant is part of the runtime loop, and retrieval over a theorem library becomes essential. 

### A useful system-design insight

These papers suggest a hierarchy of reliability:

* natural-language math solving is flexible but weakly checked,
* broad technical reasoning is stronger with better data and aggregation,
* formal proving is the most reliable but also the most constrained. 

Information not provided: none of the three papers gives a full production design for a commercial math agent that seamlessly mixes natural-language tutoring, symbolic tools, theorem proving, and human supervision in one product. That broader system design is an inference beyond the papers. 

---

## Limitations and Trade-offs

The most important trade-offs are summarized below. 

| Limitation / trade-off                | GSM8K / Verifiers | Minerva                        | LeanDojo                                         |
| ------------------------------------- | ----------------- | ------------------------------ | ------------------------------------------------ |
| Free-form reasoning errors            | High risk         | High risk                      | Much lower, because Lean checks steps            |
| Need for many samples                 | Yes               | Yes, for majority voting       | Search cost appears in proof exploration instead |
| Automatic correctness checking        | Weak              | Weak                           | Strong                                           |
| Ease of use on ordinary text problems | High              | High                           | Low                                              |
| Formal rigor                          | Low               | Low                            | High                                             |
| Data / infrastructure demands         | Moderate          | Very high model and data scale | Significant tooling and formalization effort     |

A deeper conceptual trade-off is this:

* **GSM8K** gains flexibility by staying in natural language, but loses strong correctness guarantees.
* **Minerva** improves broad quantitative capability, but still cannot formally verify its answers.
* **LeanDojo** gains rigor by moving into a proof assistant, but only for problems that can be expressed formally. 

Another interview-worthy limitation is that **verification** is not the same across papers. In GSM8K, the verifier is itself a learned model and can be wrong. In LeanDojo, Lean’s checker is not “guessing”; it is enforcing the formal rules of the proof system. This is one of the most important distinctions in the whole report. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain:

1. the difference between **informal math problem solving** and **formal theorem proving**,
2. why GSM8K used a verifier instead of trusting a single generated solution,
3. why Minerva relied on technical pretraining and majority voting,
4. why LeanDojo needs retrieval over premises,
5. why proof assistants give stronger guarantees than natural-language reasoning,
6. and why “mathematical reasoning” covers several different problem settings rather than one benchmark family. 

### Likely interview questions

**What is GSM8K really about?**
It is both a benchmark and a method paper. The benchmark is a high-quality set of grade-school word problems, and the method is to generate many candidate solutions and use a verifier to rank them. 

**Why does verification help in GSM8K?**
Because judging whether a completed solution looks correct is often easier than generating the whole reasoning chain correctly from scratch. The paper shows verification can outperform a much larger fine-tuned generator baseline. 

**What is the difference between token-level and solution-level verification?**
Solution-level verification gives one score after the full solution. Token-level verification predicts correctness throughout the solution. The paper reports that token-level verification ultimately performs better and overfits less. 

**What is Minerva’s main contribution?**
It shows that large language models can get much stronger at quantitative reasoning when further trained on technical mathematical and scientific content and evaluated with sampling plus majority voting. ([arXiv][1])

**Why does majority voting help Minerva?**
Because correct solutions often converge on the same final answer while wrong solutions are more scattered. Sampling many solutions and picking the most common answer is therefore more reliable than trusting one sample. ([arXiv][1])

**What is LeanDojo in one sentence?**
It is an open-source toolkit, benchmark, and retrieval-augmented theorem proving framework for Lean, where LLMs generate proof tactics and Lean checks whether those tactics are valid. 

**Why does LeanDojo need retrieval?**
Because proofs depend on previously known lemmas and definitions from a huge library, and the model cannot reliably memorize or fit all relevant premises into context without selecting useful ones. 

**How is LeanDojo different from Minerva?**
Minerva solves natural-language quantitative problems and outputs natural-language reasoning. LeanDojo works in a formal proof environment where each proof step must satisfy Lean. ([arXiv][1])

**Which of these papers gives the strongest correctness guarantee?**
LeanDojo, because the proof assistant checks each step formally. GSM8K and Minerva mostly rely on answer correctness, learned scoring, or benchmark evaluation rather than full symbolic proof checking. 

### Concise model answers

| Question                                           | Plain-English answer                                                                                 |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Why is GSM8K hard if it only uses elementary math? | Because the challenge is multi-step reasoning in natural language, not advanced math formulas.       |
| What is the big idea in Minerva?                   | Train on technical math/science text and use many sampled solutions instead of one greedy answer.    |
| What is the big idea in LeanDojo?                  | Use retrieval plus a proof assistant so the model reasons inside a system that can check every step. |
| Which paper is closest to formal verification?     | LeanDojo.                                                                                            |
| Which paper is most about answer ranking?          | GSM8K / verifiers.                                                                                   |
| Which paper is most about domain-specific scaling? | Minerva.                                                                                             |

These answers are syntheses grounded in the three papers. 

---

## Glossary

* **Accessible premises:** In LeanDojo, the subset of library facts that are available to the current theorem and therefore sensible retrieval candidates. 
* **Arithmetic reasoning:** Solving problems mainly using basic operations like addition, subtraction, multiplication, and division.
* **ByT5:** The encoder-decoder Transformer family used in ReProver for tactic generation. 
* **Chain of reasoning:** The sequence of intermediate steps used to reach a final answer.
* **Formal theorem proving:** Proving theorems in a precise symbolic system where proof steps can be checked mechanically. 
* **Generator:** The model that produces candidate solutions or tactics.
* **Grade-school math word problem:** A story problem stated in everyday language, as in GSM8K. 
* **Lean:** A proof assistant used for formal theorem proving. 
* **Majority voting:** Generate many outputs and choose the most common final answer. ([arXiv][1])
* **MATH:** A benchmark of middle-school and high-school competition-style math problems used heavily in Minerva. ([arXiv][1])
* **MMLU-STEM:** The STEM subset of the MMLU benchmark, covering science, technology, engineering, and mathematics. ([arXiv][1])
* **OCWCourses:** Minerva’s curated undergraduate-level STEM problem set from MIT OpenCourseWare. ([arXiv][1])
* **Pass@1:** The fraction of tasks solved on the first attempt. LeanDojo uses this as a main evaluation metric. 
* **Premise selection:** Choosing useful prior theorems, lemmas, or definitions for the current proof step. 
* **Proof assistant:** Software that checks whether a formal proof is valid. 
* **ReProver:** LeanDojo’s retrieval-augmented prover. 
* **Tactic:** A command in Lean that transforms the current proof state. 
* **Token-level verifier:** A verifier that predicts correctness throughout a generated solution rather than only after the final token. 
* **Verifier:** A model that scores whether a candidate solution is likely correct. In GSM8K this is learned rather than formal. 

---

## Recap

These three papers show that “mathematical reasoning” in LLMs spans a spectrum.

At one end, **GSM8K** studies multi-step natural-language arithmetic and shows that **generate-then-verify** can outperform stronger single-pass generation. In the middle, **Minerva** shows that technical pretraining plus majority voting can push language models much further on quantitative reasoning across math and STEM. At the most formal end, **LeanDojo** shows how to combine retrieval and LLM generation with a proof assistant so reasoning is machine-checkable rather than just plausible. 

The most important thing to remember for interviews is that these papers are not interchangeable examples of one generic “math model.” They represent three different design philosophies:

* **rank natural-language solutions,**
* **scale technical reasoning in text,**
* **or move reasoning into a formal environment with exact checking.** 

What remains limited or uncertain is also important. GSM8K’s verifier is still only a learned judge. Minerva still lacks automatic verification and external tools in its main setup. LeanDojo is rigorous, but only after the problem is formalized in Lean and proof search becomes tractable enough to handle. Those limits are exactly what make the three papers complementary rather than redundant. 

---

## Key Citations

[Training Verifiers to Solve Math Word Problems](https://arxiv.org/pdf/2110.14168)

[Solving Quantitative Reasoning Problems with Language Models](https://arxiv.org/pdf/2206.14858)

[LeanDojo: Theorem Proving with Retrieval-Augmented Language Models](https://arxiv.org/pdf/2306.15626)

[1]: https://arxiv.org/pdf/2206.14858 "Solving Quantitative Reasoning Problems With Language Models"


---
---
---

