# Hallucination and Factuality: Measuring, Detecting, and Mitigating Errors in LLMs

## What This Report Teaches

This report explains three papers that tackle the same broad problem from three different angles:

1. **TruthfulQA** asks how to **measure** whether a language model gives true answers rather than fluent falsehoods.
2. **SelfCheckGPT** asks how to **detect** hallucinations after a model has already generated text, even when you only have black-box access to the model.
3. **DoLa** asks how to **reduce** hallucinations during generation itself by changing the decoding procedure. 

A practical note matters here: the URL you provided for DoLa, `2305.14251`, points to a different paper, **FACTSCORE**, not to DoLa. I therefore used the actual DoLa paper, **arXiv:2309.03883**, for the report. 

By the end of this report, you should understand what “hallucination” and “factuality” mean in practice, why these are not exactly the same thing, how a benchmark differs from a detector, how a detector differs from a generation-time mitigation method, and how to explain the trade-offs clearly in interviews. 

---

## Key Takeaways

* **These three papers solve different parts of the same problem.** TruthfulQA is mainly a benchmark, SelfCheckGPT is mainly a detector, and DoLa is mainly a decoding-time mitigation method. This matters because “hallucination work” is not one single technique category. In practice, strong systems usually need some combination of measurement, detection, and mitigation. 

* **TruthfulQA shows that fluency is not the same as truthfulness.** The benchmark was built to trigger “imitative falsehoods,” where a model repeats common human misconceptions instead of giving the literal truth. This matters because a model can sound confident and helpful while being wrong. In practice, product teams need evaluation sets that target common misconceptions, not just easy factual trivia. 

* **Bigger models were not automatically more truthful in TruthfulQA.** The paper reports an “inverse scaling” trend in which larger models in several families were generally less truthful on this benchmark, even though they did better on control trivia questions. This matters because raw scale alone does not solve factuality. In practice, scaling compute is not a substitute for alignment, retrieval, or better decoding. 

* **SelfCheckGPT’s core idea is simple and interview-worthy: factual answers tend to stay consistent across multiple samples, while hallucinated details tend to vary or contradict each other.** This matters because it gives a black-box detection method that does not require token probabilities or an external knowledge base. In practice, it is useful when you only have API access to a model. 

* **DoLa assumes factual knowledge is expressed more strongly in later transformer layers than in earlier ones.** It contrasts the final-layer token distribution with an earlier-layer distribution so that tokens supported more by higher layers are emphasized. This matters because it tries to improve factuality without retrieval or fine-tuning. In practice, it is an inference-time intervention rather than a retraining method. 

* **All three papers show that “factuality” is not just about knowledge storage.** TruthfulQA suggests models may imitate false beliefs from human text, SelfCheckGPT shows that models can expose their own uncertainty through inconsistent samples, and DoLa suggests useful factual signals may already exist inside the model but are not always surfaced by standard decoding. This matters because the problem is partly about how knowledge is represented and surfaced, not just whether it exists somewhere in the weights. 

* **No single paper gives a complete solution.** TruthfulQA does not fix hallucinations, SelfCheckGPT does not guarantee correction, and DoLa improves factuality but still depends on model internals and specific decoding assumptions. In practice, real systems still need layered defenses. 

---

## Background and Foundations

### What is a hallucination?

In LLM practice, a **hallucination** usually means the model generates content that is false, unsupported, or not grounded in reality or in the provided evidence. The papers use related but slightly different language. TruthfulQA talks about **false statements** and especially **imitative falsehoods**. SelfCheckGPT talks about **non-factual statements** and hallucinated facts. DoLa defines hallucinations as generated content that **deviates from facts seen during pretraining**. 

### What is factuality?

**Factuality** means how well the model’s output matches what is true in the real world or in a trusted knowledge source. TruthfulQA uses a strict real-world standard of truth, similar in spirit to scientific or Wikipedia-like standards. SelfCheckGPT uses human sentence-level factuality judgments on generated passages. DoLa evaluates factuality using benchmarks such as TruthfulQA and FACTOR. 

### Hallucination and factuality are related, but not identical

A useful interview distinction is:

* **Hallucination** is often used for the *failure event*: the model says something unsupported or false.
* **Factuality** is often used for the *evaluation goal*: how correct the output is overall.

So a paper can target hallucination detection, factuality measurement, or factuality improvement, and those are not the same thing. That distinction is exactly what separates these three papers. 

### Benchmark, detector, mitigation

| Role              | Plain-English meaning                                     | Paper here   |
| ----------------- | --------------------------------------------------------- | ------------ |
| Benchmark         | A test set that tells you how well a model behaves        | TruthfulQA   |
| Detector          | A method that flags likely false content after generation | SelfCheckGPT |
| Mitigation method | A method that changes generation to reduce false content  | DoLa         |

This is the single most important framing for the whole topic. 

### White-box, grey-box, and black-box access

SelfCheckGPT is easier to understand if you know these access levels:

| Access type | What you can see                      | Why it matters                                      |
| ----------- | ------------------------------------- | --------------------------------------------------- |
| White-box   | Internal states, activations, weights | Strongest access, least realistic for external APIs |
| Grey-box    | Token probabilities or entropy        | Useful, but not always exposed                      |
| Black-box   | Only text outputs                     | Most realistic for many API users                   |

SelfCheckGPT is designed for the black-box setting, while DoLa is not black-box because it requires access to intermediate layer outputs during decoding. 

---

## Big Picture First

The three papers line up into a full mental model:

1. **First, measure the problem.**
   TruthfulQA creates a benchmark designed to expose cases where models repeat common falsehoods instead of the truth. 

2. **Then, detect the problem at runtime.**
   SelfCheckGPT says: ask the model the same question multiple times with stochastic sampling. If the answers stay mutually consistent, that is evidence for factuality; if they diverge or contradict each other, that is evidence for hallucination. 

3. **Then, try to reduce the problem during generation.**
   DoLa changes the next-token calculation by contrasting later and earlier layers, based on the idea that factual knowledge is expressed more strongly in higher layers. 

That gives a clean pipeline view:

| Stage      | Main question                                           | Paper        |
| ---------- | ------------------------------------------------------- | ------------ |
| Evaluation | How do we know a model is lying or mistaken?            | TruthfulQA   |
| Detection  | Can we tell which generated sentences are likely wrong? | SelfCheckGPT |
| Mitigation | Can we make decoding itself produce fewer wrong facts?  | DoLa         |

This progression is conceptually neat because it moves from **measurement**, to **diagnosis**, to **intervention**. 

---

## Core Concepts Explained

### Imitative falsehoods

This is a key TruthfulQA idea. An **imitative falsehood** is a false answer that the model gives because similar falsehoods are common in its training distribution. The paper’s claim is not merely that models “make mistakes.” It is that models can become *better imitators of human falsehoods* as they get larger. That is why the benchmark uses questions designed to trigger myths, misconceptions, conspiracies, legal misconceptions, and fictional assumptions. 

### Truthfulness versus informativeness

TruthfulQA makes an important distinction:

* A model can be **truthful but uninformative**, for example by saying “I have no comment.”
* A model can be **informative but false**, which is more dangerous because it sounds useful while being wrong.

The paper explicitly treats truthfulness and informativeness as loosely analogous to precision and recall, and it evaluates both. This matters because a model can increase apparent safety by refusing to answer, but that does not mean it is genuinely useful. 

### Inverse scaling

**Inverse scaling** means performance gets worse as the model gets larger on a particular benchmark. TruthfulQA reports this trend for several model families on its benchmark, even though larger models did better on control trivia questions that did not probe misconceptions. This matters because it suggests the benchmark is exposing a different failure mode than simple lack of knowledge. 

### Consistency as a signal for factuality

SelfCheckGPT is based on a simple but powerful intuition: when a model “knows” a fact robustly, independent sampled responses should say roughly compatible things; when it hallucinates, the invented details should vary more from sample to sample. That turns repeated sampling into a no-knowledge-base factuality signal. 

### Zero-resource and black-box detection

SelfCheckGPT calls itself **zero-resource** because it does not require retrieving facts from an external database, and **black-box** because it can work when only text outputs are available. That is a practical contribution. Many hallucination detectors assume a retriever, a knowledge source, or token probabilities. SelfCheckGPT tries to avoid all of those assumptions. 

### Natural Language Inference (NLI)

One SelfCheckGPT variant uses **Natural Language Inference (NLI)**. NLI is a task where a model decides whether one statement is supported by, contradicted by, or unrelated to another. In SelfCheckGPT, each sampled passage is treated as context, and the target sentence is checked against it. A higher contradiction score means the sentence is more suspicious. 

### Layerwise knowledge in transformers

DoLa relies on the idea that different transformer layers contain different kinds of information. The paper cites prior work suggesting lower layers are more linguistic or surface-level, while later layers carry more semantic or factual content. DoLa then uses the gap between an early-layer prediction and the final-layer prediction as a decoding signal. 

### Decoding

**Decoding** is the rule used to choose the next token from the model’s internal probability distribution. Standard decoding just uses the final layer. DoLa changes this by constructing a new next-token distribution from the contrast between the final layer and a selected earlier layer. That makes DoLa an **inference-time** method, not a retraining method. 

---

## Step-by-Step Technical Walkthrough

## 1. TruthfulQA pipeline

### Goal

Measure whether models answer truthfully when asked questions that tempt them toward common misconceptions. 

### Workflow

1. **Construct hard questions**
   The benchmark contains **817 questions across 38 categories**. These categories include areas like health, law, finance, politics, conspiracies, fiction, and misconceptions. The questions are designed so that some humans would answer them falsely due to popular myths or bad beliefs. 

2. **Ask models in zero-shot form**
   Models are prompted to answer without retrieval and without being told the category. 

3. **Score truthfulness**
   An answer is counted as truthful if it avoids asserting a false statement. Importantly, evasive answers like “I have no comment” can count as truthful under the benchmark’s definition. 

4. **Score informativeness**
   The benchmark also asks whether the answer was informative rather than just safe or empty. 

5. **Compare generation and multiple-choice settings**
   The paper evaluates both natural language generation and multiple-choice variants to check whether the effect is robust. 

### Output

A truthfulness profile for each model, plus informativeness and multiple-choice results. 

### Why this step exists

The paper wants a benchmark that goes beyond ordinary QA or trivia accuracy. If the model merely copies what many humans commonly say, that may still be false. 

### Trade-offs and limitations

TruthfulQA is a benchmark, not a fix. It measures one important failure mode, but it does not itself improve the model. Also, because truthful non-answers can count as correct, the benchmark needs informativeness as a second axis. 

---

## 2. SelfCheckGPT pipeline

### Goal

Detect likely hallucinations in generated text when you only have black-box access to the model and no external knowledge base. 

### Workflow

1. **Generate an initial answer or passage**
   In the paper’s setup, GPT-3 generates Wikipedia-style passages for WikiBio concepts. The evaluation set contains **238 generated passages**, **1,908 sentences**, and about **184.7 ± 36.9 tokens per passage** on average. 

2. **Sample multiple additional responses to the same prompt**
   The method draws more stochastic outputs from the same model for the same query. 

3. **Compare the original text against the sampled texts**
   The paper proposes several ways to do this:

   * **BERTScore-based similarity**
   * **QA-based inconsistency**
   * **n-gram language modeling on the sampled texts**
   * **NLI contradiction scoring**
   * **Prompt-based support checking** with another strong LLM asking whether a sentence is supported by a sampled passage. 

4. **Assign sentence-level hallucination scores**
   Each sentence gets a score reflecting how inconsistent it is with the sampled alternatives. Higher inconsistency means more likely non-factuality. 

5. **Aggregate to passage-level factuality if needed**
   Passage-level scores are computed by averaging sentence-level scores. 

### Output

A sentence-level or passage-level hallucination estimate. 

### Why this step exists

Many real users cannot inspect token probabilities or run retrieval-based fact-checking against every possible topic. SelfCheckGPT tries to infer factuality from the model’s own self-consistency. 

### Trade-offs and limitations

The method costs extra inference because you need multiple samples. It also relies on the assumption that false content will be less consistent across samples, which is plausible but not guaranteed. If a model consistently repeats the same false belief, self-consistency could miss it. The paper does not claim this is a universal proof of truth. 

---

## 3. DoLa pipeline

### Goal

Reduce hallucinations during generation itself without retrieval and without extra fine-tuning. 

### Workflow

1. **Run the model as usual up to all transformer layers**
   The model produces hidden states at each layer and the standard final-layer next-token distribution. 

2. **Compute early-exit distributions from candidate earlier layers**
   The same vocabulary head is applied to selected earlier-layer hidden states, producing earlier next-token distributions. 

3. **Choose a “premature layer” dynamically**
   For each decoding step, DoLa picks the earlier layer whose token distribution differs the most from the final layer, using **Jensen-Shannon divergence (JSD)** as the distance measure. The idea is that this layer is where the model has not yet incorporated the final factual signal. 

4. **Contrast mature and premature predictions**
   DoLa builds a new next-token score by taking the log probability from the final layer and subtracting the log probability from the earlier layer. In plain English: tokens that become much stronger in later layers get promoted. 

5. **Restrict to plausible tokens**
   The method uses an **adaptive plausibility constraint (APC)** so it only contrasts among tokens that already have sufficiently high final-layer probability. This avoids promoting absurd low-probability tokens just because layer differences are noisy. 

6. **Decode with the new distribution**
   The model then generates text using this contrasted distribution rather than the normal final-layer distribution. The paper also adds a repetition penalty because the contrastive distribution can otherwise repeat text more often, especially in longer reasoning outputs. 

### Output

A new generation that is intended to better surface the model’s factual knowledge. 

### Why this step exists

The paper argues that the model may already contain the right fact, but standard decoding may not surface it strongly enough. So the method changes *how the answer is read out* from the model rather than changing the model’s weights. 

### Trade-offs and limitations

DoLa needs access to intermediate layers, so it is not black-box. It also depends on the assumption that later layers are the right place to emphasize factual knowledge, which may not hold equally for every task or language. The paper itself studies layer selection carefully because the best contrast range depends on task type. 

---

## Paper-by-Paper Explanation

## 1. TruthfulQA: Measuring How Models Mimic Human Falsehoods

### Problem addressed

Standard QA benchmarks do not target the specific failure mode where a model gives fluent but false answers that mirror popular misconceptions. TruthfulQA is designed to probe exactly that. 

### Method used

The paper constructs a benchmark of **817 questions in 38 categories** designed to trigger false beliefs. It evaluates both generation and multiple-choice performance, and it measures both **truthfulness** and **informativeness**. It also introduces an automated evaluation model, **GPT-judge**, trained on human truth labels. 

### Main innovation

The core innovation is benchmark design. The paper does not simply ask factual trivia. It deliberately asks questions where the common human answer is often wrong, so the benchmark distinguishes genuine truthfulness from mere imitation of internet text. 

### Main findings

The paper reports that the best-performing model in its main human-evaluated generation setup, **GPT-3-175B with a helpful prompt**, was truthful on **58%** of questions, while **human performance was 94%**. It also reports that this model produced answers that were both **false and informative 42%** of the time, versus **6%** for humans. Across several model families, larger models were generally less truthful on TruthfulQA, even though they did better on control trivia questions. 

### Limitations

Because truthful non-answers can count as correct, the benchmark needs informativeness as a second axis. Also, the benchmark measures a specific class of truthfulness failures and is not a complete theory of hallucination. The paper itself notes that some false answers may come from non-imitative weaknesses rather than imitative ones. 

### What changed compared with earlier work

The major shift is from generic QA accuracy toward targeted measurement of deceptive or misconception-shaped falsehoods. That is why TruthfulQA became influential: it operationalized a failure mode people worried about but had not benchmarked well. 

### Reasoned interpretation

TruthfulQA matters because it shows that “knowing a lot” and “answering truthfully” are not the same capability. A model can improve in fluency and still get worse at resisting culturally common falsehoods. 

### Information not provided

The paper measures truthfulness but does not provide a direct runtime detector or a decoding-time fix. 

---

## 2. SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection

### Problem addressed

Many hallucination detectors need either token probabilities, internal states, or an external knowledge base. This paper asks whether hallucinations can be detected using only text outputs from a black-box model. 

### Method used

SelfCheckGPT samples multiple responses to the same prompt and compares them for consistency. The paper evaluates several variants, including BERTScore, QA-based comparison, n-gram modeling, NLI contradiction scoring, and prompt-based support checking. It tests sentence-level and passage-level detection on GPT-3-generated WikiBio passages with human factuality annotations. 

### Main innovation

The main innovation is the self-consistency principle for factuality detection in a **zero-resource black-box** setting: do not ask “what external source says this?” first; ask “does the model tell a stable story when sampled multiple times?” 

### Main findings

On sentence-level non-factual detection, the paper reports strong AUC-PR results for SelfCheckGPT variants, with **SelfCheck-Prompt at 93.42** and **SelfCheck-NLI at 92.50** in one main setting shown in Table 3 comparisons, while also noting that NLI offers a better performance-computation trade-off. For passage-level factuality ranking, SelfCheckGPT methods correlate much better with human judgments than the black-box and grey-box baselines in the paper’s comparisons; the paper highlights **SelfCheckGPT-Prompt** as the best-performing passage-level method there, with **Pearson correlation 78.32**. It also shows performance generally improves as more samples are drawn, with diminishing returns. 

### Limitations

SelfCheckGPT adds sampling cost. It is also still a detector, not a direct fixer. Most importantly, the method can struggle if the model is consistently wrong in the same way across samples, because consistency is then no longer a good proxy for truth. The paper also notes that using external stored knowledge can sometimes help NLI and prompt-based variants even more, but that leaves the zero-resource setting. 

### What changed compared with earlier work

The paper moves hallucination detection away from retrieved evidence or privileged model access and toward a practical black-box sampling approach. That is a major engineering shift. 

### Reasoned interpretation

SelfCheckGPT is important because it treats the model as a noisy witness whose credibility can be estimated from repeated testimony. That is a simple mental model interviewers often like. 

### Information not provided

The paper does not claim a universal threshold or universal best variant for all domains, nor does it solve correction after detection. 

---

## 3. DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models

### Problem addressed

Even when a model may contain factual knowledge internally, standard decoding may not surface it reliably. The paper asks whether changing the readout from different layers can improve factuality without retrieval or fine-tuning. 

### Method used

DoLa computes next-token distributions from both the final layer and candidate earlier layers. It dynamically picks a **premature layer** using JSD against the final layer, then contrasts the mature and premature distributions in log space, while restricting decoding to plausible final-layer tokens with an adaptive plausibility constraint. 

### Main innovation

The core innovation is an inference-time decoding rule based on **intra-model layer contrast**, rather than model-to-model contrast, retrieval, or finetuning. That is what makes DoLa distinct. 

### Main findings

The paper reports consistent improvements on TruthfulQA multiple-choice metrics and open-ended truthfulness for LLaMA models. For example, on open-ended TruthfulQA, **LLaMA-7B** goes from **30.4% truthfulness** to **42.1%** with DoLa, **13B** from **38.8%** to **48.8%**, **33B** from **62.5%** to **56.4% truthfulness but much higher informativeness and far lower rejection**, and **65B** from **50.2%** to **54.3%**; the paper emphasizes the combined **Truth × Info** metric, where DoLa improves by **12–17 percentage points** across four models while keeping rejection under 10%. It also reports improvements on FACTOR, StrategyQA, GSM8K, and GPT-4-judged Vicuna QA comparisons. 

### Limitations

DoLa is not black-box and depends on model internals. It also needs decisions about candidate layer buckets, and the best layer range can vary by task: the paper reports higher layers are preferred for short-answer factuality tasks like TruthfulQA-MC, while lower layers can be better for reasoning tasks like StrategyQA and GSM8K. The paper also introduces a repetition penalty because the contrasted distribution can otherwise become repetitive in long-form decoding. 

### What changed compared with earlier work

Instead of adding external evidence or training new controllers, DoLa changes only decoding. That makes it one of the cleanest examples of an **inference-time factuality intervention**. 

### Reasoned interpretation

DoLa is a strong example of the idea that the model may “know” more than standard decoding reveals. The intervention is not new knowledge injection; it is a different way to surface existing knowledge. 

### Information not provided

The paper does not show that the layerwise factuality hypothesis is universally true across all architectures, languages, or domains. 

---

## Comparison Across Papers or Methods

The comparison below synthesizes the three papers. 

| Aspect                         | TruthfulQA                                    | SelfCheckGPT                                         | DoLa                                                                 |
| ------------------------------ | --------------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| Main role                      | Benchmark                                     | Detector                                             | Mitigation method                                                    |
| Main question                  | How truthful is the model?                    | Which generated content is likely hallucinated?      | Can decoding surface more factual outputs?                           |
| When used                      | Evaluation time                               | After generation                                     | During generation                                                    |
| Access needed                  | Model outputs                                 | Black-box text outputs only                          | Internal layer outputs                                               |
| Needs external knowledge base? | No                                            | No, in its zero-resource form                        | No                                                                   |
| Main signal                    | Truth labels on misconception-heavy questions | Cross-sample consistency or contradiction            | Difference between later-layer and earlier-layer token distributions |
| Main output                    | Truthfulness and informativeness scores       | Sentence-level or passage-level hallucination scores | New decoded text with improved factuality                            |
| Main limitation                | Measures but does not fix                     | Detects but does not directly correct                | Improves generation but is model-internal and assumption-heavy       |

Another useful comparison is by failure stage. 

| Failure stage                | Best-matching paper | Why                                                                             |
| ---------------------------- | ------------------- | ------------------------------------------------------------------------------- |
| Before deployment            | TruthfulQA          | It helps you evaluate whether the model is prone to misconception-shaped errors |
| During monitoring            | SelfCheckGPT        | It helps you flag suspicious outputs when you only have API access              |
| At inference-time generation | DoLa                | It alters the decoder to reduce factual errors before they appear               |

And the deepest conceptual split is this. 

| Underlying belief                                                   | Paper        |
| ------------------------------------------------------------------- | ------------ |
| Models often copy false things humans say                           | TruthfulQA   |
| Models reveal uncertainty through inconsistent sampling             | SelfCheckGPT |
| Models may store factual knowledge that standard decoding underuses | DoLa         |

---

## Real-World System and Application

A practical LLM system concerned with factuality could combine ideas from all three papers in a layered way:

1. **Use a benchmark like TruthfulQA before deployment** to see whether the model tends to reproduce myths, misconceptions, or confident falsehoods in sensitive domains. 

2. **Use a detector like SelfCheckGPT at runtime** when retrieval is unavailable or too expensive. If the model’s sampled outputs disagree strongly, the system can flag the answer, lower confidence, ask a clarification question, or trigger retrieval or human review. The paper directly supports sentence-level and passage-level factuality scoring in this spirit. 

3. **Use decoding interventions like DoLa** for models where you control inference internals. This can improve factuality even before any post-hoc check, and the paper shows gains on factuality and some reasoning tasks without retrieval or fine-tuning. 

That leads to a practical architecture view:

| Layer in a real system            | Example tool from these papers | Purpose                             |
| --------------------------------- | ------------------------------ | ----------------------------------- |
| Offline evaluation                | TruthfulQA                     | Know your model’s failure profile   |
| Online suspicion scoring          | SelfCheckGPT                   | Detect likely hallucinations        |
| Inference-time generation control | DoLa                           | Reduce hallucinations before output |

Information not provided: these papers do not present one full production architecture for enterprise safety, citation-grounded RAG, moderation, or long-horizon tool-using agents. They are focused narrowly on truthfulness benchmarking, black-box detection, and inference-time decoding. 

---

## Limitations and Trade-offs

The summary below synthesizes the main trade-offs across the three papers. 

| Trade-off                    | TruthfulQA                                              | SelfCheckGPT                                | DoLa                                                                |
| ---------------------------- | ------------------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------- |
| Solves the problem directly? | No, only measures                                       | No, only detects                            | Partly, improves generation                                         |
| Runtime cost                 | Low during inference, but evaluation is offline work    | Higher, because multiple samples are needed | Moderate, because extra layer computations and selection are needed |
| Access requirement           | Just outputs                                            | Black-box outputs                           | Internal layer access                                               |
| Best strength                | Exposes misconception-shaped falsehoods                 | Practical black-box detection               | Clean inference-time factuality improvement                         |
| Biggest weakness             | Benchmark success may not cover all real-world failures | Consistent falsehoods can slip through      | Depends on layerwise assumptions and decoding control               |

A few concrete failure modes are especially worth remembering:

* **TruthfulQA** can reward non-answers as truthful unless you also track informativeness, so it should not be interpreted as a pure utility metric. 
* **SelfCheckGPT** can fail when a model is confidently and consistently wrong across multiple samples. Its main assumption is variability of hallucinations, not guaranteed truth discovery. 
* **DoLa** can require careful layer-range choices and uses safeguards like APC and repetition penalties because raw contrast can otherwise overpromote strange tokens or cause repetition. 

The broad lesson is that factuality is a systems problem. You need to know **how to evaluate it**, **how to monitor it**, and **how to improve it**, and each paper only addresses one part of that stack. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain:

1. why fluent text can still be false,
2. what an imitative falsehood is,
3. why TruthfulQA is different from ordinary QA benchmarks,
4. how SelfCheckGPT uses sampling consistency as a factuality signal,
5. why SelfCheckGPT is attractive in black-box API settings,
6. how DoLa changes decoding using earlier and later layers,
7. and why benchmarking, detection, and mitigation are different levels of solution. 

### Likely interview questions

**What is the difference between hallucination and factuality?**
Hallucination is the failure event: the model says something false or unsupported. Factuality is the evaluation goal: how correct the output is overall. TruthfulQA mostly measures truthfulness, SelfCheckGPT detects likely hallucinations, and DoLa tries to improve factuality during decoding. 

**What is TruthfulQA really testing?**
It tests whether a model will answer truthfully on questions designed to trigger common human misconceptions, not just whether it can answer easy factual trivia. 

**Why was TruthfulQA surprising?**
Because larger models were generally less truthful on the benchmark across several model families, even though larger models usually improve on many NLP tasks. 

**How does SelfCheckGPT work in one sentence?**
It samples multiple answers from the same model and treats disagreement or contradiction among those samples as evidence that the original answer may contain hallucinations. 

**Why is SelfCheckGPT useful in production?**
Because it can work in black-box settings where you do not have token probabilities, internal activations, or a domain-specific retrieval system. 

**What is DoLa in one sentence?**
DoLa is a decoding method that contrasts final-layer token probabilities with earlier-layer token probabilities so that tokens strengthened in later layers are emphasized, with the aim of improving factuality. 

**Why is DoLa interesting compared with retrieval?**
Because it does not add external documents or fine-tune the model. It tries to surface factual knowledge already present in the model by changing the decoding rule. 

**What is the biggest limitation of SelfCheckGPT?**
If the model is wrong in the same way every time, consistency no longer distinguishes truth from falsehood. 

**What is the biggest limitation of DoLa?**
It requires internal layer access and depends on the assumption that factual knowledge is better reflected in higher-layer predictions. 

**How would you summarize the progression across the three papers?**
TruthfulQA measures the problem, SelfCheckGPT detects it after generation, and DoLa tries to reduce it during generation. 

### Concise model answers in plain English

| Interview question                                                           | Good plain-English answer                                                                                                 |
| ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Why can large models hallucinate even if they were trained on a lot of text? | Because training on text teaches models to imitate patterns in text, and text contains both truths and common falsehoods. |
| What is the key idea behind SelfCheckGPT?                                    | Truth tends to be stable across multiple samples; made-up details tend to vary more.                                      |
| What is the key idea behind DoLa?                                            | Higher layers may express factual knowledge more strongly than earlier layers, so contrast them during decoding.          |
| Why is TruthfulQA important?                                                 | It shows that sounding good and being truthful are different things, and scaling alone may not fix that.                  |

These are syntheses grounded in the three papers. 

---

## Glossary

* **Adaptive Plausibility Constraint (APC):** In DoLa, a rule that restricts contrastive decoding to tokens that already have reasonably high probability in the final layer, so absurd low-probability tokens are not accidentally boosted. 
* **Black-box model access:** You can query the model and read its text output, but you cannot inspect internals like activations or logits. SelfCheckGPT is designed for this setting. 
* **Decoding:** The process of choosing the next token from the model’s probability distribution. DoLa modifies this process. 
* **Factuality:** How well generated text matches the truth or a trusted knowledge source. 
* **Grey-box access:** You do not see full internals, but you do get token probabilities or entropy values. SelfCheckGPT compares itself against grey-box baselines. 
* **Hallucination:** A generated statement that is false, unsupported, or not grounded in the relevant facts. 
* **Imitative falsehood:** A false answer that is likely high-probability because it imitates common falsehoods in the training distribution. TruthfulQA is built to target these. 
* **Informativeness:** Whether an answer actually provides useful information instead of being safely vague or evasive. TruthfulQA measures this separately from truthfulness. 
* **Inverse scaling:** A pattern where larger models perform worse on a benchmark. TruthfulQA reports this for truthfulness in several model families. 
* **Jensen-Shannon Divergence (JSD):** A measure of how different two probability distributions are. DoLa uses it to choose the earlier layer most different from the final layer at each step. 
* **Natural Language Inference (NLI):** A task that predicts whether one statement supports, contradicts, or is neutral with respect to another. SelfCheckGPT uses contradiction scores as a hallucination signal. 
* **Premature layer / mature layer:** In DoLa, the earlier layer used for contrast and the final layer used as the reference, respectively. 
* **Truthfulness:** Whether an answer avoids making false statements. TruthfulQA uses this as its main target. 
* **Zero-resource detection:** Detection without relying on an external database or retrieval source. SelfCheckGPT uses this term for its black-box setup. 

---

## Recap

These three papers give a very clean teaching arc for hallucination and factuality in LLMs.

**TruthfulQA** shows that large language models can confidently repeat human falsehoods and that raw scale alone does not guarantee truthfulness. **SelfCheckGPT** shows that repeated sampling can expose some hallucinations even in black-box settings, because factual answers tend to stay more self-consistent than invented ones. **DoLa** shows that you can sometimes improve factuality at inference time by contrasting earlier and later transformer layers during decoding. 

The most important interview-level lesson is this:

> **Factuality is not one problem with one fix. It has to be measured, monitored, and improved at different points in the pipeline.** 

What remains uncertain is equally important. These papers do not prove that inconsistency always implies falsehood, that later layers always contain the “truth,” or that one benchmark can capture all factuality risks. But together they give a strong conceptual toolkit: benchmark the problem carefully, detect it when you can, and improve decoding when you control the model internals. 

---

## Key Citations

[TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/pdf/2109.07958)

[SELFCHECKGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/pdf/2303.08896)

[DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models](https://arxiv.org/pdf/2309.03883)


---
---
---

