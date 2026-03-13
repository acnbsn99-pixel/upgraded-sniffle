# Reinforcement Learning from Human Feedback (RLHF): From Human Preferences to Summarization to InstructGPT

## What This Report Teaches

This report explains **Reinforcement Learning from Human Feedback (RLHF)** as a sequence of ideas that became increasingly important for modern AI systems. It starts with a robotics-and-games paper that learns rewards from human preferences, moves to a language-model paper that uses human feedback to improve summarization, and ends with the InstructGPT paper that turned this recipe into a practical method for making large language models follow instructions more helpfully. ([arXiv][1])

By the end, you should understand the core RLHF pipeline, why it exists, how reward models are trained, why methods like **supervised fine-tuning (SFT)** and **proximal policy optimization (PPO)** matter, what the main trade-offs are, and how to explain the whole system in an AI engineer or AI architect interview. ([arXiv][2])

---

## Key Takeaways

* **RLHF is a way to train a system when the “right behavior” is easier for humans to recognize than to write down as a reward function.** This matters because many useful tasks do not have a clean numeric objective. In practice, RLHF replaces hand-written rewards with a learned reward model based on human judgments. ([arXiv][1])

* **The basic RLHF loop is: generate candidates, collect human preferences, train a reward model, then optimize the policy or language model against that reward model.** This matters because all three papers use this same skeleton, even though the domain changes from robot control to text generation. In practice, this is the core mental model interviewers expect you to know. ([arXiv][1])

* **The 2017 paper showed that preference-based reward learning can scale to deep reinforcement learning with surprisingly little direct human feedback.** This matters because it established that humans do not need to label every step; they can compare short clips instead. In practice, it introduced the scalable idea of learning a reward function from preferences rather than using raw human feedback directly at every timestep. ([arXiv][1])

* **The 2020 summarization paper adapted RLHF to language models and showed that optimizing for human preference can outperform optimizing for standard proxy metrics like ROUGE.** This matters because it exposed a key alignment lesson: training loss and human quality are not the same thing. In practice, it made RLHF a serious method for text generation, not just simulated control tasks. ([arXiv][2])

* **InstructGPT added an important warm-start stage: first collect demonstrations and train an SFT model, then collect rankings, train a reward model, and finally do PPO.** This matters because starting RL from a reasonable supervised model is far more stable than starting from scratch. In practice, this three-stage recipe became the widely recognized RLHF pipeline for instruction-following language models. ([arXiv][3])

* **A learned reward model can be exploited.** This matters because a model can learn to score highly under the reward model without actually doing what humans want. In practice, this is why the papers use safeguards such as online data collection, KL penalties, reward-model regularization, and careful evaluation. ([arXiv][1])

* **RLHF aligns a model to the preferences of a particular labeling process, not to all humans in all contexts.** This matters because “aligned” does not mean universally correct, fair, or safe. In practice, interview answers that mention this limitation sound much more mature and realistic. ([arXiv][3])

* **Better alignment can beat raw scale on user preference, but it can also introduce an “alignment tax” on other benchmarks.** This matters because alignment and general capability do not always move together. In practice, system design often needs balancing terms such as pretraining-mix updates or other regularization to preserve capabilities while improving behavior. ([arXiv][3])

---

## Background and Foundations

### Why RLHF exists

In standard **reinforcement learning (RL)**, an agent interacts with an environment and receives a numeric **reward** after actions. The agent’s **policy** is the rule it learns for choosing actions. This works well when the reward function is clear and machine-readable, such as game score. But many real tasks are not like that. A robot cleaning a table, a system writing a summary, or a chatbot following instructions all involve goals that humans understand but cannot fully specify with a clean formula. The 2017 paper frames this as the central problem: humans may be able to **recognize** good behavior without being able to **demonstrate** it or encode it as a reward function. ([arXiv][1])

### Why human comparisons are attractive

The papers use **pairwise preference data**: show a human two options and ask which one is better. This is easier than asking for a perfect demonstration in a hard control task, and often easier than asking for an absolute numeric score. The 2017 paper uses comparisons between short trajectory clips; the 2020 paper uses comparisons between summaries; the 2022 paper uses rankings among several model outputs for the same prompt. The common idea is that humans are often more reliable at saying “A is better than B” than at writing a reward function or producing a globally optimal example. ([arXiv][1])

### Why language models needed RLHF

A **language model** is trained to predict the next token in text. That objective is useful, but it is not the same as “be helpful, honest, harmless, and follow instructions.” The summarization paper argues that maximum-likelihood supervised training and metrics like **ROUGE** are only rough proxies for what humans actually care about in summary quality. The InstructGPT paper makes the broader point: next-token prediction on internet text is different from following user instructions helpfully and safely. ([arXiv][2])

### How the three papers relate

Historically and conceptually, the three papers form a progression:

1. **2017:** Learn a reward from human preferences in deep RL tasks such as Atari and MuJoCo.
2. **2020:** Apply the same logic to text summarization by training a reward model on human summary comparisons and optimizing a language model with PPO.
3. **2022:** Generalize the recipe to broad instruction-following using demonstrations, rankings, reward modeling, and PPO fine-tuning of GPT-3. ([arXiv][1])

---

## Big Picture First

The simplest mental model is this:

1. Start with a model or policy that can already produce some candidate behaviors.
2. Ask humans which outputs they prefer.
3. Train a separate model, called a **reward model**, to imitate those preferences.
4. Improve the original model so it produces outputs that score highly under the reward model.
5. Use constraints and fresh data so the model does not “game” the learned reward. ([arXiv][1])

### Same skeleton, different domains

| Paper                                              | Domain                      | Human signal                         | What gets optimized                           | Why it mattered                                                                                   |
| -------------------------------------------------- | --------------------------- | ------------------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Deep Reinforcement Learning from Human Preferences | Atari + MuJoCo control      | Preference between short video clips | RL policy using predicted reward              | First clear scalable version of preference-based reward learning in deep RL. ([arXiv][1])         |
| Learning to summarize from human feedback          | Text summarization          | Preference between summaries         | GPT-style summarizer using reward model + PPO | Showed RLHF works for language generation and beats proxy metrics in human judgment. ([arXiv][2]) |
| InstructGPT                                        | Broad instruction following | Demonstrations + ranked outputs      | GPT-3 family using SFT + reward model + PPO   | Turned RLHF into a practical alignment recipe for general-purpose assistants. ([arXiv][3])        |

### What changed across the papers

The deepest change is not the reward model itself. It is the **initialization and domain**.

* In 2017, the system learns behavior in an RL environment and gets human comparisons on clips. ([arXiv][1])
* In 2020, the “environment” becomes text generation, the whole summary gets one reward, and PPO is used with a KL penalty to keep the model near a supervised baseline. ([arXiv][2])
* In 2022, the pipeline begins with demonstrations to create a strong SFT model before PPO, and the data distribution broadens from one task to many instruction-following tasks from API prompts. ([arXiv][3])

The big picture is that RLHF evolved from “learn a reward from preference” into “train a useful assistant by combining supervised demonstrations, learned preferences, and RL fine-tuning.” ([arXiv][1])

---

## Core Concepts Explained

### 1. Preference data

**What it is:** Human judgments that say which of two or more outputs is better.
**Why it exists:** Writing a perfect reward function is hard. Comparing options is easier.
**How it works:** The system shows a human candidate behaviors or texts, and the human chooses the better one.
**Where it appears:** All three papers.
**Why it matters:** This is the supervision signal that replaces hand-written reward design. ([arXiv][1])

### 2. Reward model

**What it is:** A model trained to predict which output a human would prefer.
**Why it exists:** Human feedback is expensive, so you need a learned approximation that can score many outputs cheaply.
**How it works at a high level:** Give the reward model two candidate outputs; it assigns scores, and the score difference predicts which one a human would choose. In the 2017 paper, the reward over a clip is the sum of per-step predicted rewards. In the language papers, the reward model scores the full generated summary or response.
**Where it appears:** All three papers.
**Why it matters:** It converts sparse human comparisons into a reusable training signal. ([arXiv][1])

### 3. Policy optimization

**What it is:** Updating the model so it produces outputs with higher reward-model scores.
**Why it exists:** A reward model alone does not improve the generator; you still need an optimizer.
**How it works at a high level:** In 2017, standard RL algorithms optimize the predicted reward. In 2020 and 2022, PPO is used to fine-tune a language model.
**Where it appears:** All three papers, though the specific optimizer differs.
**Why it matters:** This is the step that turns preference information into actual behavioral change. ([arXiv][1])

### 4. PPO

**What it is:** **Proximal Policy Optimization**, an RL algorithm that updates a policy in relatively controlled steps.
**Why it exists here:** PPO is a practical way to optimize a model against a reward signal without making destabilizing changes all at once.
**How it works at a high level:** Generate outputs, score them, compute gradients that increase the probability of higher-reward outputs, and restrict updates so the policy does not move too abruptly.
**Where it appears:** The 2020 summarization paper and InstructGPT.
**Why it matters:** In language RLHF, PPO became the standard optimizer for the “RL” stage. ([arXiv][2])

### 5. KL penalty

**What it is:** A term that penalizes the new policy for drifting too far from a reference model, usually the supervised model. **KL divergence** is a mathematical measure of how different two probability distributions are.
**Why it exists:** Without it, the model can exploit weaknesses in the reward model and move into strange regions where reward-model scores are high but human quality is low.
**How it works at a high level:** Final reward = reward-model score minus a penalty for moving too far from the reference model.
**Where it appears:** The 2020 paper and InstructGPT.
**Why it matters:** It is one of the main stability and anti-reward-hacking tools in RLHF. ([arXiv][2])

### 6. Supervised fine-tuning (SFT)

**What it is:** Training the model directly on human demonstrations of desired outputs.
**Why it exists:** RL works much better if it starts from a model that already does something reasonable.
**How it works at a high level:** Collect prompt-response pairs from humans and fine-tune the base language model to imitate them.
**Where it appears:** Implicitly as the starting point in the 2020 paper and explicitly as Step 1 in InstructGPT.
**Why it matters:** It provides a strong starting policy and a stable reference model for KL regularization. ([arXiv][2])

### 7. Online versus batch data collection

**What it is:** **Online** means collecting feedback on the current model as it changes; **batch** means collecting a large static dataset and training from it.
**Why it exists:** The data distribution changes as the model improves.
**How it works at a high level:** Online collection better tracks the current policy. Batch collection is operationally simpler but can lag behind the model.
**Where it appears:** The 2017 paper emphasizes the importance of online queries; the 2020 summarization paper adapts the method to a batch setting; InstructGPT uses mostly supervised-policy data plus some PPO-policy data.
**Why it matters:** This directly affects reward-model exploitation and generalization. ([arXiv][1])

### 8. Alignment tax

**What it is:** The loss in other benchmark performance that can happen when you fine-tune for alignment or preference.
**Why it exists:** Making a model more aligned to a particular human objective can move it away from behaviors that maximize benchmark scores.
**How it works at a high level:** RLHF shifts the model’s behavior toward what labelers reward, which is not identical to broad next-token modeling capability.
**Where it appears:** Most clearly in InstructGPT, where PPO caused regressions on some public NLP tasks and PPO-ptx tried to reduce those regressions by mixing in pretraining gradients.
**Why it matters:** In production systems, alignment and raw benchmark capability often need to be balanced. ([arXiv][3])

---

## Step-by-Step Technical Walkthrough

### Step 1: Start from a model or policy that can generate candidates

**Inputs:**

* 2017: a policy interacting with an environment.
* 2020: a GPT-style summarization model fine-tuned on TL;DR summaries.
* 2022: a pretrained GPT-3 model plus human demonstrations for SFT. ([arXiv][1])

**What happens:**
The system needs candidate behaviors before humans can judge anything. In language tasks, this is why SFT is so important: it gets the model into a regime where the outputs are already relevant enough to compare. ([arXiv][2])

**Outputs:**
Candidate clips, summaries, or responses. ([arXiv][1])

**Purpose:**
Human judges cannot usefully rank garbage forever. A reasonable starting policy makes the feedback much more informative. ([arXiv][3])

**Trade-off:**
A stronger initial model improves stability, but it can also anchor the final behavior around the starting model’s biases. Information not provided: the papers do not give a general theory for how strong the starting model must be; they report empirical choices. ([arXiv][2])

### Step 2: Collect preference data

**Inputs:**
Candidate outputs for the same task or prompt. ([arXiv][1])

**What happens:**

* 2017: humans compare short video clips of trajectory segments, usually 1–2 seconds long. Queries can be chosen using ensemble disagreement. Contractors answered an average query in 3–5 seconds. ([arXiv][1])
* 2020: humans compare summaries from the current policy, initial policy, references, and baselines. The task is to choose the better summary of a Reddit post. The released dataset contains 64,832 summary comparisons. ([arXiv][2])
* 2022: labelers rank 4 to 9 responses for a prompt, which yields many pairwise comparisons from a single ranking task. The training data includes about 13k SFT prompts, 33k RM prompts, and 31k PPO prompts. ([arXiv][3])

**Outputs:**
A dataset of human preferences or rankings. ([arXiv][2])

**Purpose:**
This is the signal that says what “better” means for the current task. ([arXiv][1])

**Trade-offs:**
Pairwise data is easier for humans, but it is still expensive and noisy. In 2022, most comparisons were labeled once, which the authors note as a limitation. ([arXiv][3])

### Step 3: Train a reward model from preferences

**Inputs:**
A prompt or context, candidate outputs, and a label showing which one humans preferred. ([arXiv][1])

**What happens:**
The reward model learns scores such that the preferred output gets a higher score than the rejected one. In plain English, the training rule says: “raise the score of the output humans liked, lower the score of the one they did not.”

* In 2017, the model sums predicted per-step rewards over a trajectory segment and uses a softmax-like comparison. ([arXiv][1])
* In 2020, the reward model predicts which summary is better, using the score difference between two summaries for the same post. ([arXiv][2])
* In 2022, the same pairwise logic is used, but rankings of 4–9 responses are converted efficiently into many comparisons per prompt. ([arXiv][3])

**Outputs:**
A reward model that can score new clips or text outputs without needing a human for each one. ([arXiv][1])

**Purpose:**
This amortizes human judgment. A small amount of human feedback becomes a large amount of machine-usable reward. ([arXiv][1])

**Trade-offs:**
A reward model is only an approximation. If it is wrong, optimizing it can make the generator worse, not better. The 2020 paper explicitly shows that over-optimizing against the reward model can eventually make it anti-correlated with human preference. ([arXiv][2])

### Step 4: Optimize the model against the reward model

**Inputs:**
The current policy or language model and the learned reward model. ([arXiv][1])

**What happens:**

* In 2017, the policy is updated by standard RL methods to maximize predicted reward. Atari uses A2C; MuJoCo uses TRPO. ([arXiv][1])
* In 2020 and 2022, PPO is used for language generation. A generated text gets a reward from the reward model, and the language model is updated so that outputs like that become more likely. ([arXiv][2])

**Outputs:**
A better policy or assistant model, according to the reward model. ([arXiv][3])

**Purpose:**
This is where human preferences actually change behavior at scale. ([arXiv][2])

**Trade-offs:**
The optimization can be too weak, in which case the model barely improves, or too strong, in which case it exploits the reward model. The 2020 paper shows this sharply, and the 2017 paper shows bizarre behavior when reward learning is done offline rather than online. ([arXiv][2])

### Step 5: Keep the model from drifting too far

**Inputs:**
A reference model, usually the SFT model. ([arXiv][2])

**What happens:**
A KL penalty subtracts reward when the RL-updated policy moves too far from the reference policy. In InstructGPT, PPO-ptx adds another stabilizer by mixing in gradients from the pretraining distribution. ([arXiv][2])

**Outputs:**
A model that improves on human preference while retaining more of the original language-model behavior. ([arXiv][3])

**Purpose:**
Prevent collapse, reward hacking, and capability loss. ([arXiv][2])

**Trade-offs:**
Too little regularization leads to exploitation; too much regularization blocks improvement. PPO-ptx reduces but does not eliminate benchmark regressions. ([arXiv][2])

### Step 6: Iterate

**Inputs:**
A better current policy. ([arXiv][3])

**What happens:**
Collect new human data on the improved model, retrain the reward model, and continue. The 2017 paper shows that online feedback helps because the policy’s behavior distribution changes over time. The 2022 paper says steps 2 and 3 can be iterated continuously, though in practice much of their comparison data still came from supervised policies. ([arXiv][1])

**Outputs:**
An RLHF loop instead of a one-shot training recipe. ([arXiv][1])

**Purpose:**
The current model produces the most relevant failures for humans to correct. ([arXiv][1])

**Trade-offs:**
Iteration improves relevance but adds operational complexity, cost, and data-management demands. Information not provided: the papers do not give a universal stopping rule for when RLHF should end. ([arXiv][3])

---

## Paper-by-Paper Explanation

## 1. Deep Reinforcement Learning from Human Preferences

### Problem addressed

The paper asks how to solve RL tasks when the true reward is unavailable or hard to specify, but humans can still tell which behavior they prefer. It targets settings where demonstrations may be difficult and where user feedback must be economical. ([arXiv][1])

### Method used

The method maintains a policy and a learned reward predictor at the same time. The policy generates trajectories. The system selects pairs of short segments from those trajectories and asks a human which segment is better. A reward predictor is trained on those comparisons, and the policy is updated to maximize predicted reward. The three processes run asynchronously. ([arXiv][1])

### Main innovation

The innovation is not just “use human feedback.” It is **learn a reward model from pairwise preferences and use that model inside large-scale deep RL**, with practical tricks such as ensembles, query selection by disagreement, and online collection. That made the method usable for Atari and MuJoCo rather than only very small domains. ([arXiv][1])

### Main findings

The paper reports that the approach can solve Atari and simulated robotics tasks without observing the environment reward, using feedback on less than 1% of the agent’s interactions. It also reports that non-expert humans, giving between about 15 minutes and 5 hours of feedback, were enough for many original tasks, and that some novel behaviors such as a Hopper backflip could be trained with around an hour of feedback. ([arXiv][1])

### Limitations

The learned reward can be exploited. The paper’s ablations show that removing online queries hurts performance because the policy distribution changes over time. They also note that uncertainty-based query selection is only a crude approximation and can sometimes impair performance. ([arXiv][1])

### What changed compared with earlier work

Compared with prior preference-learning work, this paper moves from small or feature-engineered settings to deep RL with nonlinear reward models and modern RL algorithms. It also uses short clips rather than whole trajectories, making the feedback process faster. ([arXiv][1])

### Directly stated facts

* Uses Atari and MuJoCo tasks. ([arXiv][1])
* Uses A2C for Atari and TRPO for robotics. ([arXiv][1])
* Uses clip comparisons and Bradley-Terry-style preference fitting. ([arXiv][1])

### Reasoned interpretation

This paper is the conceptual seed of modern RLHF: it proves that human judgments can supervise a learned reward at scale, which is exactly the core idea later reused in language models. ([arXiv][1])

### Information not provided

The paper does not provide a general theory of when preference learning will beat demonstrations, or a universal recipe for selecting the best query strategy in all domains. ([arXiv][1])

---

## 2. Learning to summarize from human feedback

### Problem addressed

The paper argues that supervised summarization and ROUGE optimization do not directly optimize what humans care about: actual summary quality. It therefore asks whether a language model can be trained to produce summaries that humans prefer, rather than summaries that only match references or score well on ROUGE. ([arXiv][2])

### Method used

The system starts from a GPT-style supervised summarizer trained on the Reddit TL;DR dataset. It collects human comparisons between summaries, trains a reward model to predict which summary humans prefer, and then uses PPO to optimize the summarizer against that reward model, with a KL penalty to keep it near the supervised model. ([arXiv][2])

### Main innovation

This paper shows that RLHF can work for **language generation**, not just control. It also makes several practical design choices that later mattered a lot: using a supervised baseline to initialize policy and reward model, using a KL penalty, using a separate value function from the policy, and studying over-optimization explicitly. ([arXiv][2])

### Main findings

The paper reports that human-feedback-trained models outperform much larger supervised models on TL;DR and are preferred even to the original human reference summaries in the dataset. It also reports transfer to CNN/DailyMail news summarization without news-specific fine-tuning, with the 6.7B TL;DR human-feedback model nearly matching T5 at similar summary lengths on CNN/DM. On TL;DR quality axes, the 6.7B PPO model receives a 7/7 overall score 45% of the time, versus 20% for the 6.7B supervised baseline and 23% for reference summaries. ([arXiv][2])

### Limitations

The most important limitation is reward-model over-optimization. The paper shows that optimizing the reward model initially improves summaries, but eventually harms true human preference and can even become anti-correlated with it. This is a central warning for all later RLHF work. The paper also notes that batch data collection differs from the online setting of earlier work. ([arXiv][2])

### What changed compared with earlier work

Compared with the 2017 paper, the domain changes from trajectories in environments to token sequences in text. The reward is now an overall summary-quality signal. The paper also releases a large public preference dataset with 64,832 comparisons and analyzes how reward-model performance scales with more data and bigger models. ([arXiv][2])

### Directly stated facts

* Uses GPT-style Transformer decoder models with 1.3B and 6.7B parameters. ([arXiv][2])
* Uses a filtered Reddit TL;DR dataset with a summary-length constraint of 24–48 tokens, and defines quality by faithful communication of the post to a reader who only sees the summary. ([arXiv][2])
* Uses a KL-regularized PPO objective and a separate value function. ([arXiv][2])

### Reasoned interpretation

This paper is the bridge between classical preference-based RL and modern LLM alignment. It is the first paper in the sequence where the RLHF recipe looks recognizably like the recipe later used for assistants. ([arXiv][2])

### Information not provided

The paper does not describe a production deployment architecture for summarization systems beyond the training and evaluation pipeline. ([arXiv][2])

---

## 3. InstructGPT / Training language models to follow instructions

### Problem addressed

The paper asks how to make GPT-3 follow a broad range of written instructions more helpfully, truthfully, and harmlessly than the base language model. It targets real API prompts rather than a single benchmark task. ([arXiv][3])

### Method used

The method has three stages:

1. Collect human demonstrations and train an SFT model.
2. Collect comparison data on model outputs and train a reward model.
3. Optimize the SFT model against the reward model using PPO.

The paper also studies **PPO-ptx**, which mixes pretraining gradients into PPO to reduce capability regressions on public NLP datasets. ([arXiv][3])

### Main innovation

The main innovation is turning RLHF into a practical instruction-following pipeline over a broad prompt distribution. It also broadens the human-feedback setup beyond pairwise-only summarization by using demonstrations first, using rankings of 4–9 outputs for reward modeling, and evaluating on real API prompts. ([arXiv][3])

### Main findings

The paper reports that labelers significantly prefer InstructGPT outputs over GPT-3 outputs across model sizes. In particular, 175B InstructGPT outputs are preferred to GPT-3 outputs 85 ± 3% of the time and to few-shot GPT-3 outputs 71 ± 4% of the time. It also reports that a 1.3B PPO-ptx model is preferred to 175B GPT-3 outputs, showing that alignment can beat raw model size on user preference. InstructGPT also improves truthfulness and reduces hallucination in some settings, shows small toxicity improvements, but does not significantly improve bias metrics over GPT-3. ([arXiv][3])

### Limitations

The paper is unusually explicit about limitations. The labelers are not the users. The labeler pool is small, mostly English-oriented, and not representative of everyone affected by the model. Most comparisons are labeled once. The models are not fully aligned or safe: they can still generate toxic, biased, sexual, violent, or fabricated content, and they often follow harmful user instructions unless additional refusal behavior is trained. PPO also causes regressions on some public NLP datasets, and PPO-ptx only partly fixes them. ([arXiv][3])

### What changed compared with earlier work

Compared with the 2020 summarization paper, InstructGPT broadens the task from one domain to many user instructions, adds an explicit demonstration stage as the first step, and studies alignment on live-style API prompts. It is less about one task doing better than baselines and more about training a generally instruction-following assistant. ([arXiv][3])

### Directly stated facts

* Uses about 40 contractors from Upwork and ScaleAI, with a screening process for sensitive prompts and ranking quality. ([arXiv][3])
* Uses about 13k SFT prompts, 33k RM prompts, and 31k PPO prompts; the data is over 96% English. ([arXiv][3])
* Uses 6B reward models because 175B reward-model training was unstable and more expensive. ([arXiv][3])

### Reasoned interpretation

This is the paper where RLHF becomes the recognizable alignment stack for LLM assistants: pretrain, SFT, reward model, PPO, and guard against drift with KL and pretraining mix. ([arXiv][3])

### Information not provided

The paper does not claim that the resulting system is aligned to universal human values, nor does it provide a complete solution for safety-critical refusal behavior. ([arXiv][3])

---

## Comparison Across Papers or Methods

| Aspect             | 2017 Preference RL                                   | 2020 Summarization RLHF                                  | 2022 InstructGPT                                     |
| ------------------ | ---------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------- |
| Main goal          | Learn behavior without a hand-coded reward           | Improve summaries according to humans, not proxy metrics | Make GPT-3 follow instructions better                |
| Human input        | Clip comparisons                                     | Summary comparisons                                      | Demonstrations + rankings/comparisons                |
| Starting point     | RL policy in environment                             | Supervised summarizer                                    | Pretrained GPT-3, then SFT                           |
| Reward model input | Trajectory segments                                  | Post + summary                                           | Prompt + completion                                  |
| Optimizer          | A2C / TRPO on predicted reward                       | PPO with KL penalty                                      | PPO and PPO-ptx with KL + pretraining mix            |
| Key strength       | Proves scalable reward learning from preferences     | Brings RLHF into language generation                     | Practical assistant-style alignment recipe           |
| Key weakness       | Reward-model exploitation and online-data dependence | Reward over-optimization                                 | Narrow labeler alignment, safety gaps, alignment tax |

The table above is a synthesis of the three papers’ methods and reported trade-offs. ([arXiv][1])

### Comparison of optimization choices

| Method                       | What it does                                  | Strength                                           | Weakness                                                        |
| ---------------------------- | --------------------------------------------- | -------------------------------------------------- | --------------------------------------------------------------- |
| Supervised fine-tuning (SFT) | Imitates human demonstrations                 | Stable starting point; easy to train               | Learns to mimic data, not necessarily maximize human preference |
| Reward model only            | Predicts human preference                     | Converts sparse human labels into cheap scores     | Does not improve behavior by itself                             |
| PPO against reward model     | Optimizes behavior for predicted preference   | Directly improves outputs under learned preference | Can exploit the reward model                                    |
| PPO + KL penalty             | Optimizes while staying near reference policy | Better stability and less reward hacking           | Too much KL can limit gains                                     |
| PPO-ptx                      | Adds pretraining gradients during PPO         | Helps preserve general capabilities                | Does not fully remove regressions                               |

This table combines the language-model findings from the 2020 summarization paper and InstructGPT. ([arXiv][2])

---

## Real-World System and Application

### Supported directly by the papers

A practical RLHF system has at least five parts:

1. **A base model or policy** that can generate candidate behavior.
2. **A data-collection interface** where humans compare outputs or provide demonstrations.
3. **A reward-model training pipeline** that turns those judgments into a scoring model.
4. **A policy-optimization pipeline** that improves the generator using the reward model.
5. **An evaluation loop** using held-out humans, metadata, or task-specific checks to see whether the model is actually improving. ([arXiv][1])

### Reasoned interpretation

For an AI assistant product, you can think of the pipeline like this:

1. Deploy a base assistant or an SFT version internally.
2. Log representative prompts.
3. Ask trained labelers to write ideal answers and rank sampled model answers.
4. Train a reward model to imitate those rankings.
5. Fine-tune the assistant with PPO while keeping it near the SFT model.
6. Re-sample difficult prompts from the current model and repeat.
7. Track side effects such as hallucination, toxicity, and benchmark regressions. ([arXiv][3])

### Information not provided

The papers do **not** provide a complete production architecture for:

* serving infrastructure,
* latency management,
* monitoring dashboards,
* policy rollout procedures,
* human-review escalation systems,
* post-training safety filters.

Those are important in real systems, but the papers mainly describe the **training and evaluation loop**, not the entire deployment stack. ([arXiv][2])

---

## Limitations and Trade-offs

| Limitation or trade-off              | Concrete meaning                                                                         | Why it matters                                                                                                                                               |
| ------------------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Reward hacking / over-optimization   | The model learns to score well under the reward model without actually satisfying humans | This is the central failure mode of RLHF and explains the need for KL penalties, online data, and careful evaluation. ([arXiv][1])                           |
| Expensive human data                 | Preferences and demonstrations require trained humans                                    | RLHF scales better than direct human reward at every step, but human data is still costly and limited. ([arXiv][1])                                          |
| Narrow alignment target              | Models align to a particular set of labelers and instructions                            | “Aligned” behavior depends on who the labelers are and what rules they were given. ([arXiv][3])                                                              |
| Helpfulness vs harmlessness conflict | A model can be helpful in following instructions that are unsafe or dishonest            | InstructGPT explicitly says training prioritized helpfulness in some stages, while final evaluations prioritized truthfulness and harmlessness. ([arXiv][3]) |
| Mostly English data                  | The instruction data is overwhelmingly English                                           | Cross-lingual or culturally broader alignment remains limited. ([arXiv][3])                                                                                  |
| Alignment tax                        | Better preference alignment can reduce performance on some public NLP tasks              | Preserving general capability while improving behavior is an engineering challenge. ([arXiv][3])                                                             |
| Safety remains incomplete            | Models can still hallucinate, produce bias, or generate harmful content                  | RLHF improves behavior, but it is not a full safety solution. ([arXiv][3])                                                                                   |

A good interview answer should emphasize that RLHF is powerful precisely because it lets humans define quality indirectly, but that same indirection creates approximation errors, labeler-dependence, and reward-exploitation risk. ([arXiv][1])

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain that RLHF exists because many important tasks lack a clean reward function, so humans provide preference judgments instead. Those preferences train a reward model, and the main model is then optimized against that learned reward, usually starting from an SFT model and using KL regularization so it does not drift too far or exploit the reward model. You should also be able to explain that this aligns the model to a particular feedback process, not to universal truth or safety. ([arXiv][1])

### Likely interview questions and concise model answers

1. **What is RLHF?**
   RLHF is a training approach where humans judge outputs, a reward model learns to predict those judgments, and the base model is then optimized to produce outputs that the reward model scores highly. It is used when the desired behavior is easier to recognize than to specify as code. ([arXiv][1])

2. **Why use comparisons instead of numeric ratings?**
   Comparisons are often easier and more reliable for humans. The 2017 paper explicitly chose clip comparisons over absolute scores, and the later language papers also center the pipeline on pairwise or ranked preferences. ([arXiv][1])

3. **What is the reward model actually learning?**
   It is learning a scoring function whose score differences match human preferences. In plain English, it tries to assign higher scores to outputs humans tend to choose. It is not learning “ground truth morality”; it is learning the preference pattern in the labeling data. ([arXiv][1])

4. **Why do we need SFT before PPO in modern LLM RLHF?**
   Because PPO needs a reasonable starting policy. SFT gives the model a good initial behavior and also provides the reference policy used in the KL penalty. InstructGPT made this explicit as the first stage of the pipeline. ([arXiv][3])

5. **Why is KL regularization important?**
   Because the model can otherwise move into strange regions where the reward model is wrong and easy to exploit. The KL term keeps the RL-updated model near the supervised model, which stabilizes training and reduces reward hacking. ([arXiv][2])

6. **What is the biggest technical risk in RLHF?**
   Reward-model exploitation. The summarization paper shows that pushing optimization too hard can make actual human preference go down even while reward-model score goes up. ([arXiv][2])

7. **What is the difference between the 2020 summarization paper and InstructGPT?**
   The summarization paper applies RLHF to one language task. InstructGPT generalizes the recipe to broad instruction following, adds demonstrations as an explicit first stage, uses broader prompt data from API usage, and studies helpfulness, truthfulness, harmlessness, and benchmark regressions. ([arXiv][2])

8. **What does “alignment tax” mean here?**
   It means that making the model better aligned to human preferences can hurt some general benchmark performance. InstructGPT observed this and introduced PPO-ptx to reduce the regressions by mixing in pretraining gradients. ([arXiv][3])

9. **Why is RLHF not the same as “human values alignment”?**
   Because the model is trained on feedback from a specific labeler group under specific instructions. The InstructGPT paper is explicit that this is alignment to a particular reference group, not a claim of universal alignment. ([arXiv][3])

---

## Glossary

| Term                         | Beginner-friendly meaning                                                                                                               |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| Reinforcement Learning (RL)  | A learning setup where a system takes actions, receives rewards, and tries to learn behavior that gets more reward.                     |
| Policy                       | The model or rule that decides what action or output to produce next.                                                                   |
| Reward function              | The scoring rule that says how good an action or output is.                                                                             |
| Trajectory segment           | A short slice of behavior over time, such as a short video clip of an agent acting.                                                     |
| Preference comparison        | A human judgment choosing which of two outputs is better.                                                                               |
| Reward model (RM)            | A learned model that predicts which output humans would prefer.                                                                         |
| Supervised Fine-Tuning (SFT) | Training the model directly on human-written examples of desired behavior.                                                              |
| PPO                          | A practical RL algorithm used to fine-tune a policy in controlled steps.                                                                |
| KL divergence                | A measure of how far one probability distribution is from another; in RLHF it is used to keep the updated model near a reference model. |
| ROUGE                        | An automatic summarization metric based mainly on overlap with reference text; useful, but only a proxy for human judgment.             |
| Hallucination                | A model output that invents facts not supported by the input or reality.                                                                |
| Alignment                    | Making model behavior better match intended human goals or preferences.                                                                 |
| Alignment tax                | The loss in some general capabilities or benchmark scores that can happen when you optimize heavily for alignment.                      |
| PPO-ptx                      | In InstructGPT, PPO with an added pretraining-loss term to help preserve general capabilities.                                          |
| Helpful, honest, harmless    | A practical framing in the InstructGPT paper for what a better-aligned assistant should do.                                             |

The paper-specific glossary items above come directly from the methods and framing used in the 2020 and 2022 language-model papers. ([arXiv][2])

---

## Recap

You should now see RLHF as one continuous idea with three stages of historical development. The 2017 paper showed that humans can teach complex behavior by comparing short clips, without directly writing rewards. The 2020 paper showed that the same preference-learning logic works for language summarization and can outperform standard proxy metrics. The 2022 InstructGPT paper turned that into the now-familiar recipe of **SFT → reward model → PPO**, using real instruction data and careful human evaluation. ([arXiv][1])

The most important concepts to retain are these: RLHF uses human preference data because exact rewards are hard to specify; the reward model is only an approximation and can be exploited; SFT gives a stable starting point; PPO changes model behavior to maximize learned reward; KL regularization and related tricks keep the model from drifting too far; and the final behavior reflects a particular feedback process rather than universal human truth. ([arXiv][1])

What remains limited or uncertain is also important: the papers do not provide a full production safety architecture, they do not solve universal alignment, and they do not eliminate bias, hallucination, harmful instruction following, or reward hacking. That realism is part of understanding RLHF well. ([arXiv][3])

---

## Key Citations

[Deep Reinforcement Learning from Human Preferences](https://arxiv.org/pdf/1706.03741)

[Learning to summarize from human feedback](https://arxiv.org/pdf/2009.01325)

[Training language models to follow instructions with human feedback](https://arxiv.org/pdf/2203.02155)

[1]: https://arxiv.org/pdf/1706.03741 "https://arxiv.org/pdf/1706.03741"
[2]: https://arxiv.org/pdf/2009.01325 "https://arxiv.org/pdf/2009.01325"
[3]: https://arxiv.org/pdf/2203.02155 "https://arxiv.org/pdf/2203.02155"



---
---
---


