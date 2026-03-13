# Offline Reinforcement Learning: From Conservative Q-Learning to Decision Transformer and Trajectory Transformer

## What This Report Teaches

This report explains three influential papers that attack the same problem - **offline reinforcement learning (offline RL)** - from two very different angles.

**Offline RL** means learning a decision-making policy from a fixed dataset of past experience, without collecting new interaction data during training. That is appealing in robotics, recommendation systems, healthcare, and other settings where online trial-and-error is expensive, dangerous, or impossible. But it is also hard: once the learner proposes actions that are not well supported by the dataset, its value estimates can become badly wrong. ([arXiv][1])

The three papers form a useful progression:

1. **Conservative Q-Learning (CQL)** is a value-based offline RL method. It stays inside the classical RL toolbox and tries to solve offline RL by making value estimates deliberately conservative. ([arXiv][1])
2. **Decision Transformer (DT)** reframes offline RL as a **sequence modeling** problem. Instead of learning a value function or policy gradient, it trains a Transformer to predict actions conditioned on a desired return. ([arXiv][2])
3. **Trajectory Transformer (TT)** pushes the sequence-modeling idea further by modeling full trajectories of states, actions, and rewards, then using **beam search** as a planning algorithm. That makes it closer to model-based planning than Decision Transformer. ([arXiv][3])

By the end, you should understand:

* what offline RL is and why it is hard,
* what a **Q-function**, **policy**, **Bellman backup**, **return-to-go**, and **beam search** are,
* how CQL differs from Decision Transformer and Trajectory Transformer,
* why the Transformer papers were exciting,
* and where sequence-modeling approaches help or struggle compared with classical offline RL methods. ([arXiv][1])

---

## Key Takeaways

* **Offline RL is hard mainly because of distribution shift.**
  Core idea: the learner trains on a fixed dataset, but the policy it is improving may choose actions not well represented in that dataset.
  Why it matters: value estimates can become unrealistically optimistic on out-of-distribution actions.
  Practical implication: offline RL methods need an explicit mechanism to stay grounded in dataset-supported behavior. ([arXiv][1])

* **CQL solves offline RL by making value estimates pessimistic on unsupported actions.**
  Core idea: it adds a conservative regularizer to the normal Bellman objective so Q-values for broadly sampled actions are pushed down, while dataset actions are comparatively favored.
  Why it matters: this reduces overestimation, which is one of the main offline RL failure modes.
  Practical implication: CQL is a strong classical baseline when reliability matters more than architectural novelty. 

* **Decision Transformer solves offline RL by treating trajectories like token sequences.**
  Core idea: it represents a trajectory as `(return-to-go, state, action, ...)` and trains a causal Transformer to predict the next action.
  Why it matters: it removes Bellman backups and policy gradients from the learning loop.
  Practical implication: offline RL can sometimes be handled with the tools of supervised sequence modeling rather than custom RL algorithms. 

* **Trajectory Transformer is more model-based than Decision Transformer.**
  Core idea: it models the joint distribution over states, actions, and rewards, then uses beam search to plan trajectories with high predicted reward.
  Why it matters: it is not just predicting actions from a target return; it is explicitly modeling and searching over future trajectories.
  Practical implication: Trajectory Transformer can act like a learned planner, not only a policy network. 

* **Decision Transformer and Trajectory Transformer are both “sequence-modeling RL,” but they are not the same method.**
  Core idea: DT is a return-conditioned policy model; TT is a trajectory model plus search.
  Why it matters: they make different trade-offs in simplicity, planning, and controllability.
  Practical implication: interviewers often lump them together, but a strong answer distinguishes them clearly. 

* **Sequence modeling does not automatically dominate classical offline RL.**
  Core idea: DT and TT perform very well on many benchmarks, but CQL remains stronger or more stable in several difficult settings, especially when conservative value estimates matter.
  Why it matters: the sequence-modeling framing is elegant, but it does not make RL’s core uncertainty problems disappear.
  Practical implication: CQL is still the right comparison point, not just a historical baseline. 

* **The papers differ most on where optimization happens.**
  Core idea: CQL optimizes values and policies through Bellman-style RL; DT learns imitation-like action generation conditioned on return; TT plans by searching in a learned trajectory model.
  Why it matters: this determines what each method can explain, optimize, and fail at.
  Practical implication: in system design, the main question is whether you want a conservative value learner, a return-conditioned policy generator, or a model-based planner. 

---

## Background and Foundations

### What is reinforcement learning?

**Reinforcement learning (RL)** is the study of how an agent chooses actions over time to maximize cumulative reward. A few core terms matter:

* **state**: the current situation the agent observes
* **action**: what the agent does next
* **reward**: immediate feedback from the environment
* **policy**: the rule the agent uses to choose actions
* **return**: total future reward, often discounted over time
* **Q-function**: the expected return of taking an action in a state and then following a policy afterward 

### What is offline RL?

In ordinary online RL, the agent keeps interacting with the environment and gathering new data. In **offline RL**, training must use only a fixed dataset collected earlier by one or more behavior policies. No new exploration is allowed. ([arXiv][1])

This is attractive because:

* real-world interaction can be expensive,
* poor exploration can be dangerous,
* and large logged datasets may already exist. 

But it causes a serious problem.

### Why offline RL is difficult: distribution shift

The dataset was collected by some old **behavior policy**. The new learned policy may choose different actions. If a value-based algorithm assigns high value to actions the dataset rarely or never contains, it can bootstrap on its own mistakes and become over-optimistic. That is the central failure mode CQL is designed to fix. ([arXiv][1])

### What is a Bellman backup?

A **Bellman backup** is the standard RL idea of updating a value estimate using:

1. observed immediate reward,
2. plus an estimate of the future value of the next state.

This is powerful, but in offline RL it becomes dangerous when the “future action” used in the backup is outside the dataset. Then the value learner may amplify errors. CQL stays in this Bellman framework but modifies it conservatively. Decision Transformer avoids Bellman backups entirely. 

### What is sequence modeling?

A **sequence model** predicts the next element in a sequence from earlier elements. In language modeling, that means predicting the next token. Decision Transformer and Trajectory Transformer both ask: can we treat RL trajectories the same way we treat sentences? ([arXiv][2])

### Two different sequence-modeling views of RL

| Method                 | What sequence is modeled?                                       | What is predicted?                                            |
| ---------------------- | --------------------------------------------------------------- | ------------------------------------------------------------- |
| Decision Transformer   | Return-to-go, state, action triplets                            | Action, conditioned on desired return                         |
| Trajectory Transformer | Full trajectories of states, actions, rewards, and reward-to-go | Next tokens of the trajectory, then plans are found by search |

This is the most important conceptual difference between the two Transformer papers. It is drawn directly from their method descriptions. 

---

## Big Picture First

A useful way to organize the three papers is by **how they decide what action to take**.

### CQL: “Learn a safe value function”

CQL learns a Q-function that intentionally underestimates unsupported actions. Then a policy is improved against that conservative Q-function. This is classical offline RL with a new pessimism mechanism. ([arXiv][1])

### Decision Transformer: “Generate actions that match a target return”

Decision Transformer does not estimate Q-values. It asks: given a desired return, the current state, and past history, what action sequence looks like trajectories in the dataset that achieved that return? ([arXiv][2])

### Trajectory Transformer: “Model possible futures, then search”

Trajectory Transformer models the whole trajectory distribution. At test time it uses beam search to find likely future trajectories with high reward. That makes it more like a planner over imagined futures. ([arXiv][3])

### The historical shift

| Paper                  | Main question                                                               |
| ---------------------- | --------------------------------------------------------------------------- |
| CQL                    | How do we make offline value learning reliable?                             |
| Decision Transformer   | Can offline RL be done as conditional sequence modeling?                    |
| Trajectory Transformer | Can one Transformer model full trajectories and support planning by search? |

This is the core storyline of the topic. ([arXiv][1])

---

## Core Concepts Explained

### Conservative value estimation

**What it is:**
A value-learning strategy that prefers underestimation to overestimation on unsupported actions. ([arXiv][1])

**Why it exists:**
Offline RL fails when the algorithm believes in actions that look good only because the value function extrapolated badly beyond the dataset. 

**Where it appears:**
This is the central idea of CQL. ([arXiv][1])

**Why it matters:**
It gives offline RL a direct defense against the main distribution-shift failure mode. ([arXiv][1])

### Return-to-go

**What it is:**
The remaining total reward expected from a timestep onward. Decision Transformer feeds this as an input token rather than feeding immediate past rewards. 

**Why it exists:**
If the model is supposed to choose actions based on desired future performance, then future-oriented reward information is more useful than only the immediate past reward. 

**Where it appears:**
Decision Transformer and Trajectory Transformer both use reward-to-go style quantities, but for different reasons. DT uses it as a conditioning signal for policy generation. TT uses reward-to-go partly as a planning heuristic to reduce beam-search myopia. 

### Autoregressive trajectory modeling

**What it is:**
Treating a trajectory like a sequence where the next token depends on previous ones. ([arXiv][2])

**Why it exists:**
Transformers are strong sequence models, so these papers ask whether the RL problem can be re-expressed in that language. ([arXiv][2])

**Where it appears:**
Decision Transformer models a sequence of return-state-action tokens; Trajectory Transformer models state, action, reward, and reward-to-go tokens more fully. 

### Beam search

**What it is:**
A search algorithm from sequence modeling that keeps several promising partial sequences at each step instead of only one. 

**Why it exists:**
Trajectory Transformer is not just a policy network. It needs a procedure for exploring candidate action sequences. Beam search plays that role. 

**Where it appears:**
Central to Trajectory Transformer; not the main mechanism in Decision Transformer. 

### Model-free vs model-based

**Model-free RL** learns a policy or value function directly, without explicitly modeling environment dynamics.
**Model-based RL** learns a model of how states evolve and can plan through it. ([arXiv][1])

CQL is model-free. Decision Transformer behaves more like a direct policy model. Trajectory Transformer is the closest to model-based planning because it models trajectories and uses search. 

---

## Step-by-Step Technical Walkthrough

## 1. Conservative Q-Learning (CQL)

### Goal

Learn from a fixed offline dataset without overestimating actions the dataset does not support. ([arXiv][1])

### Workflow

1. **Start with a standard offline RL dataset**
   The dataset contains transitions like state, action, reward, next state, collected previously by a behavior policy. 

2. **Use the normal Bellman error term**
   CQL still trains a Q-function with Bellman-style consistency. This keeps it connected to standard RL learning. 

3. **Add a conservative penalty**
   The key idea is to push down Q-values under a broad action distribution and comparatively push up Q-values on dataset-supported actions. The paper describes this as minimizing values under an appropriately chosen distribution over state-action pairs and tightening the bound with a maximization term over the data distribution. 

4. **Optionally learn a policy against that Q-function**
   In the actor-critic version, a policy is improved using the conservative Q-function. The paper also describes a Q-learning variant. 

5. **Use the conservative Q-function for action selection**
   Because out-of-distribution actions have lower values, policy improvement is less likely to exploit false optimism. ([arXiv][1])

### Practical meaning of the formula

The CQL objective has three intuitive pieces:

* a **standard Bellman fitting term** so the Q-function still learns from transitions,
* a **push-down term** on broadly sampled actions so unsupported actions do not get high Q-values for free,
* and a **push-up term** on dataset actions so the method does not become uniformly pessimistic. 

### Output

A conservative Q-function, and possibly a policy trained against it. 

### Trade-offs

* Strong at avoiding overestimation.
* Still depends on Bellman learning and its usual RL machinery.
* May be conservative enough to miss some beneficial extrapolation.
  These are partly directly stated and partly the natural interpretation of the paper’s design. ([arXiv][1])

---

## 2. Decision Transformer

### Goal

Solve offline RL by modeling trajectories as sequences and generating actions conditioned on desired return. ([arXiv][2])

### Workflow

1. **Convert each trajectory into a token sequence**
   The paper represents a trajectory as
   `(return-to-go_1, state_1, action_1, return-to-go_2, state_2, action_2, ...)`. 

2. **Embed each modality separately**
   Returns, states, and actions each get their own linear embeddings, plus an episodic timestep embedding. The model then processes them with a causal Transformer. 

3. **Train with autoregressive prediction**
   The model is trained to predict actions from the earlier sequence context, using standard supervised sequence-model training rather than Bellman updates or policy gradients. ([arXiv][2])

4. **Choose a target return at test time**
   At evaluation, you specify the desired return. The model then generates actions intended to achieve that return. After each action, the return-to-go is decremented by the observed reward. 

5. **Roll out the policy autoregressively**
   The context window carries recent history, letting the Transformer use past states, actions, and returns when choosing the next action. 

### Practical meaning of the method

Decision Transformer is easiest to understand as **return-conditioned behavior generation**. It is asking:

> “Among trajectories in the dataset that achieved this level of return, what actions tend to follow this kind of history?” ([arXiv][2])

### Output

A policy implemented as an autoregressive Transformer. 

### Trade-offs

* Very simple training loop.
* No Bellman backups.
* But performance depends on the dataset containing enough trajectories that make the return-conditioning meaningful.
  The paper’s own comparison to percentile behavior cloning suggests it is not just simple cloning, but it is still learning heavily from trajectory patterns in the dataset. 

---

## 3. Trajectory Transformer

### Goal

Model full trajectories and plan with search, treating offline RL as “one big sequence modeling problem.” ([arXiv][3])

### Workflow

1. **Tokenize trajectories at a finer level**
   The model represents states, actions, and rewards as discretized tokens. Different state and action dimensions are assigned disjoint token ranges so the model can distinguish them. 

2. **Train a Transformer on the joint distribution over trajectories**
   The paper says it trains a single high-capacity sequence model over sequences of states, actions, and rewards. Training uses teacher forcing, like standard sequence modeling. 

3. **Use the model as a long-horizon predictor**
   Because it models entire trajectories, it can predict long future rollouts better than standard one-step dynamics models in the paper’s humanoid experiments. 

4. **Run beam search for control**
   At test time, beam search explores candidate sequences. For offline RL, the search objective is changed from likelihood to reward-maximizing behavior by replacing token log-probabilities with predicted reward signals. 

5. **Add reward-to-go to reduce myopic search**
   The paper notes that plain reward-maximizing beam search can be myopic, so it augments trajectories with reward-to-go and uses cumulative reward plus reward-to-go as the search heuristic. 

### Practical meaning of the method

Trajectory Transformer is best seen as a **learned world model plus search**, although the paper describes it through sequence modeling language. It predicts what future trajectories are likely and then searches for high-reward ones. ([arXiv][3])

### Output

A planner that uses a Transformer model of trajectories and beam search to choose actions. 

### Trade-offs

* Strong long-horizon trajectory prediction.
* More planning flexibility than Decision Transformer.
* But discretization can become a bottleneck, and beam search adds inference complexity. The paper explicitly notes one halfcheetah result may suffer because the state discretization becomes too coarse. 

---

## Paper-by-Paper Explanation

## 1. Conservative Q-Learning for Offline Reinforcement Learning

### Problem addressed

Standard off-policy RL methods often fail offline because they overestimate values on actions not supported by the dataset. ([arXiv][1])

### Method used

CQL augments the standard Bellman objective with a conservative Q-value regularizer. The paper presents a framework where values are minimized under a chosen action distribution and tightened with a data-distribution maximization term. 

### Main innovation

The central innovation is learning a Q-function whose expected value under the policy lower-bounds the true policy value, rather than chasing optimistic estimates that look good only because of extrapolation error. ([arXiv][1])

### Main findings

On D4RL gym tasks, the paper says CQL performs similarly or better than prior methods on simple datasets and greatly outperforms prior methods on more complex distributions such as mixed, medium-expert, and random-expert datasets. It also reports strong gains on harder AntMaze, Adroit, kitchen, and Atari settings. For example, the paper says CQL is the only method that attains non-zero returns on harder AntMaze tasks and the only method that outperforms behavior cloning on Adroit tasks with human demonstrations. 

### Limitations

CQL is still a value-learning method with Bellman machinery, actor-critic or Q-learning components, and hyperparameters such as the conservative weight. It is more reliable than naive offline value learning, but not conceptually simpler than supervised sequence modeling. 

### What changed compared with earlier work

Earlier offline RL methods often relied on policy constraints or behavior-policy modeling. CQL instead directly penalizes Q-values to make unsupported actions unattractive, simplifying some parts of the pipeline while preserving RL structure. 

### Directly stated facts

* CQL aims to learn a conservative Q-function whose expected value lower-bounds policy value. ([arXiv][1])
* It can be implemented on top of existing deep Q-learning and actor-critic systems. ([arXiv][1])
* The paper reports especially large gains on complex and multi-modal offline datasets. 

### Reasoned interpretation

CQL is the best “classical offline RL” anchor for understanding why the later Transformer papers were surprising. It represents the strongest version of the old toolbox before the sequence-modeling reframing. ([arXiv][1])

### Information not provided

The paper does not claim that conservative value learning is universally optimal across all offline RL settings or simpler than newer sequence-modeling approaches. 

---

## 2. Decision Transformer: Reinforcement Learning via Sequence Modeling

### Problem addressed

Can offline RL be solved without value functions, Bellman backups, or policy gradients, by treating it as conditional sequence modeling? ([arXiv][2])

### Method used

Decision Transformer represents trajectories as sequences of return-to-go, state, and action tokens, and trains a causally masked Transformer to predict actions conditioned on the desired return and past history. 

### Main innovation

The main innovation is replacing the usual RL learning objective with plain autoregressive sequence modeling while keeping the test-time control handle through return conditioning. ([arXiv][2])

### Main findings

The paper reports that Decision Transformer is competitive with CQL on Atari and outperforms conventional RL baselines on almost all evaluated D4RL tasks in its table, with an average of 74.7 versus CQL’s 63.9 when Reacher is excluded. On Key-to-Door, a long-horizon sparse-reward task, it reaches 71.8% success with 1K random trajectories and 94.6% with 10K, while CQL remains near 13%. 

### Limitations

The paper itself asks whether DT is “just behavior cloning on a subset of the data” and shows it is more effective than a simple percentile behavior-cloning baseline in many settings, but the comparison makes clear that DT still relies on learning trajectory regularities from the dataset rather than explicit value optimization. It also benefits from longer context length, especially on Atari. 

### What changed compared with earlier work

Instead of asking how to stabilize Bellman learning offline, the paper asks whether we can remove Bellman learning entirely and let a Transformer model trajectory structure directly. That was a major conceptual shift. ([arXiv][2])

### Directly stated facts

* DT conditions on desired return, past states, and past actions. ([arXiv][2])
* The sequence representation is `(R̂1, s1, a1, R̂2, s2, a2, ...)`. 
* It matches or exceeds strong offline RL baselines on Atari, OpenAI Gym, and Key-to-Door. ([arXiv][2])

### Reasoned interpretation

Decision Transformer matters because it showed that offline RL could sometimes look more like supervised learning over trajectories than like classical value iteration. That changed how many researchers thought about RL problem formulation. ([arXiv][2])

### Information not provided

The paper does not claim that sequence modeling solves all RL settings, especially online exploration-heavy problems. Its evidence is concentrated on offline benchmarks. 

---

## 3. Offline Reinforcement Learning as One Big Sequence Modeling Problem (Trajectory Transformer)

### Problem addressed

Can one Transformer model the full joint distribution over trajectories well enough to support long-horizon prediction and planning, not just action imitation? ([arXiv][3])

### Method used

The paper discretizes state, action, and reward dimensions into tokens, trains a Transformer on full trajectories with teacher forcing, and uses beam search - modified for reward maximization - as a control algorithm. 

### Main innovation

The key innovation is the combination of full trajectory modeling and sequence-model search. Unlike Decision Transformer, which directly maps conditioned history to actions, Trajectory Transformer supports planning over predicted futures. ([arXiv][3])

### Main findings

The paper reports that the Trajectory Transformer is a substantially more reliable long-horizon predictor than standard feedforward dynamics models, including 100-step humanoid rollouts that remain visually close to ground truth. In offline RL on D4RL tasks, its tabulated results show strong performance, sometimes exceeding CQL - for example on hopper-mixed and walker2d-mixed - while trailing CQL on some medium-expert tasks such as halfcheetah-med-expert. It also reports 104% and 109% normalized return for imitation-style receding-horizon control on hopper and walker2d. 

### Limitations

The method introduces discretization and beam-search design choices that classical policy methods do not need. The paper itself notes that coarse discretization may hurt performance on some datasets. It is also more naturally model-based than Decision Transformer, which brings both planning power and planning overhead. 

### What changed compared with earlier work

The paper pushes sequence modeling beyond “predict the next action” toward “model the future trajectory and search through it.” That is the main conceptual step beyond Decision Transformer. ([arXiv][3])

### Directly stated facts

* It trains a model over the joint distribution of states, actions, and rewards. 
* It uses beam search as a control algorithm. 
* It shows much better long-horizon prediction than feedforward dynamics models in humanoid experiments. 

### Reasoned interpretation

Trajectory Transformer is best viewed as the most explicitly planning-oriented member of the sequence-modeling family. It is the paper that most clearly merges language-model tooling with model-based RL planning. ([arXiv][3])

### Information not provided

The paper does not claim that beam-search planning over a discretized Transformer model is always better than simpler return-conditioned policy generation. Its results are competitive, not uniformly dominant. 

---

## Comparison Across Papers or Methods

The comparison below synthesizes the three papers’ reported methods and benchmark behavior. 

| Aspect                               | CQL                                               | Decision Transformer                                       | Trajectory Transformer                                       |
| ------------------------------------ | ------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| Main framing                         | Classical offline RL                              | Conditional sequence modeling                              | Full trajectory sequence modeling                            |
| Core object learned                  | Conservative Q-function                           | Return-conditioned policy                                  | Joint trajectory model                                       |
| Bellman backups?                     | Yes                                               | No                                                         | No in the classical value-learning sense                     |
| Planning at test time?               | Indirect via learned values/policy                | Minimal; autoregressive action generation                  | Yes, beam-search trajectory planning                         |
| Main defense against offline failure | Conservative pessimism on unsupported actions     | Stay within trajectory patterns that match desired returns | Search in learned high-probability, high-reward trajectories |
| Main strength                        | Reliability and strong hard-benchmark performance | Simplicity and strong return-conditioned control           | Long-horizon prediction and planning                         |
| Main weakness                        | Still needs RL machinery and tuning               | More imitation-like dependence on dataset trajectories     | Discretization and search complexity                         |

A second comparison that interviewers often like is this one. 

| Question                           | CQL answer                       | DT answer                                      | TT answer                                  |
| ---------------------------------- | -------------------------------- | ---------------------------------------------- | ------------------------------------------ |
| How do we choose actions?          | Use conservative value estimates | Generate actions conditioned on desired return | Search through modeled future trajectories |
| Where does optimization happen?    | In Q-learning                    | In supervised sequence modeling                | In sequence modeling plus beam search      |
| What is the paper trying to avoid? | Q overestimation offline         | Bellman/value complexity                       | One-step model error and weak planning     |

---

## Real-World System and Application

These papers suggest three different deployment patterns.

### When CQL is a better fit

CQL is a strong choice when you already think in classical RL terms, need robustness under dataset shift, and care about difficult offline settings where unsupported actions can be dangerous. Robotics, logged-control systems, and conservative policy improvement are natural matches. ([arXiv][1])

### When Decision Transformer is a better fit

Decision Transformer is attractive when you have trajectory data and want a simple training pipeline that looks like supervised sequence modeling. It is especially appealing when conditioning on desired outcome is natural, such as “generate behavior that reaches roughly this return level.” ([arXiv][2])

### When Trajectory Transformer is a better fit

Trajectory Transformer is more natural when planning matters and you want a learned model of futures rather than only a direct policy. If long-horizon predictive accuracy is valuable and beam-search planning is acceptable at inference time, TT is the closest of the three to a model-based planner. 

### What these papers support directly

Directly supported by the papers:

* offline learning from fixed datasets,
* D4RL continuous-control benchmarks,
* Atari-style offline benchmarks,
* long-horizon or sparse-reward settings such as Key-to-Door and AntMaze. 

### Information not provided

These papers do not provide a full production architecture for safety monitoring, uncertainty calibration, human override, or offline RL in high-stakes real-world regulated systems. They are benchmark-centered research papers, not deployment playbooks. 

---

## Limitations and Trade-offs

| Issue                                   | CQL                                     | Decision Transformer                             | Trajectory Transformer               |
| --------------------------------------- | --------------------------------------- | ------------------------------------------------ | ------------------------------------ |
| Distribution-shift robustness           | Strong explicit mechanism               | Indirect, through dataset-conditioned generation | Indirect, through model + search     |
| Simplicity of training                  | Moderate                                | High                                             | Moderate                             |
| Planning flexibility                    | Moderate                                | Lower                                            | High                                 |
| Reliance on dataset trajectory patterns | Medium                                  | High                                             | High                                 |
| Need for search at inference            | No                                      | No                                               | Yes                                  |
| Main failure mode                       | Over-conservatism or tuning sensitivity | Weak extrapolation beyond dataset patterns       | Coarse discretization or poor search |

A few trade-offs matter especially for interviews.

First, **CQL is more principled about offline extrapolation risk** because that is its entire design target. DT and TT are elegant, but they do not make the offline-support problem disappear; they mostly avoid it by staying closer to observed trajectories. ([arXiv][1])

Second, **Decision Transformer is simpler than Trajectory Transformer** because DT turns offline RL into direct conditional action prediction, while TT adds a modeled future plus search loop. That extra planning power also adds extra complexity. ([arXiv][2])

Third, **Trajectory Transformer is the most explicitly model-based** of the three. That is why it shines in long-horizon prediction, but it also inherits model-based concerns like rollout quality and search heuristic design. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain:

1. why offline RL is hard,
2. why out-of-distribution actions break naive value learning,
3. how CQL fixes that with conservative Q-values,
4. how Decision Transformer converts RL into return-conditioned sequence modeling,
5. how Trajectory Transformer differs by modeling full trajectories and planning with beam search,
6. and why the Transformer papers are exciting without making classical offline RL obsolete. ([arXiv][1])

### Likely interview questions

**What is offline RL?**
Learning a policy from a fixed logged dataset without collecting new interaction data during training. The core problem is that the learned policy may choose actions not well covered by the dataset. ([arXiv][1])

**Why does offline RL fail for naive Q-learning?**
Because Bellman backups can assign high value to out-of-distribution actions, and then the algorithm bootstraps on those bad estimates. 

**What does CQL do in one sentence?**
It modifies Q-learning so unsupported actions get pessimistic values, making the learned Q-function conservative offline. ([arXiv][1])

**What is Decision Transformer in one sentence?**
A causal Transformer that predicts actions from the sequence of desired return, past states, and past actions, turning offline RL into conditional sequence modeling. ([arXiv][2])

**How is Decision Transformer different from behavior cloning?**
It is still supervised sequence modeling, but it conditions on return-to-go, so it can generate different behaviors depending on the desired outcome instead of just imitating average behavior. The paper also shows it is more effective than percentile behavior cloning on many tasks. 

**What is Trajectory Transformer in one sentence?**
A Transformer trained on full trajectories of states, actions, and rewards, used together with beam search to plan high-reward trajectories. ([arXiv][3])

**How does Trajectory Transformer differ from Decision Transformer?**
DT is basically a return-conditioned policy model; TT is a trajectory model plus search-based planning. TT explicitly reasons over future sequences, while DT directly predicts the next action. 

**Which of the three is most classical RL?**
CQL, because it stays in the Q-learning / Bellman-backup framework. 

**Which is most planning-oriented?**
Trajectory Transformer, because it uses a learned trajectory model with beam search. 

**Which is simplest to train?**
Decision Transformer is the simplest conceptually because it reduces offline RL to supervised sequence modeling over trajectories. ([arXiv][2])

### Concise model answers

| Question                                       | Plain-English answer                                                                                |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Why is CQL conservative?                       | It would rather underestimate uncertain actions than overestimate them and break offline training.  |
| Why is DT called a transformer?                | Because it uses a causal Transformer exactly like sequence models in NLP, but on trajectory tokens. |
| Why is TT called a trajectory transformer?     | Because it models whole trajectories, not just next actions.                                        |
| What is the main difference between DT and TT? | DT is a return-conditioned policy; TT is a trajectory model plus planner.                           |

These are teaching-oriented summaries of the paper-level ideas. 

---

## Glossary

* **Actor-critic:** An RL setup with both a policy (actor) and a value estimator (critic). CQL includes an actor-critic variant. 
* **Beam search:** A search method that keeps several promising partial sequences instead of only one. TT uses it for planning. 
* **Bellman backup:** A value update using immediate reward plus estimated next-state value. CQL uses Bellman-style learning; DT does not. 
* **Behavior cloning (BC):** Supervised imitation of actions from dataset states. All three papers compare against it in some form. 
* **D4RL:** A benchmark suite and dataset collection for offline RL. All three papers report D4RL results. 
* **Distribution shift:** The mismatch between dataset actions and the actions a learned policy wants to take. This is the main offline RL difficulty. ([arXiv][1])
* **Offline RL:** Learning from a fixed logged dataset with no new environment interaction during training. ([arXiv][1])
* **Policy:** A mapping from states to actions. 
* **Q-function:** The expected long-term return of taking an action in a state and then following a policy. CQL learns this directly. ([arXiv][1])
* **Return-to-go:** Remaining future cumulative reward from the current timestep onward. DT uses it as an input token. 
* **Sequence modeling:** Predicting the next element of a sequence from previous elements. DT and TT apply this idea to RL trajectories. ([arXiv][2])
* **Teacher forcing:** Standard supervised sequence-model training where the true previous tokens are fed during training. TT uses this. 

---

## Recap

These three papers capture a major transition in offline RL.

**CQL** represents the strongest classical answer: keep Bellman learning, but make it conservative enough to survive offline distribution shift. **Decision Transformer** shows that offline RL can sometimes be reframed as conditional sequence modeling over return-state-action tokens. **Trajectory Transformer** pushes that view further by modeling full trajectories and using search for control. ([arXiv][1])

The most important interview-level takeaway is this:

**Offline RL can be attacked either by fixing value estimation under distribution shift or by rephrasing control as sequence modeling. CQL is the clearest example of the first path; Decision Transformer and Trajectory Transformer are the clearest examples of the second.** 

What remains limited is also important. Sequence models are elegant, but they still depend on the structure of the offline dataset and do not magically solve unsupported-action problems. Conservative value methods are robust, but they keep the complexity of classical RL machinery. That is why these papers are complementary rather than redundant. ([arXiv][1])

---

## Key Citations

Decision Transformer: Reinforcement Learning via Sequence Modeling. ([arXiv][2])

Offline Reinforcement Learning as One Big Sequence Modeling Problem. ([arXiv][3])

Conservative Q-Learning for Offline Reinforcement Learning. ([arXiv][1])

[1]: https://arxiv.org/abs/2006.04779?utm_source=chatgpt.com "Conservative Q-Learning for Offline Reinforcement Learning"
[2]: https://arxiv.org/abs/2106.01345?utm_source=chatgpt.com "Decision Transformer: Reinforcement Learning via Sequence Modeling"
[3]: https://arxiv.org/abs/2106.02039?utm_source=chatgpt.com "Offline Reinforcement Learning as One Big Sequence Modeling Problem"
