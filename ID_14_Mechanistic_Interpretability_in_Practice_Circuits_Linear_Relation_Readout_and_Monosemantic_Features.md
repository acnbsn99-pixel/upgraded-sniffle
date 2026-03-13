# Mechanistic Interpretability in Practice: Circuits, Linear Relation Readout, and Monosemantic Features

## What This Report Teaches

This report explains three influential directions in mechanistic interpretability: finding a concrete circuit for one language task inside GPT-2 small, showing that some factual and linguistic relations can be decoded by a simple linear map inside transformer representations, and extracting sparse, human-interpretable features from a production model using sparse autoencoders. Together, these sources show three different levels of explanation: **task-level circuits**, **relation-level linear structure**, and **feature-level decomposition**. By the end, you should understand what each level tries to explain, how the methods work, what they can and cannot prove, and how to discuss the trade-offs in an AI engineer or AI architect interview. 

A source note is important here: the first two sources are standard research papers available as PDFs, while the third item is best accessed through Anthropic’s primary Transformer Circuits Thread and companion Anthropic research posts. I treat that third source as the primary source for the Claude 3 Sonnet monosemanticity work. 

---

## Key Takeaways

* The IOI paper shows that, for one carefully designed task, a surprisingly small set of attention heads in GPT-2 small can explain most of the behavior; this matters because it turns “the model is a black box” into a more concrete engineering problem, and the practical implication is that circuit tracing can sometimes isolate a real algorithm rather than just produce pretty visualizations. 

* The LRE paper shows that many subject-to-object relations inside language models are approximately handled by an **affine linear map** from a subject representation to an object representation; this matters because it suggests some knowledge retrieval is simpler and more structured than a fully mysterious nonlinear process, and the practical implication is that engineers can think about some facts as being linearly readable and even linearly editable. 

* The monosemanticity work argues that model activations can be decomposed into many sparse features that are more interpretable than raw neurons; this matters because single neurons are often too entangled to mean much on their own, and the practical implication is that sparse autoencoders give a scalable way to inspect model internals at the feature level. ([Anthropic][1])

* None of the papers claims full understanding of a modern model; this matters because interpretability results are easy to overstate, and the practical implication is that in interviews you should talk about **partial, validated explanations** rather than pretending these methods fully “solve” interpretability. 

* Validation is central. The IOI paper uses faithfulness, completeness, and minimality; the LRE paper uses faithfulness and causality; the Claude 3 Sonnet work uses behavioral interventions on extracted features; this matters because interpretability without causal testing can be misleading, and the practical implication is that strong answers should always ask, “Did changing the claimed component actually change behavior?” 

* The three sources fit together as a progression in scale: one task in one small model, many relations across several models, then millions of features in a production model; this matters because it shows the field moving from hand-crafted analysis toward scalable tooling, and the practical implication is that future interpretability systems will likely combine circuit analysis, linear readouts, and feature dictionaries. This connection is a reasoned synthesis across the sources rather than a claim made by any one paper. 

---

## Background and Foundations

Mechanistic interpretability is the attempt to explain a neural network in terms of the internal pieces that actually perform computation. In transformer language models, those pieces might be attention heads, multi-layer perceptrons (MLPs), residual stream vectors, or higher-level learned features. The goal is not just to know that a model predicts correctly, but to understand **how** internal information flows from input to output. The IOI paper frames this as finding a circuit inside a computational graph, the LRE paper studies how a model transforms a subject representation into an object prediction, and the Claude 3 Sonnet work studies how to decompose internal activations into sparse interpretable features. 

To follow the papers, you need a few transformer basics. A transformer processes tokens through layers. At each layer, information lives in the **residual stream**, which is the running hidden state that attention heads and MLPs both read from and write back to. An **attention head** decides which earlier tokens matter for the current token and writes a transformed summary into the residual stream. At the end, the model converts the final hidden state into logits, which are scores for candidate next tokens. The IOI paper explicitly describes GPT-2 small as a 12-layer decoder-only transformer with 12 attention heads per layer and focuses mainly on attention heads. 

The third source adds two important background ideas. The **linear representation hypothesis** says meaningful concepts may correspond to directions in activation space. The **superposition hypothesis** says a network may pack more concepts into a space than there are raw dimensions by overlapping them in near-orthogonal directions. If those ideas are true, then dictionary learning, and especially **sparse autoencoders (SAEs)**, become natural tools for trying to recover those hidden concepts. ([Transformer Circuits][2])

Historically and conceptually, the sources are related but not identical. The IOI paper is a classic circuit-discovery result focused on one crisp linguistic behavior. The LRE paper is less about tracing individual heads and more about discovering a reusable geometric structure for relation decoding. The Claude 3 Sonnet work is about scaling feature discovery to a real deployed model. That “from circuits to geometry to sparse features” framing is an interpretation that helps connect the papers; it is not a slogan stated verbatim in the sources. 

---

## Big Picture First

A useful mental model is that the three sources study interpretability at three different resolutions:

| Resolution     | What is being explained?          | Main object of study               | Typical question                      |
| -------------- | --------------------------------- | ---------------------------------- | ------------------------------------- |
| Task level     | One behavior on one prompt family | Circuit of heads                   | “Which components solve this task?”   |
| Relation level | Subject-to-object decoding        | Linear map between representations | “How is this relation read out?”      |
| Feature level  | Reusable semantic building blocks | Sparse features from activations   | “What concepts are represented here?” |

This table is a synthesis across the sources. 

At a high level, all three methods try to answer the same deeper question: **what internal variables actually matter?** The IOI paper answers by finding named attention-head roles like Duplicate Token Heads and Name Mover Heads. The LRE paper answers by saying that for many relations, a single affine map from subject state to object state is already a strong approximation. The Claude 3 Sonnet work answers by saying that raw neurons are too entangled, but sparse learned features can often correspond to coherent concepts and can sometimes be behaviorally manipulated. 

What changes across the papers is scale and granularity. The IOI paper is deep and narrow. The LRE paper is broader across relation types and models. The monosemanticity work is much broader still, using sparse autoencoders to extract millions of features from the middle layer of Claude 3.0 Sonnet and to inspect phenomena such as multilingual concepts, multimodal concepts, safety-relevant features, and behavior steering. 

---

## Core Concepts Explained

### Mechanistic Interpretability

Mechanistic interpretability tries to reverse engineer the actual algorithm a model uses internally. It is not the same as post-hoc explanation methods that only correlate attention weights or saliency with outputs. In these papers, the emphasis is on **causal interventions**: patching activations, knocking out heads, editing representations, or amplifying features to see whether behavior changes in the predicted way. 

### Circuit

A **circuit** is a subgraph of model computation responsible for a behavior. In the IOI paper, the authors define a circuit as a subgraph of the model’s computational graph, where nodes are model components and edges are their interactions. This matters because it gives you a concrete target: not “the whole model,” but “the pieces that actually implement this behavior.” 

### Indirect Object Identification (IOI)

The IOI task uses sentences like: “When Mary and John went to the store, John gave a drink to …” where the correct continuation is “Mary.” The paper’s plain-English algorithm is: identify names, remove the duplicated subject name, output the remaining name. This task matters because it is linguistically meaningful but still simple enough to plausibly reverse engineer. 

### Path Patching

**Path patching** is a causal intervention method introduced in the IOI paper. The basic idea is to replace information along a chosen computational path with information from another input and measure how much the output changes. It exists because directly asking “which head matters?” is often too crude; path patching lets you ask “which head matters along this specific route of influence?” 

### Faithfulness, Completeness, Minimality, Causality

These are validation ideas.

* **Faithfulness** means the explanation reproduces the model’s behavior well.
* **Completeness** means the explanation did not leave out important components.
* **Minimality** means the explanation does not contain unnecessary pieces.
* **Causality** in the LRE paper means the proposed internal description supports interventions that predictably change model output. 

These distinctions matter because a story can be faithful but incomplete. The IOI paper’s Backup Name Mover Heads are a direct example: the original circuit looked good until ablating regular Name Mover Heads revealed backup machinery. 

### Linear Relational Embedding (LRE)

An **LRE** is an affine map of the form (R(s)=W_rs+b_r). In plain English, it says: once the model has built an internal representation of the subject, the step that decodes the relevant relation may behave like “apply one relation-specific matrix and bias, then decode.” The LRE paper estimates this map from the model’s local derivative, or Jacobian, and then checks whether it predicts behavior on new subjects. 

### Attribute Lens

The **attribute lens** is built from an LRE. Instead of decoding a hidden state as “what next token does this state predict right now?”, it decodes “what object token distribution does this state imply for relation r?” This matters because it can reveal that the model internally knows an attribute even when the final output is distracted or wrong. 

### Sparse Autoencoder (SAE)

A sparse autoencoder is a two-layer model trained to reconstruct activations while using only a small number of active features for each example. The encoder maps an activation vector into many candidate features, a sparsity penalty encourages only a few to activate, and the decoder reconstructs the original activation from those active features. The goal is to trade raw entangled neurons for a sparse set of more interpretable features. ([Transformer Circuits][2])

### Monosemanticity

In this context, **monosemanticity** means a feature corresponds to one coherent concept rather than a messy mixture. The Claude 3 Sonnet work does not claim perfect one-feature-one-concept coverage of the model; instead, it argues that sparse autoencoders can recover many highly interpretable features, including abstract, multilingual, multimodal, and safety-relevant ones. ([Anthropic][1])

---

## Step-by-Step Technical Walkthrough

### 1. IOI Circuit Discovery in GPT-2 Small

#### Goal

Explain how GPT-2 small solves the IOI task by identifying the heads responsible for moving information from the correct name to the output. The paper reports a circuit of 26 attention heads, about 1.1% of all head-position pairs considered for the task. 

#### Pipeline

1. **Start from the output logits.**
   The authors begin at the final prediction and ask which heads directly affect the logit difference between the correct indirect object and the wrong subject. This leads to **Name Mover Heads** and **Negative Name Mover Heads**. Name Mover Heads have copy scores above 95%, versus less than 20% for an average head. Negative Name Mover Heads behave similarly but push in the opposite direction. 

2. **Trace backward into the attention mechanism.**
   They ask what influences the Name Mover Heads’ attention, especially their query at the final token position. Using path patching, they identify four **S-Inhibition Heads** that reduce attention to the duplicated subject and shift attention toward the indirect object. 

3. **Disentangle what S-Inhibition Heads are writing.**
   The paper argues that these heads transmit both a **token signal** and a **position signal**, with the position signal being more important. In plain English: the model is not only recognizing which name is duplicated, but also where the earlier duplicate occurred. 

4. **Trace the source of the position signal.**
   Looking backward again, the authors identify **Duplicate Token Heads**, **Induction Heads**, and **Previous Token Heads**. Duplicate Token Heads appear to detect that the current name token has appeared before and write positional information. Induction Heads and Previous Token Heads provide an alternate route using repeated-pattern behavior. 

5. **Test for missed components.**
   When the regular Name Mover Heads are all knocked out, task performance drops by only about 5% in logit difference, revealing **Backup Name Mover Heads**. This is a major lesson: apparent explanation can hide redundancy. 

6. **Validate the proposed circuit.**
   The paper evaluates the explanation using faithfulness, completeness, and minimality. The circuit is much better than a naive faithful circuit, but the most difficult tests still reveal gaps. 

#### Why this exists

Without this process, you might say “attention focuses on Mary.” That would be too shallow. The paper instead identifies a multi-step algorithm distributed across head types: duplicate detection, subject suppression, and correct-name copying. 

#### Main trade-off

The explanation is detailed and causal, but narrow. It is powerful for one task in one small model, yet labor-intensive and still incomplete. 

---

### 2. Linear Relation Decoding in Transformer Language Models

#### Goal

Explain how a model goes from “representation of subject” to “prediction of related object” for relations such as country-capital, person-instrument, or adjective-superlative. The paper studies 47 relations across more than 10,000 facts and reports robust LREs for 48% of the relations tested. 

#### Pipeline

1. **Choose a relation and prompts.**
   Each relation has subject-object pairs and a prompt template such as “[subject] plays the …” or “[country]’s capital is …”. The authors restrict evaluation to cases where the model itself already predicts the correct object. 

2. **Extract the subject representation.**
   At some intermediate layer, the subject token has been enriched with relevant knowledge. The paper treats that hidden state as the input (s). 

3. **Approximate the model’s relation-decoding step.**
   The model’s true computation from subject state to object state is written as (F(s,c)). The paper estimates its Jacobian with respect to (s), averages over examples, and turns this into an affine approximation (LRE(s)=\beta W_rs+b_r). In plain English, it is fitting the best local linear map for a relation and then correcting its scale with the factor (\beta). 

4. **Evaluate faithfulness.**
   Does the LRE predict the same object token as the full model on new subjects? Faithfulness measures that agreement rate. 

5. **Evaluate causality.**
   If the LRE is a good description, then inverting it should let you edit the subject representation so the model predicts a different object. This is an intervention test, not just a probe score. 

6. **Build the attribute lens.**
   Once an LRE exists, you can decode hidden states at each layer into relation-specific object predictions. This reveals where knowledge becomes available internally, even when the final output is wrong because of distractions. On “repetition-distracted” and “instruction-distracted” prompts, the raw prompted model almost never outputs the true fact, but the attribute lens recovers much stronger top-k recall. 

#### Why this exists

Earlier work suggested models store factual knowledge in subject representations, but not exactly how that knowledge is read out for a requested relation. The LRE paper proposes that, for many relations, the readout step is close to linear. 

#### Main trade-off

This method is broader and more scalable than circuit tracing, but less mechanistically specific. It tells you **that** a relation can be decoded linearly, not necessarily which exact heads and MLPs implemented that decoding. It also fails on many relations that the model still answers correctly, showing that “model knows the fact” does not imply “the fact is linearly decoded this way.” 

---

### 3. Sparse Autoencoders and Monosemantic Features in Claude 3 Sonnet

#### Goal

Decompose activations of a large deployed language model into sparse, more interpretable features. Anthropic reports extracting millions of features from the middle layer of Claude 3.0 Sonnet and finding many abstract, multilingual, multimodal, and safety-relevant features. ([Anthropic][1])

#### Pipeline

1. **Collect model activations.**
   The work trains SAEs on Claude 3 Sonnet activations, specifically from a middle layer according to the retrieved primary-source descriptions. The exact middle-layer index is not provided in the retrieved excerpts. ([Anthropic][1])

2. **Normalize the activations.**
   As a preprocessing step, activations are scaled so their average squared norm matches the residual-stream dimension. This helps stabilize SAE training. ([Transformer Circuits][3])

3. **Encode into many candidate features.**
   The encoder applies a learned linear map plus ReLU to produce feature activations. In plain English, it asks: which feature directions are present in this token’s hidden state? ([Transformer Circuits][2])

4. **Reconstruct the original activation.**
   The decoder linearly combines active features to rebuild the original activation. Training minimizes reconstruction error plus an L1 sparsity penalty, which encourages only a small number of active features per example. ([Transformer Circuits][2])

5. **Inspect and label features.**
   After training, the researchers inspect the highest-activating examples and neighboring features. They report features for entities like the Golden Gate Bridge, cities, scientists, code structures, secrecy, gender bias discussions, scam emails, sycophantic praise, manipulation, and dangerous content. ([Anthropic][1])

6. **Test whether features matter behaviorally.**
   The Anthropic article reports that amplifying some features changes model behavior. For example, amplifying a Golden Gate Bridge-related feature can make the model talk as if it were the bridge, and strongly activating a scam-email-related feature can override the normal refusal and produce a scam email. This is presented as evidence that the features are not only correlated with text patterns but causally involved in behavior. ([Anthropic][1])

7. **Scale the method.**
   The work uses scaling-law ideas to choose SAE design and reports engineering bottlenecks such as shuffling very large activation datasets and building feature-visualization pipelines. Anthropic’s companion engineering post mentions 100TB of training data in earlier scaling stages and a 100M-token dataset for visualization. ([Anthropic][4])

#### Why this exists

Single neurons are often polysemantic: one neuron can participate in many unrelated concepts, and one concept can spread across many neurons. Sparse autoencoders aim to recover a more interpretable basis of features. ([Anthropic][1])

#### Main trade-off

This is far more scalable than hand-tracing circuits, but it is also further from a full algorithmic explanation. It tells you that a feature exists and can influence behavior, but not automatically which full circuit uses it or how all features interact during reasoning. Anthropic explicitly says that understanding representations does not by itself tell us how the model uses them, and that finding a full set of features with current methods would be cost-prohibitive. ([Anthropic][1])

---

## Paper-by-Paper Explanation

### 1. Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2

| Item                         | Explanation                                                                                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Problem addressed            | How does GPT-2 small solve a natural language task where it must identify the indirect object rather than the repeated subject?                                                          |
| Method used                  | Circuit analysis centered on path patching, activation patching, head knockouts, attention-pattern analysis, and embedding-space projections.                                            |
| Main innovation              | An end-to-end circuit explanation for a real task in GPT-2 small, organized into 7 classes of heads and validated with explicit criteria.                                                |
| Main findings                | A 26-head circuit explains much of the task; key classes include Duplicate Token, S-Inhibition, Name Mover, Negative Name Mover, Previous Token, Induction, and Backup Name Mover heads. |
| Limitations                  | Redundancy complicates explanation; backup heads appear after ablation; MLP roles were not deeply traced; the circuit still fails the hardest validation tests.                          |
| What changed vs earlier work | The paper pushes beyond very small toy behaviors and beyond loose large-model stories by attempting a detailed end-to-end reverse engineering of one in-the-wild behavior.               |

Table content is drawn from the paper and its validation sections. 

### 2. Linearity of Relation Decoding in Transformer Language Models

| Item                         | Explanation                                                                                                                                                                                            |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Problem addressed            | How do transformers decode relations such as capital cities, occupations, colors, or instruments from internal subject representations?                                                                |
| Method used                  | Estimate a relation-specific affine map from the model Jacobian, test it with faithfulness and causal editing, and visualize it with the attribute lens.                                               |
| Main innovation              | The claim that some relation decoding inside language models is well-approximated by a simple linear relational embedding.                                                                             |
| Main findings                | Robust LREs exist for 48% of the 47 tested relations; they generalize across factual, commonsense, linguistic, and bias-related relations, but many other relations remain nonlinear in practice.      |
| Limitations                  | Only 47 relations were tested; evaluation uses first-token correctness; some relations have multiple valid objects; the method describes relation-level structure rather than exact internal circuits. |
| What changed vs earlier work | Earlier work showed facts can be stored in representations; this paper focuses on the decoding step from representation to object prediction.                                                          |



### 3. Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet

| Item                         | Explanation                                                                                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Problem addressed            | Can sparse feature discovery scale from toy models to a production large language model?                                                                                                                                                         |
| Method used                  | Train sparse autoencoders on Claude 3 Sonnet activations, inspect highest-activating examples and feature neighborhoods, and test feature interventions on behavior.                                                                             |
| Main innovation              | Scaling dictionary learning to Claude 3.0 Sonnet and extracting millions of interpretable features from a deployed model.                                                                                                                        |
| Main findings                | Features can be abstract, multilingual, multimodal, and behaviorally causal; the work highlights features tied to entities, code structure, bias, deception, sycophancy, dangerous content, and self-representation.                             |
| Limitations                  | Features cover only a subset of the model’s concepts; full feature recovery is currently too expensive; feature understanding does not automatically reveal full circuits; training also produces dead features, especially at larger SAE sizes. |
| What changed vs earlier work | It extends sparse-feature interpretability from tiny or modest models to a modern production model and emphasizes engineering scale as a core challenge.                                                                                         |

Anthropic reports dead-feature rates of roughly 2% for the 1M SAE, 35% for the 4M SAE, and 65% for the 34M SAE. ([Transformer Circuits][2])

---

## Comparison Across Papers or Methods

| Dimension        | IOI Circuit Paper                                           | LRE Paper                                              | Monosemanticity / Claude 3 Sonnet          |
| ---------------- | ----------------------------------------------------------- | ------------------------------------------------------ | ------------------------------------------ |
| Main goal        | Explain one specific behavior                               | Explain one class of computation across many relations | Recover reusable interpretable features    |
| Unit of analysis | Attention heads and their paths                             | Hidden-state geometry for a relation                   | Sparse learned features                    |
| Typical output   | A named circuit                                             | A relation-specific affine map                         | A feature dictionary / browser             |
| Validation style | Patching, knockouts, faithfulness, completeness, minimality | Faithfulness, causal representation editing            | Behavioral steering and feature inspection |
| Strength         | Deep causal story                                           | Broad relation coverage                                | Scales to a deployed model                 |
| Weakness         | Narrow and labor-intensive                                  | Less algorithmically specific                          | Does not automatically recover circuits    |
| Best use         | Reverse engineering a particular failure mode or task       | Studying factual retrieval structure                   | Building scalable interpretability tooling |

This table is a synthesis across the sources. 

A second useful comparison is **what each method can prove**:

| Method           | Strongest plausible claim                                          | Claim it does **not** justify on its own                        |
| ---------------- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
| Circuit analysis | “These components causally contribute to this task.”               | “We fully understand the whole model.”                          |
| LREs             | “This relation is approximately decoded linearly here.”            | “We know the exact mechanistic circuit that implements it.”     |
| SAEs             | “These sparse features correlate with and can influence behavior.” | “We have a complete map of all concepts and how they interact.” |



---

## Real-World System and Application

No single source gives one full production blueprint that combines all three methods. The pipeline below is therefore a **reasoned interpretation** that connects the papers into a plausible practical interpretability stack. 

1. **Collect activations and task traces.**
   For a target model and prompt family, record residual-stream states, attention outputs, and logits. This supports both relation-level and feature-level analysis. 

2. **Use sparse autoencoders to build a feature dictionary.**
   This helps replace uninterpretable neuron activations with more coherent features that can be browsed, clustered, and searched. ([Transformer Circuits][2])

3. **Use lenses or linear readouts to ask targeted questions.**
   If you care about a relation like “capital city” or “occupation,” LREs and attribute lenses can tell you where the model seems to know that attribute, even if the output is distracted. 

4. **Trace a specific failure mode with circuit methods.**
   If the model behaves badly on a narrow task, path patching and knockouts can identify the exact route of influence for the error or capability. 

5. **Validate causally.**
   Do not trust feature labels or linear probes alone. Edit representations, amplify or suppress features, or patch internal paths and check whether output changes as predicted. 

6. **Apply to safety or reliability.**
   Anthropic explicitly suggests uses such as monitoring for dangerous behaviors, debiasing, or identifying harmful capabilities that remain latent after training. Information about concrete deployed safeguards built from these methods is not provided in the retrieved sources. ([Anthropic][1])

---

## Limitations and Trade-offs

### Concrete limitations from the sources

* **IOI circuit work:** redundancy and backup mechanisms make completeness hard; some head roles remain only partially characterized; MLP contributions were not fully reverse engineered. 

* **LRE work:** only 47 relations were studied; evaluation mostly checks the first object token; some relations may have multiple correct answers; some correctly predicted relations still do not admit a good linear approximation. 

* **Monosemanticity work:** feature extraction is expensive, incomplete, and does not by itself reveal circuits; larger SAE dictionaries produce many dead features; engineering scale is a bottleneck. ([Anthropic][1])

### Bigger interpretability trade-offs

| Trade-off                         | Why it matters                                                                                                                                    |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Depth vs scale                    | Circuit work gives deep local understanding, SAE work gives broad scalable access, and LREs sit somewhere in between.                             |
| Correlation vs causation          | Pretty feature labels or probe scores are not enough; all three sources emphasize interventions.                                                  |
| Simplicity vs completeness        | A simple explanation is attractive, but backup pathways and nonlinear exceptions show the model may be doing more than your clean story captures. |
| Human-readability vs faithfulness | Interpretable stories are useful only if they survive patching, editing, and ablation tests.                                                      |

This table is a synthesis grounded in the sources. 

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain the difference between a **circuit**, a **linear readout**, and a **sparse feature**; describe why causal validation matters; explain the IOI algorithm in plain English; describe what an LRE is and why the Jacobian is used; explain why raw neurons are often not interpretable enough; and discuss why sparse autoencoders are promising but incomplete. 

### Likely interview questions and concise model answers

1. **What is mechanistic interpretability?**
   It is the attempt to explain model behavior in terms of the internal computations actually being performed, such as specific heads, pathways, or learned features, rather than only looking at inputs and outputs. 

2. **What did the IOI paper show?**
   It showed that GPT-2 small’s behavior on indirect object identification can be largely explained by a 26-head circuit with roles like duplicate detection, subject suppression, and name copying. 

3. **Why is faithfulness not enough?**
   Because an explanation can match behavior while still missing important redundant pathways. The IOI paper’s Backup Name Mover Heads are the example: the original story looked good until ablations revealed extra machinery. 

4. **What is an LRE in simple terms?**
   It is a relation-specific affine map that takes a subject’s hidden representation and approximates the model’s internal step for producing the related object representation. 

5. **What is the practical value of the attribute lens?**
   It can show what relation-specific knowledge is present in hidden states layer by layer, even when the model’s final output is wrong because the prompt distracts it. 

6. **Why are sparse autoencoders useful for interpretability?**
   They turn one dense activation vector into a sparse combination of features, which is often easier for humans to inspect than raw neurons because the same concept may be spread across many neurons. ([Anthropic][1])

7. **Did Anthropic prove full monosemanticity in Claude 3 Sonnet?**
   No. They showed that many high-quality interpretable features can be extracted and that some are behaviorally causal, but they also say the recovered features are only a subset and that understanding features alone does not reveal full circuits. ([Anthropic][1])

8. **How would you compare the three approaches?**
   Circuit tracing is best for detailed local explanation, LREs are good for broad relation-specific structure, and sparse autoencoders are best for scalable feature discovery. They answer different questions and are complementary. This is a synthesis across the sources. 

---

## Glossary

| Term                     | Beginner-friendly definition                                                                                                    |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| Attention head           | A sub-component in a transformer that looks at other tokens and writes a summary into the hidden state.                         |
| Residual stream          | The running hidden representation that carries information through the transformer.                                             |
| Circuit                  | A subset of model components and connections that together implement a behavior.                                                |
| Path patching            | A causal method that swaps information along a chosen internal path to test its effect on output.                               |
| Activation patching      | A broader intervention method that replaces internal activations from one run with those from another.                          |
| IOI                      | Indirect object identification: choosing the correct recipient rather than the repeated subject in a sentence.                  |
| Name Mover Head          | In the IOI paper, a head that copies the relevant name toward the output position.                                              |
| S-Inhibition Head        | In the IOI paper, a head that suppresses attention to the duplicated subject name.                                              |
| Induction head           | A head type known for handling repeated-sequence patterns; in the IOI paper it contributes positional information in a new way. |
| Jacobian                 | A matrix of local sensitivities: it tells you how changing one representation would change another.                             |
| Affine map               | A linear transformation plus a bias term.                                                                                       |
| LRE                      | Linear relational embedding; a relation-specific affine approximation from subject representation to object representation.     |
| Faithfulness             | How well an explanation reproduces model behavior.                                                                              |
| Completeness             | Whether the explanation includes all important components.                                                                      |
| Minimality               | Whether the explanation excludes unnecessary components.                                                                        |
| Causality                | Whether intervening on the proposed mechanism changes behavior in the predicted way.                                            |
| Attribute lens           | A tool built from an LRE that decodes relation-specific object predictions from hidden states.                                  |
| Sparse autoencoder (SAE) | A model trained to reconstruct activations using only a small number of active learned features.                                |
| Feature                  | A learned direction or component in activation space that corresponds to a recognizable pattern or concept.                     |
| Monosemanticity          | The idea that one feature corresponds to one coherent concept, rather than a messy mixture.                                     |
| Superposition            | The idea that networks may store many concepts in overlapping directions within the same space.                                 |
| Dead feature             | A learned feature that rarely or never activates, so it does not contribute much useful explanation.                            |

Definitions are aligned with the retrieved sources and their usage. 

---

## Recap

You should now see mechanistic interpretability not as one single method, but as a family of approaches that answer different questions. The IOI paper shows that one narrow behavior can be reverse engineered into a causal circuit of attention heads. The LRE paper shows that some relational knowledge is decoded by a surprisingly simple linear mechanism, even inside a highly nonlinear model. The monosemanticity work shows that large-model activations can be decomposed into many sparse, interpretable features that sometimes causally affect behavior. 

What matters most for interviews is the pattern behind all three: modern interpretability is moving from hand-crafted explanation toward **validated internal measurement**. The strongest answers are the ones that distinguish representation from computation, correlation from causation, and local explanation from full understanding. 

What remains uncertain is just as important. None of these works provides a complete map of a large language model. The sources explicitly leave open questions about missing pathways, nonlinear relations, incomplete feature coverage, MLP roles, and the cost of scaling these methods further. 

---

## Key Citations

[Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2](https://arxiv.org/pdf/2211.00593)

[Linearity of Relation Decoding in Transformer Language Models](https://arxiv.org/pdf/2308.09124)

[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)

[Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model)

[The Engineering Challenges of Scaling Interpretability](https://www.anthropic.com/research/engineering-challenges-interpretability)

[1]: https://www.anthropic.com/research/mapping-mind-language-model "Mapping the Mind of a Large Language Model \ Anthropic"
[2]: https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html?utm_source=chatgpt.com "Scaling Monosemanticity: Extracting Interpretable Features from Claude ..."
[3]: https://transformer-circuits.pub/2024/scaling-monosemanticity/?utm_source=chatgpt.com "Scaling Monosemanticity: Extracting Interpretable Features from Claude ..."
[4]: https://www.anthropic.com/research/engineering-challenges-interpretability "The engineering challenges of scaling interpretability \ Anthropic"

---
---
---


