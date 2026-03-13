# Embodied AI in Robotics: How SayCan, PaLM-E, and RT-2 Connect Language, Perception, and Action

## What This Report Teaches

This report explains three influential steps in embodied AI for robotics. **SayCan** shows how a large language model can help choose high-level robot skills, but only after those choices are grounded by what the robot can actually do in the current world. **PaLM-E** goes further by feeding robot observations directly into a large language model as multimodal inputs, so the model can reason over text plus perception in one system. **RT-2** goes further still by turning robot actions themselves into tokens and training a single vision-language-action model that can directly output robot actions while also benefiting from web-scale vision-language pretraining. Together, these papers trace a clear progression from **language-guided skill selection**, to **multimodal embodied reasoning**, to **end-to-end action generation**. ([arXiv][1])

---

## Key Takeaways

* **SayCan separates “what would make sense” from “what is actually feasible.”** The language model scores which skill would be useful for the instruction, while affordance or value functions score whether the robot can successfully execute that skill now. This matters because language models often suggest plausible but physically impossible actions. The practical implication is that robot planning needs both semantic reasoning and grounded feasibility checks. ([say-can.github.io][2])

* **PaLM-E makes grounding more direct by inserting robot observations into the language model itself.** This matters because many embodied tasks depend on spatial layout and current scene state, which pure text prompts do not capture. The practical implication is that a single model can reason across text, images, states, and robot tasks in one shared token sequence. 

* **RT-2 turns robotic control into token prediction.** It represents actions as text-like tokens and co-fine-tunes vision-language models on both robot trajectories and internet-scale vision-language data. This matters because it lets robotics inherit semantic knowledge from web-trained models. The practical implication is that robot control can gain new generalization abilities, such as understanding unseen objects, symbols, and simple semantic reasoning tasks. ([robotics-transformer2.github.io][3])

* **The three papers differ mainly in where grounding happens.** In SayCan, grounding happens outside the language model through affordance/value scores. In PaLM-E, grounding happens inside the model through multimodal input tokens. In RT-2, grounding extends all the way to the output by making actions part of the model’s native prediction space. This matters because it shows an architectural shift from modular pipelines toward more unified models. The practical implication is that interview answers should compare where perception, planning, and control are fused. ([say-can.github.io][2])

* **Transfer from non-robot data is a major theme.** SayCan reuses a large language model’s world knowledge, PaLM-E benefits from joint training with vision-language and language tasks, and RT-2 explicitly transfers web-scale vision-language knowledge into robot control. This matters because robot data is expensive and limited. The practical implication is that embodied AI increasingly depends on combining scarce robot data with abundant internet data. ([arXiv][1])

* **End-to-end is not always the first step.** SayCan deliberately uses a modular design with a fixed skill library. PaLM-E still assumes low-level skills or planners when it is used for control. RT-2 is the clearest move toward directly predicting robot actions. This matters because robotics often needs reliable intermediate abstractions before fully end-to-end control becomes practical. The practical implication is that system decomposition is still a core robotics design choice. ([say-can.github.io][2])

* **Evaluation in embodied AI is about real execution, not only benchmark scores.** SayCan reports real-world task execution in kitchen environments, PaLM-E evaluates across multiple robot domains plus general vision-language tasks, and RT-2 reports more than 6,000 robotic trials along with generalization experiments. The practical implication is that embodied AI papers should be read as both machine learning papers and robot systems papers. ([say-can.github.io][2])

---

## Background and Foundations

Embodied AI is the branch of AI that studies agents that must **perceive the world, reason about it, and act in it**. In robotics, this means the model cannot stop at generating text or class labels. It must help produce behavior that succeeds in a physical environment with objects, geometry, failures, and changing state. That makes embodied AI harder than ordinary language or vision tasks, because mistakes are not just wrong predictions; they become failed actions in the world. ([arXiv][1])

A helpful beginner distinction is between **high-level planning** and **low-level control**. High-level planning decides things like “find the sponge,” “open the drawer,” or “bring the can.” Low-level control decides motor commands, end-effector motions, grasps, and trajectories. SayCan mainly helps with high-level skill choice. PaLM-E mainly helps with multimodal reasoning and text plan generation that can condition low-level skills. RT-2 pushes closer to low-level closed-loop control by directly outputting action tokens. ([say-can.github.io][2])

Another core concept is **grounding**. A model is grounded when its symbols connect to real observations and real action consequences. A text-only model may know that a sponge is useful for cleaning a spill, but that does not mean the robot currently sees a sponge, can reach it, or can grasp it. SayCan treats this as the central problem. PaLM-E addresses it by feeding observations directly into the model. RT-2 addresses it by learning a joint mapping from visual observations and language to actual actions. ([arXiv][1])

These papers are also part of a broader historical shift in robotics. Earlier robot learning systems often used narrow policies trained on one embodiment or one task family. These papers instead explore how large pretrained models, multimodal tokens, and internet data can be reused to make robots more general. A reasonable interpretation is that the field is moving from **task-specific robot policies** toward **generalist embodied foundation models**, though the three papers still sit at different points along that path. 

---

## Big Picture First

The simplest way to understand the three papers is to ask a single question:

**At what stage does the language model meet the robot?**

| Paper  | Where language helps                                             | What the robot side provides                                      | Main output                                                   |
| ------ | ---------------------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------- |
| SayCan | Chooses the next useful skill                                    | Affordance/value functions estimate which skills are feasible now | A sequence of high-level skills                               |
| PaLM-E | Reasons over text plus perception in one model                   | Continuous observations become multimodal tokens inside the LM    | Textual answers or text plans that condition low-level skills |
| RT-2   | Predicts actions directly using a vision-language model backbone | Robot trajectories are turned into action-token supervision       | Direct robot action tokens                                    |

Sources: SayCan website and abstract, PaLM-E paper, RT-2 project page and blog. ([say-can.github.io][2])

A second useful mental model is that the papers gradually remove boundaries:

1. **SayCan** keeps a strong boundary between semantic planning and physical feasibility. ([say-can.github.io][2])
2. **PaLM-E** softens that boundary by putting perception into the same sequence model as language. 
3. **RT-2** softens it even more by making robot actions just another token sequence the model can generate. ([robotics-transformer2.github.io][3])

That progression is a synthesis across the papers, but it is a faithful one. It is the cleanest big-picture story you can tell in an interview. ([say-can.github.io][2])

---

## Core Concepts Explained

### Affordances

An **affordance** is a way of describing what actions are possible or likely to succeed in the current situation. In SayCan, each robot skill has an associated value or affordance function that estimates how likely that skill is to work from the current state. This exists because “useful” is not the same as “possible.” A sponge may be the right object semantically, but if the sponge is not reachable, the next action should be something else, such as first finding it. This matters because affordances are SayCan’s main grounding mechanism. ([say-can.github.io][2])

### Skill libraries

A **skill library** is a collection of reusable low-level behaviors, such as finding objects, opening drawers, picking objects up, or bringing them to a person. SayCan assumes such skills already exist and selects among them. PaLM-E also assumes low-level policies or planners when it is used for embodied planning and control. This matters because both papers are not solving robotics “from motor torques upward.” They rely on compositional reuse of existing skills. ([say-can.github.io][2])

### Multimodal sentences

PaLM-E introduces the idea of **multimodal sentences**, meaning sequences where text tokens and observation tokens appear together in one stream. The observation tokens can come from images, neural scene representations, or state information. This exists because a language model can only reason over what it receives as input. If real-world observations are turned into tokens, the same model can reason over words and percepts together. This matters because it is PaLM-E’s main architectural idea. 

### Embodied reasoning

**Embodied reasoning** means reasoning that depends on the current physical scene, current robot embodiment, and current task. A pure language model might know commonsense facts, but embodied reasoning needs more: where the object is, whether the drawer is open, whether a path is blocked, and what the robot can do. PaLM-E is built specifically to improve this kind of reasoning, and RT-2 shows that such reasoning can influence direct action generation. 

### Vision-language-action models

A **vision-language-action (VLA) model** is a model that takes visual and language input and outputs actions. RT-2 defines this family and instantiates it by representing robot actions as token strings. This exists because standard vision-language models usually output text, not control commands. By turning actions into tokens, RT-2 uses the same modeling machinery for both language and control. This matters because it is the paper’s central simplification. ([robotics-transformer2.github.io][3])

### Co-fine-tuning

**Co-fine-tuning** means fine-tuning a pretrained model on robotics data while keeping some of the original web-scale vision and language data in the training mix. RT-2 uses this so the model does not forget the broader semantic knowledge learned from internet-scale pretraining. This matters because robot-only fine-tuning could otherwise narrow the model too much and weaken the generalization benefit of web pretraining. ([robotics-transformer2.github.io][3])

### Positive transfer

**Positive transfer** means performance improves on one task because training included other tasks or data sources. PaLM-E explicitly studies this idea and shows that joint training across robot tasks and general vision-language tasks can improve embodied performance. This matters because robot data is scarce, so transfer is one of the few practical routes to broader capability. 

---

## Step-by-Step Technical Walkthrough

### 1. SayCan: from instruction to grounded skill sequence

#### Inputs

SayCan receives a high-level language instruction, such as helping clean a spill or bringing a drink, along with a fixed set of robot skills and their associated affordance or value estimates in the current state. ([say-can.github.io][2])

#### What happens

1. The language model scores which candidate skill would make the most semantic progress toward the user’s instruction. ([say-can.github.io][2])
2. A learned value or affordance function estimates how likely each skill is to succeed now. ([Proceedings of Machine Learning Research][4])
3. SayCan combines these two signals so the selected skill is both useful and feasible. ([say-can.github.io][2])
4. The chosen skill is executed on the robot. ([say-can.github.io][2])
5. The chosen step is appended to the interaction history, and the system repeats the process until it outputs a termination step such as “done.” ([say-can.github.io][2])

#### Outputs

The output is a sequence of high-level skills, not raw motor commands. The low-level controllers for those skills already exist outside the language model. ([say-can.github.io][2])

#### Why each step exists

The language model provides commonsense task decomposition, but affordances prevent unrealistic or currently impossible choices. The repeated loop allows the plan to adapt as the scene changes. This is why SayCan can handle long-horizon instructions instead of a single one-shot decision. ([say-can.github.io][2])

#### Trade-offs

The main trade-off is modularity versus flexibility. SayCan is interpretable and grounded, but it can only choose from its existing skill set. A reasonable interpretation is that it cannot invent new low-level behaviors on its own; its competence is bounded by the skill library and the quality of the affordance models. ([say-can.github.io][2])

---

### 2. PaLM-E: from multimodal observations to embodied reasoning

#### Inputs

PaLM-E takes text plus continuous observations, which may include image embeddings, neural scene representations, and state information, and interleaves them into a multimodal sentence. 

#### What happens

1. Visual or state observations are encoded into continuous embeddings. 
2. These embeddings are inserted alongside text tokens in a single token sequence. 
3. A decoder-only large language model processes the whole sequence autoregressively. 
4. The model outputs text, which could be an answer, a caption, or a plan expressed as textual steps. 
5. When used for robot control, the generated text conditions low-level skills or planners that execute the chosen behavior. 

#### Outputs

PaLM-E outputs text, but that text can function as a plan for a robot when paired with a low-level execution system. It is therefore more grounded than a text-only LLM, but still not purely direct motor control. 

#### Why each step exists

The critical design choice is letting the model directly “see” through multimodal tokens. That helps with tasks where scene layout matters, such as reasoning about block positions, drawers, or object references. It also allows one model to cover robot planning, visual question answering, and captioning. 

#### Trade-offs

The model is more unified than SayCan, but the output side is still partly modular because robot execution relies on downstream skills or planners. Another trade-off is that smaller multimodal models can forget some pure language ability during multimodal training, though PaLM-E reports that this catastrophic forgetting becomes much smaller as model size increases. 

---

### 3. RT-2: from observations and instructions to action tokens

#### Inputs

RT-2 takes robot camera images and language instructions, and it is trained on both robot trajectories and internet-scale vision-language tasks. ([robotics-transformer2.github.io][3])

#### What happens

1. Start from a pretrained vision-language model such as PaLM-E or PaLI-X. ([robotics-transformer2.github.io][3])
2. Represent robot actions as token strings, using the same discretized action format as RT-1 but serialized into text-like tokens. ([robotics-transformer2.github.io][3])
3. Co-fine-tune the model on both robot data and web-scale vision-language data. ([robotics-transformer2.github.io][3])
4. At inference time, the model predicts action tokens directly from the current observation and instruction. ([robotics-transformer2.github.io][3])
5. The output tokens are de-tokenized back into robot actions for closed-loop control. ([robotics-transformer2.github.io][3])

#### Outputs

The output is not a textual plan for a separate planner. It is the action itself, expressed in a tokenized format that the robot can execute after de-tokenization. ([robotics-transformer2.github.io][3])

#### Why each step exists

The core idea is to avoid changing the input-output structure of the underlying vision-language model too much. If actions are just another token sequence, then the same pretrained model can be reused for control. This is what lets RT-2 inherit more semantic and visual knowledge from web data than a robot-only model would. ([robotics-transformer2.github.io][3])

#### Trade-offs

The approach is elegant, but it depends on discretized action representations and substantial robot trajectory data for co-fine-tuning. A reasonable interpretation is that RT-2 gains generalization from web knowledge, but still depends on the robot action space and embodiment being encoded in a form the model can learn reliably. ([robotics-transformer2.github.io][3])

---

## Paper-by-Paper Explanation

## 1. SayCan: *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances*

### Problem addressed

SayCan addresses the gap between what a language model can describe and what a robot can actually do in a particular environment. The paper starts from the observation that large language models contain useful semantic knowledge, but they are not grounded in robot embodiment or scene-specific feasibility. ([arXiv][1])

### Method used

The system combines a language model that scores which skill is useful with skill-specific affordance or value functions that score whether the skill can succeed now. It then iteratively selects and executes the highest-scoring feasible skill until the task is done. ([say-can.github.io][2])

### Main innovation

The main innovation is the separation and recombination of two types of knowledge: **semantic task knowledge** from the language model and **physical feasibility knowledge** from robot affordances. That lets the robot use abstract language instructions without trusting the language model blindly. ([say-can.github.io][2])

### Main findings

SayCan was evaluated on 101 real-world kitchen tasks. The updated PaLM-SayCan system chooses the correct sequence of skills 84% of the time and executes successfully 74% of the time, and the authors report that affordance grounding nearly doubles performance over non-grounded baselines. ([say-can.github.io][2])

### Limitations

The method depends on pretrained skill libraries and good affordance estimates. It does not give the language model direct visual access to the scene in the way later multimodal models do, and it does not directly predict low-level actions. Those are not flaws in the paper’s logic; they reflect its deliberate modular design. ([say-can.github.io][2])

### What changed compared with earlier work

Compared with using an LLM alone, SayCan adds grounding through affordances. Compared with older robot planners that lacked large language knowledge, it brings in commonsense task decomposition from an LLM. This makes it one of the clearest early examples of combining LLMs with robot control without retraining the whole planning stack end to end. ([say-can.github.io][2])

---

## 2. PaLM-E: *An Embodied Multimodal Language Model*

### Problem addressed

PaLM-E addresses a limitation of approaches like SayCan: the planner itself only sees text. Many robot problems depend on the actual observed scene, so the model needs a way to reason over language plus perception together. 

### Method used

PaLM-E inserts visual, state, and other continuous observation embeddings directly into a decoder-only language model as multimodal tokens. It trains these encodings end to end together with the pretrained language model on a mixture of embodied tasks and general vision-language tasks. 

### Main innovation

The main innovation is the **multimodal sentence** interface: robot observations become part of the language model’s token stream. This is a simple idea conceptually, but it changes the role of the language model from text reasoner to embodied multimodal reasoner. 

### Main findings

PaLM-E shows positive transfer across robot domains and general vision-language tasks. In the paper’s generalist setting, PaLM-E-562B reaches 66.1 on OK-VQA and 80.0 on VQAv2 test-dev while retaining more language capability than smaller multimodal variants; the paper also reports that larger-scale models suffer much less catastrophic forgetting, with only a 3.9% relative drop on natural language generation tasks for PaLM-E-562B compared with 87.3% for PaLM-E-12B. 

### Limitations

PaLM-E is more integrated than SayCan, but for robot control it still outputs text that conditions low-level skills or planners, rather than directly emitting continuous control commands. The paper also shows that smaller models can lose pure language ability during multimodal training, which means scaling is part of the solution. 

### What changed compared with earlier work

The key change is that grounding moves from an external feasibility module into the input sequence of the model itself. PaLM-E does not just ask the language model to choose among textual skills; it lets the model reason over what the robot currently sees and senses. 

---

## 3. RT-2: *Vision-Language-Action Models Transfer Web Knowledge to Robotic Control*

### Problem addressed

RT-2 addresses the next step after PaLM-E: if vision-language models can reason over images and language, can they be turned directly into robot controllers that also inherit web-scale semantic knowledge? ([robotics-transformer2.github.io][3])

### Method used

RT-2 co-fine-tunes pretrained vision-language models on robotic trajectories and internet-scale vision-language tasks. It represents robot actions as token strings so the same model can produce natural language tokens and action tokens in one unified output space. The paper presents RT-2 variants built on PaLM-E and PaLI-X. ([robotics-transformer2.github.io][3])

### Main innovation

The main innovation is very simple and very powerful: **treat actions as tokens**. This avoids building a special separate action head that breaks the pretrained model’s structure and makes it possible to reuse web-pretrained VLM backbones more directly. ([robotics-transformer2.github.io][3])

### Main findings

RT-2 is evaluated in more than 6,000 robotic trials and shows strong gains in generalization. The project and blog report more than 3x better performance on emergent skill evaluations relative to earlier baselines, improvement on unseen scenarios from RT-1’s 32% to 62%, and 90% success on the Language Table simulation benchmark compared with earlier baselines at 72%, 74%, and 77%. The paper also highlights new behaviors such as symbol understanding, simple reasoning, human recognition, and chain-of-thought-style multi-stage reasoning for control. ([robotics-transformer2.github.io][3])

### Limitations

RT-2 is closer to end-to-end control, but it still depends on discretized action representations and robot-specific fine-tuning data. The public summaries emphasize generalization gains, but they do not claim that the model is a universal robot controller across arbitrary embodiments or open-world safety-critical settings. Information about production safety mechanisms is not provided. ([robotics-transformer2.github.io][3])

### What changed compared with earlier work

Compared with SayCan and PaLM-E, RT-2 moves the action interface inside the model’s native token space. That is the clearest point where the architecture stops being “a planner that talks to robot skills” and starts becoming “a multimodal token model that directly outputs robot actions.” ([robotics-transformer2.github.io][3])

---

## Comparison Across Papers or Methods

| Dimension               | SayCan                                                                | PaLM-E                                                          | RT-2                                                                      |
| ----------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Main goal               | Ground LLM planning in robot affordances                              | Put perception into a language model for embodied reasoning     | Turn a vision-language model into a robot action model                    |
| Where grounding happens | External affordance/value functions                                   | Inside the model input via multimodal tokens                    | Inside both input and output, since actions are tokenized                 |
| Main input              | Text instruction + skill/value information                            | Text + images + states + scene representations                  | Images + text instructions + robot trajectory supervision during training |
| Main output             | High-level skill choice                                               | Textual answers or plans                                        | Direct action tokens                                                      |
| Control style           | Modular plan-and-act                                                  | Multimodal reasoning with downstream execution                  | End-to-end token-based control                                            |
| Main strength           | Interpretable and feasibility-aware                                   | Strong multimodal transfer and generalist behavior              | Best direct transfer of web-scale VLM knowledge to robot actions          |
| Main weakness           | Limited by skill library and no direct perceptual grounding inside LM | Still relies on low-level skills/planners for control use cases | Still depends on discretized action representation and robot fine-tuning  |

Sources: SayCan site and abstract, PaLM-E paper, RT-2 project and blog. ([say-can.github.io][2])

A useful interview summary is:

* **SayCan**: “LLM chooses the next useful skill, affordances decide whether it can work.” ([say-can.github.io][2])
* **PaLM-E**: “Robot observations become tokens inside a large language model.” 
* **RT-2**: “Robot actions become tokens too, so the same model can directly output control.” ([robotics-transformer2.github.io][3])

---

## Real-World System and Application

A practical embodied AI system built from these ideas would likely have four layers:

1. **Perception**, which turns images or states into structured embeddings. 
2. **Task reasoning**, which interprets user language in the context of the current scene. ([say-can.github.io][2])
3. **Action selection or action generation**, which either chooses a skill from a library or directly predicts action tokens. ([say-can.github.io][2])
4. **Execution**, which uses either low-level skill controllers or de-tokenized robot actions to move the hardware. 

If you were building a real robot system today, SayCan is the most natural fit when you already have a reliable skill library and want better semantic planning. PaLM-E is attractive when you want a single multimodal model that can reason across perception and language and still hand off to lower-level controllers. RT-2 is most attractive when you want to push toward a unified vision-language-action policy that can directly benefit from web-scale pretraining. That comparison is a reasoned synthesis across the papers. ([say-can.github.io][2])

The papers support several real-world use cases: long-horizon mobile manipulation in kitchens, tabletop rearrangement and pushing tasks, visually grounded question answering, and symbolic or semantic robot commands involving unseen objects or concepts. ([say-can.github.io][2])

Information about full production deployment stacks, fleet management, fail-safe control layers, safety certification, and human oversight interfaces is not provided in enough operational detail by these papers to draw stronger implementation conclusions. ([say-can.github.io][2])

---

## Limitations and Trade-offs

The most important trade-off across the papers is **modularity versus unification**. SayCan is modular and interpretable: you can inspect skill candidates, language scores, and affordance scores. But that modularity limits flexibility because the system cannot go beyond its skill library. RT-2 is more unified and more direct, but that also makes the model itself responsible for more of the control stack. PaLM-E sits in the middle. ([say-can.github.io][2])

A second trade-off is **data efficiency versus direct grounding**. SayCan needs less end-to-end robot data because it reuses language knowledge and pretrained skills. PaLM-E and RT-2 gain more direct grounding by integrating perception and actions more tightly, but they also rely on more multimodal and robot-specific training. ([arXiv][1])

A third trade-off is **generality versus embodiment specificity**. Large pretrained models know many things from the web, but robots still have specific embodiments, specific grippers, specific action spaces, and specific environments. SayCan handles this with affordances, PaLM-E with multimodal observations, and RT-2 with co-fine-tuning on real robot trajectories. None of the papers claims that language or web pretraining alone solves embodiment. ([arXiv][1])

There is also a **reasoning versus execution** trade-off. A model may produce a correct high-level plan but still fail in execution because perception is wrong, the skill fails, or the physical world changes. SayCan explicitly separates these layers. PaLM-E and RT-2 reduce some of that separation, but embodied AI still has to survive physical failures in a way ordinary LLM tasks do not. ([say-can.github.io][2])

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain that SayCan grounds language planning with affordance functions, PaLM-E feeds perception into a large language model as multimodal tokens, and RT-2 makes actions part of the model’s token output space so a web-pretrained vision-language model can directly control a robot. ([say-can.github.io][2])

### Likely interview questions

#### 1. What problem does SayCan solve?

SayCan solves the problem that an LLM may propose a semantically reasonable next step that is not actually executable by the robot in the current environment. It fixes this by combining language-model usefulness scores with affordance or value scores for skills. ([say-can.github.io][2])

#### 2. Why is SayCan called “Do As I Can, Not As I Say”?

Because the robot should not blindly do whatever the language model “says” sounds good. It should do what it actually **can** do, given its current affordances and skills. That phrase captures the central grounding idea of the paper. ([arXiv][1])

#### 3. What is PaLM-E’s main architectural idea?

PaLM-E turns observations into embeddings and inserts them into the language model’s token stream, creating multimodal sentences. This lets one decoder-only model reason over text plus perception. 

#### 4. Is PaLM-E an end-to-end low-level robot controller?

Not in the strongest sense. When used for robot control, it outputs text that conditions low-level skills or planners. It is more grounded than a text-only LLM, but still usually sits above a lower-level execution layer. 

#### 5. What makes RT-2 different from PaLM-E?

RT-2 pushes one step further by representing actions as tokens and training the model to output them directly. PaLM-E reasons over multimodal input; RT-2 uses a similar foundation but makes actions native outputs. ([robotics-transformer2.github.io][3])

#### 6. Why is tokenizing actions useful?

Because it lets existing vision-language models be reused with minimal structural change. If actions are just another token sequence, the model can learn language and control in a unified output space. ([robotics-transformer2.github.io][3])

#### 7. What does “transfer web knowledge to robotic control” mean in RT-2?

It means the robot policy benefits from semantic and visual knowledge learned from internet-scale vision-language data, such as understanding symbols, object categories, or simple reasoning prompts that were not fully covered in the robot data. ([Google DeepMind][5])

#### 8. How do the papers differ in system design philosophy?

SayCan is a grounded planner over fixed skills. PaLM-E is a multimodal embodied reasoner that still typically uses downstream execution modules. RT-2 is the most unified, since it directly predicts robot actions. That is the cleanest cross-paper comparison. ([say-can.github.io][2])

---

## Glossary

| Term                               | Beginner-friendly definition                                                                    |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| Embodied AI                        | AI for agents that perceive and act in the real or simulated world                              |
| Grounding                          | Connecting symbols or language to actual observations and action consequences                   |
| Affordance                         | A measure of whether an action is possible or likely to succeed in the current state            |
| Value function                     | A learned function that estimates how good or feasible a skill is from the current state        |
| Skill library                      | A set of reusable low-level robot behaviors such as pick, open, or move                         |
| Multimodal                         | Involving more than one kind of input, such as text, images, and robot state                    |
| Multimodal sentence                | PaLM-E’s term for a token sequence that interleaves text with continuous observation embeddings |
| Embodied reasoning                 | Reasoning that depends on current perception, embodiment, and physical context                  |
| Vision-language model (VLM)        | A model trained jointly on visual and language tasks                                            |
| Vision-language-action model (VLA) | A model that takes vision and language input and outputs actions                                |
| Co-fine-tuning                     | Fine-tuning on a new task while still training on some older pretraining data                   |
| Action tokens                      | Discrete token representations of robot actions used by RT-2                                    |
| Closed-loop control                | Repeatedly choosing actions using fresh observations from the environment                       |
| Catastrophic forgetting            | Losing old abilities while training on new tasks                                                |
| Positive transfer                  | Improvement on one task because related tasks or datasets were trained jointly                  |

Sources: terminology and architecture descriptions across SayCan, PaLM-E, and RT-2 materials. ([say-can.github.io][2])

---

## Recap

These three papers give a very clear teaching progression for embodied AI in robotics. **SayCan** shows how to use an LLM as a high-level planner only after grounding it with affordances. **PaLM-E** shows how to put robot observations directly into a large language model so it can reason over text and perception together. **RT-2** shows how to unify perception, language, and action more tightly by turning actions into tokens and directly predicting them with a pretrained vision-language backbone. ([say-can.github.io][2])

The biggest conceptual lesson is that embodied AI is not just “put an LLM on a robot.” It is about how semantic knowledge, perception, embodiment, and control are connected. SayCan connects them through external affordances, PaLM-E connects them through multimodal input tokens, and RT-2 connects them through a shared token space that includes actions. If you can explain that progression clearly, you have an interview-ready understanding of where modern embodied AI is heading. ([say-can.github.io][2])

---

## Key Citations

* *Do As I Can, Not As I Say: Grounding Language in Robotic Affordances* (arXiv abstract). ([arXiv][1])

* *SayCan: Grounding Language in Robotic Affordances* (project page). ([say-can.github.io][2])

* *PaLM-E: An Embodied Multimodal Language Model* (PMLR paper PDF). 

* *RT-2: Vision-Language-Action Models* (project page). ([robotics-transformer2.github.io][3])

* *RT-2: New model translates vision and language into action* (Google DeepMind blog). ([Google DeepMind][5])

[1]: https://arxiv.org/abs/2204.01691?utm_source=chatgpt.com "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"
[2]: https://say-can.github.io/ "SayCan: Grounding Language in Robotic Affordances"
[3]: https://robotics-transformer2.github.io/ "RT-2: Vision-Language-Action Models"
[4]: https://proceedings.mlr.press/v205/ichter23a/ichter23a.pdf?utm_source=chatgpt.com "Do As I Can, Not As I Say: Grounding Language in Robotic Affordances"
[5]: https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/ "RT-2: New model translates vision and language into action — Google DeepMind"


---
---
---

