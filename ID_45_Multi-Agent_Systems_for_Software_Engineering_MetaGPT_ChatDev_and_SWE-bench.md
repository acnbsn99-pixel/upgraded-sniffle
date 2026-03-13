# Multi-Agent Systems for Software Engineering: MetaGPT, ChatDev, and SWE-bench

## What This Report Teaches

This report explains three influential works that sit at the intersection of **multi-agent systems** and **software engineering**, but they play different roles in the ecosystem. **MetaGPT** is a multi-agent framework that tries to organize software development like a structured company workflow with role-specialized agents and standard operating procedures. **ChatDev** is another multi-agent software-development framework, but it emphasizes communication itself: who talks to whom, what they talk about, and how communication can reduce hallucinations during collaborative coding. **SWE-bench** is different from both: it is not a multi-agent framework for building software, but a benchmark for evaluating whether language models can resolve real GitHub issues in real repositories. ([arXiv][1])

Two source corrections are important. The user-provided arXiv ID `2307.07924` matches **ChatDev**, not MetaGPT. The correct MetaGPT paper is arXiv **2308.00352**. The user-provided arXiv ID `2310.03533` is a survey paper, not SWE-bench. The correct SWE-bench paper is arXiv **2310.06770**. I use the papers that match the supplied titles so the report stays on the intended topic. ([arXiv][2])

By the end, you should understand how MetaGPT and ChatDev structure multi-agent collaboration for software work, why standardized workflows and communication protocols matter, why SWE-bench changed how people evaluate software-engineering agents, and what interviewers usually want when they ask about “multi-agent software engineering systems.” ([ar5iv][3])

---

## Key Takeaways

* **MetaGPT and ChatDev are systems for doing software engineering, while SWE-bench is a system for measuring whether models can actually solve real software issues.** This matters because research progress in agent frameworks is easy to overstate without a hard benchmark. The practical implication is that a strong agent story needs both an architecture and a realistic evaluation setup. ([ar5iv][3])

* **MetaGPT’s main idea is to encode human software workflows as Standardized Operating Procedures (SOPs).** This matters because many naïve multi-agent systems fail through unstructured conversations and cascading hallucinations. The practical implication is that role design and workflow structure are as important as the base model. ([arXiv][1])

* **ChatDev’s main idea is that software development can be organized as controlled communication among specialized agents.** It uses a “chat chain” to decide what to communicate and “communicative dehallucination” to guide how agents communicate. This matters because uncontrolled multi-agent discussion can amplify errors instead of fixing them. The practical implication is that communication policy is a core design variable in software agents. ([arXiv][4])

* **MetaGPT and ChatDev both imitate parts of real software organizations, but they do it differently.** MetaGPT emphasizes assembly-line workflow, structured artifacts, and SOP-driven handoffs. ChatDev emphasizes a waterfall-like process with staged multi-turn dialogue among software-role agents. The practical implication is that “multi-agent software engineering” is not one pattern; it ranges from workflow automation to communication orchestration. ([arXiv][1])

* **SWE-bench raised the difficulty of software-engineering evaluation dramatically.** It contains 2,294 real GitHub issue–pull request pairs from 12 Python repositories, and solving an instance often requires edits across multiple functions, classes, or files. The practical implication is that repo-level issue resolution is much harder than function-level code generation benchmarks like HumanEval. 

* **Early results on SWE-bench were extremely low, even for strong proprietary models.** In the original paper, Claude 2 resolved 4.8% of issues with oracle retrieval, while GPT-4 resolved 1.74%. This matters because it showed that real-world software repair was far from solved. The practical implication is that impressive demo-based coding agents can still perform poorly on realistic, execution-verified maintenance tasks. 

* **MetaGPT reports very strong benchmark results for code generation and stronger software-project statistics than ChatDev on its SoftwareDev benchmark.** This matters because it suggests that structured SOP-based orchestration can improve both code quality and project-level execution. The practical implication is that disciplined multi-agent coordination can outperform freer-form multi-agent collaboration. ([ar5iv][3])

* **The central systems lesson across these papers is that software engineering agents need structure, communication control, and hard evaluation.** The practical implication is that building a useful software agent is not just about making the LLM stronger; it is about designing roles, handoffs, artifact formats, execution loops, and benchmarks. ([ar5iv][3])

---

## Background and Foundations

### Why software engineering is a special testbed for agents

Software engineering is attractive for agent systems because it combines language, planning, formal structure, and verifiability. A model can read requirements, write code, edit repositories, run tests, and inspect failures. But it is also difficult because real software work is not just “generate one function.” It often involves long contexts, many files, hidden dependencies, vague issue descriptions, and iterative debugging. SWE-bench makes exactly this point: resolving real GitHub issues often requires coordination across multiple functions, classes, and files, plus interaction with execution environments. 

### What makes a system “multi-agent” here

A **multi-agent system** in this context means a framework where multiple language-model-driven roles cooperate, rather than one monolithic model doing everything in one prompt. Examples include roles like product manager, architect, engineer, code reviewer, or tester. MetaGPT and ChatDev both use role specialization, but with different organizational theories. MetaGPT frames this as SOP-guided collaboration and assembly-line task decomposition. ChatDev frames it as language-based collaboration across software roles inside staged development phases. ([arXiv][1])

### Why benchmark choice matters

Before SWE-bench, many coding evaluations focused on short, self-contained tasks such as function completion. Those tasks are useful, but they do not capture real repo-level maintenance work. SWE-bench was introduced because traditional benchmarks were too limited for evaluating practical software agents. That makes it especially relevant in a report about software-engineering multi-agent systems: it tells you whether those systems are solving real maintenance problems, not only polished toy tasks. 

### How these three papers fit together

Conceptually, the papers form a useful triangle:

1. **MetaGPT** proposes a structured multi-agent framework for software development. ([arXiv][1])
2. **ChatDev** proposes a communication-centered multi-agent framework for software development. ([arXiv][4])
3. **SWE-bench** proposes a realistic benchmark for whether models can actually resolve real software issues in codebases. 

That relationship matters because it separates **system design** from **system evaluation**.

---

## Big Picture First

The easiest way to understand the topic is to ask three different questions:

1. **How should a software-development agent organization be structured?** MetaGPT answers with SOPs, roles, and artifact-oriented handoffs. ([arXiv][1])
2. **How should agents communicate during software development?** ChatDev answers with chat chains and communicative dehallucination inside staged development phases. ([arXiv][4])
3. **How should we tell whether these systems really work on real software maintenance?** SWE-bench answers with execution-verified GitHub issue resolution at repository scale. 

| Paper     | Primary role          | Core idea                                                                    | What it contributes most       |
| --------- | --------------------- | ---------------------------------------------------------------------------- | ------------------------------ |
| MetaGPT   | Multi-agent framework | Encode human software workflows as SOPs and structured roles                 | Stronger process structure     |
| ChatDev   | Multi-agent framework | Use staged dialogue plus controlled communication among software-role agents | Stronger communication design  |
| SWE-bench | Benchmark             | Use real GitHub issues and tests to evaluate repo-level issue resolution     | Stronger realism in evaluation |

The table summarizes the intended role of each paper in the ecosystem. ([ar5iv][3])

A second big-picture point is that **software engineering agents are not only code generators**. MetaGPT generates artifacts like requirements and technical designs before code. ChatDev includes design, coding, testing, and documentation in its framing. SWE-bench then checks whether a model can actually repair codebases when given only an issue description and the repository. ([arXiv][1])

---

## Core Concepts Explained

### 1. Standardized Operating Procedures (SOPs)

**What it is:** A structured workflow that specifies how roles collaborate and in what order they produce outputs. MetaGPT encodes SOPs into prompt sequences. ([arXiv][1])

**Why it exists:** Naïvely chaining agents often creates logic inconsistencies and cascading hallucinations. SOPs are meant to reduce that chaos. ([arXiv][1])

**How it works at a high level:** Different agents play specialized roles, follow predefined steps, and exchange structured outputs such as requirements, designs, APIs, and code. ([ar5iv][3])

**Why it matters:** It turns multi-agent collaboration from free-form chatting into process engineering.

### 2. Assembly-line paradigm

**What it is:** MetaGPT’s metaphor for software development as staged role-specialized production. ([arXiv][1])

**Why it exists:** Large tasks are easier to handle when decomposed into subtasks assigned to different roles. ([arXiv][1])

**How it works at a high level:** Product roles analyze requirements, architect roles define designs and interfaces, engineering roles implement, and feedback mechanisms debug and revise outputs. ([ar5iv][3])

**Why it matters:** It gives a concrete organizational theory for why multiple agents might outperform a single one.

### 3. Chat chain

**What it is:** ChatDev’s mechanism for deciding **what** agents should communicate at each stage. ([arXiv][4])

**Why it exists:** Multi-agent conversations can drift or become redundant if they are not structured. 

**How it works at a high level:** The development process is split into stages and then into smaller subtasks, and the chat chain organizes which sub-dialogues happen in sequence. ([arXiv][4])

**Why it matters:** It treats communication planning itself as a systems component.

### 4. Communicative dehallucination

**What it is:** ChatDev’s mechanism for controlling **how** agents communicate so hallucinations are reduced. ([arXiv][2])

**Why it exists:** Coding hallucinations can produce incomplete, unexecutable, or inaccurate software artifacts. 

**How it works at a high level:** The paper frames it as communication rules that help agents validate and correct one another during dialogue. ([arXiv][2])

**Why it matters:** It highlights that agent reliability is partly a communication-design problem, not only a model-capability problem.

### 5. Executability

**What it is:** A software-quality measure used by MetaGPT’s SoftwareDev evaluation. It rates whether the generated project runs correctly, on a scale where 4 means flawless. ([ar5iv][3])

**Why it exists:** In software engineering, “looks plausible” is not enough. The system has to run. ([ar5iv][3])

**Why it matters:** It is much closer to real developer concerns than surface-level code similarity.

### 6. Repository-level issue resolution

**What it is:** SWE-bench’s core task: given a repository snapshot and an issue description, generate a patch that resolves the issue and passes the relevant tests. 

**Why it exists:** Real software maintenance usually happens at repository scale, not as isolated functions. 

**Why it matters:** It is one of the clearest realism upgrades over earlier code-generation benchmarks.

### 7. Retrieval setting: BM25 vs oracle

**What it is:** SWE-bench evaluates models under different context-retrieval conditions. **BM25** is a standard lexical retrieval baseline; **oracle retrieval** supplies the relevant files more directly. 

**Why it exists:** Repo-level code is too large to fit naively into the context window, so retrieval becomes part of the task. 

**Why it matters:** It separates “can the model patch the right code once shown it?” from “can the system find the right code in the first place?”

---

## Step-by-Step Technical Walkthrough

## 1. MetaGPT

### Inputs

MetaGPT starts from a user requirement and a team of role-specialized agents. The framework uses SOPs to coordinate the flow from requirement analysis to code production and revision. ([arXiv][1])

### What happens

1. **Role assignment.** Different agents take roles such as product, architect, and engineer. ([ar5iv][3])
2. **SOP-driven workflow.** The task is broken into stages with structured intermediate outputs, such as PRD generation, technical design generation, and API interface generation. ([ar5iv][3])
3. **Code generation.** Engineering agents implement the solution. ([ar5iv][3])
4. **Executive feedback.** The framework executes and debugs code during runtime, which the paper reports improves MBPP performance by 5.4 absolute points. ([ar5iv][3])

### Outputs

MetaGPT outputs both software artifacts and code. The paper emphasizes that it can generate not only code but also structured engineering artifacts. ([ar5iv][3])

### Purpose

The main purpose is to reduce unproductive collaboration and cascading errors by imposing a human-like engineering workflow on the agent team. ([ar5iv][3])

### Trade-offs

The framework is more structured and, in benchmark settings, more effective than looser alternatives, but it also uses more tokens than ChatDev on SoftwareDev. On the main SoftwareDev table, MetaGPT uses 31,255 tokens versus ChatDev’s 19,292, though it produces many more code lines and lower tokens-per-line productivity cost. ([ar5iv][3])

---

## 2. ChatDev

### Inputs

ChatDev starts from a user requirement and a virtual company of role-specific software agents. ([arXiv][4])

### What happens

1. **Stage decomposition.** The framework mirrors a waterfall-like process with stages such as design, coding, testing, and, in the abstract framing, documentation. ([arXiv][4])
2. **Chat chain.** Each stage is broken into smaller subtasks with guided conversations about what should be discussed. ([arXiv][4])
3. **Communicative dehallucination.** The conversation structure is also designed to reduce coding hallucinations and improve reliability. ([arXiv][2])
4. **Artifact generation.** The agents collaboratively produce designs, code, tests, and related outputs through multi-turn dialogues. 

### Outputs

The system outputs complete software projects. In the paper’s analysis of 70 tasks, it generated an average of 17.04 files per software, took 409.84 seconds on average, and cost $0.2967 on average. ([OpenReview][5])

### Purpose

The purpose is to use language as a unifying medium for software development across multiple roles and phases, instead of building separate specialized models for each software-engineering stage. ([arXiv][4])

### Trade-offs

ChatDev is fast and cheap, and the paper emphasizes that it can finish the full software-development process in under seven minutes for less than one dollar on average. But later comparison tables in MetaGPT suggest that freer-form chat-based collaboration is weaker than SOP-driven collaboration on more demanding software-development evaluations. ([arXiv][4])

---

## 3. SWE-bench

### Inputs

SWE-bench provides a repository snapshot, a GitHub issue description, and the repository’s tests. The model or agent must generate a code patch to resolve the issue. 

### What happens

1. **Task construction.** The benchmark links real GitHub issues to merged pull requests and associated tests. After filtering, it keeps 2,294 task instances from 12 Python repositories. 
2. **Context retrieval.** The model may receive BM25-retrieved context or oracle-retrieved files. 
3. **Patch generation.** The model edits the repository to address the issue. 
4. **Execution-based evaluation.** The patch is tested against the repository’s test framework. 

### Outputs

The output is a candidate patch, and success is measured by whether the repository’s tests pass appropriately after applying the patch. 

### Purpose

The benchmark’s purpose is to measure whether language models can handle realistic software-engineering tasks involving long contexts, file localization, multi-file reasoning, and execution feedback. 

### Trade-offs

SWE-bench is much more realistic than small code-generation benchmarks, but that realism makes the task much harder. In the original paper, all tested models struggle badly, and performance depends strongly on retrieval quality and context length. 

---

## Paper-by-Paper Explanation

## 1. MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework

### The problem addressed

MetaGPT starts from the observation that many LLM-based multi-agent systems can handle simple dialogue tasks but become unreliable on more complex tasks because hallucinations cascade across agents. The paper asks whether software-style workflow structure can make multi-agent collaboration more reliable. ([arXiv][1])

### The method used

MetaGPT encodes Standardized Operating Procedures into prompt sequences and organizes a team of specialized agents in an assembly-line style workflow. It also adds an executive feedback mechanism that executes and debugs code during runtime. ([arXiv][1])

### The main innovation

The main innovation is the claim that **human workflow structure** is a missing ingredient in many multi-agent systems. Instead of only adding more agents, MetaGPT adds more process. ([arXiv][1])

### The main findings

MetaGPT reports state-of-the-art Pass@1 performance of 85.9% on HumanEval and 87.7% on MBPP. On its SoftwareDev evaluation, it outperforms ChatDev on most metrics, including executability, runtime, productivity, and human revision cost. In one reported table, MetaGPT achieves executability 3.75 versus ChatDev’s 2.25 and runtime 541 seconds versus 762 seconds. ([arXiv][6])

### The limitations

MetaGPT is evaluated partly on a self-generated SoftwareDev benchmark, not only on third-party benchmarks. It also uses more tokens than ChatDev in the reported SoftwareDev comparison. Information not provided: the paper does not establish that SOP-driven multi-agent collaboration is universally best for every software task or repository maintenance scenario. ([ar5iv][3])

### What changed compared with earlier work

Compared with looser multi-agent frameworks, MetaGPT adds explicit SOPs, structured artifacts, and runtime feedback. It pushes the field from “multiple agents talking” toward “multiple agents following a software process.” ([ar5iv][3])

---

## 2. Communicative Agents for Software Development (ChatDev)

### The problem addressed

ChatDev asks whether software engineering can be driven end to end by multiple communicating LLM agents instead of separate specialized systems for each software-development phase. It also focuses on the risk of **coding hallucinations** in autonomous multi-agent development. ([arXiv][4])

### The method used

The framework creates a virtual chat-powered software company with specialized software-role agents. It follows staged development, uses a chat chain to structure subtasks, and applies communicative dehallucination to guide how agents communicate. ([arXiv][4])

### The main innovation

The main innovation is communication design. The paper argues that software development among agents should be mediated through carefully structured language exchange, not only through independent role prompts. ([arXiv][2])

### The main findings

ChatDev reports that it can complete the full software-development process in under seven minutes at a cost of less than one dollar on average. On a 70-task analysis, it generated an average of 17.04 files per software, alleviated potential vulnerabilities caused by code hallucinations 13.23 times on average, had a software production time of 409.84 seconds, and incurred an average cost of $0.2967. ([arXiv][4])

### The limitations

ChatDev is impressive in speed and cost, but later MetaGPT comparisons suggest that it underperforms more structured SOP-based systems on project executability and revision cost in their SoftwareDev evaluation. Information not provided: the paper does not show performance on a real-repository maintenance benchmark like SWE-bench. ([ar5iv][3])

### What changed compared with earlier work

Compared with single-agent coding or loosely role-played frameworks, ChatDev treats communication itself as a first-class system component. It is one of the clearest early examples of software development as orchestrated multi-agent dialogue. ([arXiv][2])

---

## 3. SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

### The problem addressed

SWE-bench addresses an evaluation gap. Existing coding benchmarks did not reflect real software maintenance, especially issue-driven repository editing with tests. The paper asks whether language models can actually resolve real-world GitHub issues. 

### The method used

The authors build a benchmark from real GitHub issues, corresponding pull requests, repository snapshots, and tests. Models are asked to generate patches and are evaluated through execution. The dataset contains 2,294 issues from 12 popular Python repositories. 

### The main innovation

The main innovation is realism. SWE-bench upgrades evaluation from isolated code snippets to real codebases, long contexts, retrieval, and test-based verification. 

### The main findings

The benchmark is hard. In the original paper, Claude 2 achieves 4.80% resolution in the oracle retrieval setting and 1.96% in the BM25 setting. GPT-4 achieves 1.74% in the oracle setting. The paper also shows that performance drops as context length increases and that even oracle retrieval is far from enough to make current models reliable issue resolvers. 

### The limitations

SWE-bench is a benchmark, not an agent framework. It tells you whether a system works, but not how to build one. Also, retrieval strategy and context formatting influence results substantially, so the benchmark measures both code-editing ability and repository-context handling. 

### What changed compared with earlier work

Compared with HumanEval-style benchmarks, SWE-bench moved the field toward real repository maintenance. That shift strongly influenced the later wave of software-engineering agents and scaffolds. 

---

## Comparison Across Papers or Methods

### Comparison by role in the ecosystem

| Aspect                    | MetaGPT                           | ChatDev                                        | SWE-bench                      |
| ------------------------- | --------------------------------- | ---------------------------------------------- | ------------------------------ |
| What it is                | Software-agent framework          | Software-agent framework                       | Software-engineering benchmark |
| Main organizing principle | SOPs and role workflow            | Communication structure                        | Execution-verified repo tasks  |
| Software scope            | Requirements to code and revision | Design, coding, testing, documentation framing | Repository issue resolution    |
| Primary contribution      | Better process structure          | Better communication structure                 | Better evaluation realism      |

This table compares the papers by what they contribute to the software-agent stack. ([ar5iv][3])

### Comparison by how they treat software engineering

| Question                             | MetaGPT                                         | ChatDev                                   | SWE-bench                                            |
| ------------------------------------ | ----------------------------------------------- | ----------------------------------------- | ---------------------------------------------------- |
| What is software engineering mainly? | A structured organizational workflow            | A staged communication problem            | A real-world repo maintenance challenge              |
| What does success look like?         | High Pass@1 and stronger project executability  | Fast, cheap end-to-end project completion | Passing real repository tests                        |
| What is the main failure source?     | Unstructured collaboration and missing workflow | Hallucination during agent communication  | Long context, localization, and execution complexity |

The comparison above is a synthesis of each paper’s framing and evaluation focus. ([ar5iv][3])

### What changed historically

A useful way to explain the progression is:

1. **ChatDev** showed that software development could be organized as multi-agent language collaboration. ([arXiv][4])
2. **MetaGPT** argued that collaboration alone was not enough, and that SOPs and structured workflows improved reliability and code quality. ([arXiv][1])
3. **SWE-bench** then raised the bar on evaluation, showing that even strong models remained very weak on realistic repository maintenance. 

That historical story is a reasoned synthesis across the three papers.

---

## Real-World System and Application

A practical multi-agent software-engineering system inspired by these papers would have at least five layers:

1. **Requirement interpretation** by a planner or product role.
2. **Structured artifact generation** such as PRDs, designs, or interface specs.
3. **Implementation and review** by coding roles.
4. **Execution and debugging** via tests and runtime feedback.
5. **Evaluation against realistic issue-resolution tasks** like those represented in SWE-bench. ([ar5iv][3])

MetaGPT contributes most to layers 1–4 through structured workflow and feedback. ChatDev contributes most to layers 1–4 through communication design and staged dialogue. SWE-bench contributes most to layer 5 by forcing systems to prove themselves on real repositories. ([ar5iv][3])

Information not provided: these papers do not give a complete production architecture for permissioning, sandbox security, developer approval loops, merge governance, audit trails, or rollback strategy for autonomous code changes. They are strong research artifacts, but not full enterprise deployment blueprints. ([ar5iv][3])

---

## Limitations and Trade-offs

| Limitation or trade-off  | Concrete meaning                                                       | Why it matters                                                       |
| ------------------------ | ---------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Structure vs flexibility | SOP-heavy systems can be more reliable, but may be more rigid          | Strong process can reduce hallucination but may reduce improvisation |
| Communication overhead   | More agent dialogue can improve reasoning, but costs tokens and time   | Multi-agent systems can become expensive or unstable                 |
| Benchmark mismatch       | Demo-style project generation is different from repo-level maintenance | A system that looks good in demos may still fail on SWE-bench        |
| Retrieval dependence     | Repo-scale issue solving depends on finding the right files            | Good coding is useless if the wrong context is retrieved             |
| Evaluation difficulty    | Executability, cost, and real test passing measure different things    | There is no single perfect “software agent” metric                   |

The first two points are reasoned engineering interpretations supported by the system designs and reported token/runtime trade-offs. The last three are directly motivated by the contrast between MetaGPT/ChatDev project generation and SWE-bench’s execution-verified repo tasks. ([ar5iv][3])

A strong interview answer should say one thing clearly: **multi-agent software engineering is not solved by adding more agents**. The hard part is organizing roles, constraining communication, handling execution feedback, and proving performance on realistic benchmarks. ([ar5iv][3])

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain that MetaGPT and ChatDev are two different answers to the question “how should software-development agents collaborate?” MetaGPT says collaboration should follow SOPs and structured engineering artifacts. ChatDev says collaboration should be built around controlled multi-turn communication among specialized roles. Then you should be able to explain that SWE-bench is not another agent framework, but the benchmark that checks whether such systems can actually solve real repository issues. ([ar5iv][3])

### Likely interview questions

#### 1. What is MetaGPT’s main idea?

MetaGPT encodes Standardized Operating Procedures into multi-agent workflows so specialized roles collaborate more like a real software organization and less like a loose group chat. ([arXiv][1])

#### 2. What is the difference between MetaGPT and ChatDev?

MetaGPT emphasizes structured workflow, artifact generation, and SOP-guided handoffs. ChatDev emphasizes staged dialogue, chat chains, and communication rules that reduce hallucinations. ([ar5iv][3])

#### 3. What is “communicative dehallucination”?

It is ChatDev’s idea that agents need rules for how they communicate so that hallucinations in coding and debugging are reduced during collaboration. ([arXiv][2])

#### 4. Why are SOPs important in MetaGPT?

Because unstructured multi-agent collaboration can create cascading errors. SOPs make roles, steps, and intermediate outputs explicit, which improves robustness. ([arXiv][1])

#### 5. Why is SWE-bench important if it is not a multi-agent framework?

Because it gives a realistic way to evaluate whether software agents actually solve real GitHub issues in repositories, instead of only performing well on toy code-generation tasks. 

#### 6. What makes SWE-bench harder than HumanEval?

SWE-bench requires repository understanding, file localization, long-context processing, multi-file edits, and execution-based verification, while HumanEval mostly tests short self-contained programming tasks. 

#### 7. What were the main early SWE-bench results?

In the original paper, Claude 2 resolved 4.8% of instances in the oracle retrieval setting, while GPT-4 resolved 1.74%, showing that real repo-level issue resolution was still extremely hard. 

#### 8. What does MetaGPT contribute beyond code generation?

It contributes requirements generation, technical design generation, API interface generation, role-based task management, code review, and precompilation execution in the framework comparison. ([ar5iv][3])

#### 9. Why is retrieval important on SWE-bench?

Because repositories are too large to fit naively into the model context, so the system must find the relevant files before it can patch them. Performance drops significantly under weaker retrieval settings. 

#### 10. What is the main lesson across all three papers?

Good software agents need a combination of structured collaboration, controlled communication, execution feedback, and realistic evaluation. No single ingredient is enough by itself. ([ar5iv][3])

---

## Glossary

| Term                              | Beginner-friendly definition                                                              |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| Multi-agent system                | A system where multiple LLM-driven roles cooperate to solve a task                        |
| SOP                               | Standardized Operating Procedure; a predefined workflow for how agents should collaborate |
| Assembly-line paradigm            | Organizing agents into sequential specialized roles, like a production pipeline           |
| Chat chain                        | ChatDev’s mechanism for structuring which sub-conversations happen and in what order      |
| Communicative dehallucination     | Communication rules intended to reduce hallucinations during agent collaboration          |
| Executability                     | Whether generated software actually runs correctly                                        |
| Repository-level issue resolution | Fixing a real issue inside a full software repository, not just generating one function   |
| Patch                             | A set of code changes applied to an existing codebase                                     |
| BM25 retrieval                    | A standard lexical method for retrieving relevant files or documents                      |
| Oracle retrieval                  | A setting where the system is given highly relevant files directly                        |
| Pass@1                            | The probability that the first generated solution is correct                              |
| Human revision cost               | How much manual debugging or correction is needed after generation                        |
| Code hallucination                | Code that looks plausible but is incomplete, wrong, or unexecutable                       |
| SoftwareDev                       | MetaGPT’s project-style software-development benchmark                                    |
| SWE-Llama                         | A fine-tuned code model evaluated in the original SWE-bench paper                         |

The definitions above come from how the papers frame their systems and evaluations. ([ar5iv][3])

---

## Recap

You should now see the structure of this topic clearly. **MetaGPT** is the workflow-structure paper. **ChatDev** is the communication-structure paper. **SWE-bench** is the realism-and-evaluation paper. Together, they show that software-engineering agents are not only about code generation; they are about roles, communication, execution, and hard repo-level validation. ([ar5iv][3])

The most important practical lesson is that agent systems for software engineering need to be judged at the level of **real codebase maintenance**, not only on polished demos or function-level tasks. MetaGPT and ChatDev show different ways to organize collaborative software agents, while SWE-bench shows how far the field still had to go on realistic issue resolution. ([ar5iv][3])

What remains limited is also important. These papers do not provide a complete production framework for safe autonomous software delivery, and the benchmark results make clear that repo-level autonomous repair was still very weak in the original SWE-bench setting. That realism is part of understanding the field well. 

---

## Key Citations

MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework. ([arXiv][1])

Communicative Agents for Software Development. ([arXiv][4])

SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ([arXiv][7])

Source mismatch note for user-supplied `2307.07924` and `2310.03533`. ([arXiv][2])

[1]: https://arxiv.org/abs/2308.00352?utm_source=chatgpt.com "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework"
[2]: https://arxiv.org/abs/2307.07924?utm_source=chatgpt.com "[2307.07924] ChatDev: Communicative Agents for Software Development"
[3]: https://ar5iv.labs.arxiv.org/html/2308.00352 "[2308.00352] MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework"
[4]: https://export.arxiv.org/abs/2307.07924v4 "[2307.07924v4] Communicative Agents for Software Development"
[5]: https://openreview.net/pdf?id=yW0AZ5wPji&utm_source=chatgpt.com "arXiv:2307.07924v4 [cs.SE] 19 Dec 2023 - OpenReview"
[6]: https://arxiv.org/html/2308.00352v6?utm_source=chatgpt.com "MetaGPT: Meta Programming for a Multi-Agent Collaborative Framework"
[7]: https://arxiv.org/abs/2310.06770?utm_source=chatgpt.com "SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"

---
---
---

