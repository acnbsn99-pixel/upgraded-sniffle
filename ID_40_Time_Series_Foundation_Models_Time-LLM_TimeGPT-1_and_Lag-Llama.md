# Time Series Foundation Models: Time-LLM, TimeGPT-1, and Lag-Llama

## What This Report Teaches

This report explains three different ideas that are often grouped under the broad theme of **time series foundation models**, even though they are not the same kind of system. **Time-LLM** repurposes a pretrained large language model for forecasting without changing the frozen backbone. **TimeGPT-1** is a native time-series forecasting model trained on a very large corpus of time series data. **Lag-Llama** is another native time-series foundation model, but it focuses specifically on **probabilistic** forecasting and uses a LLaMA-style decoder-only Transformer with lag features as covariates. ([arXiv][1])

Two source notes matter before we begin. The user-provided URL for **Time-LLM** (`2302.00861`) actually points to **SimMTM**, not Time-LLM, and the user-provided URL for **Lag-Llama** (`2310.10196`) actually points to a survey paper, not Lag-Llama. I used the papers that match the supplied titles: **Time-LLM** is arXiv **2310.01728**, and **Lag-Llama** is arXiv **2310.08278**. The provided **TimeGPT-1** URL is correct. ([arXiv][2])

By the end, you should understand what makes a time-series model “foundation-model-like,” how these three systems differ in architecture and training philosophy, how they handle **zero-shot**, **few-shot**, and **fine-tuned** forecasting, and what trade-offs matter if you were asked about them in an AI engineer or AI architect interview. ([arXiv][1])

---

## Key Takeaways

* **Time-LLM is not a native time-series foundation model in the same sense as TimeGPT or Lag-Llama.** It keeps a pretrained LLM backbone frozen and learns a small reprogramming layer that turns time-series patches into text-like prototype representations. This matters because it tests whether language-model capabilities can be reused for forecasting without time-series pretraining from scratch. The practical implication is that it is a transfer and alignment method, not a pure time-series pretraining story. ([arXiv][1])

* **TimeGPT-1 is a specialized time-series Transformer, not an LLM.** The paper explicitly says TimeGPT is not based on an existing LLM, even though it follows the same broad foundation-model idea of large-scale pretraining. This matters because the name can mislead people into thinking it is a text model adapted to time series. The practical implication is that TimeGPT should be described as a time-series-native foundation model. ([arXiv][3])

* **Lag-Llama is a probabilistic forecasting model, not just a point forecaster.** It predicts a probability distribution over future values, uses a Student’s t distribution head, and is evaluated with **CRPS** rather than only deterministic error metrics like MSE. This matters because uncertainty is central in many real forecasting systems. The practical implication is that Lag-Llama is especially relevant when downstream decisions need calibrated uncertainty, not only a single future estimate. ([arXiv][4])

* **The three papers sit at different points in the design space.** Time-LLM is about **cross-modality reprogramming** of a frozen LLM; TimeGPT-1 is about **large-scale native pretraining for zero-shot forecasting**; Lag-Llama is about **probabilistic pretraining plus adaptation across domains**. This matters because “time series foundation model” is not one architecture family. The practical implication is that interview answers should distinguish repurposed LLMs from native time-series foundation models. ([arXiv][1])

* **Data scale and diversity are central to the native-model papers.** TimeGPT-1 reports training on over **100 billion** publicly available time-series data points across many domains, while Lag-Llama pretrains on **27 datasets**, **7,965** univariate time series, and roughly **352 million** data windows. This matters because both papers argue that broad pretraining enables cross-domain transfer. The practical implication is that foundation-model behavior in time series depends heavily on pretraining breadth, not just architecture choice. ([arXiv][3])

* **Time-LLM’s strongest claim is data-efficiency.** The paper reports strong few-shot and zero-shot behavior and says improvements over GPT4TS become larger as data becomes scarcer: about **7.7%** in 10% few-shot, **8.4%** in 5% few-shot, and **22%** in zero-shot settings. This matters because time-series datasets are often small or domain-specific. The practical implication is that Time-LLM is most compelling when labeled training data is limited. ([arXiv][1])

* **TimeGPT-1’s strongest claim is operational simplicity.** It performs zero-shot forecasting on unseen series and reports an average GPU inference speed of about **0.6 milliseconds per series**, while many baselines require a train-then-predict pipeline. This matters because real forecasting workflows often care about deployment simplicity as much as raw benchmark accuracy. The practical implication is that zero-shot inference can reduce training overhead dramatically. ([arXiv][3])

* **Lag-Llama’s strongest claim is general-purpose probabilistic transfer.** The paper reports comparable zero-shot performance to strong baselines and the best average rank after fine-tuning, with especially strong few-shot adaptation across different data-history levels. This matters because it suggests one pretrained model can serve as a default starting point across many new datasets. The practical implication is that Lag-Llama is a strong candidate when you want one probabilistic forecaster that can adapt broadly rather than a separate model for each dataset. ([arXiv][4])

---

## Background and Foundations

### What is a time series forecasting model?

A **time series** is data ordered over time: electricity demand by hour, web traffic by minute, stock price by day, or sensor readings by second. **Forecasting** means predicting future values from historical values, and sometimes from extra input variables such as calendar information, promotions, weather, or events. TimeGPT formalizes forecasting as learning a function that maps past target values and optional covariates to future target values. Lag-Llama formalizes the probabilistic version: predict a **distribution** over future values, not just one number. ([arXiv][3])

### Deterministic forecasting vs probabilistic forecasting

A **deterministic** forecaster outputs one predicted future path, or one number per future step. Time-LLM is evaluated mainly with **MSE** and **MAE**, which are standard point-forecast metrics. A **probabilistic** forecaster outputs a full predictive distribution, which lets you estimate uncertainty intervals and risk. Lag-Llama does this with a parametric distribution head and evaluates using **CRPS** (Continuous Ranked Probability Score), a standard probabilistic forecasting metric where lower is better. TimeGPT-1 is mainly presented as a point-forecasting system in the paper’s core benchmark tables, but it also states that it uses conformal prediction based on historic errors to estimate prediction intervals. ([arXiv][1])

### Zero-shot, few-shot, and fine-tuning

These papers use transfer-learning language borrowed partly from NLP.

1. **Zero-shot** means apply the pretrained model to a new dataset without updating the model weights on that dataset. TimeGPT-1 and Lag-Llama both emphasize this. ([arXiv][3])
2. **Few-shot** means adapt the model using only a small fraction of the available target-domain history. Time-LLM and Lag-Llama both evaluate this. ([arXiv][1])
3. **Fine-tuning** means update pretrained parameters on the downstream dataset, usually using more target data than in the few-shot setting. TimeGPT-1 and Lag-Llama both evaluate this. ([arXiv][3])

### What makes a model “foundation-model-like” here?

In these papers, the idea is not exactly the same as in language or vision, but the common theme is **broad pretraining plus transfer**. TimeGPT-1 and Lag-Llama both pretrain on large and diverse time-series corpora, then test transfer to unseen datasets. Time-LLM is different: it does not pretrain a new time-series model at all. It repurposes a pretrained LLM through input alignment and prompting. That still fits the broader foundation-model conversation, but by a different route. ([arXiv][1])

---

## Big Picture First

The cleanest mental model is that these papers answer three different questions:

1. **Time-LLM:** Can a frozen LLM be turned into a time-series forecaster by rewriting the input into something more language-like? ([arXiv][1])
2. **TimeGPT-1:** Can a large, specialized Transformer trained on a massive time-series corpus act as a general zero-shot forecaster? ([arXiv][5])
3. **Lag-Llama:** Can a decoder-only Transformer pretrained on diverse time-series data become a general-purpose probabilistic forecaster with strong zero-shot and adaptation performance? ([arXiv][4])

| Paper     | What it is really trying to do                            | Backbone                                | Output type                                      | Main transfer story                                                       |
| --------- | --------------------------------------------------------- | --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------- |
| Time-LLM  | Reuse a frozen LLM for forecasting                        | Frozen Llama/GPT-style LLM              | Point forecast                                   | Reprogram input time series into text-like prototypes                     |
| TimeGPT-1 | Build a native time-series foundation model               | Specialized Transformer encoder-decoder | Mostly point forecast, plus prediction intervals | Large-scale pretraining for zero-shot transfer                            |
| Lag-Llama | Build a native probabilistic forecasting foundation model | LLaMA-style decoder-only Transformer    | Predictive distribution                          | Large-scale pretraining plus zero-shot / few-shot / fine-tuned adaptation |

The comparison above is synthesized from the three papers’ method descriptions and evaluation setups. ([arXiv][1])

---

## Core Concepts Explained

### 1. Reprogramming

**What it is:** Changing the input interface to a pretrained model so the model can solve a new modality’s task without changing the main backbone. Time-LLM uses this idea. ([arXiv][1])

**Why it exists:** LLMs were trained on tokens, not raw time-series values. The model therefore needs help aligning numeric time-series patterns with its text-trained internal representations. ([arXiv][1])

**How it works at a high level:** Time-LLM breaks a time series into patches, embeds them, maps them into learned **text prototypes**, prepends natural-language prompt information, passes them through a frozen LLM, and projects the LLM outputs back into forecast values. ([arXiv][1])

**Why it matters:** It is one of the clearest examples of using a language model as a general sequence reasoner rather than retraining a domain-native model from scratch. ([arXiv][1])

### 2. Prompt-as-Prefix (PaP)

**What it is:** Time-LLM’s prompting scheme that adds natural-language context before the reprogrammed time-series tokens. ([arXiv][1])

**Why it exists:** The paper argues that LLM reasoning can be improved when the model is given extra context such as dataset description, task instruction, and input statistics. ([arXiv][1])

**How it works at a high level:** PaP prepends information about the dataset, the forecasting task, and the input statistics to guide how the LLM should interpret the transformed time-series patches. ([arXiv][1])

**Why it matters:** In the ablations, removing prompting hurts forecasting quality substantially, especially in scarce-data settings. ([arXiv][1])

### 3. Native time-series foundation model

**What it is:** A model trained directly on large collections of time-series data so that the model itself learns transferable temporal patterns. TimeGPT-1 and Lag-Llama both fit this pattern more clearly than Time-LLM. ([arXiv][3])

**Why it exists:** Time-series data has characteristics that differ from text, such as seasonality, trend, changing frequency, sparsity, and exogenous covariates. A native architecture can be specialized to those properties. ([arXiv][3])

**Why it matters:** It avoids the modality-mismatch problem that Time-LLM must solve with reprogramming. ([arXiv][1])

### 4. Lag features

**What they are:** Previous values of the same time series used as explicit inputs to the model. Lag-Llama constructs tokens from selected lag indices and date-time covariates. ([arXiv][4])

**Why they exist:** Time series often depend strongly on earlier values at meaningful time gaps, such as one hour ago, one day ago, or one week ago. Feeding lag values directly helps the model express such dependencies. ([arXiv][4])

**Why they matter:** Lag-Llama builds its whole tokenization around them, which is one of the main differences from generic language-model tokenization. ([arXiv][4])

### 5. Probabilistic forecasting

**What it is:** Predicting not just one future value, but a distribution of plausible future values. Lag-Llama models the next-step distribution with a parametric head and samples autoregressively to build forecast trajectories. ([arXiv][4])

**Why it exists:** Real decisions often need uncertainty, such as “what is the 90% interval for future demand?” rather than only “what is the single best guess?” ([arXiv][4])

**Why it matters:** It makes Lag-Llama better suited to risk-aware applications than a pure point forecaster. ([arXiv][4])

### 6. Zero-shot forecasting

**What it is:** Forecasting a new dataset without retraining on that dataset. TimeGPT-1 and Lag-Llama emphasize this heavily, and Time-LLM studies a related cross-domain zero-shot setting. ([arXiv][3])

**Why it exists:** In many practical settings, you do not have time, labels, or infrastructure to retrain a separate model for every new series or dataset. ([arXiv][3])

**Why it matters:** This is the strongest “foundation-model” behavior these papers are trying to demonstrate. ([arXiv][5])

---

## Step-by-Step Technical Walkthrough

## 1. Time-LLM

### Inputs

Time-LLM takes a multivariate time series, normalizes it, splits it into patches, and then embeds those patches before sending them to a frozen LLM backbone. Its default reported backbone is **Llama-7B**. ([arXiv][1])

### What happens

1. **Patch and embed the time series.** The input sequence is partitioned into patches and mapped into an embedding space. ([arXiv][1])
2. **Reprogram the patches with learned text prototypes.** A cross-attention mechanism maps time-series patch embeddings into a form better aligned with the LLM’s hidden space. ([arXiv][1])
3. **Add Prompt-as-Prefix.** Natural-language prompt prefixes supply dataset context, task instruction, and input statistics. ([arXiv][1])
4. **Run the frozen LLM.** The backbone itself is kept intact and frozen. ([arXiv][1])
5. **Project outputs back to forecasts.** The model discards the prefix outputs, keeps the transformed patch representations, flattens them, and linearly projects them to the forecast horizon. ([arXiv][1])

### Outputs

The system outputs point forecasts, and the paper evaluates them mainly with MSE and MAE on common long-term forecasting datasets. ([arXiv][1])

### Why each step exists

The patching step reduces raw-sequence complexity. The text-prototype reprogramming step aligns continuous temporal data with a model trained on discrete language tokens. The prompting step gives the frozen LLM additional semantic context. The output projection step converts language-model hidden states back into future numeric values. ([arXiv][1])

### Trade-offs

The paper’s main advantage is parameter efficiency: the reprogramming network uses fewer than **6.6 million** trainable parameters, about **0.2%** of Llama-7B. But the method still depends on successful cross-modality alignment, and its overall efficiency is capped by the heavy LLM backbone. ([arXiv][1])

---

## 2. TimeGPT-1

### Inputs

TimeGPT takes historical target values and optional exogenous variables as inputs. It is designed for new, unseen time series without retraining in the zero-shot case. ([arXiv][3])

### What happens

1. **Train a large specialized Transformer on a huge time-series corpus.** The paper reports over **100 billion** public time-series data points across domains such as finance, economics, healthcare, weather, IoT, energy, web traffic, sales, transport, and banking. ([arXiv][3])
2. **Use a Transformer encoder-decoder architecture.** The model uses self-attention, residual connections, layer normalization, local positional encoding, and a linear output layer. The figure also shows CNN components around the encoder-decoder stack. ([arXiv][3])
3. **Run zero-shot inference on unseen datasets.** The paper evaluates on over **300 thousand** unseen time series from multiple domains. ([arXiv][3])
4. **Optionally fine-tune.** The paper also studies fine-tuning on subsets of target data. ([arXiv][3])
5. **Estimate prediction intervals.** The system uses conformal prediction based on historic errors for intervals. ([arXiv][3])

### Outputs

The main paper emphasizes point-forecast performance through **rMAE** and **rRMSE**, which are relative to the Seasonal Naive baseline and allow frequency-wise comparison. It also provides prediction intervals through conformal methods. ([arXiv][3])

### Why each step exists

Large-scale pretraining is meant to expose the model to diverse temporal patterns. The specialized Transformer architecture is meant to model time-series structure directly rather than inherit it indirectly from language. Relative metrics are used because they make results more interpretable across monthly, weekly, daily, and hourly settings. ([arXiv][3])

### Trade-offs

TimeGPT’s biggest strength is simplicity at inference time: the paper stresses that zero-shot use collapses the usual train-select-predict pipeline into direct invocation of a pretrained model. The trade-off is that the paper is about forecasting only; it is not a general multimodal or multitask time-series foundation model in the way language foundation models are general across many tasks. That second point is a reasoned interpretation; the paper itself is narrowly focused on forecasting. ([arXiv][3])

---

## 3. Lag-Llama

### Inputs

Lag-Llama is built for **univariate probabilistic time-series forecasting**. It uses lagged values and date-time covariates as token features. ([arXiv][4])

### What happens

1. **Collect a diverse pretraining corpus.** The paper pretrains on **27 datasets** from six broad domains and reports **7,965** univariate time series and about **352 million** training windows. ([arXiv][4])
2. **Tokenize using lag features and time covariates.** Each token contains selected past lag values and date-time features such as hour-of-day or month. ([arXiv][4])
3. **Run a decoder-only LLaMA-style Transformer.** The architecture uses causal masking, RMSNorm, and RoPE. ([arXiv][4])
4. **Predict a parametric next-step distribution.** The paper uses a Student’s t distribution head with parameters for degrees of freedom, mean, and scale. ([arXiv][4])
5. **Decode autoregressively.** Future values are sampled step by step to generate trajectories and uncertainty intervals. ([arXiv][4])
6. **Evaluate zero-shot, fine-tuned, and few-shot adaptation.** The main benchmark uses CRPS and average rank across unseen datasets. ([arXiv][4])

### Outputs

The model outputs probability distributions over future values, from which samples and uncertainty summaries can be derived. ([arXiv][4])

### Why each step exists

Lag tokenization makes periodic structure and recency explicit. The decoder-only setup keeps the architecture simple. The Student’s t head handles heavy-tailed uncertainty better than a fixed point prediction. The diverse pretraining corpus is meant to support cross-domain transfer. ([arXiv][4])

### Trade-offs

Lag-Llama is focused on **univariate** forecasting, not general multivariate forecasting. It also requires a sufficient lag window because lag-based tokenization needs enough prior history. These are direct consequences of the architecture and tokenization design described in the paper. ([arXiv][4])

---

## Paper-by-Paper Explanation

## Time-LLM: Time Series Forecasting by Reprogramming Large Language Models

### Problem addressed

The paper addresses a modality mismatch problem: LLMs are strong on text, but time series are continuous numerical signals. The question is whether one can still exploit a frozen LLM’s sequence reasoning abilities for forecasting without training a large time-series model from scratch. ([arXiv][1])

### Method used

Time-LLM keeps the backbone LLM intact, converts time-series patches into learned text-prototype representations, adds Prompt-as-Prefix context, and projects the LLM outputs back into future values. The main backbone used in experiments is Llama-7B. ([arXiv][1])

### Main innovation

The main innovation is the **reprogramming** view: treat forecasting as another “language-like” task without fine-tuning the backbone model. Prompt-as-Prefix is the second key idea, because it gives the frozen LLM extra task and dataset context. ([arXiv][1])

### Main findings

On long-term forecasting benchmarks, Time-LLM reports strong results against specialized time-series models, including strong ETT and Weather performance. The paper also reports that its advantage over GPT4TS becomes larger in data-scarce settings: about **7.7%** better in 10% few-shot, **8.4%** in 5% few-shot, and **22%** in zero-shot forecasting. ([arXiv][1])

### Limitations

Time-LLM is not a native time-series foundation model and depends on successful alignment between time-series patches and language-model hidden space. It also inherits the computational weight of the LLM backbone, even though the trainable part is small. Information not provided: the paper does not present large-scale probabilistic uncertainty modeling like Lag-Llama does. ([arXiv][1])

### What changed compared with earlier work

Compared with task-specific time-series Transformers, Time-LLM moves the effort from architecture design toward modality alignment and prompting. Compared with native pretraining approaches, it avoids time-series pretraining from scratch. ([arXiv][1])

### Directly stated facts

The paper states that the LLM backbone is frozen, that the input is reprogrammed with text prototypes, that Prompt-as-Prefix enriches context, and that the trainable reprogramming network is under **6.6M** parameters, about **0.2%** of Llama-7B. ([arXiv][1])

### Reasoned interpretation

Time-LLM is best understood as a **foundation-model transfer** paper rather than a **native time-series foundation model** paper. That distinction is important and often blurred in casual discussion. ([arXiv][1])

### Information not provided

The paper does not claim that frozen LLM reprogramming is universally better than native time-series pretraining for all forecasting regimes or all deployment budgets. Information not provided. ([arXiv][1])

---

## TimeGPT-1

### Problem addressed

TimeGPT-1 addresses a long-standing forecasting systems problem: most time-series models are trained or tuned per dataset, while other AI domains have started to benefit from large pretrained models with strong transfer. The paper asks whether a single large pretrained time-series model can forecast unseen data accurately without retraining. ([arXiv][5])

### Method used

TimeGPT is a specialized Transformer-based forecasting model with an encoder-decoder structure, local positional encoding, residual connections, and layer normalization. It is trained on over **100 billion** public time-series data points across many domains, then evaluated in zero-shot and fine-tuned settings. The paper explicitly says it is **not** based on an existing LLM. ([arXiv][3])

### Main innovation

The main innovation is the scale-and-transfer claim: build a native time-series forecasting model large enough and broad enough that zero-shot forecasting on unseen series becomes practical. It is less about inventing a novel micro-architecture than about bringing the foundation-model training paradigm into forecasting. ([arXiv][5])

### Main findings

The paper reports that TimeGPT ranks among the top-3 performers across frequencies in zero-shot evaluation on over **300 thousand** unseen series, and that it does so with very fast inference. It reports around **0.6 ms per series** GPU inference, compared with around **57 ms** per series for global models when training and inference are included, and about **600 ms** per series for optimized statistical pipelines including training and inference. ([arXiv][3])

### Limitations

TimeGPT is specialized for forecasting and does not claim to be a general-purpose multimodal model. Also, while it provides prediction intervals through conformal prediction, the paper’s core benchmark emphasis is still on relative point-forecast errors. Information not provided: the paper does not present the kind of explicit probabilistic distribution-head design that Lag-Llama does. ([arXiv][3])

### What changed compared with earlier work

Compared with classical time-series pipelines, TimeGPT removes the need to train a separate model for every new dataset in the zero-shot setting. Compared with Time-LLM, it is a time-series-native model rather than a reprogrammed language model. ([arXiv][3])

### Directly stated facts

The paper states that TimeGPT is the first foundation model for time series, that it is not based on an existing LLM, that it was trained on over **100 billion** publicly available time-series points, and that it supports zero-shot transfer to unseen domains. ([arXiv][5])

### Reasoned interpretation

TimeGPT is the clearest “native foundation model” paper in this set. Its contribution is mostly about the **training regime and transfer paradigm**, not about borrowing language-model reasoning through reprogramming. ([arXiv][3])

### Information not provided

The paper does not give a detailed open breakdown of model size, training recipe, or pretraining mixture at the same level of architectural detail that Lag-Llama provides for its setup. Information not provided. ([arXiv][3])

---

## Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting

### Problem addressed

Lag-Llama targets probabilistic forecasting across unseen datasets and domains. The paper asks whether a simple decoder-only Transformer pretrained on diverse univariate time-series data can generalize well in zero-shot settings and become a strong general-purpose forecaster after adaptation. ([arXiv][4])

### Method used

Lag-Llama uses a decoder-only Transformer based on LLaMA, tokenizes series with lag features and date-time covariates, uses RMSNorm and RoPE, and predicts parameters of a Student’s t distribution via a parametric head. It is pretrained on **27 datasets**, **7,965** univariate time series, and around **352 million** training windows. ([arXiv][4])

### Main innovation

The main innovation is the combination of large-scale pretraining with a simple, probabilistic decoder-only architecture specialized for time series through lag-based tokenization. It is a strong example of “train from scratch on time-series data, but keep the architecture simple.” ([arXiv][4])

### Main findings

The paper reports that in the zero-shot setting Lag-Llama is comparable to baselines with an average rank of **6.714**. After fine-tuning, it achieves state-of-the-art performance on **3** datasets and the best average rank of **2.786**, around **2 points** better than the best supervised model in that comparison. It also reports strong few-shot adaptation, with the best average rank across different levels of available history. ([arXiv][4])

### Limitations

Lag-Llama is restricted to **univariate** forecasting, and lag-based tokenization requires sufficient historical context to compute the lag features. It also focuses on forecasting rather than broader time-series tasks like classification or anomaly detection. ([arXiv][4])

### What changed compared with earlier work

Compared with Time-LLM, Lag-Llama does not rely on language-model reprogramming. Compared with TimeGPT, it puts stronger emphasis on probabilistic forecasting and transparent architectural choices such as lag features and a Student’s t head. ([arXiv][4])

### Directly stated facts

The paper states that Lag-Llama is a foundation model for **univariate probabilistic forecasting**, uses a simple decoder-only Transformer architecture, is pretrained on a broad diverse corpus, and uses a Student’s t distribution head with CRPS evaluation. ([arXiv][4])

### Reasoned interpretation

Lag-Llama is the most **forecasting-theoretically grounded** of the three for uncertainty-aware applications, because it is probabilistic by design rather than treating intervals as an add-on. ([arXiv][4])

### Information not provided

The paper does not claim that decoder-only probabilistic forecasting is universally best for all multivariate, event-rich, or multimodal forecasting settings. Information not provided. ([arXiv][4])

---

## Comparison Across Papers or Methods

### Comparison by model philosophy

| Aspect                              | Time-LLM                            | TimeGPT-1                                 | Lag-Llama                                               |
| ----------------------------------- | ----------------------------------- | ----------------------------------------- | ------------------------------------------------------- |
| Core philosophy                     | Reuse a frozen LLM                  | Pretrain a native time-series Transformer | Pretrain a native probabilistic time-series Transformer |
| Is the backbone time-series-native? | No                                  | Yes                                       | Yes                                                     |
| Main transfer mechanism             | Reprogramming + prompting           | Large-scale zero-shot transfer            | Large-scale zero-shot + few-shot + fine-tuned transfer  |
| Output style                        | Point forecast                      | Mostly point forecast, plus intervals     | Full predictive distribution                            |
| Main benchmark emphasis             | Long-term deterministic forecasting | Zero-shot forecasting across frequencies  | Probabilistic forecasting across unseen datasets        |

The table above is synthesized from the papers’ model descriptions and evaluation sections. ([arXiv][1])

### Comparison by data and scale

| Aspect             | Time-LLM                            | TimeGPT-1                                                             | Lag-Llama                                           |
| ------------------ | ----------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------- |
| Pretraining source | Pretrained text LLM                 | Over 100B public time-series data points                              | 27 datasets, 7,965 univariate series, ~352M windows |
| Default backbone   | Llama-7B in main setup              | Specialized encoder-decoder Transformer                               | Decoder-only LLaMA-style Transformer                |
| Domain scope       | Forecasting via modality transfer   | Diverse domains including finance, weather, energy, web traffic, etc. | Six broad domains with held-out unseen datasets     |
| Zero-shot focus    | Cross-domain zero-shot and few-shot | Strong zero-shot emphasis                                             | Zero-shot plus strong adaptation emphasis           |

The table above combines directly stated scale details from the three papers. ([arXiv][1])

### Comparison by strengths and weaknesses

| Method    | Biggest strength                                                    | Biggest weakness                                            |
| --------- | ------------------------------------------------------------------- | ----------------------------------------------------------- |
| Time-LLM  | Strong transfer in scarce-data settings without tuning the backbone | Heavy frozen LLM backbone and modality-alignment dependence |
| TimeGPT-1 | Extremely simple zero-shot forecasting workflow with strong speed   | Less explicit uncertainty modeling than Lag-Llama           |
| Lag-Llama | Probabilistic forecasting with strong cross-domain adaptation       | Univariate-only focus and lag-window requirements           |

This final comparison is partly directly stated and partly a reasoned engineering interpretation of the papers’ designs and results. ([arXiv][1])

---

## Real-World System and Application

These papers support three different practical deployment stories. **Time-LLM** fits settings where an organization already trusts a large pretrained language model and wants to adapt it cheaply to forecasting without backbone fine-tuning. **TimeGPT-1** fits settings where zero-shot forecasting and operational simplicity matter most, because the paper emphasizes direct invocation instead of full retraining pipelines. **Lag-Llama** fits settings where uncertainty matters and you want one pretrained probabilistic forecaster that can be adapted across many downstream datasets. ([arXiv][1])

A realistic forecasting platform could even combine ideas from these papers. One could use a native foundation model like TimeGPT or Lag-Llama as the main forecasting engine, while borrowing prompt-style or metadata-conditioning ideas like Time-LLM’s Prompt-as-Prefix for richer task descriptions and domain context. That is a reasoned interpretation rather than a pipeline explicitly proposed by the papers. ([arXiv][1])

Information not provided: these papers do not specify a full production design for data ingestion, drift monitoring, forecast governance, retraining triggers, probabilistic calibration monitoring, or service-level reliability. They are model papers, not end-to-end forecasting platform papers. ([arXiv][1])

---

## Limitations and Trade-offs

| Limitation or trade-off                  | Concrete meaning                                                                                                        | Why it matters                                                           |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| Reprogramming vs native modeling         | Time-LLM must align time-series structure to a text-trained model                                                       | Transfer is elegant, but modality mismatch is real                       |
| Zero-shot simplicity vs specialization   | TimeGPT is extremely easy to use zero-shot, but it is specialized for forecasting rather than general time-series tasks | Great for forecasting products, less clearly general beyond that         |
| Point forecast vs probabilistic forecast | Time-LLM and TimeGPT emphasize deterministic accuracy more directly, while Lag-Llama models uncertainty explicitly      | Choice depends on whether downstream decisions need confidence estimates |
| Univariate vs multivariate               | Lag-Llama is univariate; Time-LLM directly studies multivariate forecasting                                             | Representation scope changes what tasks the model fits naturally         |
| Backbone cost                            | Time-LLM trains few parameters but still runs a large LLM backbone                                                      | Parameter efficiency is not the same as cheap inference                  |
| Pretraining cost                         | Native foundation models require very large and diverse time-series corpora                                             | Strong transfer often comes from expensive pretraining                   |

The first, third, fifth, and sixth points follow directly from the methods and reported setups; the second and fourth are reasoned engineering interpretations of what each paper optimizes for. ([arXiv][1])

---

## Interview-Ready Understanding

### What you should be able to explain

You should be able to explain that there are **two different ways** to get “foundation-model-like” behavior in time series:

1. **Repurpose an existing foundation model** from another modality, which is what Time-LLM does with a frozen LLM. ([arXiv][1])
2. **Pretrain a native time-series foundation model**, which is what TimeGPT-1 and Lag-Llama do. ([arXiv][3])

You should also be able to explain that **TimeGPT-1** is time-series-native and optimized for zero-shot forecasting simplicity, while **Lag-Llama** is time-series-native and optimized for probabilistic forecasting and broad adaptation. ([arXiv][3])

### Likely interview questions

#### 1. Is Time-LLM a real time-series foundation model?

Not in the same sense as TimeGPT or Lag-Llama. Time-LLM repurposes a frozen pretrained LLM through reprogramming and prompting, rather than pretraining a time-series-native model from scratch. ([arXiv][1])

#### 2. What is the core idea of Time-LLM?

Patch the time series, map those patches into learned text-prototype representations, prepend prompt context, run a frozen LLM, and project its hidden states back into forecasts. ([arXiv][1])

#### 3. Why does Time-LLM use Prompt-as-Prefix?

Because the LLM needs extra semantic context to interpret reprogrammed numeric inputs. The prompt provides dataset background, task instruction, and input statistics. ([arXiv][1])

#### 4. Is TimeGPT an LLM?

No. The paper explicitly says TimeGPT is not based on an existing LLM. It is a specialized Transformer architecture trained directly for time-series forecasting. ([arXiv][3])

#### 5. What makes TimeGPT foundation-model-like?

It is pretrained on a massive and diverse time-series corpus and can perform accurate zero-shot forecasting on unseen series without retraining. ([arXiv][3])

#### 6. What is Lag-Llama’s main idea?

Use a simple decoder-only Transformer, feed it lag features and time covariates, predict a distribution over future values, and rely on broad pretraining for transfer. ([arXiv][4])

#### 7. Why are lag features important in Lag-Llama?

They explicitly expose useful periodic and autoregressive structure to the model, which is especially helpful in time series where specific offsets like one day or one week ago matter. ([arXiv][4])

#### 8. What is the main difference between TimeGPT and Lag-Llama?

TimeGPT emphasizes zero-shot forecasting simplicity and relative point-error benchmarks, while Lag-Llama is designed as a probabilistic forecaster with a distribution head and CRPS evaluation. ([arXiv][3])

#### 9. Which paper is best if I care about uncertainty?

Lag-Llama, because uncertainty modeling is part of the core architecture through its parametric distribution head and probabilistic evaluation. ([arXiv][4])

#### 10. Which paper is best if I care about not training a new model?

Time-LLM if you want to reuse a frozen LLM, or TimeGPT if you want zero-shot use of a pretrained time-series model. The better answer depends on whether you already have an LLM backbone or want a native forecaster. ([arXiv][1])

---

## Glossary

| Term                          | Beginner-friendly definition                                                                    |
| ----------------------------- | ----------------------------------------------------------------------------------------------- |
| Time series                   | Data points ordered over time                                                                   |
| Forecasting                   | Predicting future values from past values and optional extra inputs                             |
| Foundation model              | A model pretrained broadly enough to transfer to many downstream settings                       |
| Zero-shot forecasting         | Forecasting a new dataset without training on that dataset                                      |
| Few-shot forecasting          | Forecasting after adapting with only a small amount of target data                              |
| Fine-tuning                   | Updating pretrained model parameters on a downstream dataset                                    |
| Reprogramming                 | Changing the input interface to reuse a pretrained model for a new modality or task             |
| Prompt-as-Prefix (PaP)        | Time-LLM’s prompt strategy for adding dataset and task context before the reprogrammed sequence |
| Patch                         | A small contiguous chunk of the input time series                                               |
| Text prototype                | A learned representation used by Time-LLM to align time-series patches with LLM hidden space    |
| Exogenous variables           | Additional input features besides the target series, such as weather or events                  |
| Point forecast                | A single predicted value or path                                                                |
| Probabilistic forecast        | A predictive distribution over future values                                                    |
| CRPS                          | Continuous Ranked Probability Score, a standard metric for probabilistic forecasting            |
| rMAE / rRMSE                  | Relative error metrics used by TimeGPT, normalized against a baseline                           |
| Lag feature                   | A past value of the time series used explicitly as an input feature                             |
| Covariate                     | Any extra feature that may help prediction                                                      |
| Student’s t distribution head | A neural output layer that predicts parameters of a Student’s t forecast distribution           |
| Conformal prediction          | A way to produce prediction intervals using historical error behavior                           |

The glossary above is derived from how the three papers define their methods and evaluations. ([arXiv][1])

---

## Recap

You should now see that “time series foundation model” covers at least two different research directions. One direction is **borrow a foundation model from another modality and adapt it**, which is what Time-LLM does. The other is **build a native foundation model for time series**, which is what TimeGPT-1 and Lag-Llama do. ([arXiv][1])

The most important conceptual distinction is this: **Time-LLM is about transfer by reprogramming, TimeGPT-1 is about zero-shot forecasting at scale, and Lag-Llama is about probabilistic transfer with a simple decoder-only architecture.** That is the cleanest way to explain the progression in an interview. ([arXiv][1])

What remains limited is also important. These papers are all about forecasting, not the full spectrum of time-series tasks. They do not define a universal time-series foundation model for classification, anomaly detection, imputation, and multimodal reasoning all at once. Information not provided. ([arXiv][6])

---

## Key Citations

*Time-LLM: Time Series Forecasting by Reprogramming Large Language Models.* ([arXiv][1])

*TimeGPT-1.* ([arXiv][5])

*Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting.* ([arXiv][4])

*Source mismatch note: provided URL `2302.00861` points to SimMTM, not Time-LLM.* ([arXiv][2])

*Source mismatch note: provided URL `2310.10196` points to a survey paper, not Lag-Llama.* ([arXiv][6])

[1]: https://arxiv.org/pdf/2310.01728 "https://arxiv.org/pdf/2310.01728"
[2]: https://arxiv.org/pdf/2302.00861 "https://arxiv.org/pdf/2302.00861"
[3]: https://arxiv.org/pdf/2310.03589v1 "https://arxiv.org/pdf/2310.03589v1"
[4]: https://arxiv.org/pdf/2310.08278 "https://arxiv.org/pdf/2310.08278"
[5]: https://arxiv.org/pdf/2310.03589 "https://arxiv.org/pdf/2310.03589"
[6]: https://arxiv.org/pdf/2310.10196 "https://arxiv.org/pdf/2310.10196"


---
---
---

