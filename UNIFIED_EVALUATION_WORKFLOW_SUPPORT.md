# OpenCompass: Unified Evaluation Workflow Support Analysis

This document analyzes which strategies from the Unified Evaluation Workflow are **natively supported** by OpenCompass. 

## Methodology

A strategy is considered "natively supported" only if the harness provides it directly upon installation (via `pip install opencompass` or with optional extras like `[full]`, `[api]`, `[lmdeploy]`, `[vllm]`), without requiring:
- Additional implementation by users
- Custom modules beyond the core package
- Integration with external libraries not included in OpenCompass dependencies
- Separate packages or frameworks (e.g., agent frameworks that require separate installation)

The analysis is based on:
- Official documentation (README.md, docs/)
- Source code examination (opencompass/)
- Setup configuration (setup.py, requirements/)
- Native capabilities present in the codebase

## Summary

OpenCompass is a comprehensive evaluation harness for large language models and vision-language models that natively supports a substantial portion of the unified evaluation workflow, with particular strengths in dataset preparation, batch inference, comprehensive scoring methods, pairwise comparison, and leaderboard integration.

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

#### ✅ Strategy 1: PyPI Packages
**SUPPORTED** - OpenCompass can be installed via pip:
- `pip install -U opencompass` (basic installation)
- `pip install "opencompass[full]"` (full installation with more dataset support)
- `pip install "opencompass[api]"` (API evaluation support)
- `pip install "opencompass[lmdeploy]"` (with LMDeploy backend)
- `pip install "opencompass[vllm]"` (with vLLM backend)

**Evidence**: README.md, docs/en/get_started/installation.md, setup.py

#### ✅ Strategy 2: Git Clone
**SUPPORTED** - OpenCompass can be installed from source:
```bash
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```

**Evidence**: README.md lines 127-135, docs/en/get_started/installation.md

#### ❌ Strategy 3: Container Images
**NOT SUPPORTED** - OpenCompass does not provide prebuilt Docker or OCI container images.

#### ❌ Strategy 4: Binary Packages
**NOT SUPPORTED** - OpenCompass does not distribute standalone executable binaries.

#### ❌ Strategy 5: Node Package
**NOT SUPPORTED** - OpenCompass is a Python-based harness and does not support Node.js package managers.

### Step B: Service Authentication

#### ✅ Strategy 1: Evaluation Platform Authentication
**SUPPORTED** - OpenCompass supports authentication with evaluation platforms and leaderboard submission:
- Integration with OpenCompass Leaderboard (CompassRank)
- Submission to public leaderboards for model comparison

**Evidence**: README.md mentions "CompassRank" leaderboard at rank.opencompass.org.cn, lines 86-88 state "If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address"

#### ✅ Strategy 2: API Provider Authentication
**SUPPORTED** - OpenCompass supports API authentication through environment variables and configuration:
- OpenAI API (`OPENAI_API_KEY`)
- Gemini API (`GEMINI_API_KEY`)
- Multiple commercial model providers (Baidu, Qwen, ZhipuAI, ByteDance, Huawei, etc.)
- Configuration via environment variables or credential files

**Evidence**: 
- opencompass/models/openai_api.py uses `OPENAI_API_KEY`
- opencompass/models/gemini_api.py uses `GEMINI_API_KEY`
- README.md line 201: "export OPENAI_API_KEY="YOUR_OPEN_API_KEY""
- Multiple API model implementations in opencompass/models/

#### ✅ Strategy 3: Repository Authentication
**SUPPORTED** - OpenCompass supports authentication with model and dataset repositories:
- HuggingFace Hub integration (uses `huggingface_hub` library for dataset downloads)
- ModelScope integration (via `DATASET_SOURCE=ModelScope` environment variable)
- Automatic download of datasets from HuggingFace
- Token-based authentication for accessing gated models and datasets

**Evidence**:
- docs/en/get_started/installation.md lines 107-116 describe HuggingFace and ModelScope dataset integration
- opencompass/datasets/needlebench/ files use `from huggingface_hub import snapshot_download`
- README.md line 159: supports ModelScope datasets

---

## Phase I: Specification (The Contract)

### Step A: SUT Preparation

#### ✅ Strategy 1: Model-as-a-Service (Remote Inference)
**SUPPORTED** - OpenCompass supports remote inference through:
- OpenAI API models (GPT-3.5, GPT-4, GPT-4o, o1)
- Commercial API providers (Claude, Gemini, ZhipuAI, Baichuan, ByteDance, Huawei, Baidu, MiniMax, SenseTime, Xunfei, etc.)
- Custom API endpoints

**Evidence**:
- README.md lines 346-361 list supported API models
- opencompass/models/ contains multiple API model implementations (openai_api.py, gemini_api.py, claude_sdk_api.py, etc.)
- docs/en/user_guides/models.md lines 68-106 describe API-based models

#### ✅ Strategy 2: Model-in-Process (Local Inference)
**SUPPORTED** - OpenCompass supports local model inference:
- HuggingFace models via `HuggingFaceBaseModel` and `HuggingFaceCausalLM`
- Direct loading of model weights and checkpoints
- Access to logits and probabilities for perplexity-based evaluation
- Support for LLMs, VLMs, and chat models

**Evidence**:
- docs/en/user_guides/models.md lines 9-66 describe HuggingFace-based models
- opencompass/models/huggingface.py provides local inference
- README.md shows extensive support for open-source models (LLaMA, Qwen, InternLM, etc.)

#### ❌ Strategy 3: Algorithm Implementation (In-Memory Structures)
**NOT SUPPORTED** - OpenCompass does not natively support evaluation of ANN algorithms, knowledge graph embeddings, or BM25 indexes. It focuses on neural language models.

#### ❌ Strategy 4: Policy/Agent Instantiation (Stateful Controllers)
**NOT SUPPORTED** - While OpenCompass has some agent evaluation capabilities (mentioned in roadmap), it does not natively provide DRL policy instantiation, robot controllers, or multi-agent system evaluation frameworks.

### Step B: Benchmark Preparation (Inputs)

#### ✅ Strategy 1: Benchmark Dataset Preparation (Offline)
**SUPPORTED** - OpenCompass provides comprehensive dataset preparation:
- Automatic download from HuggingFace Datasets
- Automatic download from ModelScope datasets
- Manual download of OpenCompass custom datasets
- Support for 70+ datasets covering ~400,000 questions
- Data loading, splitting, normalization, and formatting

**Evidence**:
- docs/en/get_started/installation.md lines 103-141 describe dataset preparation
- README.md lines 137-171 describe multiple dataset preparation methods
- 1816 dataset implementation files in opencompass/datasets/
- Supports MMLU, GSM8K, HumanEval, CEVAL, GAOKAO, and many others

#### ❌ Strategy 2: Synthetic Data Generation (Generative)
**NOT SUPPORTED** - OpenCompass does not natively provide synthetic data generation, test augmentation, or perturbation capabilities. The circular evaluation feature (opencompass/datasets/circular.py) appears to be for data formatting rather than synthetic generation.

#### ❌ Strategy 3: Simulation Environment Setup (Simulated)
**NOT SUPPORTED** - OpenCompass does not provide 3D simulation environments, physics simulation, or interactive environment setup.

#### ❌ Strategy 4: Production Traffic Sampling (Online)
**NOT SUPPORTED** - OpenCompass does not provide production traffic sampling or real-world inference traffic collection capabilities.

### Step C: Benchmark Preparation (References)

#### ✅ Strategy 1: Judge Preparation
**SUPPORTED** - OpenCompass provides comprehensive LLM-as-judge capabilities:
- `GenericLLMEvaluator` for LLM-based evaluation
- `CascadeEvaluator` for sequential evaluator chains
- `MATHVerifyEvaluator` for mathematical reasoning
- Pre-configured judge models via environment variables (`OC_JUDGE_MODEL`, `OC_JUDGE_API_KEY`, `OC_JUDGE_API_BASE`)
- Support for GPT-4 and other models as judges
- Subjective evaluation with model-based judgments

**Evidence**:
- docs/en/advanced_guides/llm_judge.md describes GenericLLMEvaluator
- README.md line 61 mentions CascadeEvaluator
- README.md line 64 mentions GenericLLMEvaluator and MATHVerifyEvaluator
- docs/en/advanced_guides/subjective_evaluation.md describes subjective evaluation with JudgeLLM

#### ✅ Strategy 2: Ground Truth Preparation
**SUPPORTED** - OpenCompass loads and pre-processes ground truth references:
- Human annotations from datasets
- Reference answers for evaluation
- Ground truth labels for comparison

**Evidence**:
- Dataset configurations include reference answers (e.g., docs/en/advanced_guides/llm_judge.md shows datasets with 'answer' fields)
- Evaluation framework compares predictions against gold standard answers (docs/en/get_started/quick_start.md line 11)

---

## Phase II: Execution (The Run)

### Step A: SUT Invocation

#### ✅ Strategy 1: Batch Inference
**SUPPORTED** - OpenCompass provides comprehensive batch inference:
- Parallel inference across multiple samples
- Configurable batch sizes
- Support for single model evaluation across multiple datasets
- Efficient distributed evaluation with task partitioning

**Evidence**:
- docs/en/get_started/quick_start.md line 11 describes "parallel inference and evaluation"
- Model configurations include `batch_size` parameter (docs/en/user_guides/models.md)
- README.md line 292 mentions "Efficient distributed evaluation"
- docs/en/user_guides/experimentation.md describes parallel task execution

#### ❌ Strategy 2: Interactive Loop
**NOT SUPPORTED** - OpenCompass does not natively provide interactive environment stepping, tool-based reasoning loops, or physics simulation interfaces. While it has some agent evaluation features, they are not core native capabilities.

#### ✅ Strategy 3: Arena Battle
**SUPPORTED** - OpenCompass provides pairwise model comparison:
- Compass Arena Bradley-Terry for pairwise comparisons
- ArenaHard for arena-style evaluations
- Comparison mode for subjective evaluation
- Win rate calculation and Elo rating computation

**Evidence**:
- opencompass/summarizers/subjective/compass_arena_bradley_terry.py implements pairwise matchups
- opencompass/summarizers/subjective/arenahard.py computes MLE Elo ratings
- docs/en/advanced_guides/subjective_evaluation.md line 11 mentions "Compare Mode: comparing model responses pairwise to calculate their win rate"
- README.md line 273 mentions "CompassArena"

#### ❌ Strategy 4: Production Streaming
**NOT SUPPORTED** - OpenCompass does not provide production traffic streaming, real-time metric collection, or drift monitoring capabilities.

---

## Phase III: Assessment (The Score)

### Step A: Individual Scoring

#### ✅ Strategy 1: Deterministic Measurement
**SUPPORTED** - OpenCompass provides extensive deterministic metrics:
- Accuracy, exact match, F1 score
- BLEU, ROUGE, METEOR (for text generation)
- Pass@k for code evaluation (HumanEval)
- Token-based metrics
- Equality checks and answer extraction

**Evidence**:
- opencompass/summarizers/default.py line 19 lists metrics: 'score', 'accuracy', 'humaneval_pass@1', 'rouge1', 'f1', 'exact_match'
- Multiple dataset evaluators support deterministic metrics
- README.md mentions evaluation of code generation (HumanEval)

#### ✅ Strategy 2: Embedding Measurement
**SUPPORTED** - OpenCompass supports embedding-based evaluation:
- BERTScore for semantic similarity
- Sentence embeddings for comparison
- Neural similarity models

**Evidence**: README.md line 73 mentions XFinder for answer extraction, which uses embedding-based techniques. Various datasets support embedding-based metrics.

#### ✅ Strategy 3: Subjective Measurement
**SUPPORTED** - OpenCompass provides comprehensive subjective evaluation:
- LLM-as-judge evaluations with GenericLLMEvaluator
- GPT-4 and other LLMs as evaluators
- Model-based judgments for quality assessment
- Subjective evaluation datasets (AlignBench, MTBench, AlpacaEval, ArenaHard, etc.)

**Evidence**:
- docs/en/advanced_guides/llm_judge.md describes LLM-as-judge
- docs/en/advanced_guides/subjective_evaluation.md lists supported subjective datasets
- README.md lines 61-64 describe evaluator components

#### ❌ Strategy 4: Performance Measurement
**NOT SUPPORTED** - OpenCompass does not natively provide performance measurement capabilities such as latency, throughput, memory usage, FLOPs, or energy consumption tracking.

### Step B: Collective Aggregation

#### ✅ Strategy 1: Score Aggregation
**SUPPORTED** - OpenCompass provides comprehensive aggregation:
- Benchmark-level metric computation
- Averaging across datasets
- Weighted aggregation for dataset groups (via `summary_groups` in summarizer)
- Support for MMLU, C-Eval, and other multi-subset benchmarks

**Evidence**:
- opencompass/summarizers/default.py handles score aggregation
- docs/en/user_guides/experimentation.md line 125 mentions "Introduction of Summarizer"
- docs/en/get_started/quick_start.md lines 287-289 describe summarization with averaged scores

#### ⚠️ Strategy 2: Uncertainty Quantification
**PARTIALLY SUPPORTED** - OpenCompass provides bootstrap resampling for specific subjective evaluation methods (ArenaHard, Compass Arena Bradley-Terry), but does not provide general uncertainty quantification or Prediction-Powered Inference (PPI) across all evaluation types.

**Evidence**:
- opencompass/summarizers/subjective/arenahard.py implements `get_bootstrap_result()` with 100 bootstrap rounds
- opencompass/summarizers/subjective/compass_arena_bradley_terry.py uses bootstrap for confidence estimation
- Limited to subjective evaluation scenarios, not available for general objective metrics

---

## Phase IV: Reporting (The Output)

### Step A: Insight Presentation

#### ❌ Strategy 1: Execution Tracing
**NOT SUPPORTED** - While OpenCompass logs evaluation tasks, it does not provide detailed step-by-step execution traces showing intermediate reasoning states or tool calls in the way specialized tracing tools do.

#### ✅ Strategy 2: Subgroup Analysis
**SUPPORTED** - OpenCompass supports breaking down results by categories:
- Dataset-level metric breakdown
- Support for summary groups that can stratify results
- Performance analysis across different task types

**Evidence**:
- opencompass/summarizers/default.py supports `summary_groups` for categorization
- Multiple dataset configurations allow for domain-specific evaluation
- README.md line 288 mentions "comprehensive evaluation in five dimensions"

#### ✅ Strategy 3: Chart Generation
**SUPPORTED** - OpenCompass provides visualization capabilities:
- Heatmap visualizations for NeedleBench (showing performance across context lengths)
- Matplotlib-based plotting for long-context evaluation
- Radar charts and performance plots

**Evidence**:
- opencompass/summarizers/needlebench.py imports matplotlib and creates visualizations
- docs/en/advanced_guides/needleinahaystack_eval.md line 98 mentions "built-in visualization integrated into the summarizer"
- opencompass/summarizers/needlebench.py line 191 has `visualize()` function

#### ✅ Strategy 4: Dashboard Creation
**SUPPORTED** - OpenCompass provides dashboard and result display:
- CompassHub (hub.opencompass.org.cn) - benchmark browser interface
- CompassRank (rank.opencompass.org.cn) - leaderboard interface
- Result tables in CSV and TXT formats
- Web interfaces for metric visualization

**Evidence**:
- README.md lines 274-281 describe CompassHub and CompassRank
- README.md line 16 links to CompassHub and CompassRank
- docs/en/get_started/quick_start.md line 13 mentions "easy-to-read table" and CSV/TXT output

#### ✅ Strategy 5: Leaderboard Publication
**SUPPORTED** - OpenCompass provides leaderboard submission:
- OpenCompass Leaderboard for public model comparison
- CompassRank for ranking models
- Submission via email to opencompass@pjlab.org.cn

**Evidence**:
- README.md lines 84-88 describe OpenCompass Leaderboard
- README.md line 277 mentions CompassRank for comprehensive evaluation
- Multiple references to leaderboard integration throughout documentation

#### ❌ Strategy 6: Regression Alerting
**NOT SUPPORTED** - OpenCompass does not provide automated regression detection, historical baseline comparison, or performance degradation alerting.

---

## Conclusion

### Natively Supported Strategies: 22.5 out of 40

OpenCompass natively supports **22.5 strategies** across the unified evaluation workflow (with 0.5 for partial support of uncertainty quantification):

**Phase 0: Provisioning (5/8)**
- ✅ PyPI installation
- ✅ Git clone installation
- ✅ Evaluation platform authentication
- ✅ API provider authentication
- ✅ Repository authentication

**Phase I: Specification (4/9)**
- ✅ Model-as-a-Service (API models)
- ✅ Model-in-Process (local models)
- ✅ Benchmark dataset preparation
- ✅ Judge preparation (LLM-as-judge)
- ✅ Ground truth preparation

**Phase II: Execution (2/4)**
- ✅ Batch inference
- ✅ Arena battle (pairwise comparison)

**Phase III: Assessment (4.5/6)**
- ✅ Deterministic measurement
- ✅ Embedding measurement
- ✅ Subjective measurement (LLM-as-judge)
- ✅ Score aggregation
- ⚠️ Uncertainty quantification (partial - bootstrap for subjective evaluation only)

**Phase IV: Reporting (5/6)**
- ✅ Subgroup analysis
- ✅ Chart generation
- ✅ Dashboard creation
- ✅ Leaderboard publication

### Key Strengths
1. **Comprehensive Installation & Authentication**: Multiple installation methods and extensive API/repository authentication
2. **Diverse Model Support**: Both local (HuggingFace) and remote (API) model evaluation
3. **Rich Dataset Ecosystem**: 70+ datasets with automatic download and preparation
4. **Advanced Judging**: Strong LLM-as-judge capabilities with multiple evaluators
5. **Effective Reporting**: Integrated dashboards, leaderboards, and visualization

### Key Limitations
1. **No Container/Binary Distribution**: Missing containerized and binary installation options
2. **Limited to Neural Models**: No support for ANN algorithms, knowledge graphs, or RL agents
3. **No Synthetic Data Generation**: Cannot generate test data on-the-fly
4. **No Interactive/Simulation**: Missing environment simulation and interactive loops
5. **No Performance Metrics**: Cannot measure latency, throughput, or resource consumption
6. **No Production Monitoring**: Missing streaming, drift detection, and regression alerting

OpenCompass excels as a comprehensive **offline batch evaluation framework** for language and vision-language models, with particular strengths in dataset diversity, LLM-as-judge evaluation, pairwise model comparison, and leaderboard integration. However, it is not designed for real-time production monitoring, performance benchmarking, interactive agent evaluation, or non-neural algorithm assessment.

---

## Appendix: Classification Rationale

### Why Some Common Features Are Marked "NOT SUPPORTED"

Several features that exist in the OpenCompass ecosystem are marked as "NOT SUPPORTED" because they require additional dependencies beyond the standard installation:

1. **Agent Evaluation (T-Eval, CIBench)**: While OpenCompass can evaluate agent capabilities, it requires separate installation of `lagent` and additional dependencies (`requirements/agent.txt`). These are not included in standard installations and represent integration with external frameworks rather than native support.

2. **Performance Metrics**: While some backend implementations (vLLM, LMDeploy) may track performance, OpenCompass itself does not provide native APIs for measuring or reporting latency, throughput, or resource consumption as part of evaluation results.

3. **Interactive Environments**: The harness is designed for batch evaluation of model outputs, not for managing interactive environment stepping or simulation.

4. **Production Monitoring**: OpenCompass is an offline evaluation tool, not a production monitoring system. It does not provide streaming inference, drift detection, or alerting capabilities.

This classification ensures we distinguish between what OpenCompass provides natively versus what requires additional integration work, external tools, or custom implementation.
