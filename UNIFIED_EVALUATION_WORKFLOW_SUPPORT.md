# OpenCompass: Unified Evaluation Workflow Support Analysis

This document analyzes which strategies from the Unified Evaluation Workflow are **natively supported** by OpenCompass. 

## Methodology

This analysis uses a three-tier classification system to distinguish between different levels of support:

### Classification Criteria

#### **Native Support (Natively Supported)**
Strategies that meet ALL of the following requirements:
- Available immediately after `pip install "opencompass[full]"`
- Requires only import statements and minimal configuration (≤2 lines)
- No external dependencies beyond the `[full]` installation
- No custom implementation or glue code required

**Example:**
```python
from opencompass.models import HuggingFaceCausalLM
# model config ready to use
```

#### **Integrated Support (Supported via Third-Party Integration)**
Strategies that meet ALL of the following requirements:
- Requires installing ≥1 external package(s) beyond `[full]` (e.g., `[api]`, `[lmdeploy]`, `[vllm]`)
- Requires glue code or additional configuration (typically ≤10 lines)
- Has documented integration pattern or official example in OpenCompass
- Functionality enabled through third-party tools rather than the harness alone

**Example:**
```python
# Requires: pip install "opencompass[api]"
from opencompass.models import OpenAI
# + API key configuration
```

#### **Not Supported**
Strategies that:
- Are not available even with third-party integration
- Require significant custom implementation (>10 lines of glue code)
- Have no documented integration pattern
- Are fundamentally incompatible with the harness design

### Installation Baseline

The analysis uses `pip install "opencompass[full]"` as the baseline for native support, which includes:
- Core runtime dependencies (`requirements/runtime.txt`)
- Extended dataset support (`requirements/extra.txt`)
- Additional evaluation capabilities (alpaca-eval, human-eval, faiss, etc.)

Additional installation options (`[api]`, `[lmdeploy]`, `[vllm]`) are considered third-party integrations.

The analysis is based on:
- Official documentation (README.md, docs/)
- Source code examination (opencompass/)
- Setup configuration (setup.py, requirements/)
- Integration examples and patterns

## Summary

OpenCompass is a comprehensive evaluation harness for large language models and vision-language models. This analysis classifies support into three tiers:

- **Natively Supported**: Available with `pip install "opencompass[full]"` with minimal configuration
- **Supported via Third-Party Integration**: Requires additional packages (e.g., `[api]`, `[lmdeploy]`) or external tools, but documented
- **Not Supported**: Not available even with integrations

OpenCompass provides strong native support for dataset preparation, batch inference, scoring methods, and leaderboard integration. Additional capabilities like API models and accelerated inference are available through well-documented third-party integrations.

---

## Phase 0: Provisioning (The Runtime)

### Step A: Harness Installation

#### ✅ Strategy 1: PyPI Packages
**NATIVELY SUPPORTED** - OpenCompass can be installed via pip:
- `pip install -U opencompass` (basic installation)
- `pip install "opencompass[full]"` **(recommended full installation with extended dataset support)**
- `pip install "opencompass[api]"` (API evaluation support)
- `pip install "opencompass[lmdeploy]"` (with LMDeploy backend)
- `pip install "opencompass[vllm]"` (with vLLM backend)

**Evidence**: README.md, docs/en/get_started/installation.md, setup.py

#### ✅ Strategy 2: Git Clone
**NATIVELY SUPPORTED** - OpenCompass can be installed from source:
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
**NATIVELY SUPPORTED** - OpenCompass supports authentication with evaluation platforms and leaderboard submission:
- Integration with OpenCompass Leaderboard (CompassRank)
- Submission to public leaderboards for model comparison

**Evidence**: README.md mentions "CompassRank" leaderboard at rank.opencompass.org.cn. Lines 86-88 state "If you would like to join the evaluation, please provide the model repository URL or a standard API interface to the email address opencompass@pjlab.org.cn"

#### 🔌 Strategy 2: API Provider Authentication
**SUPPORTED VIA THIRD-PARTY INTEGRATION** - OpenCompass supports API authentication through the `[api]` installation extra:
- Requires: `pip install "opencompass[api]"`
- OpenAI API (`OPENAI_API_KEY`) - also available in base installation
- Commercial providers (Claude/Anthropic, Gemini, Qwen/Dashscope, ZhipuAI, Baichuan, ByteDance, Huawei, Baidu, MiniMax, SenseTime, Xunfei, etc.)
- Configuration via environment variables or credential files

**Evidence**: 
- requirements/api.txt includes: anthropic, dashscope, zhipuai, volcengine, etc.
- opencompass/models/ contains API model implementations requiring these dependencies
- README.md line 201: "export OPENAI_API_KEY="YOUR_OPEN_API_KEY""
- docs/en/get_started/installation.md describes API installation: `pip install "opencompass[api]"`

#### ✅ Strategy 3: Repository Authentication
**NATIVELY SUPPORTED** - OpenCompass supports authentication with model and dataset repositories:
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

#### 🔌 Strategy 1: Model-as-a-Service (Remote Inference)
**SUPPORTED VIA THIRD-PARTY INTEGRATION** - OpenCompass supports remote inference through the `[api]` installation extra:
- Requires: `pip install "opencompass[api]"`
- OpenAI API models (GPT-3.5, GPT-4, GPT-4o, o1) - also available in base installation
- Commercial API providers (Claude, Gemini, ZhipuAI, Baichuan, ByteDance, Huawei, Baidu, MiniMax, SenseTime, Xunfei, etc.)
- Custom API endpoints

**Evidence**:
- requirements/api.txt lists required packages (anthropic, dashscope, zhipuai, etc.)
- README.md lines 346-361 list supported API models
- opencompass/models/ contains multiple API model implementations
- docs/en/user_guides/models.md lines 68-106 describe API-based models

#### ✅ Strategy 2: Model-in-Process (Local Inference)
**NATIVELY SUPPORTED** - OpenCompass supports local model inference:
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
**NATIVELY SUPPORTED** - OpenCompass provides comprehensive dataset preparation:
- Automatic download from HuggingFace Datasets
- Automatic download from ModelScope datasets
- Manual download of OpenCompass custom datasets
- Support for 70+ datasets covering ~400,000 questions
- Data loading, splitting, normalization, and formatting

**Evidence**:
- docs/en/get_started/installation.md lines 103-141 describe dataset preparation
- README.md lines 137-171 describe multiple dataset preparation methods
- Extensive dataset implementations in opencompass/datasets/ directory (hundreds of files)
- Supports MMLU, GSM8K, HumanEval, CEVAL, GAOKAO, and many others

#### ❌ Strategy 2: Synthetic Data Generation (Generative)
**NOT SUPPORTED** - OpenCompass does not natively provide synthetic data generation, test augmentation, or perturbation capabilities. The circular evaluation feature (opencompass/datasets/circular.py) appears to be for data formatting rather than synthetic generation.

#### ❌ Strategy 3: Simulation Environment Setup (Simulated)
**NOT SUPPORTED** - OpenCompass does not provide 3D simulation environments, physics simulation, or interactive environment setup.

#### ❌ Strategy 4: Production Traffic Sampling (Online)
**NOT SUPPORTED** - OpenCompass does not provide production traffic sampling or real-world inference traffic collection capabilities.

### Step C: Benchmark Preparation (References)

#### ✅ Strategy 1: Judge Preparation
**NATIVELY SUPPORTED** - OpenCompass provides comprehensive LLM-as-judge capabilities:
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
**NATIVELY SUPPORTED** - OpenCompass loads and pre-processes ground truth references:
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
**NATIVELY SUPPORTED** - OpenCompass provides comprehensive batch inference with `[full]` installation:
- Parallel inference across multiple samples with HuggingFace models
- Configurable batch sizes
- Support for single model evaluation across multiple datasets
- Efficient distributed evaluation with task partitioning

**Additional Integration**: Accelerated inference backends available via separate installation:
- vLLM: `pip install "opencompass[vllm]"`
- LMDeploy: `pip install "opencompass[lmdeploy]"`

**Evidence**:
- docs/en/get_started/quick_start.md line 11 describes "parallel inference and evaluation"
- Model configurations include `batch_size` parameter (docs/en/user_guides/models.md)
- README.md line 292 mentions "Efficient distributed evaluation"
- docs/en/user_guides/experimentation.md describes parallel task execution
- README.md line 212 shows accelerated evaluation with LMDeploy/vLLM

#### ❌ Strategy 2: Interactive Loop
**NOT SUPPORTED** - OpenCompass does not natively provide interactive environment stepping, tool-based reasoning loops, or physics simulation interfaces. While it has some agent evaluation features, they are not core native capabilities.

#### ✅ Strategy 3: Arena Battle
**NATIVELY SUPPORTED** - OpenCompass provides pairwise model comparison:
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
**NATIVELY SUPPORTED** - OpenCompass provides extensive deterministic metrics:
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
**NATIVELY SUPPORTED** - OpenCompass supports embedding-based evaluation:
- BERTScore for semantic similarity
- Sentence embeddings for comparison
- Neural similarity models

**Evidence**: README.md line 73 mentions XFinder for answer extraction, which uses embedding-based techniques. Various datasets support embedding-based metrics.

#### ✅ Strategy 3: Subjective Measurement
**NATIVELY SUPPORTED** - OpenCompass provides comprehensive subjective evaluation:
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
**NATIVELY SUPPORTED** - OpenCompass provides comprehensive aggregation:
- Benchmark-level metric computation
- Averaging across datasets
- Weighted aggregation for dataset groups (via `summary_groups` in summarizer)
- Support for MMLU, C-Eval, and other multi-subset benchmarks

**Evidence**:
- opencompass/summarizers/default.py handles score aggregation
- docs/en/user_guides/experimentation.md line 125 mentions "Introduction of Summarizer"
- docs/en/get_started/quick_start.md lines 287-289 describe summarization with averaged scores

#### ⚠️ Strategy 2: Uncertainty Quantification
**PARTIALLY NATIVELY SUPPORTED** - OpenCompass provides bootstrap resampling for specific subjective evaluation methods (ArenaHard, Compass Arena Bradley-Terry), but does not provide general uncertainty quantification or Prediction-Powered Inference (PPI) across all evaluation types.

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
**NATIVELY SUPPORTED** - OpenCompass supports breaking down results by categories:
- Dataset-level metric breakdown
- Support for summary groups that can stratify results
- Performance analysis across different task types

**Evidence**:
- opencompass/summarizers/default.py supports `summary_groups` for categorization
- Multiple dataset configurations allow for domain-specific evaluation
- README.md line 288 mentions "comprehensive evaluation in five dimensions"

#### ✅ Strategy 3: Chart Generation
**NATIVELY SUPPORTED** - OpenCompass provides visualization capabilities:
- Heatmap visualizations for NeedleBench (showing performance across context lengths)
- Matplotlib-based plotting for long-context evaluation
- Radar charts and performance plots

**Evidence**:
- opencompass/summarizers/needlebench.py imports matplotlib and creates visualizations
- docs/en/advanced_guides/needleinahaystack_eval.md line 98 mentions "built-in visualization integrated into the summarizer"
- opencompass/summarizers/needlebench.py line 191 has `visualize()` function

#### ✅ Strategy 4: Dashboard Creation
**NATIVELY SUPPORTED** - OpenCompass provides dashboard and result display:
- CompassHub (hub.opencompass.org.cn) - benchmark browser interface
- CompassRank (rank.opencompass.org.cn) - leaderboard interface
- Result tables in CSV and TXT formats
- Web interfaces for metric visualization

**Evidence**:
- README.md lines 274-281 describe CompassHub and CompassRank
- README.md line 16 links to CompassHub and CompassRank
- docs/en/get_started/quick_start.md line 13 mentions "easy-to-read table" and CSV/TXT output

#### ✅ Strategy 5: Leaderboard Publication
**NATIVELY SUPPORTED** - OpenCompass provides leaderboard submission:
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

### Support Summary

OpenCompass provides support across the unified evaluation workflow through three tiers:

#### **Natively Supported: 20.5 out of 40 strategies**

Strategies available immediately with `pip install "opencompass[full]"`:

**Phase 0: Provisioning (3/8)**
- ✅ PyPI installation
- ✅ Git clone installation
- ✅ Repository authentication

**Phase I: Specification (3/9)**
- ✅ Model-in-Process (local models via HuggingFace)
- ✅ Benchmark dataset preparation (70+ datasets)
- ✅ Judge preparation (LLM-as-judge)
- ✅ Ground truth preparation

**Phase II: Execution (2/4)**
- ✅ Batch inference (HuggingFace models)
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

#### **Supported via Third-Party Integration: 2 out of 40 strategies**

Strategies requiring additional installation (`[api]`, `[lmdeploy]`, `[vllm]`):

**Phase 0: Provisioning (2/8)**
- 🔌 API provider authentication (requires `[api]`)
- 🔌 Evaluation platform authentication (for some platforms)

**Phase I: Specification (1/9)**
- 🔌 Model-as-a-Service (requires `[api]` for most providers)

**Note**: Batch inference with accelerated backends (vLLM, LMDeploy) is also available as an integration but is counted under native batch inference support.

#### **Not Supported: 17.5 out of 40 strategies**

**Phase 0: Provisioning (3/8)**
- ❌ Container images
- ❌ Binary packages
- ❌ Node package

**Phase I: Specification (5/9)**
- ❌ Algorithm implementation (ANN, knowledge graphs)
- ❌ Policy/Agent instantiation (RL agents)
- ❌ Synthetic data generation
- ❌ Simulation environment setup
- ❌ Production traffic sampling

**Phase II: Execution (2/4)**
- ❌ Interactive loop
- ❌ Production streaming

**Phase III: Assessment (1/6)**
- ❌ Performance measurement (latency, throughput)

**Phase IV: Reporting (1/6)**
- ❌ Regression alerting

### Key Strengths
1. **Comprehensive Dataset Support**: 70+ datasets with automatic download and preparation
2. **Flexible Model Evaluation**: Both local (HuggingFace) and API models supported
3. **Advanced Judging**: Strong LLM-as-judge capabilities with multiple evaluators
4. **Rich Scoring Methods**: Deterministic, embedding-based, and subjective measurement
5. **Effective Reporting**: Integrated dashboards, leaderboards, and visualization
6. **Well-Documented Integrations**: Clear patterns for API models and accelerated inference

### Key Limitations
1. **No Container/Binary Distribution**: Missing containerized and binary installation options
2. **Limited to Neural Models**: No support for ANN algorithms, knowledge graphs, or RL agents
3. **No Synthetic Data Generation**: Cannot generate test data on-the-fly
4. **No Interactive/Simulation**: Missing environment simulation and interactive loops
5. **No Performance Metrics**: Cannot measure latency, throughput, or resource consumption
6. **No Production Monitoring**: Missing streaming, drift detection, and regression alerting

### Use Case Fit

**Excellent for:**
- Offline batch evaluation of LLMs and VLMs
- Benchmark-based model comparison
- Subjective evaluation with LLM judges
- Academic research and model development

**Not suitable for:**
- Real-time production monitoring
- Performance/latency benchmarking
- Interactive agent evaluation
- Non-neural algorithm assessment (ANN, knowledge graphs, etc.)

OpenCompass excels as a comprehensive **offline batch evaluation framework** for language and vision-language models, with particular strengths in dataset diversity, LLM-as-judge evaluation, pairwise model comparison, and leaderboard integration.

---

## Appendix: Classification Details

### Three-Tier Classification System

This analysis distinguishes between native support, third-party integration, and unsupported features:

#### **Tier 1: Native Support**
Features included with `pip install "opencompass[full]"`:
- Core runtime dependencies (`requirements/runtime.txt`)
- Extended dataset support (`requirements/extra.txt`):
  - `alpaca-eval` for AlpacaEval subjective evaluation
  - `human-eval` for code evaluation (HumanEval, HumanEval+)
  - `math-verify` for mathematical verification
  - `faiss_gpu` for in-context learning retrieval
  - Various dataset-specific libraries (ltp, pypinyin, rdkit, etc.)

#### **Tier 2: Third-Party Integration**
Features requiring additional installation beyond `[full]`:

1. **API Models** (`pip install "opencompass[api]"`):
   - Includes: anthropic, dashscope, zhipuai, volcengine, etc.
   - Use case: Evaluating commercial API models (Claude, Gemini, Qwen, etc.)
   - Integration: Simple configuration (API keys + model config)

2. **Accelerated Inference** (`pip install "opencompass[lmdeploy]"` or `"opencompass[vllm]"`):
   - Includes: lmdeploy or vllm packages
   - Use case: Faster inference for large models
   - Integration: Backend specification in model config

3. **Agent Evaluation** (requires `requirements/agent.txt`):
   - Includes: lagent-cibench, tensorflow, jupyter
   - Use case: T-Eval, CIBench agent benchmarks
   - **Status**: Listed as example but not counted as supported integration due to complexity

#### **Tier 3: Not Supported**
Features that are either:
- Not available even with third-party packages
- Require significant custom implementation (>10 lines)
- Have no documented integration pattern
- Are fundamentally incompatible with harness design

### Why Some Features Are Not Counted as Integrated Support

1. **Agent Evaluation (T-Eval, CIBench)**: While packages exist in `requirements/agent.txt`, the integration requires:
   - Separate environment setup (conflicts with main dependencies)
   - Complex agent framework integration (lagent-cibench)
   - No simple documented pattern in main docs
   - Typically >10 lines of glue code

2. **Performance Metrics**: While backends like vLLM may track performance internally, OpenCompass does not expose APIs for measuring or reporting latency, throughput, or resource consumption as part of evaluation results.

3. **Interactive Environments & Simulation**: These require external simulation frameworks (e.g., robot simulators, game engines) that are beyond the scope of evaluation harness integration.

4. **Production Monitoring**: OpenCompass is designed as an offline evaluation tool, not a production monitoring system. Real-time streaming, drift detection, and alerting require fundamentally different architectures.

### Counting Methodology

- **Native**: Counted if available with `[full]` installation
- **Integrated**: Counted if requires `[api]`, `[lmdeploy]`, or `[vllm]` extras AND has documented integration pattern
- **Not Supported**: All other cases

This ensures clear distinction between out-of-the-box functionality (native), well-documented extensions (integrated), and unavailable features (not supported).
