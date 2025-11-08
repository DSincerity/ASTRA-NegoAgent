# ASTRA-NegoAgent

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue.svg)](https://2025.emnlp.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2503.07129-b31b1b.svg)](https://arxiv.org/abs/2503.07129)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository contains the implementation of **ASTRA** (Adaptive Strategic Reasoning with Action), a sophisticated multi-agent negotiation framework that enables AI agents to engage in strategic resource allocation negotiations using Large Language Models (LLMs).

## 📄 Paper Information

**Title:** [ASTRA: A Negotiation Agent with Adaptive and Strategic Reasoning via Tool-integrated Action for Dynamic Offer Optimization](https://arxiv.org/abs/2503.07129)

**Authors:** Deuksin Kwon, Jiwon Hae, Emma Clift, Daniel Shamsoddini, Jonathan Gratch, Gale M. Lucas

**Conference:** Main Conference of EMNLP 2025

<div align="center">
<img src="images/main_fig.png" alt="ASTRA Framework Overview" width="700"/>
</div>

## 🎯 Key Features

- **Multi-LLM Support**: Compatible with OpenAI GPT, Google Gemini, and Anthropic Claude models
- **Strategic Reasoning**: Three-stage ASTRA module for adaptive negotiation strategies
- **Linear Programming Integration**: Optimal offer generation using mathematical optimization
- **Priority Consistency Checking**: Real-time validation of partner behavior patterns
- **Flexible Negotiation Types**: Support for integrative, distributive, and mixed negotiations
- **Comprehensive Evaluation**: Built-in metrics for negotiation outcome analysis

For any questions, please contact: **Brian Deuksin Kwon** (deuksink@usc.edu)

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.10.16 or higher
- **Package Manager**: pip or conda
- **Recommended**: pyenv + virtualenv for Python version management

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ASTRA-NegoAgent.git
cd ASTRA-NegoAgent

# Create and activate virtual environment
pyenv virtualenv 3.10.16 ASTRA-env
pyenv activate ASTRA-env

# Install dependencies
pip install -r requirements.txt
```

### 2. API Keys Configuration

Set up environment variables for the LLM APIs you plan to use:

> 💡 **Tip**: You only need to set keys for the models you intend to use.

```bash
# For OpenAI models (GPT-4o, GPT-4o-mini)
export OPENAI_API_KEY="your-openai-api-key"

# For Anthropic models (Claude-3.5-Sonnet)
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# For Google models (Gemini-1.5-Flash, Gemini-2.0-Flash)
export GEMINI_API_KEY="your-google-gemini-api-key"
```

### 3. Verify Installation

```bash
# Test the installation
python agent_agent_simulation.py --help
```

## 🧪 Running Experiments

### Basic Experiment
```bash
# Run a simple negotiation with default settings
python agent_agent_simulation.py \
  --n_exp 1 \
  --n_round 15 \
  --engine-STR gpt-4o-mini \
  --engine-partner gpt-4o-mini \
  --negotiation-type integrative \
  --STR
```

### Advanced Configuration
```bash
# Run with ASTRA strategic reasoning enabled
python agent_agent_simulation.py \
  --n_exp 3 \
  --n_round 10 \
  --engine-STR gpt-4o-mini \
  --engine-partner gpt-4o-mini \
  --negotiation-type integrative \
  --fine-grained-OSAD \
  -w1 0.35 \
  -w2 0.65 \
  --top_n 5 \
  --STR
```

### Batch Experiments
```bash
# Run predefined experiment configurations
bash run_a2a_experiment.sh
```

### Configuration Options

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--engine-STR` | LLM for strategic reasoning agent | `gpt-4o`, `gpt-4o-mini`, `gemini-2.0-flash`, `claude-3-5-sonnet-20241022` |
| `--engine-partner` | LLM for partner agent | Same as above |
| `--negotiation-type` | Type of negotiation | `integrative`, `distributive`, `mixed` |
| `--partner-agent-personality` | Partner personality | `base`, `greedy`, `fair` |
| `--STR` | Enable ASTRA strategic reasoning | Flag (no value needed) |
| `--fine-grained-OSAD` | Enable fine-grained assessment for the Partner's Acceptance Probability (PAP) | Flag (no value needed) |
| `-w1`, `-w2` | Weights for OSAD (for PAP) and Self-Assessment (SA) | Float values (e.g., 0.35, 0.65) |

## 🏗️ Project Architecture

The ASTRA framework consists of several key components organized in a modular structure:

```
ASTRA-NegoAgent/
├── agent/                          # Core agent implementations
│   ├── __init__.py                 # Agent package exports
│   ├── agent.py                    # Main negotiation agent classes
│   └── base_dialog_agent.py        # Base LLM dialog interface
├── components/                     # Strategic reasoning modules
│   ├── astra.py                    # ASTRA strategic reasoning pipeline
│   ├── priority_consistency_check.py  # Opponent modeling: Partner behavior validation
│   ├── partner_preference_asker.py # Opponent modeling: Partner preference inquiry system
│   └── partner_preference_updater.py  # Opponent modeling: Partner preference update logic
├── prompt/                         # Prompt templates and builders
├── async_lib_api.py               # Unified LLM API wrapper
├── tools.py                       # Linear programming optimization
├── utils.py                       # Common utilities and helpers
├── agent_agent_simulation.py      # Main simulation runner
└── run_a2a_experiment.sh          # Experiment configuration script
```

### Core Components

- **🤖 Agent System**: Multi-LLM negotiation agents with strategic reasoning capabilities
- **🧠 ASTRA Module**: Three-stage strategic reasoning (fairness prediction, LP optimization, offer selection)
- **🔍 Consistency Checker**: Real-time validation of partner priority inference
- **❓ Preference Inquiry**: Dynamic partner preference questioning system
- **🔄 Preference Updates**: Adaptive partner preference update mechanisms
- **⚙️ LP Optimizer**: Mathematical optimization for offer generation
- **🌐 API Layer**: Unified interface for multiple LLM providers with retry logic

## 📊 Output and Results

Experiment results are saved in JSON format in the `a2a_results/` directory with detailed information including:

- Negotiation outcomes (accept/reject/walk-away)
- Complete dialog history and offer exchanges
- Agent priority predictions and accuracy
- Strategic reasoning logs and decision rationales
- Performance metrics and evaluation scores

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## 📄 Citation

If you use this code or data in your research, please cite our paper:

```bibtex
@inproceedings{kwon-etal-2025-astra,
    title = "{ASTRA}: A Negotiation Agent with Adaptive and Strategic Reasoning via Tool-integrated Action for Dynamic Offer Optimization",
    author = "Kwon, Deuksin and Hae, Jiwon and Clift, Emma and Shamsoddini, Daniel and Gratch, Jonathan and Lucas, Gale",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.821/",
    pages = "16228--16249",
    ISBN = "979-8-89176-332-6",
}

```

## 📜 License

Please refer to the LICENSE file in the root directory for more details.
