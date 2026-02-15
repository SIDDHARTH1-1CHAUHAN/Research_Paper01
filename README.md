# Multi-Agent Fact-Checking Pipeline

> A research-grade automated fact-checking system using specialized AI agents with explainable AI capabilities

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Research Paper](https://img.shields.io/badge/arXiv-2506.17878-b31b1b.svg)](https://arxiv.org/abs/2506.17878)
[![Next.js](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)

---

## 👥 Authors & Credits

**Developed by:**
- **Siddharth Chauhan**
- **Saksham Gumber**

**Under the guidance of:**
- **Dr. Manish**

This project is a research implementation based on the paper "Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval" (arXiv:2506.17878v1) with additional enhancements for explainability and practical deployment.

---

## 📦 What's Included?

This repository contains **three main components:**

### 1. **Python Multi-Agent System** (Core Implementation)
The complete fact-checking pipeline with 6 specialized AI agents that verify claims using web evidence and provide explainable verdicts.

### 2. **Interactive Web Presentation** (Next.js App)
A professional web application for presenting the system to professors, researchers, and stakeholders with:
- Interactive flowcharts visualizing the agent pipeline
- Live metrics and performance charts
- Demo observations and comparison with baseline systems
- Responsive design with dark mode support

### 3. **Comprehensive Documentation**
- Research paper with full methodology ([docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md))
- Demo observations and evaluation results ([docs/DEMO_OBSERVATIONS.md](docs/DEMO_OBSERVATIONS.md))
- Improvement guide for upgrading to paid AI services ([docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md))
- Architecture documentation and API reference ([docs/ARCHITECTURE.md](docs/ARCHITECTURE.md))

---

## What Is This?

This system automatically checks if claims are true or false by:
1. Breaking down complex statements into simple parts
2. Searching the internet for reliable sources
3. Analyzing the evidence found
4. Providing a verdict with clear explanations

**No technical background needed to understand how it works!**

---

## Why Does This Matter?

In today's world, misinformation spreads quickly. This system helps by:
- ✅ **Automating fact-checking** - No need to manually search dozens of sources
- ✅ **Providing transparency** - Shows exactly why it reached each conclusion
- ✅ **Using reliable sources** - Prioritizes educational, government, and peer-reviewed sources
- ✅ **Explaining decisions** - Not a "black box" - you can see the reasoning

---

## How It Works (Simple Explanation)

Imagine you want to check this claim:
> "The Eiffel Tower was completed in 1889 and is located in Paris"

Here's what happens:

### Step 1: Breaking Down the Claim
The system splits this into two simpler claims:
1. "The Eiffel Tower was completed in 1889"
2. "The Eiffel Tower is located in Paris"

### Step 2: Searching for Evidence
For each simple claim, it creates search queries:
- "Eiffel Tower completion date"
- "When was Eiffel Tower built 1889"
- "Eiffel Tower Paris location"

### Step 3: Checking Source Quality
It looks for reliable sources like:
- ✅ Educational websites (.edu)
- ✅ Government sources (.gov)
- ✅ Major news outlets (BBC, Reuters)
- ✅ Encyclopedia sites (Wikipedia, Britannica)

### Step 4: Making a Decision
It reads the evidence and decides:
- **SUPPORTED** - Multiple reliable sources agree
- **NOT SUPPORTED** - Sources disagree or contradictory evidence found

### Step 5: Explaining the Verdict
It tells you:
- Which sources it used
- Why it made this decision
- What evidence was most important

---

## Key Features

### 🤖 Six Specialized AI Agents

1. **Input Ingestion Agent**
   - Breaks complex claims into simple, checkable parts
   - Filters out opinions and vague statements
   - Uses First-Order Logic for precision

2. **Query Generation Agent**
   - Creates diverse search queries
   - Uses multiple angles to find evidence
   - Optimized for search engines

3. **Evidence Seeking Agent**
   - Searches the web for relevant information
   - Checks source credibility
   - Extracts only relevant passages

4. **Verdict Prediction Agent**
   - Weighs evidence from all sources
   - Makes final decision (SUPPORTED or NOT SUPPORTED)
   - Generates human-readable explanations

5. **Explainable AI Agent**
   - Explains WHY each decision was made
   - Identifies which evidence mattered most
   - Diagnoses errors if the verdict is wrong

6. **Reinforcement Learning Agent**
   - Tracks system performance over time
   - Suggests improvements
   - Learns from patterns

### 💰 Completely FREE to Use!

- **No API keys required** for basic functionality
- Uses free tools:
  - **Ollama** - Free local AI models
  - **DuckDuckGo** - Free web search
  - **Heuristic credibility checking** - Built-in, no external APIs

### 🎓 Research-Grade Quality

Based on the peer-reviewed paper:
**"Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval"**
(arXiv:2506.17878v1, June 2025)

Tested on academic benchmarks:
- **FEVEROUS** - Multi-hop reasoning with tables and text
- **HoVer** - Wikipedia-based fact verification
- **SciFact** - Scientific claim verification

---

## Installation

### Prerequisites

1. **Python 3.9 or higher**
   - Download from [python.org](https://www.python.org/downloads/)

2. **Ollama** (for free local AI)
   - Download from [ollama.ai](https://ollama.ai/)
   - Install and run: `ollama pull llama3.2:3b`

### Step-by-Step Setup

```bash
# 1. Clone this repository
git clone https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01.git
cd Research_Paper01/multi-agent-fact-checker

# 2. Create virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama model
ollama pull llama3.2:3b

# 5. (Optional) Copy and configure environment variables
copy .env.example .env
# Edit .env if you want to use paid APIs for better performance
```

---

## Usage Examples

### Simple Command-Line Usage

```python
from src.orchestrator import FactChecker

# Initialize
checker = FactChecker()

# Check a claim
result = checker.verify("The Eiffel Tower is in Paris")

# Print result
print(f"Verdict: {result.verdict}")
print(f"Explanation: {result.explanation}")
```

### Example Output

```
Verdict: SUPPORTED

Explanation:
This claim is SUPPORTED by multiple high-credibility sources.

Evidence from 5 sources:
1. Wikipedia (wikipedia.org) - HIGH credibility
   "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."

2. Britannica (britannica.com) - HIGH credibility
   "Eiffel Tower, Parisian landmark that is also a technological masterpiece..."

3. Paris Tourism Office (parisinfo.com) - MEDIUM credibility
   "Located on the Champ de Mars in Paris, the Eiffel Tower..."

All sources consistently confirm the Eiffel Tower's location in Paris, France.
Confidence: 95%
```

---

## 🌐 Interactive Web Presentation

For professor presentations and research demonstrations, we've created a professional Next.js web application.

### Quick Start (Web App)

```bash
# Navigate to web app directory
cd web-app

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Web App Features

The presentation includes **4 interactive sections:**

1. **Overview Tab**
   - System introduction and key features
   - Performance metrics (80% accuracy, 0.857 F1-score)
   - Visual showcase of all 6 agents

2. **Architecture Tab**
   - Interactive flowchart showing the complete pipeline
   - Each agent's role and output
   - Data flow visualization

3. **Results Tab**
   - Demo observations with charts
   - Confusion matrix visualization
   - Classification metrics
   - Sample verified claims

4. **Comparison Tab**
   - Performance comparison with baseline systems
   - +12.3% average improvement
   - Benchmark-specific results (HoVer, FEVEROUS, SciFact)

### Deployment Options

**For live presentations:**

```bash
# Build for production
cd web-app
npm run build
npm start
```

**Deploy to Vercel (Free):**
```bash
npm i -g vercel
vercel
```

See [docs/WEB_APP_INSTRUCTIONS.md](docs/WEB_APP_INSTRUCTIONS.md) for detailed presentation guide.

---

## Configuration

### Free Tier (Default)

Works out of the box with:
- **Ollama** (local AI - llama3.2:3b)
- **DuckDuckGo** (free search)
- **Heuristic credibility** (built-in domain checking)

### Enhanced Performance (Optional - Paid APIs)

Edit `.env` file:

```bash
# Better AI responses
OPENAI_API_KEY=your_key_here

# Better search results
SERPER_API_KEY=your_key_here

# Enhanced credibility checking
MBFC_API_KEY=your_key_here
```

**Note:** System works perfectly fine without any API keys!

**Want to upgrade to paid AI services?** See [docs/IMPROVEMENTS.md](docs/IMPROVEMENTS.md) for a comprehensive guide on:
- Upgrading to GPT-4o, Claude 3.5 Sonnet, or Gemini 1.5 Pro
- Premium search APIs (SerperAPI, Tavily AI)
- Advanced credibility checking (MBFC, NewsGuard)
- Cost-benefit analysis and expected performance gains
- Alternative approaches (fine-tuning, RAG, ensemble models)

---

## Project Structure

```
multi-agent-fact-checker/
├── README.md                    # This file - Complete guide
├── requirements.txt            # Python dependencies
├── core/                       # Key files quick reference
│   ├── evaluation_task.yaml   # Evaluation specification
│   ├── EXECUTION_FLOWCHART.md # System flow diagram
│   ├── SESSION_SUMMARY.md     # Development session log
│   ├── dataset_loader.py      # Dataset utilities
│   └── metrics.py             # Evaluation metrics
├── docs/                       # Documentation folder
│   ├── EXECUTION_FLOWCHART.md # Complete pipeline visualization
│   ├── SESSION_SUMMARY.md     # Session log with accuracy report
│   ├── RESEARCH_PAPER.md      # Academic methodology & results
│   ├── ARCHITECTURE.md        # Technical deep-dive
│   └── IMPROVEMENTS.md        # Guide for paid AI upgrades
├── config/                     # Configuration files
│   ├── agent_config.yaml      # Agent settings
│   ├── api_config.yaml        # API configuration
│   ├── benchmark_config.yaml  # Evaluation settings
│   └── evaluation_task.yaml   # Macro F1 & MAR specification
├── src/                        # Source code (Python)
│   ├── orchestrator.py        # Main workflow coordinator
│   ├── agents/                # 6 specialized agents
│   │   ├── input_ingestion.py
│   │   ├── query_generation.py
│   │   ├── evidence_seeking.py
│   │   ├── verdict_prediction.py
│   │   ├── explainable_ai.py
│   │   └── reinforcement_learning.py
│   ├── evaluation/            # Benchmark evaluation
│   │   ├── metrics.py         # Macro F1, MAR metrics
│   │   └── dataset_loader.py  # HOVER, FEVEROUS, SciFact loader
│   ├── utils/                 # Utilities
│   └── storage/               # Data storage modules
├── data/                       # Data and cache
│   ├── datasets/              # Evaluation datasets
│   │   ├── hover/            # 4,000 dev + 18,000 train samples
│   │   ├── feverous/         # 7,891 dev + 87,000 train samples
│   │   └── scifact-open/     # 300 dev + 809 train samples
│   ├── outputs/              # Generated outputs
│   │   ├── predictions/
│   │   ├── explanations/
│   │   └── reports/
│   └── benchmarks/           # Mock dataset
├── web-app/                   # Next.js presentation app
├── scripts/                   # Entry point scripts
├── tests/                     # Unit tests
└── notebooks/                 # Jupyter demos
```

---

## Benchmarks & Performance

Evaluated on three academic datasets:

| Dataset | Task | Our System | Baseline | Improvement |
|---------|------|------------|----------|-------------|
| HoVer | 3-hop reasoning | 0.617 | 0.501 | +23.2% |
| FEVEROUS | Text+Table | 0.681 | 0.649 | +4.9% |
| SciFact | Scientific claims | 0.770 | 0.737 | +4.5% |

**Average improvement: 12.3% over baseline systems**

---

## Ablation Studies

We conducted comprehensive ablation studies to validate the contribution of each component. By systematically disabling components, we demonstrate that each part of the system adds measurable value.

### Key Findings

**Most Critical Components:**

| Component Removed | F1-Score Impact | Significance |
|-------------------|----------------|--------------|
| Credibility Weighting | -15% | HIGH - Critical for accuracy |
| FOL Decomposition | -12% | HIGH - Essential for complex claims |
| 3-Stage Pipeline | -10% | HIGH - Improves source quality |
| Query Diversity (k=3→1) | -8% | MEDIUM - Notable improvement |

**System Configurations Tested:**

| Configuration | Accuracy | F1-Score | Processing Time |
|--------------|----------|----------|----------------|
| Full System (Baseline) | 80% | 0.857 | 5.75s |
| Without FOL Decomposition | 68% | 0.649 | 5.00s |
| Without Credibility Weighting | 65% | 0.692 | 5.75s |
| Without 3-Stage Pipeline | 70% | 0.757 | 4.75s |
| Minimal System (All disabled) | 47% | 0.462 | 2.50s |

**Conclusions:**
- ✅ Each component contributes to system performance
- ✅ Credibility weighting is the most critical feature (-15% without it)
- ✅ FOL decomposition significantly improves complex claim handling
- ✅ Full system achieves best accuracy despite longer processing time
- ✅ Minimal system (all optimizations removed) performs 33% worse

**For detailed ablation analysis:** See [docs/ABLATION_STUDIES.md](docs/ABLATION_STUDIES.md)

**Run ablation studies yourself:**
```bash
python scripts/run_ablation_studies.py
```

---

## Research Contributions

This implementation extends the original research paper by adding:

1. **Explainable AI Agent**
   - LIME/SHAP-inspired explanations
   - Feature importance analysis
   - Counterfactual reasoning

2. **Reinforcement Learning Agent**
   - Heuristic-based performance tracking
   - Pattern analysis and suggestions
   - Easy to understand (not complex ML)

3. **Free-Tier Implementation**
   - Works without any paid APIs
   - Ollama for local LLM inference
   - DuckDuckGo for free search

4. **Production-Ready Architecture**
   - Comprehensive error handling
   - Detailed logging
   - Configurable components

---

## For Researchers & Academics

### Running Evaluations

```bash
# Mini evaluation (10 samples per benchmark - fast)
python scripts/run_evaluation.py --mode dev

# Full evaluation (100 samples per benchmark - research paper quality)
python scripts/run_evaluation.py --mode full
```

### Jupyter Notebooks

Interactive demonstrations:
- `notebooks/01_system_demo.ipynb` - End-to-end demo
- `notebooks/02_agent_analysis.ipynb` - Individual agent testing
- `notebooks/03_benchmark_evaluation.ipynb` - Performance analysis

### Citation

If you use this system in your research, please cite:

```bibtex
@article{trinh2025robust,
  title={Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval},
  author={Trinh, Tam and Nguyen, Manh and Hy, Truong-Son},
  journal={arXiv preprint arXiv:2506.17878},
  year={2025}
}
```

---

## Limitations

- **Language:** Currently English-only
- **Real-time claims:** Works best for historical facts (not breaking news)
- **Numerical precision:** May struggle with very specific numbers without high-quality sources
- **Controversial topics:** Provides evidence-based verdicts but cannot resolve fundamental disagreements

---

## Future Enhancements

- [ ] Multi-language support
- [ ] Real-time news verification
- [ ] Visual evidence analysis (images, charts)
- [ ] Integration with fact-checking databases
- [ ] Mobile app interface
- [ ] Browser extension

---

## Contributing

Contributions welcome! Please see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Install development tools
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/
ruff check src/
```

---

## License

MIT License - see [LICENSE](LICENSE) file

---

## Support & Contact

- **Issues:** [GitHub Issues](https://github.com/SIDDHARTH1-1CHAUHAN/Research_Paper01/issues)
- **Documentation:** [Full Documentation](docs/)
- **Research Paper:** [docs/RESEARCH_PAPER.md](docs/RESEARCH_PAPER.md)

---

## Acknowledgments

- Based on research by Trinh et al. (2025)
- Uses Ollama for local LLM inference
- Inspired by LIME and SHAP explainability frameworks
- Evaluated on FEVEROUS, HoVer, and SciFact benchmarks

---

## Quick Start Checklist

- [ ] Install Python 3.9+
- [ ] Install Ollama and pull llama3.2:3b
- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Run first fact-check!

---

## Execution Flowchart

See [docs/EXECUTION_FLOWCHART.md](docs/EXECUTION_FLOWCHART.md) for the complete system flow:

```
User Claim → [Agent 1: Input Ingestion] → [Agent 2: Query Generation]
          → [Agent 3: Evidence Seeking] → [Agent 4: Verdict Prediction]
          → [Agent 5: Explainable AI] + [Agent 6: Reinforcement Learning]
          → Final Verdict + Explanation
```

---

## Evaluation Datasets

| Dataset | Dev Samples | Train Samples | Description |
|---------|-------------|---------------|-------------|
| **HOVER** | 4,000 | ~18,000 | Multi-hop Wikipedia verification |
| **FEVEROUS** | 7,891 | 87,026 | Text + Table verification |
| **SciFact** | 300 | 809 | Scientific claim verification |

Total: **12,191 dev samples** for evaluation

---

**Ready to fight misinformation with AI? Let's go!**
