# Complete Setup and Running Guide

## Step-by-Step Instructions for Running the Multi-Agent Fact-Checking System

This guide will take you from zero to a fully working fact-checking system.

---

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Windows 10/11 (or Mac/Linux)
- [ ] Python 3.9+ installed
- [ ] Internet connection
- [ ] 4GB+ free disk space
- [ ] Terminal/Command Prompt access

---

## STEP 1: Verify Python Installation

Open Command Prompt (Windows) or Terminal (Mac/Linux):

```bash
python --version
```

**Expected output:** `Python 3.9.x` or higher (3.10, 3.11, 3.12 are all fine)

**If Python is not installed:**
1. Go to https://python.org/downloads
2. Download the latest version (3.12 recommended)
3. During installation, CHECK "Add Python to PATH"
4. Restart your terminal

---

## STEP 2: Navigate to Project Directory

```bash
cd "C:\Users\Dell\Desktop\2026\Research_Paper_01\multi-agent-fact-checker"
```

**Verify you're in the right place:**
```bash
dir
```

You should see: `demo.py`, `requirements.txt`, `src/`, `config/`, etc.

---

## STEP 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv
```

**Activate the virtual environment:**

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**You should see `(venv)` at the start of your command line.**

---

## STEP 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**This will install ~50 packages. Expected time: 2-5 minutes**

**If you encounter errors:**

1. **SSL Error:**
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

2. **Build Error (lxml/numpy):**
```bash
pip install wheel
pip install -r requirements.txt
```

3. **Memory Error:**
```bash
pip install --no-cache-dir -r requirements.txt
```

---

## STEP 5: (Optional but Recommended) Install Ollama

Ollama provides free local LLM inference. The system works without it (fallback mode), but is better with it.

### Windows:
1. Download from https://ollama.ai/download
2. Run the installer
3. Open new terminal and run:
```bash
ollama pull llama3.2:3b
```

### Verify Ollama is running:
```bash
ollama list
```

**If Ollama is not running, start it:**
```bash
ollama serve
```

---

## STEP 6: Run the Demo

```bash
python demo.py
```

### Expected Output:

```
================================================================================
MULTI-AGENT FACT-CHECKING SYSTEM - DEMO
================================================================================
Start Time: 2026-01-30 XX:XX:XX

Loaded 10 claims from mock dataset
Initializing Multi-Agent System...
Input Ingestion Agent initialized
Query Generation Agent initialized (k=3, lang=en)
Evidence Seeking Agent initialized (lang=en, region=wt-wt)
Verdict Prediction Agent initialized (lang=en)
Explainable AI Agent initialized
Reinforcement Learning Agent initialized (heuristic-based)
✓ All agents initialized successfully

================================================================================
CLAIM 1/10
================================================================================
Text: The Eiffel Tower was completed in 1889 and is located in Paris, France
Ground Truth: SUPPORTED

[1/6] Input Ingestion Agent - Decomposing claim...
✓ Found 2 verifiable subclaims

[2/6] Query Generation Agent - Creating search queries...
✓ Generated 6 search queries

[3/6] Evidence Seeking Agent - Retrieving evidence...
✓ Retrieved 6 evidence items

[4/6] Verdict Prediction Agent - Aggregating evidence...
✓ Verdict: SUPPORTED (confidence: 0.85)

[5/6] Explainable AI Agent - Generating explanations...
✓ Explanation quality: 0.85

[6/6] Reinforcement Learning Agent - Recording performance...
✓ Run recorded (accuracy: 1.00)

VERDICT: SUPPORTED
   Confidence: 85.00%
   Correct: ✓

... (continues for all 10 claims)

================================================================================
COMPREHENSIVE EVALUATION METRICS
================================================================================

CLASSIFICATION METRICS
----------------------------------------
Accuracy:   0.8000 (80.00%)
Precision:  1.0000
Recall:     0.7143
F1-Score:   0.8333

PERFORMANCE METRICS
----------------------------------------
Mean Queries per Claim:      6.0
Mean Evidence per Claim:     6.0

EXPLANATION QUALITY METRICS
----------------------------------------
Overall Quality:  0.8500

================================================================================
DEMO COMPLETE
================================================================================

✓ Detailed observations saved to: DEMO_OBSERVATIONS.md
```

---

## STEP 7: Check Results

After running, you'll have:

1. **DEMO_OBSERVATIONS.md** - Detailed results file
2. **demo_log.txt** - Execution log

Open DEMO_OBSERVATIONS.md to see:
- Executive summary
- Claim-by-claim results
- Technical metrics
- RL analysis and suggestions

---

## Troubleshooting Common Issues

### Issue: "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Config file not found"
The system works with defaults. This warning is safe to ignore.

### Issue: "LLM initialization failed"
The system will use fallback mode (heuristic-based). This is normal without Ollama.

### Issue: "Permission denied" on Windows
Run Command Prompt as Administrator.

### Issue: Demo is slow
- With Ollama: ~30-60 seconds for 10 claims
- Without Ollama (fallback): ~10-20 seconds for 10 claims

---

## Advanced Usage

### Run with Fewer Claims (Quick Test)

Edit `demo.py` line 276:
```python
# Change from:
run_demo(num_claims=10)

# To:
run_demo(num_claims=3)
```

### Verify a Single Claim

```python
from src.orchestrator import verify_claim

result = verify_claim(
    "The Earth is flat",
    ground_truth="NOT_SUPPORTED"
)

print(f"Verdict: {result['verdict']['final_verdict']}")
print(f"Confidence: {result['verdict']['overall_confidence']}")
```

### Batch Processing

```python
from src.orchestrator import FactCheckingOrchestrator

orchestrator = FactCheckingOrchestrator()

claims = [
    ("Python was created in 1991", "SUPPORTED"),
    ("The moon is made of cheese", "NOT_SUPPORTED"),
    ("Water boils at 100C at sea level", "SUPPORTED"),
]

results = orchestrator.batch_verify(claims)
```

---

## Directory Structure After Setup

```
multi-agent-fact-checker/
├── venv/                    # Virtual environment (created in Step 3)
├── demo.py                  # Main demo script
├── demo_log.txt            # Log file (created after running)
├── DEMO_OBSERVATIONS.md    # Results file (created after running)
├── requirements.txt        # Dependencies
├── README.md              # Project overview
├── src/
│   ├── orchestrator.py    # Main coordinator
│   ├── agents/            # 6 AI agents
│   ├── utils/             # Utilities
│   └── evaluation/        # Metrics
├── config/
│   ├── agent_config.yaml  # Agent settings
│   └── languages/         # Multilingual support
├── data/
│   └── benchmarks/        # Test datasets
└── docs/                  # Documentation
```

---

## Quick Commands Reference

```bash
# Activate environment (Windows)
venv\Scripts\activate

# Activate environment (Mac/Linux)
source venv/bin/activate

# Run demo
python demo.py

# Deactivate environment
deactivate

# Check installed packages
pip list

# Update a package
pip install --upgrade package_name
```

---

## What the Demo Demonstrates

1. **6 Specialized AI Agents** working in coordination
2. **FOL-based claim decomposition** for complex statements
3. **3-stage evidence retrieval** (search, credibility, extraction)
4. **Weighted verdict aggregation** based on source quality
5. **LIME/SHAP-inspired explanations** for transparency
6. **Performance tracking** with improvement suggestions

---

## Next Steps

After successfully running the demo:

1. **Read DEMO_OBSERVATIONS.md** - Understand the results
2. **Read docs/RESEARCH_PAPER.md** - Academic methodology
3. **Read docs/ARCHITECTURE.md** - Technical deep-dive
4. **Read NEURIPS_PREPARATION_GUIDE.md** - Steps for publication

---

## Getting Help

If you encounter issues:

1. Check `demo_log.txt` for detailed error messages
2. Read the troubleshooting section above
3. Open an issue on GitHub
4. Check existing documentation in `docs/`

---

**Congratulations! You now have a working multi-agent fact-checking system!**
