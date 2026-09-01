# 🚀 `buffdata`

**AI Training Data Optimizer powered by Google Gemini API**

`buffdata` is a high-throughput, modular CLI and Python framework designed to optimize, score, refine, evolve, and deduplicate AI datasets for Supervised Fine-Tuning (SFT), Pre-training, and Preference Tuning (DPO / RLHF) using state-of-the-art Google Gemini models (`gemini-3.7-flash`, `gemini-3.5-flash-lite`, and `text-embedding-004`).

---

## 🌟 Key Features

- 🎯 **LLM-as-a-Judge Quality Scoring**: Multi-dimensional evaluation (Clarity, Factual Accuracy, Reasoning Depth, Instruction Adherence, Safety) with structured Pydantic schemas.
- 🧹 **Deep Data Refinement & Cleansing**: Eliminates AI conversational boilerplate (*"As an AI..."*, *"Certainly!"*), expands chain-of-thought reasoning, and fixes syntax/markdown errors.
- 🧬 **Evol-Instruct Data Synthesis**: Complexifies datasets through reasoning deepening, constraint addition, concretization, and domain expansion.
- ⚖️ **Automated DPO Preference Builder**: Automatically synthesizes high-contrast `chosen` vs. `rejected` pairs with explicit defect explanations.
- 🔍 **Semantic & Lexical Deduplication**: Exact SHA-256 hash, MinHash n-gram LSH, and Gemini dense embedding similarity deduplication.
- ⚡ **Async High-Throughput Engine**: Token-bucket rate limiting (RPM/TPM), concurrency control, and auto-resuming checkpoints.
- 📊 **Interactive HTML & Markdown Reports**: Rich visual analytics with score distribution charts, identified issue breakdown, and Before vs. After diff viewers.

---

## 📦 Installation

```bash
# Clone the repository and navigate to directory
cd ~/projects/buffdata

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e ".[dev]"
```

---

## 🔑 Setup API Key

Set your Google Gemini API key:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

Or create a `.env` file in the project directory:

```env
GEMINI_API_KEY=your-gemini-api-key
BUFFDATA_DEFAULT_MODEL=gemini-3.7-flash
```

---

## 💻 CLI Quickstart

### 1. Score & Filter Dataset
```bash
buffdata score examples/alpaca_sample.jsonl -o filtered.jsonl --min-score 7.5 --filter
```

### 2. Refine & Elevate Quality
```bash
buffdata refine examples/alpaca_sample.jsonl -o refined.jsonl --mode all
```

### 3. Evolve Reasoning Complexity (Evol-Instruct)
```bash
buffdata evolve examples/alpaca_sample.jsonl -o evolved.jsonl --strategy deepen_reasoning
```

### 4. Build DPO / RLHF Preference Pairs
```bash
buffdata dpo examples/alpaca_sample.jsonl -o dpo_dataset.jsonl
```

### 5. Deduplicate
```bash
buffdata dedup examples/alpaca_sample.jsonl -o deduped.jsonl --method minhash --threshold 0.85
```

### 6. Run Full Multi-Stage Pipeline
```bash
buffdata pipeline examples/pipeline_config.yaml -i examples/alpaca_sample.jsonl -o optimized.jsonl
```

### 7. View Stats & Generate HTML Audit Report
```bash
# Print summary to terminal
buffdata stats examples/alpaca_sample.jsonl

# Generate interactive HTML report
buffdata report examples/alpaca_sample.jsonl -o audit_report.html
```

---

## 🐍 Python SDK Example

```python
from buffdata import GeminiClient, QualityScorer, DataRefiner, read_dataset, write_dataset

# Load dataset
items = read_dataset("examples/alpaca_sample.jsonl")

# Initialize client and optimizer
client = GeminiClient(default_model="gemini-3.7-flash")
refiner = DataRefiner(client=client)

# Refine batch asynchronously
import asyncio
refined_items = asyncio.run(refiner.refine_batch_async(items, mode="all"))

# Save refined dataset
write_dataset(refined_items, "refined_output.jsonl")
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📄 License
MIT License.
