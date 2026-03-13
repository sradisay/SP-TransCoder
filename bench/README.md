# SP-TransCoder Benchmarking & Evaluation

This directory contains the evaluation suite for the SP-TransCoder research project. These scripts are designed to assess the performance of the translation models between Python and C++ using both qualitative visual checks and quantitative metrics (CodeBLEU and execution-based testing).

## Files Overview

### 1. `evaluate.py`
**Purpose:** Calculates the CodeBLEU score for an SP-TransCoder model on a paired dataset. 
**How it works:** It loads tokenized parallel code data, translates the source language to the target language in batches, caches the predictions to a JSON file (to save time on repeated runs), and computes the CodeBLEU metric (evaluating n-gram, weighted n-gram, syntax, and dataflow matches).

**Basic Usage:**
```bash
python evaluate.py --m <path_to_model_or_hf_id> --test_dir <path_to_test_data_dir> --source Python --target C++
```
*Note: The `--test_dir` must contain the `test-C++-Python-tok.py` and `test-C++-Python-tok.cpp` files.*

---

### 2. `naive_copy.py`

**Purpose:** Establishes a "do-nothing" baseline CodeBLEU score.
**How it works:** Instead of translating the code, this script simply copies the source language input and evaluates it directly against the target language reference. This is highly useful for understanding the absolute baseline overlap between the two languages in your dataset before any machine learning is applied.

**Basic Usage:**

```bash
python naive_copy.py --test_dir <path_to_test_data_dir> --source Python --target C++
```

---

### 3. `unit_testing.py`

**Purpose:** Generates translations for execution-based unit testing using the `THUDM/humaneval-x` dataset.
**How it works:** It fetches the HumanEval-X dataset from Hugging Face, formats the source code alongside the target signature, and generates the translated functions. The output is saved as a `.jsonl` file, which is specifically formatted to be passed into the CodeGeeX evaluation harness to test if the translated code actually compiles and passes unit tests.

**Basic Usage:**

```bash
python unit_testing.py --m <path_to_model_or_hf_id> --out samples.jsonl --source Python --target C++
```

---

### 4. `visual_inspection.py`

**Purpose:** Performs a quick, qualitative sanity check on a model checkpoint.
**How it works:** It runs a series of hardcoded, fundamental code snippets (e.g., simple addition functions, list/vector declarations, and basic `main()` blocks) in both Python to C++ and C++ to Python directions. It prints the input and the model's generated output directly to the terminal for immediate visual feedback.

**Basic Usage:**

```bash
python visual_inspection.py --m <path_to_model_or_hf_id>
```
