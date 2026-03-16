# SP-TransCoder: Python to C++ Code Translation

**Course**: CS 175: Project in Artificial Intelligence, Winter 2026

**Author/Team**: Sterling Radisay, Elian Hijmans Malessy, Sia Aggarwal

This repository contains the software, models, and evaluation code for SP-TransCoder, a project exploring artificial intelligence techniques for translating code between Python and C++.

## Libraries Used
* `torch` (PyTorch) - Deep learning framework for model training and inference.
* `transformers` (Hugging Face) - Model architecture (`T5ForConditionalGeneration`) and tokenization.
* `datasets` (Hugging Face) - Downloading and streaming training/evaluation data.
* `codebleu` - Calculating syntactic and semantic similarity metrics.
* `tqdm` - Progress bars for training and evaluation loops.

## Publicly Available Codes Used
* **Salesforce/codet5-small** (https://huggingface.co/Salesforce/codet5-small): Used as the base pre-trained LLM architecture and tokenizer. Unmodified.
* **bigcode/the-stack-smol** (https://huggingface.co/datasets/bigcode/the-stack-smol): Unpaired monolingual code dataset used for Unsupervised Back-Translation. Unmodified.
* **iNeil77/CodeNet** (https://huggingface.co/datasets/iNeil77/CodeNet): Dataset used to mine paired Python/C++ code. Unmodified.
* **THUDM/humaneval-x** (https://huggingface.co/datasets/THUDM/humaneval-x): Dataset used for execution-based unit testing evaluation. Unmodified.

## Scripts/Functions Written by Our Team

All original source code for this project is located within the `src/` directory.

**Data Preparation & Loading**
* `src/dataset.py`: Central data loader containing classes for unpaired (`The Stack`), paired (`CodeNet`), and snippet (`XLCoST`) datasets (approx. 110 lines).
* `src/prepare_paired_data.py`: Utility script that mines the CodeNet dataset to align "Accepted" Python and C++ solutions matching the same problem ID (approx. 70 lines).

**Model Trainers**
* `src/models/supervised_trainer.py`: Algorithm for standard Supervised Fine-Tuning on paired datasets (approx. 80 lines).
* `src/models/backtranslation_trainer.py`: Algorithm for Unsupervised Iterative Back-Translation using noisy generation (approx. 90 lines).
* `src/models/denoising_trainer.py`: Algorithm for Denoising Auto-Encoding (DAE) via token dropping, masking, and shuffling (approx. 110 lines).

**Training Pipelines**
* `src/train_supervised.py`: Main execution loop for supervised training (approx. 40 lines).
* `src/train_backtranslation.py`: Main execution loop for back-translation (approx. 30 lines).
* `src/train_semi_supervised.py`: Main execution loop alternating supervised and back-translation phases (approx. 50 lines).
* `src/train_semi_supervised_with_denoising.py`: Main execution loop running a 3-phase cycle of DAE, supervised, and back-translation (approx. 60 lines).

**Benchmarking & Evaluation (`src/bench/`)**
* `src/bench/evaluate.py`: Calculates the CodeBLEU score for models using tokenized parallel data (approx. 160 lines).
* `src/bench/naive_copy.py`: Establishes a "do-nothing" baseline CodeBLEU score (approx. 60 lines).
* `src/bench/unit_testing.py`: Generates execution-based translations for the CodeGeeX evaluation harness (approx. 80 lines).
* `src/bench/visual_inspection.py`: Performs qualitative sanity checks on model checkpoints using basic code snippets (approx. 70 lines).
* `src/eval.sh`: Bash script to automate batched evaluation across multiple saved checkpoints (approx. 40 lines).