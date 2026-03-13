# SP-TransCoder Training Models

This directory contains the core training algorithms (Trainers) used to develop and fine-tune the SP-TransCoder model. These files do not contain standalone executable scripts; rather, they define the training logic, loss calculations, and forward/backward passes for different learning paradigms. 

These trainers are designed to be imported and executed by the four main training scripts located at the root of the project.

## Trainers Overview

### 1. `supervised_trainer.py`
**Algorithm:** Standard Supervised Fine-Tuning (SFT)

**Purpose:** Trains the model using explicitly paired data (e.g., a Python snippet and its exact C++ equivalent from a dataset like CodeNet or XLCost).

**Mechanism:** * Takes a batch of source code and a batch of corresponding target code.
* Prepares prompt-based inputs (e.g., `Translate Python to C++: [code]`).
* Calculates standard Cross-Entropy loss against the provided target code.
* Processes both `Python -> C++` and `C++ -> Python` directions simultaneously within a single batch to maintain bidirectional capabilities.

### 2. `backtranslation_trainer.py`
**Algorithm:** Unsupervised Iterative Back-Translation

**Purpose:** Leverages large amounts of *unpaired* monolingual code to improve translation quality, addressing the scarcity of high-quality parallel data.

**Mechanism:** 
* **Step 1 (Generation):** Takes real code in a source language (e.g., Python) and uses the model in `eval` mode to generate a synthetic, "noisy" translation in the target language (e.g., C++).
* **Step 2 (Training):** Switches the model to `train` mode. It then trains the model to translate the *synthetic* target code back into the *original*, perfect source code. 
* By learning to reconstruct clean source code from noisy, machine-generated translations, the model inherently improves its understanding of both languages and their structural mappings.

### 3. `denoising_trainer.py`
**Algorithm:** Denoising Auto-Encoding (DAE)

**Purpose:** Helps the model learn the fundamental syntax, vocabulary, and grammar of a language before or during cross-lingual training.

**Mechanism:** * Takes a sequence of valid code and artificially corrupts it. The `_corrupt_sequence` function applies three types of noise:

* **Token Dropping:** Randomly deletes tokens (`drop_prob`).
* **Token Masking:** Replaces tokens with a `<extra_id_0>` mask (`mask_prob`).
* **Token Shuffling:** Randomly rearranges the order of tokens within a defined distance (`shuffle_dist`).
* The model is then trained to take this corrupted string and reconstruct the original, uncorrupted code. 

## Technical Details (Common Across Trainers)
All trainers in this directory utilize the following optimization techniques to manage memory and speed up training:
* **Gradient Checkpointing:** Enabled on the T5 model to save memory at the cost of slight compute overhead during the backward pass.
* **Automatic Mixed Precision (AMP):** Utilizes `torch.amp.autocast('cuda')` and `GradScaler` for faster operations and reduced VRAM footprint.
* **Gradient Accumulation:** Accumulates gradients over micro-batches (default 64) to achieve the target effective batch size without running out of GPU memory.