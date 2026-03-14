# SP-TransCoder

SP-TransCoder is a machine learning project dedicated to intelligent code translation between Python and C++. Leveraging the `Salesforce/codet5-small` architecture as a foundation, this repository explores several training paradigms—ranging from standard supervised fine-tuning to advanced semi-supervised cycles incorporating back-translation and denoising—to improve cross-lingual code generation.

## Repository Structure

The project is organized into core training scripts at the root, supported by dedicated modules for model algorithms and benchmarking:

* **`/models/`**: Contains the core algorithmic trainers (Supervised, Back-Translation, and Denoising Auto-Encoder). See the [`models/README.md`](src/models/README.md) for a deep dive into the training mechanics.
* **`/bench/`**: Contains the evaluation suite, including CodeBLEU scoring, baseline generation, unit testing generation, and visual inspection tools. See the [`bench/README.md`](src/bench/README.md) for evaluation instructions.
* **Root Directory**: Contains the datasets, data preparation scripts, the main execution scripts for various training loops, and batch evaluation scripts.

## Data Management

The project handles multiple types of datasets to facilitate supervised and unsupervised learning phases:

* **`dataset.py`**: The central data loader.
    * `UnpairedCodeDataset`: Loads monolingual Python and C++ data from `bigcode/the-stack-smol` for unsupervised training phases.
    * `PairedCodeDataset`: Loads pre-aligned parallel data from a local JSON file.
    * `XLCoSTSnippetDataset`: Loads, decodes, and cleans tokenized C++ and Python snippet pairs from the XLCoST dataset.
* **`prepare_paired_data.py`**: A utility script that mines the `iNeil77/CodeNet` dataset on Hugging Face. It extracts "Accepted" Python solutions and aligns them with matching "Accepted" C++ solutions for the exact same problem ID to construct a high-quality parallel dataset.

## Training Pipelines

The root directory features four distinct training scripts that combine the trainers from the `models/` directory into full execution loops:

* **`train_supervised.py`**: Executes a standard Supervised Fine-Tuning (SFT) loop using the paired XLCoST snippet dataset.
* **`train_backtranslation.py`**: Runs an unsupervised iterative back-translation loop using the unpaired Stack dataset.
* **`train_semi_supervised.py`**: Implements a cyclical semi-supervised approach. Each epoch alternates between a supervised phase (using paired XLCoST data) and a back-translation phase (using unpaired Stack data).
* **`train_semi_supervised_with_denoising.py`**: The most comprehensive pipeline. It runs a three-phase cycle every epoch: Denoising Auto-Encoding (DAE) to reinforce core language syntax, followed by Supervised training, and concluding with Back-translation. 

*Note: All training scripts are configured to use a batch size of 128, a maximum sequence length of 256, and save their epoch checkpoints to the `./checkpoints/` directory.*

## Evaluation

* **`eval.sh`**: A bash script that automates the evaluation of multiple model checkpoints. It sequentially runs both Python-to-C++ and C++-to-Python CodeBLEU evaluations (via `bench/evaluate.py`) for the various trained models and logs the results in the `./eval_results/` directory.