import argparse
import torch
import re
import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from codebleu import calc_codebleu
from tqdm import tqdm

CONFIG = {
    "test_dir": "../data/pair_data_tok_1/C++-Python/",
    "max_len": 256,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_samples": 500,
    "batch_size": 16
}


def _decode_and_clean(code_string):
    tokens = code_string.split()
    decoded_chars = []
    indent_level = 0
    is_new_line = True

    for token in tokens:
        if token == 'NEW_LINE':
            decoded_chars.append('\n')
            is_new_line = True
        elif token == 'INDENT':
            indent_level += 1
        elif token == 'DEDENT':
            indent_level = max(0, indent_level - 1)
        else:
            if is_new_line:
                decoded_chars.append('    ' * indent_level)
                is_new_line = False
            elif decoded_chars and decoded_chars[-1] != '\n':
                decoded_chars.append(' ')
            decoded_chars.append(token)

    raw_decoded_string = "".join(decoded_chars).strip()
    cleaned_string = re.sub(r'\s+([,.:;\]\)\}])', r'\1', raw_decoded_string)
    cleaned_string = re.sub(r'([\[\(\{])\s+', r'\1', cleaned_string)
    cleaned_string = re.sub(r'\+\s+\+', '++', cleaned_string)
    cleaned_string = re.sub(r'-\s+-', '--', cleaned_string)

    return cleaned_string.strip()


def load_test_data(limit=None):
    py_path = os.path.join(CONFIG["test_dir"], "test-C++-Python-tok.py")
    cpp_path = os.path.join(CONFIG["test_dir"], "test-C++-Python-tok.cpp")

    sources, references = [], []

    print(f"Loading test data from {CONFIG['test_dir']}...")
    with open(py_path, 'r', encoding='utf-8') as f_py, \
            open(cpp_path, 'r', encoding='utf-8') as f_cpp:

        for py_line, cpp_line in zip(f_py, f_cpp):
            py_code = _decode_and_clean(py_line)
            cpp_code = _decode_and_clean(cpp_line)

            if py_code and cpp_code:
                sources.append(py_code)
                references.append(cpp_code)

            if limit and len(sources) >= limit:
                break

    return sources, references


def main(model_name, tokenizer_name, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached predictions from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
            references = cache_data["references"]
            predictions = cache_data["predictions"]
    else:
        print(f"Loading SP-TransCoder from {model_name} on {CONFIG['device']}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(CONFIG["device"])

        print(f"Loading tokenizer from {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        sources, references = load_test_data(limit=CONFIG["num_samples"])
        predictions = []
        batch_size = CONFIG["batch_size"]

        print(f"Starting batched evaluation on {len(sources)} samples (Batch Size: {batch_size})...")

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(sources), batch_size), desc="Translating Batches"):
                batch_sources = sources[i:i + batch_size]
                input_texts = [f"Translate Python to C++: {src}" for src in batch_sources]

                inputs = tokenizer(
                    input_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=CONFIG["max_len"]
                ).to(CONFIG["device"])

                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=CONFIG["max_len"],
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty=1.0
                )

                pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                predictions.extend(pred_texts)

        print(f"\nSaving predictions to {cache_file}...")
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"references": references, "predictions": predictions}, f, indent=4)

    print("\nCalculating CodeBLEU Score...")
    result = calc_codebleu(
        [[ref] for ref in references],
        predictions,
        lang="cpp",
        weights=(0.25, 0.25, 0.25, 0.25),
        tokenizer=None
    )

    print("\n" + "=" * 40)
    print("SP-TransCoder Evaluation Results")
    print("=" * 40)
    print(f"Total CodeBLEU Score: {result['codebleu'] * 100:.2f}")
    print(f"   - N-Gram Match:    {result['ngram_match_score'] * 100:.2f}")
    print(f"   - Weighted N-Gram: {result['weighted_ngram_match_score'] * 100:.2f}")
    print(f"   - Syntax Match:    {result['syntax_match_score'] * 100:.2f}")
    print(f"   - Dataflow Match:  {result['dataflow_match_score'] * 100:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SP-TransCoder CodeBLEU score.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Path or HuggingFace ID for the model checkpoint."
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=False,
        default="",
        help="Path or HuggingFace ID for the tokenizer. Defaults to model_name if omitted."
    )
    parser.add_argument(
        "--c",
        type=str,
        required=False,
        default="predictions_cache.json",
        help="Path to the cache file. Loads from it if it exists, or caches predictions there."
    )

    args = parser.parse_args()

    final_tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name

    main(args.model_name, final_tokenizer_name, args.c)