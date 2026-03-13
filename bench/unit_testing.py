import argparse
import torch
import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset


CONFIG = {
    "max_len": 512,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 16
}


def load_humanevalx_translation_data(source_lang, target_lang):
    lang_map = {"Python": "python", "C++": "cpp"}
    src_hf_lang = lang_map.get(source_lang, source_lang.lower())
    tgt_hf_lang = lang_map.get(target_lang, target_lang.lower())

    ds_src = load_dataset("THUDM/humaneval-x", src_hf_lang, split = "test")
    ds_tgt = load_dataset("THUDM/humaneval-x", tgt_hf_lang, split = "test")

    tasks = []
    for src_item, tgt_item in zip(ds_src, ds_tgt):
        assert src_item['task_id'] == tgt_item['task_id']

        src_full_code = src_item['prompt'] + src_item['canonical_solution']
        tgt_prompt = tgt_item['prompt']

        tasks.append({
            "task_id": tgt_item['task_id'],
            "source_code": src_full_code,
            "target_prompt": tgt_prompt
        })
    return tasks


def main(model_name, tokenizer_name, output_file, source_lang, target_lang):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(CONFIG["device"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tasks = load_humanevalx_translation_data(source_lang, target_lang)
    batch_size = CONFIG["batch_size"]

    print(f"Generating translations for {len(tasks)} tasks...")
    model.eval()

    with open(output_file, "w", encoding = "utf-8") as f:
        with torch.no_grad():
            for i in tqdm(range(0, len(tasks), batch_size), desc = "Translating"):
                batch_tasks = tasks[i:i + batch_size]
                input_texts = [
                    f"Translate {source_lang} to {target_lang}:\n{task['source_code']}\n\nTarget Signature:\n{task['target_prompt']}"
                    for task in batch_tasks
                ]

                inputs = tokenizer(
                    input_texts, return_tensors = "pt", padding = True, truncation = True,
                    max_length = CONFIG["max_len"]
                ).to(CONFIG["device"])

                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask = inputs.attention_mask,
                    max_length = CONFIG["max_len"],
                    num_beams = 4,
                    early_stopping = True
                )

                pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens = True)

                for task, generation in zip(batch_tasks, pred_texts):
                    output_obj = {
                        "task_id": task["task_id"],
                        "generation": generation
                    }
                    f.write(json.dumps(output_obj) + "\n")

    print(f"\nGenerations saved to {output_file}. You can now pass this to the CodeGeeX evaluation harness.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Generate HumanEval-X translations for execution.")
    parser.add_argument("--m", type = str, required = True, help = "path or hf ID for the model")
    parser.add_argument("--t", type = str, required = False, default = "", help = "tokenizer")
    parser.add_argument("--out", type = str, default = "samples.jsonl", help = "Output JSONL file")
    parser.add_argument("--source", type = str, default = "Python")
    parser.add_argument("--target", type = str, default = "C++")

    args = parser.parse_args()
    final_tokenizer_name = args.t if args.t else args.m
    main(args.m, final_tokenizer_name, args.out, args.source, args.target)
