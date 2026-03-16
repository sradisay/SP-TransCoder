import argparse
import torch
import os
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
import textwrap
CONFIG = {
    "max_len": 512,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "batch_size": 16
}


def load_humanevalx_translation_data(source_lang, target_lang):
    lang_map = {"Python": "python", "C++": "cpp"}
    src_hf_lang = lang_map.get(source_lang, source_lang.lower())
    tgt_hf_lang = lang_map.get(target_lang, target_lang.lower())

    ds_src = load_dataset("THUDM/humaneval-x", src_hf_lang, split="test", trust_remote_code=True)
    ds_tgt = load_dataset("THUDM/humaneval-x", tgt_hf_lang, split="test", trust_remote_code=True)

    src_dict = {item['task_id'].split('/')[-1]: item for item in ds_src}

    tasks = []
    for tgt_item in ds_tgt:
        num_id = tgt_item['task_id'].split('/')[-1]

        if num_id in src_dict:
            src_item = src_dict[num_id]
            src_full_code = src_item['declaration'] + src_item['canonical_solution']

            tasks.append({
                "task_id": tgt_item['task_id'],
                "source_code": src_full_code.strip(),
                "target_declaration": tgt_item['declaration'],
                "prompt": tgt_item['prompt']
            })
        else:
            print(f"Warning: Task ID {num_id} missing from source dataset.")

    return tasks


def main(model_name, tokenizer_name, output_file, source_lang, target_lang, limit):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(CONFIG["device"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    tasks = load_humanevalx_translation_data(source_lang, target_lang)

    # Apply the dry run limit if one was provided
    if limit > 0:
        tasks = tasks[:limit]
        print(f"\n--- DRY RUN MODE: Limiting to {limit} tasks ---\n")

    batch_size = CONFIG["batch_size"]

    print(f"Generating translations for {len(tasks)} tasks...")
    model.eval()

    with open(output_file, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for i in tqdm(range(0, len(tasks), batch_size), desc="Translating"):
                batch_tasks = tasks[i:i + batch_size]
                input_texts = [
                    f"Translate {source_lang} to {target_lang}: {task['source_code']}"
                    for task in batch_tasks
                ]

                inputs = tokenizer(
                    input_texts, return_tensors="pt", padding=True, truncation=True,
                    max_length=CONFIG["max_len"]
                ).to(CONFIG["device"])

                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=CONFIG["max_len"],
                    num_beams=4,
                    early_stopping=True,
                    repetition_penalty = 1.2
                )

                pred_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for task, generation in zip(batch_tasks, pred_texts):
                    final_generation = generation.strip()
                    target_lang_lower = target_lang.lower()

                    if target_lang_lower in ["c++", "cpp"]:
                        idx = final_generation.find('{')
                        if idx != -1:
                            final_generation = final_generation[idx + 1:].strip()
                    elif target_lang_lower == "python":
                        def_idx = final_generation.find('def ')
                        if def_idx != -1:
                            doc_end = final_generation.find('"""', def_idx + 4)
                            if doc_end != -1:
                                second_doc_end = final_generation.find('"""', doc_end + 3)
                                if second_doc_end != -1:
                                    final_generation = final_generation[second_doc_end + 3:].strip('\n')
                            else:
                                colon_idx = final_generation.find(':', def_idx)
                                if colon_idx != -1:
                                    final_generation = final_generation[colon_idx + 1:].strip('\n')

                        dedented_code = textwrap.dedent(final_generation)
                        lines = dedented_code.split('\n')
                        final_generation = '\n'.join(['    ' + line for line in lines if line.strip()])

                    output_obj = {
                        "task_id": task["task_id"],
                        "prompt": task["prompt"],
                        "generation": final_generation
                    }
                    f.write(json.dumps(output_obj) + "\n")

    print(f"\nGenerations saved to {output_file}. You can now pass this to the CodeGeeX evaluation harness.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate HumanEval-X translations for execution.")
    parser.add_argument("--m", type=str, required=True, help="path or hf ID for the model")
    parser.add_argument("--t", type=str, required=False, default="", help="tokenizer")
    parser.add_argument("--out", type=str, default="samples.jsonl", help="Output JSONL file")
    parser.add_argument("--source", type=str, default="Python")
    parser.add_argument("--target", type=str, default="C++")
    parser.add_argument("--limit", type=int, default=0, help="Limit the number of tasks for testing (0 = full dataset)")

    args = parser.parse_args()
    final_tokenizer_name = args.t if args.t else args.m
    main(args.m, final_tokenizer_name, args.out, args.source, args.target, args.limit)