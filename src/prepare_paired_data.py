import json
import os
from datasets import load_dataset


def extract_and_save_codenet(output_filepath="data/codenet_paired.json", limit_pairs=50000):
    print(f"Starting CodeNet extraction. Target: {limit_pairs} pairs.")
    pairs = []

    try:
        print("Loading CodeNet subsets from Hugging Face... this may take a moment.")
        py_ds = load_dataset("iNeil77/CodeNet", "Python", split="train", streaming=True, trust_remote_code=True)
        cpp_ds = load_dataset("iNeil77/CodeNet", "C++", split="train", streaming=True, trust_remote_code=True)

        py_accepted = {}
        print("Gathering Accepted Python solutions...")
        for row in py_ds:
            status = row.get("status", row.get("state", ""))
            if "Accepted" in status:
                pid = row["p_id"]
                if pid not in py_accepted:
                    py_accepted[pid] = []

                code = row.get("code", row.get("source_code", ""))
                py_accepted[pid].append(code)

                if len(py_accepted) > limit_pairs * 1.5:
                    break

        print("Matching with Accepted C++ solutions...")
        for row in cpp_ds:
            status = row.get("status", row.get("state", ""))
            if "Accepted" in status:
                pid = row["p_id"]

                if pid in py_accepted and len(py_accepted[pid]) > 0:
                    py_code = py_accepted[pid].pop()
                    cpp_code = row.get("code", row.get("source_code", ""))

                    if len(cpp_code) < 400 and len(py_code) < 400 and cpp_code.count("#include") < 4:
                        pairs.append({
                            "python": py_code.strip(),
                            "c++": cpp_code.strip()
                        })

                    if not py_accepted[pid]:
                        del py_accepted[pid]

            if len(pairs) >= limit_pairs:
                break

        print(f"Successfully aligned {len(pairs)} Python-C++ pairs!")

    except Exception as e:
        print(f"Failed to load or align CodeNet: {e}")
        return

    if not pairs:
        print("Error: No pairs were generated. Check your connection or dataset logic.")
        return

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    print(f"Saving {len(pairs)} pairs to {output_filepath}...")

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, indent=4)



if __name__ == "__main__":
    extract_and_save_codenet("../data/codenet_paired_50k.json", limit_pairs=50000)