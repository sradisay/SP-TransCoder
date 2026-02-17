from datasets import load_dataset
import random
import json

class UnpairedCodeDataset:
    def __init__(self, languages=["python", "c++"]):
        self.langs = languages
        self.data = {}
        for lang in languages:
            # Map standard language names to The Stack's data directories
            # The Stack uses "python" and "c++" (lowercase), so your config matches.
            try:
                # We use data_dir to fetch specific languages efficiently
                ds = load_dataset(
                    "bigcode/the-stack-smol",
                    data_dir=f"data/{lang}", 
                    split="train", 
                    trust_remote_code=True
                )
                # The Stack uses 'content' column for code, not 'text'
                self.data[lang] = [row["content"] for row in ds]
                print(f"Loaded {len(self.data[lang])} examples for {lang}")
            except Exception as e:
                print(f"Failed to load {lang}: {e}")
                self.data[lang] = []

    def sample_batch(self, lang, batch_size):
        lang_data = self.data.get(lang, [])
        if not lang_data:
            return [""] * batch_size # Handle missing data gracefully
        return random.sample(lang_data, min(batch_size, len(lang_data)))


class PairedCodeDataset:
    def __init__(self, json_filepath="data/codenet_paired_50k.json"):
        self.pairs = []

        try:
            print(f"Loading paired data from {json_filepath}...")
            with open(json_filepath, 'r', encoding='utf-8') as f:
                self.pairs = json.load(f)
            print(f"Successfully loaded {len(self.pairs)} pairs!")

        except FileNotFoundError:
            print(f"Error: {json_filepath} not found.")
            print("Please run 'python prepare_paired_data.py' first to generate the dataset.")
            self.pairs = [
                {
                    "python": "def add(a, b):\n    return a + b",
                    "c++": "int add(int a, int b) {\n    return a + b;\n}"
                }
            ]

    def sample_paired_batch(self, batch_size):
        if not self.pairs:
            return [""] * batch_size, [""] * batch_size

        batch = random.sample(self.pairs, min(batch_size, len(self.pairs)))

        py_batch = [item["python"] for item in batch]
        cpp_batch = [item["c++"] for item in batch]

        return py_batch, cpp_batch
