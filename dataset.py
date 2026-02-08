from datasets import load_dataset
import random

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
