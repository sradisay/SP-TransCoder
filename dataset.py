import random
from datasets import load_dataset

class UnpairedCodeDataset:
    def __init__(self, languages=("Python", "C++")):
        self.langs = languages
        self.data = {}
        
        for lang in languages:
            try:
                # The Stack requires lowercase directories (e.g., 'python', 'c++')
                ds = load_dataset(
                    "bigcode/the-stack-smol", 
                    data_dir=f"data/{lang.lower()}", 
                    split="train", 
                    trust_remote_code=True
                )
                self.data[lang] = ds["content"]
                print(f"Loaded {len(self.data[lang])} examples for {lang}")
            except Exception as e:
                print(f"Failed to load {lang}: {e}")
                self.data[lang] = []

    def sample_batch(self, lang: str, batch_size: int):
        lang_data = self.data.get(lang, [])
        if not lang_data:
            return [""] * batch_size
        return random.sample(lang_data, min(batch_size, len(lang_data)))
