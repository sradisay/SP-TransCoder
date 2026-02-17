import random
from datasets import load_dataset

class UnpairedCodeDataset:
    def __init__(self, languages=("Python", "C++")):
        self.langs = languages
        self.data = {}
        
        for lang in languages:
            try:
                print(f"Loading {lang} from The Stack...")
                ds = load_dataset(
                    "bigcode/the-stack-smol", 
                    data_dir=f"data/{lang.lower()}", 
                    split="train", 
                    trust_remote_code=True
                )
                self.data[lang] = list(ds["content"])
                print(f"Successfully loaded {len(self.data[lang])} examples for {lang}")
            except Exception as e:
                print(f"Failed to load {lang}: {e}")
                self.data[lang] = []

    def sample_batch(self, lang: str, batch_size: int):
        lang_data = self.data.get(lang, [])
        if not lang_data or len(lang_data) == 0:
            return [""] * batch_size
        
        # Avoid sampling more than available
        size = min(batch_size, len(lang_data))
        return random.sample(lang_data, size)
