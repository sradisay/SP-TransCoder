import os
import json
import multiprocessing
import random
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader

class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = self._infinite_generator()

    def _infinite_generator(self):
        while True:
            for batch in self.dataloader:
                yield batch

    def next_batch(self):
        return next(self.iterator)

class PairedCodeDataset:
    def __init__(self, json_filepath="data/codenet_paired_50k.json", batch_size=16):
        print(f"\nLoading Paired CodeNet data from {json_filepath}...")
        try:
            with open(json_filepath, 'r', encoding='utf-8') as f:
                pairs = json.load(f)
            print(f"Successfully loaded {len(pairs)} exact Python-C++ pairs.")
        except FileNotFoundError:
            print(f"Warning: {json_filepath} not found. Creating a dummy dataset for testing.")
            pairs = [{"python": "def add(a, b):\n    return a + b", "c++": "int add(int a, int b) {\n    return a + b;\n}"}] * 100

        # Convert to HuggingFace dataset for easy dataloader integration
        ds = Dataset.from_list(pairs)
        
        num_workers = min(4, int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())) // 2)
        
        dl = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=max(0, num_workers),
            drop_last=True
        )
        self.dataloader = InfiniteDataLoader(dl)

    def sample_batch(self):
        """Returns a dict with 'python' and 'c++' keys containing paired code."""
        return self.dataloader.next_batch()

class UnpairedCodeDataset:
    def __init__(self, languages=("Python", "C++"), batch_size=16):
        self.langs = languages
        self.dataloaders = {}
        
        self.num_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        print(f"\nUnpaired Dataset using {self.num_proc} CPU workers for preprocessing.")
        
        processed_datasets = {}
        
        for lang in languages:
            try:
                print(f"Loading {lang} from The Stack...")
                ds = load_dataset(
                    "bigcode/the-stack-smol", 
                    data_dir=f"data/{lang.lower()}", 
                    split="train", 
                    trust_remote_code=True
                )
                
                def process_and_chunk(batch, current_lang=lang):
                    valid_chunks = []
                    for content in batch["content"]:
                        clean_content = self._strip_headers(content)
                        lines = clean_content.split('\n')
                        chunk_size_lines = 50 
                        
                        chunks = [
                            '\n'.join(lines[i:i + chunk_size_lines]) 
                            for i in range(0, len(lines), chunk_size_lines)
                        ]
                        
                        for chunk in chunks:
                            if self._is_valid_code(chunk, current_lang):
                                valid_chunks.append(chunk)
                                
                    return {"clean_content": valid_chunks}

                final_ds = ds.map(
                    process_and_chunk, 
                    batched=True, 
                    num_proc=self.num_proc,
                    desc=f"Processing {lang} files",
                    remove_columns=ds.column_names 
                ).with_format("python")
                
                print(f"Successfully processed {len(final_ds)} VALID chunks for {lang}")
                processed_datasets[lang] = final_ds
                
            except Exception as e:
                print(f"Failed to process {lang}: {e}")

        num_workers = min(4, max(0, self.num_proc // len(languages))) 
        
        for lang, final_ds in processed_datasets.items():
            try:
                dl = DataLoader(
                    final_ds, # type: ignore
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    prefetch_factor=4 if num_workers > 0 else None,                
                    persistent_workers=True if num_workers > 0 else False,          
                    pin_memory=False,                
                    drop_last=True     
                )
                
                self.dataloaders[lang] = InfiniteDataLoader(dl)
            except Exception as e:
                print(f"Failed to create DataLoader for {lang}: {e}")

    def sample_batch(self, lang: str, limit: int = None):
        if lang not in self.dataloaders:
            raise ValueError(f"Language {lang} not loaded.")
        
        strings = self.dataloaders[lang].next_batch()["clean_content"]
        
        if limit is not None:
            return strings[:limit]
            
        return strings

    def _strip_headers(self, text):
        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith(('#', '//', '/*', '*', '"""', "'''", 'import ', 'from ', '#include')):
                start_idx = i
                break
        return '\n'.join(lines[start_idx:])

    def _is_valid_code(self, text, lang):
        if not text or len(text) < 50: 
            return False
        if lang == "Python":
            return any(kw in text for kw in ["def ", "class ", "return ", "for ", "if "])
        elif lang == "C++":
            return any(kw in text for kw in ["int ", "void ", "class ", "return ", "for (", "if ("])
        return True
