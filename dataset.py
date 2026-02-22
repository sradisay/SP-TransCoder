import os
import multiprocessing
from datasets import load_dataset
from torch.utils.data import DataLoader

class InfiniteDataLoader:
    """
    Idiomatic PyTorch wrapper to yield batches infinitely using a generator.
    Avoids try/except overhead and cleanly integrates with persistent workers.
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = self._infinite_generator()

    def _infinite_generator(self):
        while True:
            for batch in self.dataloader:
                yield batch

    def next_batch(self):
        return next(self.iterator)

class UnpairedCodeDataset:
    def __init__(self, languages=("Python", "C++"), batch_size=16):
        self.langs = languages
        self.dataloaders = {}
        
        # SLURM automatically provisions this env var.
        self.num_proc = int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count()))
        print(f"Dataset using {self.num_proc} CPU workers for preprocessing.")
        
        for lang in languages:
            try:
                print(f"\nLoading {lang} from The Stack...")
                ds = load_dataset(
                    "bigcode/the-stack-smol", 
                    data_dir=f"data/{lang.lower()}", 
                    split="train", 
                    trust_remote_code=True
                )
                
                # 1. ONE-TO-MANY MULTIPROCESSING
                # We strip, chunk, and filter in a single pass.
                def process_and_chunk(batch):
                    valid_chunks = []
                    
                    for content in batch["content"]:
                        # Strip headers
                        clean_content = self._strip_headers(content)
                        
                        # Fix: LINE-BASED CHUNKING to preserve syntax boundaries
                        lines = clean_content.split('\n')
                        chunk_size_lines = 50 # roughly 1000-1500 characters depending on code density
                        
                        # Group lines into chunks
                        chunks = [
                            '\n'.join(lines[i:i + chunk_size_lines]) 
                            for i in range(0, len(lines), chunk_size_lines)
                        ]
                        
                        # Filtering
                        for chunk in chunks:
                            if self._is_valid_code(chunk, lang):
                                valid_chunks.append(chunk)
                                
                    return {"clean_content": valid_chunks}

                # Apply map function across multiple CPU cores
                final_ds = ds.map(
                    process_and_chunk, 
                    batched=True, 
                    num_proc=self.num_proc,
                    desc=f"Processing {lang} files",
                    remove_columns=ds.column_names 
                ).with_format("python")
                
                print(f"Successfully processed {len(final_ds)} VALID chunks for {lang}")
                
                # 2. CREATE HIGH-PERFORMANCE DATALOADERS (Fixed Worker Logic)
                num_workers = min(4, max(0, self.num_proc // len(languages))) # Allow 0 for local testing
                
                dl = DataLoader(
                    final_ds, # type: ignore
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    # FIX: Conditionally apply multi-processing args to prevent crashes
                    prefetch_factor=4 if num_workers > 0 else None,               
                    persistent_workers=True if num_workers > 0 else False,         
                    pin_memory=False,                
                    drop_last=True     
                )

                # Apply map function across multiple CPU cores
                final_ds = ds.map(
                    process_and_chunk, 
                    batched=True, 
                    num_proc=self.num_proc,
                    desc=f"Processing {lang} files",
                    remove_columns=ds.column_names # Drop old raw columns to save RAM
                ).with_format("python")
                
                print(f"Successfully processed {len(final_ds)} VALID chunks for {lang}")
                
                # 2. CREATE HIGH-PERFORMANCE DATALOADERS
                num_workers = min(4, max(1, self.num_proc // len(languages))) 
                
                dl = DataLoader(
                    final_ds, # type: ignore
                    batch_size=batch_size, 
                    shuffle=True, 
                    num_workers=num_workers,
                    prefetch_factor=4,               # Increased since we have 64GB RAM
                    persistent_workers=True,         # CRITICAL: Keeps workers alive across epochs!
                    pin_memory=False,                # Strings cannot be pinned
                    drop_last=True     
                )
                
                self.dataloaders[lang] = InfiniteDataLoader(dl)
                
            except Exception as e:
                print(f"Failed to load {lang}: {e}")

    def sample_batch(self, lang: str, limit: int | None = None):
        """Fetches instantly from the background RAM cache."""
        if lang not in self.dataloaders:
            raise ValueError(f"Language {lang} not loaded.")
        
        # Fetch pre-batched list of strings
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
