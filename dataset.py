import os
import multiprocessing
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
    def __init__(self, data_dir="./data", batch_size=16, num_problems=2, min_solutions=5000):
        print(f"\nScanning CodeNet data from {data_dir}...")
        self.pairs = []
        
        if not os.path.exists(data_dir):
            print(f"Warning: {data_dir} not found. Creating a dummy dataset for testing.")
            self.pairs = [{"python": "def add(a, b):\n    return a + b", "c++": "int add(int a, int b) {\n    return a + b;\n}"}] * 100
        else:
            # Find all problem subdirectories (e.g., p03497)
            problem_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            
            selected_problems = []
            
            for p_id in problem_dirs:
                py_dir = os.path.join(data_dir, p_id, "Python")
                cpp_dir = os.path.join(data_dir, p_id, "C++")
                
                # Check if both language directories exist for this problem
                if os.path.isdir(py_dir) and os.path.isdir(cpp_dir):
                    # Get all file paths
                    py_files = [os.path.join(py_dir, f) for f in os.listdir(py_dir) if f.endswith('.py')]
                    cpp_files = [os.path.join(cpp_dir, f) for f in os.listdir(cpp_dir) if f.endswith('.cpp')]
                    
                    if len(py_files) >= min_solutions and len(cpp_files) >= min_solutions:
                        selected_problems.append({
                            "p_id": p_id,
                            "py_files": py_files,
                            "cpp_files": cpp_files
                        })
                        
                    if len(selected_problems) == num_problems:
                        break
            
            if not selected_problems:
                print(f"Warning: Could not find {num_problems} problems with >= {min_solutions} solutions. Using dummy data.")
                self.pairs = [{"python": "def add(a, b):\n    return a + b", "c++": "int add(int a, int b) {\n    return a + b;\n}"}] * 100
            else:
                for prob in selected_problems:
                    py_files = prob["py_files"]
                    cpp_files = prob["cpp_files"]
                    
                    # Take the minimum to form exact 1-to-1 semantic pairs
                    num_pairs = min(len(py_files), len(cpp_files))
                    print(f"Problem {prob['p_id']}: Found {len(py_files)} Python & {len(cpp_files)} C++ solutions. Creating {num_pairs} pairs.")
                    
                    for i in range(num_pairs):
                        try:
                            with open(py_files[i], 'r', encoding='utf-8', errors='ignore') as f_py:
                                py_code = f_py.read().strip()
                            with open(cpp_files[i], 'r', encoding='utf-8', errors='ignore') as f_cpp:
                                cpp_code = f_cpp.read().strip()
                                
                            # Only append if neither file was empty
                            if py_code and cpp_code:
                                self.pairs.append({"python": py_code, "c++": cpp_code})
                        except Exception as e:
                            continue

                print(f"Successfully loaded a total of {len(self.pairs)} semantic Python-C++ pairs.")

        # Convert to HuggingFace dataset for fast Arrow-backed dataloader integration
        ds = Dataset.from_list(self.pairs)
        
        # Allocate workers safely
        num_workers = min(4, max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", multiprocessing.cpu_count())) // 2))
        
        dl = DataLoader(
            ds, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
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
        print(f"\nUnpaired Dataset using {self.num_proc} CPU workers for preprocessing The Stack.")
        
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
