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
                
                valid_examples = []
                for content in ds["content"]:
                    clean_content = self._strip_headers(content)
                    if self._is_valid_code(clean_content, lang):
                        valid_examples.append(clean_content)
                        
                self.data[lang] = valid_examples
                print(f"Successfully loaded {len(self.data[lang])} VALID code examples for {lang}")
            except Exception as e:
                print(f"Failed to load {lang}: {e}")
                self.data[lang] = []

    def _strip_headers(self, text):
        """Removes top-of-file comments, licenses, and empty lines."""
        lines = text.split('\n')
        start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Skip empty lines, comments, and basic imports
            if stripped and not stripped.startswith(('#', '//', '/*', '*', '"""', "'''", 'import ', 'from ', '#include')):
                start_idx = i
                break
        return '\n'.join(lines[start_idx:])

    def _is_valid_code(self, text, lang):
        """Ensures the remaining chunk actually contains structural logic."""
        if not text or len(text) < 50: 
            return False
            
        if lang == "Python":
            return any(kw in text for kw in ["def ", "class ", "return ", "for "])
        elif lang == "C++":
            return any(kw in text for kw in ["int ", "void ", "class ", "return ", "for ("])
        return True

    def sample_batch(self, lang: str, batch_size: int):
        lang_data = self.data.get(lang, [])
        if not lang_data or len(lang_data) == 0:
            return [""] * batch_size
        return random.sample(lang_data, min(batch_size, len(lang_data)))
