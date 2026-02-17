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
                
                # FIX 1: Filter out license headers and comment-only files
                valid_examples = []
                for content in ds["content"]:
                    if self._is_valid_code(content, lang):
                        valid_examples.append(content)
                        
                self.data[lang] = valid_examples
                print(f"Successfully loaded {len(self.data[lang])} VALID code examples for {lang}")
            except Exception as e:
                print(f"Failed to load {lang}: {e}")
                self.data[lang] = []

    def _is_valid_code(self, text, lang):
        """Filters out files that are mostly comments, licenses, or too short."""
        if not text or len(text) < 30: 
            return False
            
        # Strip out common comment lines to see what's left
        lines = text.split('\n')
        code_lines = [l for l in lines if not l.strip().startswith(('#', '//', '*', '/*'))]
        clean_text = '\n'.join(code_lines)
        
        # Enforce language-specific structural keywords
        if lang == "Python":
            return any(kw in clean_text for kw in ["def ", "class ", "import ", "from "])
        elif lang == "C++":
            return any(kw in clean_text for kw in ["#include", "int ", "void ", "std::", "class "])
        return True

    def sample_batch(self, lang: str, batch_size: int):
        lang_data = self.data.get(lang, [])
        if not lang_data or len(lang_data) == 0:
            return [""] * batch_size
        return random.sample(lang_data, min(batch_size, len(lang_data)))
