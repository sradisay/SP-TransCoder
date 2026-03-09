from datasets import load_dataset, Dataset
import random
import json
import re
import os

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


class XLCoSTSnippetDataset:
    def __init__(self, base_dir="data/pair_data_tok_1/C++-Python/"):
        self.pairs = []

        py_path = os.path.join(base_dir, "train-C++-Python-tok.py")
        cpp_path = os.path.join(base_dir, "train-C++-Python-tok.cpp")

        print(f"Loading and decoding paired data from:\n- {py_path}\n- {cpp_path}")

        try:
            with open(py_path, 'r', encoding='utf-8') as f_py, \
                    open(cpp_path, 'r', encoding='utf-8') as f_cpp:

                for py_line, cpp_line in zip(f_py, f_cpp):
                    py_code = self._decode_and_clean(py_line)
                    cpp_code = self._decode_and_clean(cpp_line)

                    if py_code and cpp_code:
                        self.pairs.append({
                            "python": py_code,
                            "c++": cpp_code
                        })

            print(f"Successfully loaded and decoded {len(self.pairs)} snippet pairs!")

        except FileNotFoundError as e:
            print(f"Error: Could not find the dataset files. {e}")
            self.pairs = [
                {
                    "python": "def add(a, b):\n    return a + b",
                    "c++": "int add(int a, int b) {\n    return a + b;\n}"
                }
            ]

    def _decode_and_clean(self, code_string):
        tokens = code_string.split()
        decoded_chars = []
        indent_level = 0
        is_new_line = True

        for token in tokens:
            if token == 'NEW_LINE':
                decoded_chars.append('\n')
                is_new_line = True
            elif token == 'INDENT':
                indent_level += 1
            elif token == 'DEDENT':
                indent_level = max(0, indent_level - 1)
            else:
                if is_new_line:
                    decoded_chars.append('    ' * indent_level)
                    is_new_line = False
                elif decoded_chars and decoded_chars[-1] != '\n':
                    decoded_chars.append(' ')
                decoded_chars.append(token)

        raw_decoded_string = "".join(decoded_chars).strip()

        cleaned_string = re.sub(r'\s+([,.:;\]\)\}])', r'\1', raw_decoded_string)
        cleaned_string = re.sub(r'([\[\(\{])\s+', r'\1', cleaned_string)
        cleaned_string = re.sub(r'\+\s+\+', '++', cleaned_string)
        cleaned_string = re.sub(r'-\s+-', '--', cleaned_string)

        return cleaned_string.strip()

    def sample_paired_batch(self, batch_size):
        if not self.pairs:
            return [""] * batch_size, [""] * batch_size

        batch = random.sample(self.pairs, min(batch_size, len(self.pairs)))

        py_batch = [item["python"] for item in batch]
        cpp_batch = [item["c++"] for item in batch]

        return py_batch, cpp_batch

class CPPDataset:
    def __init__(self, csv_file_path = ""):
        self.data = []

        try:
            self.data = Dataset.from_csv(csv_file_path).filter(lambda x: x["response"] < 250)
            self.data 
        except FileNotFoundError:
            print("File not found")
        
    def filtercpp(self, code):
        # filter code for include statements comments
        lines = code.split("\n")
        filter = []
        for line in lines:
            l = line.strip()
            if l.startswith("/*") or l.startswith("//") or l.startswith("#") or l.startswith("using") or l.startswith("import"):
                continue
            filter.append(l.strip())

        return "\n".join(filter)
            
        # data in self.data["response"]

class PythonDataset:
    def __init__(self, csv_file_path="", json_f="", file = ""):
        self.data = []

        try:
            dataset = Dataset.from_csv(csv_file_path).filter(lambda x: x["code_length"] < 250)
            r = open(json_f).read()
            if json_f:
                jr = json.loads(r)
                for i in dataset["python_solutions"]:
                    self.data.append("def "+ i.split("def")[1].strip())
                self.data.extend([i for i in jr if len(i) < 250][:9000])
            if file:
                i = 1
                fp = open(file)
                for code in fp:
                    if i < 9000 and len(code) < 250:
                        self.data.append(code)
                    i += 1

        except FileNotFoundError:
            print("File not found")
    
    def filtercpp(self, code):
        # filter code for include statements comments
        lines = code.split("\n")
        filter = []
        for line in lines:
            l = line.strip()
            if l.startswith("/*") or l.startswith("//") or l.startswith("#") or l.startswith("using") or l.startswith("import"):
                continue
            filter.append(l.strip())

        return "\n".join(filter)
